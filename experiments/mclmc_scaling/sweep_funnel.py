"""Funnel panel: unadjusted MCLMC vs adjusted_mclmc_dynamic at GT-dense IMM.

Quantifies whether the MH correction in ``adjusted_mclmc_dynamic`` gives
*graceful degradation* on funnel/heavy-tail geometry — lower bias than
unadjusted MCLMC, divergences flagged (``is_divergent``) rather than
silently biasing.

Context (catalog mclmc-routing-taxonomy.md §4):
  - Category C (heavy-tail: horseshoe) → adjusted_mclmc_dynamic (REVIEW)
  - Category D (hierarchical funnels: neals_funnel, irt_2pl) → NUTS (MCLMC honest-null)
  This sweep QUANTIFIES the honest-null claim: does adjusted_dynamic reduce
  the EEVPD/bias vs unadjusted?  Is the routing boundary principled?

Panel definition:
  Full: neals_funnel, eight_schools_ncp, irt_2pl, stoch_vol, horseshoe
  Smoke (--smoke): neals_funnel, eight_schools_ncp

Per (model × variant) output:
  max_bias, mean_bias, min_ess, ess/grad, acc (adj only), div_rate, eevpd, wall

Smoke usage:
  JAX_PLATFORM_NAME=cpu uv run python sweep_funnel.py --smoke

Full panel (TL runs this):
  JAX_PLATFORM_NAME=cpu uv run python sweep_funnel.py
"""

import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Add experiment dir to sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Change to tuningfork repo root so catalog paths resolve correctly
_REPO_ROOT = os.path.join(_HERE, "..", "..")
os.chdir(_REPO_ROOT)

from gt_imm import gt_from_draws
from run_fixed_imm import run_adj_dynamic_fixed_imm, run_mclmc_fixed_imm

# ---------------------------------------------------------------------------
# Panel definition
# ---------------------------------------------------------------------------

# Full funnel panel
FULL_PANEL = [
    "neals_funnel",  # d=10, v~N(0,9), theta_i|v~N(0,exp(v)), clean funnel
    "eight_schools_ncp",  # d=10, hierarchical, mild funnel (NCP)
    "irt_2pl",  # d=144, hierarchical, correlated funnel
    "stoch_vol",  # d=503, AR(1) banded, funnel-like
    "horseshoe",  # d=204, heavy-tail, Category C (adj_dynamic REVIEW)
]

# Smoke: only the two fast small models
SMOKE_PANEL = ["neals_funnel", "eight_schools_ncp"]

SMOKE = "--smoke" in sys.argv
PANEL = SMOKE_PANEL if SMOKE else FULL_PANEL

if SMOKE:
    N_WARMUP = 200
    N_SAMPLES = 300
    NUM_CHAINS = 2
else:
    N_WARMUP = 2000
    N_SAMPLES = 3000
    NUM_CHAINS = 4

SEED = 20260616

# Scaling law prefactors (from S3 isotropic canonical run) for context
A_REF = 1.22
B_REF = 0.85

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

print(
    f"sweep_funnel | smoke={SMOKE} | n_warmup={N_WARMUP} n_samples={N_SAMPLES} "
    f"num_chains={NUM_CHAINS} seed={SEED}"
)
print("Panel:", PANEL)
print()
print("Variants: unadjusted MCLMC (mclmc) vs adjusted_mclmc_dynamic (adj_dyn)")
print("Both run at GT-dense IMM (k=d from GT draws).")
print(
    "adj_dyn: unadjusted step x0.55 init (§7 rule), frac_tune2=0, "
    "target_accept=0.9, real is_divergent flag."
)
print()

# ---------------------------------------------------------------------------
# Column header
# ---------------------------------------------------------------------------

_hdr = (
    f"{'model':22s} {'variant':12s} {'d':>5s} "
    f"{'step':>8s} {'L':>8s} {'max_bias':>9s} {'mean_bias':>9s} "
    f"{'min_ess':>8s} {'ess/grad':>10s} {'acc':>6s} "
    f"{'div_rate':>9s} {'eevpd':>10s} {'wall':>6s}"
)
print(_hdr)
print("-" * len(_hdr))
sys.stdout.flush()

results = []


def _fmt_row(model_name, variant, d, r, wall):
    """Format a result row."""
    acc_str = f"{r.get('acceptance_rate', float('nan')):6.3f}"
    eevpd_str = (
        f"{r['eevpd']:10.2e}"
        if not math.isnan(r.get("eevpd", float("nan")))
        else f"{'nan':>10s}"
    )
    return (
        f"{model_name:22s} {variant:12s} {d:>5d} "
        f"{r['step_size']:>8.3f} {r['L']:>8.3f} "
        f"{r['max_bias']:>9.4f} {r['mean_bias']:>9.4f} "
        f"{r['min_bulk_ess']:>8.1f} {r['ess_per_grad']:>10.6f} "
        f"{acc_str} {r['div_rate']:>9.4f} {eevpd_str} {wall:>6.1f}s"
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for model_name in PANEL:

    # ----------------------------------------------------------------
    # Step 1: load GT draws, build dense IMM
    # ----------------------------------------------------------------
    try:
        imm, gt_var, gt_mean, d = gt_from_draws(model_name)
    except FileNotFoundError as exc:
        print(f"{model_name:22s}  ERROR loading GT draws: {exc}")
        sys.stdout.flush()
        continue
    except Exception as exc:
        print(f"{model_name:22s}  ERROR: {exc}")
        sys.stdout.flush()
        continue

    # ----------------------------------------------------------------
    # Variant 1: unadjusted MCLMC
    # ----------------------------------------------------------------
    t0 = time.time()
    try:
        r_unadj = run_mclmc_fixed_imm(
            model_name,
            imm,
            n_warmup=N_WARMUP,
            n_samples=N_SAMPLES,
            num_chains=NUM_CHAINS,
            seed=SEED,
            gt_mean=gt_mean,
            gt_var=gt_var,
        )
    except Exception as exc:
        r_unadj = None
        print(f"{model_name:22s} {'mclmc':12s}  ERROR: {exc}")
        sys.stdout.flush()

    wall_unadj = round(time.time() - t0, 1)

    if r_unadj is not None:
        row_unadj = {
            "model": model_name,
            "variant": "mclmc",
            "d": d,
            **r_unadj,
            "wall_s": wall_unadj,
        }
        results.append(row_unadj)
        print(_fmt_row(model_name, "mclmc", d, r_unadj, wall_unadj))
        sys.stdout.flush()

    # ----------------------------------------------------------------
    # Variant 2: adjusted_mclmc_dynamic
    # ----------------------------------------------------------------
    t1 = time.time()
    try:
        r_adj_dyn = run_adj_dynamic_fixed_imm(
            model_name,
            imm,
            n_warmup=N_WARMUP,
            n_samples=N_SAMPLES,
            num_chains=NUM_CHAINS,
            seed=SEED,
            adj_target=0.9,
            step_scale=0.55,  # catalog §7: validated on horseshoe, ~94% acceptance
            gt_mean=gt_mean,
            gt_var=gt_var,
        )
    except Exception as exc:
        r_adj_dyn = None
        print(f"{model_name:22s} {'adj_dyn':12s}  ERROR: {exc}")
        sys.stdout.flush()

    wall_adj_dyn = round(time.time() - t1, 1)

    if r_adj_dyn is not None:
        row_adj_dyn = {
            "model": model_name,
            "variant": "adj_dyn",
            "d": d,
            **r_adj_dyn,
            "wall_s": wall_adj_dyn,
        }
        results.append(row_adj_dyn)
        print(_fmt_row(model_name, "adj_dyn", d, r_adj_dyn, wall_adj_dyn))
        sys.stdout.flush()

    print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * len(_hdr))
print("FUNNEL SWEEP SUMMARY")
print("  bias: |Var_mcmc - gt_var| / gt_var  (lower = more accurate)")
print("  div_rate: NaN proxy (mclmc) or is_divergent (adj_dyn)")
print("  eevpd nan = adjusted sampler (metric not meaningful)")
print()

# Group by model
models_done = []
for r in results:
    if r["model"] not in models_done:
        models_done.append(r["model"])

for m in models_done:
    rows_m = [r for r in results if r["model"] == m]
    print(f"  {m}:")
    for row in rows_m:
        flags = []
        if row.get("div_rate", 0) > 0.01:
            flags.append("HIGH-DIV")
        if (
            not math.isnan(row.get("eevpd", float("nan")))
            and row.get("eevpd", 0) > 5e-3
        ):
            flags.append("HIGH-EEVPD")
        if row.get("max_bias", 0) > 0.1:
            flags.append("HIGH-BIAS")
        flag_str = " [" + ",".join(flags) + "]" if flags else ""
        print(
            f"    {row['variant']:12s}  max_bias={row['max_bias']:.4f}  "
            f"min_ess={row['min_bulk_ess']:.1f}  "
            f"ess/grad={row['ess_per_grad']:.4e}  "
            f"acc={row.get('acceptance_rate', float('nan')):.3f}  "
            f"div={row['div_rate']:.4f}{flag_str}"
        )
    print()

print()
if SMOKE:
    print("SMOKE COMPLETE. TL runs full panel + NUTS reference.")
else:
    print("FULL FUNNEL PANEL COMPLETE. TL adds NUTS reference.")
