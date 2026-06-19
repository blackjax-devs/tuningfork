"""S3-real: test whether the sqrt-d scaling law holds on real posteriors.

Canonical isotropic result (S3): MCLMC optimal step ≈ 1.22·√d and L ≈ 0.85·√d.
This script runs the same fixed-GT-dense-IMM MCLMC harness on real models and
checks whether the EEVPD-tuned (step, L) fall near those prefactors.

For each model in the panel:
  1. Load GT draws from catalog/<model>/groundtruth_samples/blackjax/draws.npz
     (draws are in UNCONSTRAINED space — confirmed by signed tau in eight_schools_ncp).
  2. Build the GT-dense IMM via gt_from_draws(model, k=d).
  3. Run unadjusted MCLMC with the fixed IMM (tuned step + L).
  4. Print: d, step, L, max_bias, min_ess, ess/grad, eevpd, div,
            step/(1.22*sqrt(d)), L/(0.85*sqrt(d))

Panel:
  Smooth/global-metric class first (expect law to hold):
    german_credit, logistic_synthetic, eight_schools_ncp
  Higher-dimensional:
    irt_2pl, irt_1pl, stoch_vol
  Funnels / heavy-tail (negative controls — expect law to FAIL or ess/grad to collapse):
    horseshoe

Usage:
  # Smoke test (tiny N, 2 models, fast):
  JAX_PLATFORM_NAME=cpu uv run python sweep_s3_real.py --smoke

  # Full panel (TL runs this, not SWE):
  JAX_PLATFORM_NAME=cpu uv run python sweep_s3_real.py
"""

import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Add experiment dir to sys.path so gt_imm / run_fixed_imm imports work
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gt_imm import gt_from_draws
from run_fixed_imm import run_mclmc_fixed_imm

# ---------------------------------------------------------------------------
# Panel definition
# ---------------------------------------------------------------------------

FULL_PANEL = [
    "german_credit",  # d=26,  GLM, mild-corr, smooth
    "logistic_synthetic",  # d=3,   tiny-d control
    "eight_schools_ncp",  # d=10,  hierarchical, mild funnel (NCP)
    "irt_2pl",  # d=144, hierarchical, correlated funnel
    "irt_1pl",  # d=500, large-d
    "stoch_vol",  # d=503, AR(1) banded, diagonal-useless
    "horseshoe",  # d=204, heavy-tail, structural negative control
]

SMOKE_PANEL = ["german_credit", "eight_schools_ncp"]

SMOKE = "--smoke" in sys.argv

PANEL = SMOKE_PANEL if SMOKE else FULL_PANEL

# Scaling law prefactors from the isotropic S3 canonical run
A_REF = 1.22  # step ≈ 1.22·√d
B_REF = 0.85  # L    ≈ 0.85·√d

if SMOKE:
    N_WARMUP = 200
    N_SAMPLES = 300
    NUM_CHAINS = 2
else:
    N_WARMUP = 2000
    N_SAMPLES = 3000
    NUM_CHAINS = 4

SEED = 20260616

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

print(
    f"S3-real | smoke={SMOKE} | n_warmup={N_WARMUP} n_samples={N_SAMPLES} "
    f"num_chains={NUM_CHAINS} seed={SEED}"
)
print(f"Law: step ≈ {A_REF}·√d,  L ≈ {B_REF}·√d  (from isotropic S3)")
print()

_hdr = (
    f"{'model':22s} {'d':>5s} {'step':>8s} {'L':>8s} {'max_bias':>9s} "
    f"{'min_ess':>8s} {'ess/grad':>10s} {'eevpd':>10s} {'div':>5s} "
    f"{'step/1.22√d':>12s} {'L/0.85√d':>10s} {'wall':>6s}"
)
print(_hdr)
print("-" * len(_hdr))

results = []

for model_name in PANEL:
    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: load GT draws, build dense IMM
    # ------------------------------------------------------------------
    try:
        imm, gt_var, gt_mean, d = gt_from_draws(model_name)
        # gt_from_draws returns k=None -> dense (k=d) by default
    except Exception as exc:
        print(f"{model_name:22s}  ERROR loading GT draws: {exc}")
        sys.stdout.flush()
        continue

    # ------------------------------------------------------------------
    # Step 2: run MCLMC with fixed GT-dense IMM
    # ------------------------------------------------------------------
    try:
        r = run_mclmc_fixed_imm(
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
        print(f"{model_name:22s}  ERROR running MCLMC: {exc}")
        sys.stdout.flush()
        continue

    wall = round(time.time() - t0, 1)

    # ------------------------------------------------------------------
    # Step 3: compute √d-law residuals
    # ------------------------------------------------------------------
    sqrt_d = math.sqrt(d)
    step = r["step_size"]
    L = r["L"]
    step_residual = step / (A_REF * sqrt_d)  # ≈ 1.0 if law holds
    L_residual = L / (B_REF * sqrt_d)  # ≈ 1.0 if law holds

    row = {
        "model": model_name,
        "d": d,
        "step": step,
        "L": L,
        "max_bias": r["max_bias"],
        "min_ess": r["min_bulk_ess"],
        "ess_per_grad": r["ess_per_grad"],
        "eevpd": r["eevpd"],
        "div": r["div_rate"],
        "step_over_law": step_residual,
        "L_over_law": L_residual,
        "wall_s": wall,
    }
    results.append(row)

    print(
        f"{model_name:22s} {d:>5d} {step:>8.3f} {L:>8.3f} "
        f"{r['max_bias']:>9.4f} {r['min_bulk_ess']:>8.1f} "
        f"{r['ess_per_grad']:>10.6f} {r['eevpd']:>10.2e} "
        f"{r['div_rate']:>5.3f} {step_residual:>12.3f} {L_residual:>10.3f} {wall:>6.1f}s"
    )
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * len(_hdr))
print("SUMMARY: step/(1.22√d) and L/(0.85√d) ≈ 1.0 means the law holds.")
for row in results:
    flag = ""
    if row.get("div", 0) > 0.01:
        flag += " [HIGH-DIV]"
    if row.get("eevpd", 0) > 5e-3:
        flag += " [HIGH-EEVPD]"
    if row.get("max_bias", 0) > 0.1:
        flag += " [HIGH-BIAS]"
    print(
        f"  {row['model']:22s}  step_res={row['step_over_law']:.3f}  "
        f"L_res={row['L_over_law']:.3f}  ess/grad={row['ess_per_grad']:.4e}{flag}"
    )
print()
if SMOKE:
    print("SMOKE COMPLETE (tiny N). TL runs the full panel.")
else:
    print("FULL PANEL COMPLETE.")
