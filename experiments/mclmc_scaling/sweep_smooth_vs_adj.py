"""Decisive: unadjusted MCLMC vs adjusted_mclmc_dynamic on SMOOTH targets.

Question (user 2026-06-17): is "always use adjusted_mclmc_dynamic for real samples"
the right default? The only place unadjusted can legitimately win is SMOOTH high-d
(its no-rejection O(d^1/4) edge). Both prior "adjusted worse" claims were confounded
(adjusted_mclmc at N=1 = MALA; or diagonal IMM). This runs the dynamic variant at the
GT IMM on the smooth panel, fair ess/grad accounting, vs unadjusted.

Decision rule: "always adjusted_dynamic" is a good default if, on smooth targets,
adj_dyn ess/grad is within ~2x of unadjusted AND its bias is <= unadjusted everywhere.

Smoke:  JAX_PLATFORM_NAME=cpu uv run python sweep_smooth_vs_adj.py --smoke
Full:   JAX_PLATFORM_NAME=cpu uv run python sweep_smooth_vs_adj.py
"""

import os
import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.chdir(os.path.join(_HERE, "..", ".."))

from gt_imm import gt_from_draws
from run_fixed_imm import run_adj_dynamic_fixed_imm, run_mclmc_fixed_imm

SMOKE = "--smoke" in sys.argv
# Smooth panel, ascending d. irt_1pl (d=500) is the crux (high-d smooth).
FULL_PANEL = ["mvn_10", "german_credit", "ill_cond_50", "eight_schools_ncp", "irt_1pl"]
SMOKE_PANEL = ["mvn_10", "german_credit"]
PANEL = SMOKE_PANEL if SMOKE else FULL_PANEL

if SMOKE:
    N_WARMUP, N_SAMPLES, NUM_CHAINS = 200, 300, 2
else:
    N_WARMUP, N_SAMPLES, NUM_CHAINS = 2000, 2000, 4
SEED = 20260617

print(
    f"sweep_smooth_vs_adj | smoke={SMOKE} | nw={N_WARMUP} ns={N_SAMPLES} chains={NUM_CHAINS}"
)
print("Panel:", PANEL)
hdr = (
    f"{'model':18s} {'variant':9s} {'d':>4s} {'step':>7s} {'L':>7s} "
    f"{'max_bias':>9s} {'min_ess':>8s} {'ess/grad':>10s} {'acc':>6s} {'nstep':>6s} {'wall':>6s}"
)
print(hdr)
print("-" * len(hdr))
sys.stdout.flush()

results = {}
for m in PANEL:
    try:
        imm, gt_var, gt_mean, d = gt_from_draws(m)
    except Exception as exc:
        print(f"{m:18s}  ERROR loading GT: {exc}")
        sys.stdout.flush()
        continue

    t0 = time.time()
    r_u = run_mclmc_fixed_imm(
        m,
        imm,
        n_warmup=N_WARMUP,
        n_samples=N_SAMPLES,
        num_chains=NUM_CHAINS,
        seed=SEED,
        gt_mean=gt_mean,
        gt_var=gt_var,
    )
    w_u = time.time() - t0
    print(
        f"{m:18s} {'mclmc':9s} {d:>4d} {r_u['step_size']:>7.2f} {r_u['L']:>7.2f} "
        f"{r_u['max_bias']:>9.4f} {r_u['min_bulk_ess']:>8.1f} {r_u['ess_per_grad']:>10.6f} "
        f"{'-':>6s} {'1':>6s} {w_u:>5.1f}s"
    )
    sys.stdout.flush()

    t1 = time.time()
    r_a = run_adj_dynamic_fixed_imm(
        m,
        imm,
        n_warmup=N_WARMUP,
        n_samples=N_SAMPLES,
        num_chains=NUM_CHAINS,
        seed=SEED,
        adj_target=0.9,
        step_scale=0.55,
        gt_mean=gt_mean,
        gt_var=gt_var,
    )
    w_a = time.time() - t1
    print(
        f"{m:18s} {'adj_dyn':9s} {d:>4d} {r_a['step_size']:>7.2f} {r_a['L']:>7.2f} "
        f"{r_a['max_bias']:>9.4f} {r_a['min_bulk_ess']:>8.1f} {r_a['ess_per_grad']:>10.6f} "
        f"{r_a['acceptance_rate']:>6.3f} {r_a['n_steps_median']:>6.0f} {w_a:>5.1f}s"
    )
    sys.stdout.flush()
    results[m] = (r_u, r_a, d)
    print()

print("=" * len(hdr))
print("VERDICT TABLE: ess/grad ratio (adj_dyn / unadj) and bias comparison")
print(
    f"{'model':18s} {'d':>4s} {'unadj e/g':>10s} {'adj e/g':>10s} {'ratio':>7s} "
    f"{'unadj bias':>10s} {'adj bias':>9s} {'verdict':>20s}"
)
for m, (r_u, r_a, d) in results.items():
    ratio = r_a["ess_per_grad"] / max(r_u["ess_per_grad"], 1e-30)
    bias_ok = r_a["max_bias"] <= r_u["max_bias"] * 1.05
    eff_ok = ratio >= 0.5
    if bias_ok and eff_ok:
        v = "adj_dyn DEFAULT ok"
    elif bias_ok and not eff_ok:
        v = f"adj costs {1/ratio:.1f}x grad"
    else:
        v = "unadj better"
    print(
        f"{m:18s} {d:>4d} {r_u['ess_per_grad']:>10.6f} {r_a['ess_per_grad']:>10.6f} "
        f"{ratio:>7.2f} {r_u['max_bias']:>10.4f} {r_a['max_bias']:>9.4f} {v:>20s}"
    )
print("\nDONE")
