"""S2 smoke test (§4.6 pre-flight, tiny N).

Sub-study S2: optimal step size at fixed IMM = GT-dense (k=d).

Smokes BOTH S2 modes at tiny N (n_warmup=20, n_samples=50, num_chains=2)
on ill_cond_50 with the GT-dense IMM (gt_lrd_imm(Sigma, k=50)):

  (a) Unadjusted MCLMC at 3 fixed step values (fixed_step_size + fixed_L):
      tests the skip-tuning path added in the S2 extension.

  (b) Adjusted MCLMC (tuned): tests the adjusted=True path.

Asserts: exit 0, all mandatory fields populated, no NaN in key metrics.

Usage:
    JAX_PLATFORM_NAME=cpu PYTHONUNBUFFERED=1 uv run python sweep_s2_smoke.py
"""

import os
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gt_imm import gt_cov, gt_lrd_imm
from run_fixed_imm import run_mclmc_fixed_imm

# ---------------------------------------------------------------------------
# Smoke parameters (tiny N per §4.6)
# ---------------------------------------------------------------------------
MODEL = "ill_cond_50"
NW = 20
NS = 50
NC = 2
SEED = 20260616

# GT-dense IMM: k = d (full rank -> kappa_eff = 1.0)
Sigma, _ = gt_cov(MODEL)
d = Sigma.shape[0]
imm_dense = gt_lrd_imm(Sigma, k=d)

print("=" * 70)
print(
    f"sweep_s2_smoke | model={MODEL} d={d} n_warmup={NW} n_samples={NS} num_chains={NC}"
)
print("=" * 70)
sys.stdout.flush()

# ---------------------------------------------------------------------------
# (a) Unadjusted MCLMC — fixed-step grid (3 values)
# ---------------------------------------------------------------------------
# Use small step values appropriate for ill_cond_50 (oracle ~8.0 at k=50).
# At smoke N these are too small/medium/large on purpose; we just verify the
# skip-tuning code path runs and returns valid fields.
STEP_VALUES = [2.0, 5.0, 10.0]
# L = step * some_factor to avoid DA-ceiling issues
L_VALUES = [s * 3.0 for s in STEP_VALUES]

REQUIRED_KEYS_UNADJ = {
    "step_size",
    "L",
    "max_bias",
    "mean_bias",
    "min_bulk_ess",
    "ess_per_grad",
    "eevpd",
    "div_rate",
    "n_warmup_grads",
    "n_sampling_grads",
    "total_grads",
}

print("\n--- (a) Unadjusted fixed-step sweep ---")
print(
    f"{'step':>8} {'L':>8} {'max_bias':>10} {'min_ess':>10} "
    f"{'ess/grad':>12} {'n_wup_grads':>12}"
)

all_ok = True
for step, L in zip(STEP_VALUES, L_VALUES):
    result = run_mclmc_fixed_imm(
        MODEL,
        imm_dense,
        n_warmup=NW,
        n_samples=NS,
        num_chains=NC,
        seed=SEED,
        adjusted=False,
        fixed_step_size=step,
        fixed_L=L,
    )

    # Check all required keys present
    missing = REQUIRED_KEYS_UNADJ - set(result.keys())
    if missing:
        print(f"  FAIL step={step}: missing keys {missing}")
        all_ok = False
        continue

    # Check warmup grads = 0 (skip-tuning path)
    if result["n_warmup_grads"] != 0:
        print(
            f"  FAIL step={step}: n_warmup_grads={result['n_warmup_grads']} (expected 0 in fixed-step mode)"
        )
        all_ok = False

    # Check step_size and L match what we provided
    if abs(result["step_size"] - step) > 1e-10:
        print(
            f"  FAIL step={step}: returned step_size={result['step_size']:.6f} != {step}"
        )
        all_ok = False

    # Check no NaN in key fields (bias and ESS can be finite even if large at tiny N)
    for key in ("max_bias", "mean_bias", "min_bulk_ess", "ess_per_grad"):
        if np.isnan(result[key]):
            print(f"  FAIL step={step}: {key} is NaN")
            all_ok = False

    print(
        f"  step={step:6.1f} L={L:6.1f} | "
        f"max_bias={result['max_bias']:10.4f} min_ess={result['min_bulk_ess']:10.2f} "
        f"ess/grad={result['ess_per_grad']:12.6f} n_wup_grads={result['n_warmup_grads']:12d}"
    )
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# (b) Adjusted MCLMC — tuned (no fixed step; use the warmup tuner)
# ---------------------------------------------------------------------------
print("\n--- (b) Adjusted MCLMC (tuned) ---")

REQUIRED_KEYS_ADJ = REQUIRED_KEYS_UNADJ | {"acceptance_rate", "n_steps_median"}

result_adj = run_mclmc_fixed_imm(
    MODEL,
    imm_dense,
    n_warmup=NW,
    n_samples=NS,
    num_chains=NC,
    seed=SEED,
    adjusted=True,
    floor_factor=1.5,
)

missing_adj = REQUIRED_KEYS_ADJ - set(result_adj.keys())
if missing_adj:
    print(f"  FAIL adjusted: missing keys {missing_adj}")
    all_ok = False

# Check no NaN in key fields (eevpd is nan by design for adjusted)
for key in (
    "max_bias",
    "mean_bias",
    "min_bulk_ess",
    "ess_per_grad",
    "acceptance_rate",
    "n_steps_median",
):
    if np.isnan(result_adj.get(key, float("nan"))):
        print(f"  FAIL adjusted: {key} is NaN")
        all_ok = False

print(f"  step={result_adj['step_size']:.4f} L={result_adj['L']:.4f}")
print(
    f"  max_bias={result_adj['max_bias']:.4f}  mean_bias={result_adj['mean_bias']:.4f}"
)
print(
    f"  min_ess={result_adj['min_bulk_ess']:.2f}  ess/grad={result_adj['ess_per_grad']:.6f}"
)
print(f"  acceptance_rate={result_adj['acceptance_rate']:.4f}")
print(f"  n_steps_median={result_adj['n_steps_median']:.1f}")
print(
    f"  n_warmup_grads={result_adj['n_warmup_grads']}  total_grads={result_adj['total_grads']}"
)

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
if all_ok:
    print("SMOKE PASSED — all assertions OK, no NaN in mandatory fields.")
else:
    print("SMOKE FAILED — see above for details.")
print("=" * 70)
sys.stdout.flush()

sys.exit(0 if all_ok else 1)
