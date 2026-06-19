"""Smoke test for the full S1 flow (AGENT_CHECKLIST §4.6).

Runs at tiny N to verify every code path before any real-N sweep:
  - gt_imm: gt_cov, gt_lrd_imm, kappa_eff — all correctness gates
  - run_fixed_imm: end-to-end with tiny n_warmup/n_samples/num_chains
  - Both models (ill_cond_50 and mvn_10)
  - k ∈ {0, 5, DIM} for each model

MANDATORY assertions:
  1. κ_eff correctness gates (from gt_imm):
       ill_cond_50 k=0 → ≈863; k=d → 1.0; mvn_10 all k → 1.0
  2. All output-dict fields present and non-NaN
  3. ESS basis declared (arviz bulk)
  4. Exit 0 on success

Tiny-N parameters: n_warmup=5, n_samples=20, num_chains=2
These are the AGENT_CHECKLIST §4.6 smoke values.
"""

import math
import os
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

# Add experiment dir to sys.path so relative imports work
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from gt_imm import gt_cov, gt_lrd_imm, kappa_eff, kappa_eff_table
from run_fixed_imm import run_mclmc_fixed_imm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_WARMUP = 5
N_SAMPLES = 20
NUM_CHAINS = 2
SEED = 0

# Required output keys
REQUIRED_KEYS = {
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

# ESS basis declaration (mandatory per study protocol)
ESS_BASIS = "arviz.ess(method='bulk')"

print("=" * 70)
print("S1 smoke test — AGENT_CHECKLIST §4.6")
print(f"n_warmup={N_WARMUP}, n_samples={N_SAMPLES}, num_chains={NUM_CHAINS}")
print(f"ESS basis declared: {ESS_BASIS}")
print("=" * 70)


# ---------------------------------------------------------------------------
# Section 1: κ_eff correctness gates
# ---------------------------------------------------------------------------

print("\n--- Section 1: κ_eff correctness gates ---")

# Gate 1a: ill_cond_50 k=0 → ≈863
Sigma_ic, _ = gt_cov("ill_cond_50")
d_ic = Sigma_ic.shape[0]
imm_ic_diag = gt_lrd_imm(Sigma_ic, k=0)
keff_ic_diag = kappa_eff(Sigma_ic, imm_ic_diag)
print(f"  ill_cond_50 k=0 (diagonal):  κ_eff = {keff_ic_diag:.3f}  (expected ≈863)")
assert (
    abs(keff_ic_diag - 863) < 30
), f"GATE 1a FAILED: κ_eff={keff_ic_diag:.1f}, expected ≈863 (±30)"
print("  GATE 1a PASS ✓")

# Gate 1b: ill_cond_50 k=d → 1.0
imm_ic_dense = gt_lrd_imm(Sigma_ic, k=d_ic)
keff_ic_dense = kappa_eff(Sigma_ic, imm_ic_dense)
print(f"  ill_cond_50 k=d (dense):     κ_eff = {keff_ic_dense:.8f}  (expected = 1.0)")
assert (
    abs(keff_ic_dense - 1.0) < 1e-4
), f"GATE 1b FAILED: κ_eff={keff_ic_dense:.6f}, expected = 1.0 (±1e-4)"
print("  GATE 1b PASS ✓")

# Gate 1c: mvn_10 all k → 1.0
Sigma_mv, _ = gt_cov("mvn_10")
d_mv = Sigma_mv.shape[0]
for k in [0, d_mv // 2, d_mv]:
    imm_mv = gt_lrd_imm(Sigma_mv, k)
    keff_mv = kappa_eff(Sigma_mv, imm_mv)
    print(
        f"  mvn_10 k={k:2d}:                  κ_eff = {keff_mv:.8f}  (expected = 1.0)"
    )
    assert (
        abs(keff_mv - 1.0) < 1e-6
    ), f"GATE 1c FAILED: mvn_10 k={k} κ_eff={keff_mv:.8f}, expected = 1.0 (±1e-6)"
print("  GATE 1c PASS ✓")


# ---------------------------------------------------------------------------
# Section 2: run_mclmc_fixed_imm — end-to-end at tiny N
# ---------------------------------------------------------------------------

print("\n--- Section 2: end-to-end run_mclmc_fixed_imm ---")

# k values to test per model
k_values_ic = [0, 5, d_ic]
k_values_mv = [0, d_mv // 2, d_mv]

MODELS = [
    ("ill_cond_50", Sigma_ic, d_ic, k_values_ic),
    ("mvn_10", Sigma_mv, d_mv, k_values_mv),
]

all_results = {}

for model_name, Sigma, d, k_vals in MODELS:
    print(f"\n  Model: {model_name} (d={d})")
    all_results[model_name] = {}

    for k in k_vals:
        imm = gt_lrd_imm(Sigma, k)
        struct = "diagonal" if k == 0 else ("dense" if k == d else f"low-rank(k={k})")
        print(f"    k={k:2d} ({struct}) ...", end=" ", flush=True)

        result = run_mclmc_fixed_imm(
            model_name,
            imm,
            n_warmup=N_WARMUP,
            n_samples=N_SAMPLES,
            num_chains=NUM_CHAINS,
            seed=SEED,
        )

        # Assert all required keys present
        missing = REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

        # Assert no NaN values in numeric fields
        for key in [
            "step_size",
            "L",
            "max_bias",
            "mean_bias",
            "min_bulk_ess",
            "ess_per_grad",
            "eevpd",
        ]:
            val = result[key]
            assert not math.isnan(val), f"NaN in result['{key}'] for {model_name} k={k}"
            assert not math.isinf(val), f"Inf in result['{key}'] for {model_name} k={k}"

        # Assert grad counts are positive and consistent
        assert (
            result["n_warmup_grads"] == 2 * N_WARMUP * NUM_CHAINS
        ), f"warmup grad count wrong: {result['n_warmup_grads']}"
        assert (
            result["n_sampling_grads"] == 2 * N_SAMPLES * NUM_CHAINS
        ), f"sampling grad count wrong: {result['n_sampling_grads']}"
        assert (
            result["total_grads"]
            == result["n_warmup_grads"] + result["n_sampling_grads"]
        )

        # Assert step_size > 0 (adaptation didn't collapse)
        assert result["step_size"] > 0, f"step_size = {result['step_size']}"
        assert result["L"] > 0, f"L = {result['L']}"

        all_results[model_name][k] = result
        print(
            f"OK  (step={result['step_size']:.3f}, L={result['L']:.3f}, "
            f"bias_max={result['max_bias']:.3f})"
        )


# ---------------------------------------------------------------------------
# Section 3: κ_eff table printout (both models)
# ---------------------------------------------------------------------------

print("\n--- Section 3: κ_eff(k) tables ---")

print(f"\n  ill_cond_50 (d={d_ic}):")
rows_ic = kappa_eff_table("ill_cond_50")
print(f"  {'k':>4}  {'structure':<22}  {'κ_eff':>10}")
print(f"  {'-'*4}  {'-'*22}  {'-'*10}")
for r in rows_ic:
    print(f"  {r['k']:>4}  {r['structure']:<22}  {r['kappa_eff']:>10.4f}")

print(f"\n  mvn_10 (d={d_mv}):")
rows_mv = kappa_eff_table("mvn_10")
print(f"  {'k':>4}  {'structure':<22}  {'κ_eff':>10}")
print(f"  {'-'*4}  {'-'*22}  {'-'*10}")
for r in rows_mv:
    print(f"  {r['k']:>4}  {r['structure']:<22}  {r['kappa_eff']:>10.4f}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("SMOKE TEST PASSED — all sections complete, no NaN/missing fields")
print(f"ESS basis: {ESS_BASIS}")
print("Correctness gates: ill_cond_50 k=0→863, k=d→1.0; mvn_10→1.0 [ALL PASS]")
print("=" * 70)

sys.exit(0)
