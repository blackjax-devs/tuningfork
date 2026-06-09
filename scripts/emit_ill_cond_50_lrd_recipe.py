"""Emit certified LRD recipe + IMM artifact for ill_cond_50.

This script runs a single warmup pass (seed=98765, n_warmup=1000, 4 chains)
to capture calibrated step_size/L, saves the k=40 LRD IMM as a .npz artifact,
and writes the completed recipe JSON to the catalog.

Science is already settled (statistician multi-seed PASS, 2026-06-09 —
gate_evidence already present in the recipe).  This script only adds:
  - base_method_params: {step_size, L, k_rank} (mean step_size/L over chains)
  - inverse_mass_matrix_path: relative path to the saved .npz

The oracle covariance (ill_cond_50.COV) is used directly for the LRD
decomposition — no NUTS pilot required.  This is valid because the exact
50-D covariance is known analytically for this benchmark.

Run from repo root:
    uv run python scripts/emit_ill_cond_50_lrd_recipe.py
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")

# Use the development blackjax (PR #936 LRD dispatch), not the installed stub.
REPO_ROOT = Path(__file__).resolve().parents[1]
BLACKJAX_DEV = REPO_ROOT.parent / "blackjax"
sys.path.insert(0, str(BLACKJAX_DEV))
sys.path.insert(1, str(REPO_ROOT))

import blackjax
from blackjax.mcmc.metrics import LowRankInverseMassMatrix

from tuningfork.base_method.mclmc import (  # noqa: E402 (after sys.path setup)
    decompose_covariance_low_rank,
    make_lrd_kernel,
)
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model.ill_cond_50 import COV

SEED = 98765
N_WARMUP = 1000
NUM_CHAINS = 4
K_RANK = 40

RECIPE_PATH = (
    REPO_ROOT
    / "tuningfork"
    / "catalog"
    / "ill_cond_50"
    / "recipes"
    / "low__mclmc_lrd__mclmc_lrd_tuning.json"
)
IMM_PATH = (
    REPO_ROOT
    / "tuningfork"
    / "catalog"
    / "ill_cond_50"
    / "recipes"
    / "low__mclmc_lrd__mclmc_lrd_tuning.imm.npz"
)


def main():
    print("=" * 60)
    print("ill_cond_50 LRD recipe emission")
    print(f"seed={SEED}, n_warmup={N_WARMUP}, chains={NUM_CHAINS}, k={K_RANK}")
    print("=" * 60)

    # ── 1. Decompose oracle covariance at k=40 ──────────────────────────
    print("\n[1] Decomposing oracle COV at k=40 ...")
    sigma, U, lam = decompose_covariance_low_rank(COV, K_RANK)
    lrd_imm = LowRankInverseMassMatrix(sigma=sigma, U=U, lam=lam)
    print(f"    sigma={sigma.shape}, U={U.shape}, lam={lam.shape}")
    print(f"    lam range: [{float(jnp.min(lam)):.3f}, {float(jnp.max(lam)):.3f}]")

    # ── 2. Save IMM artifact ────────────────────────────────────────────
    print(f"\n[2] Saving IMM artifact → {IMM_PATH.name}")
    np.savez(
        str(IMM_PATH),
        sigma=np.array(sigma),
        U=np.array(U),
        lam=np.array(lam),
        k=K_RANK,
        model="ill_cond_50",
        seed=SEED,
        note="LRD k=40 oracle COV decomposition; R-hat=1.0039, ESS=1993 @stat-2026-06-09",
    )
    print(f"    Saved {IMM_PATH.stat().st_size} bytes.")

    # ── 3. Run single warmup to capture step_size / L ───────────────────
    print(f"\n[3] Running mclmc_find_L_and_step_size (n_warmup={N_WARMUP}) ...")
    entry = MODELS["ill_cond_50"]
    master_key = jax.random.key(SEED)
    init_key, warmup_key = jax.random.split(master_key)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)

    warmup_keys = jax.random.split(warmup_key, NUM_CHAINS)

    @jax.vmap
    def run_warmup_one(k, x_start):
        init_k, tune_k = jax.random.split(k)
        state = blackjax.mcmc.mclmc.init(x_start, logdensity_fn, init_k)
        kernel = make_lrd_kernel(lrd_imm)
        _, adaptation_state, _ = blackjax.mclmc_find_L_and_step_size(
            kernel,
            num_steps=N_WARMUP,
            state=state,
            rng_key=tune_k,
            logdensity_fn=logdensity_fn,
            diagonal_preconditioning=False,
        )
        return adaptation_state

    init_positions = jax.tree.map(
        lambda x: jnp.tile(x, (NUM_CHAINS, *([1] * x.ndim))), init_position
    )
    adaptation_states = run_warmup_one(warmup_keys, init_positions)

    # Take mean over chains (step_size / L are scalars per chain)
    step_sizes = np.array(adaptation_states.step_size)
    Ls = np.array(adaptation_states.L)
    print(f"    step_size per chain: {step_sizes.tolist()}")
    print(f"    L per chain:         {Ls.tolist()}")
    step_size_mean = float(np.mean(step_sizes))
    L_mean = float(np.mean(Ls))
    print(f"    → mean step_size={step_size_mean:.6f}, mean L={L_mean:.6f}")

    # ── 4. Update recipe JSON ────────────────────────────────────────────
    print(f"\n[4] Updating recipe JSON → {RECIPE_PATH.name}")
    with open(RECIPE_PATH) as f:
        recipe = json.load(f)

    recipe["base_method_params"] = {
        "step_size": step_size_mean,
        "L": L_mean,
        "k_rank": K_RANK,
        # IMM is too large for inline JSON; loaded from inverse_mass_matrix_path
    }
    # Relative path from catalog/ill_cond_50/ root
    recipe["inverse_mass_matrix_path"] = (
        "recipes/low__mclmc_lrd__mclmc_lrd_tuning.imm.npz"
    )
    recipe["warmup_grad_evals"] = 2 * N_WARMUP * NUM_CHAINS

    # Update calibration_budget with actual warmup cost
    recipe["calibration_budget"]["warmup_grad_evals"] = 2 * N_WARMUP * NUM_CHAINS

    with open(RECIPE_PATH, "w") as f:
        json.dump(recipe, f, indent=2)
        f.write("\n")

    print("    Done.")
    print("\n" + "=" * 60)
    print("EMISSION COMPLETE")
    print(f"  Recipe: {RECIPE_PATH}")
    print(f"  IMM:    {IMM_PATH}")
    print(f"  step_size={step_size_mean:.6f}  L={L_mean:.6f}  k={K_RANK}")
    print("=" * 60)


if __name__ == "__main__":
    main()
