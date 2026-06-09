"""Calibrate german_credit LRD-MCLMC recipe.

Protocol (statistician-approved, 2026-06-09):
  - k=8 (true low-rank; full-rank k=26 overfits the pilot → R-hat culprit)
  - n_warmup=2000, n_samples=2000, num_chains=4
  - NUTS pilot: n_warmup=1000, n_samples=1000 (single chain)
  - Seeds 42, 99, 777 (multi-seed hardening)

Emits:
  tuningfork/catalog/german_credit/recipes/low__mclmc_lrd__mclmc_lrd_tuning.json
  tuningfork/catalog/german_credit/recipes/low__mclmc_lrd__mclmc_lrd_tuning.imm.npz

Run from repo root:
    uv run python scripts/calibrate_german_credit_lrd.py
"""

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[1]
BLACKJAX_DEV = REPO_ROOT.parent / "blackjax"
sys.path.insert(0, str(BLACKJAX_DEV))
sys.path.insert(1, str(REPO_ROOT))

import blackjax
from blackjax.mcmc.metrics import LowRankInverseMassMatrix

from tuningfork.base_method.mclmc import (
    extract_lrd_from_samples,
    make_lrd_kernel,
    run_pilot_nuts,
)
from tuningfork.calibration.statistician_gate import auto_gate
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn

# ── Hyperparameters (statistician-approved greenlit protocol) ──────────────
SEEDS = [42, 99, 777]
K_RANK = 8
PILOT_N_WARMUP = 1000
PILOT_N_SAMPLES = 1000
N_WARMUP = 2000
N_SAMPLES = 2000
NUM_CHAINS = 4

MODEL_NAME = "german_credit"
CATALOG_DIR = REPO_ROOT / "tuningfork" / "catalog"
RECIPE_DIR = CATALOG_DIR / MODEL_NAME / "recipes"
RECIPE_PATH = RECIPE_DIR / "low__mclmc_lrd__mclmc_lrd_tuning.json"
IMM_PATH = RECIPE_DIR / "low__mclmc_lrd__mclmc_lrd_tuning.imm.npz"
GT_DRAWS_PATH = (
    CATALOG_DIR / MODEL_NAME / "groundtruth_samples" / "blackjax" / "draws.npz"
)


def _load_gt_summaries() -> dict:
    """Load ground-truth mean/std for german_credit from the committed draws."""
    _ref = np.load(str(GT_DRAWS_PATH))
    # draws.npz has key 'beta', shape (40000, 26) — unconstrained logistic-reg params
    beta = _ref["beta"]
    return {
        "beta": {
            "mean": beta.mean(axis=0),
            "std": beta.std(axis=0),
            "n_samples": int(beta.shape[0]),
        }
    }


def _warmup_one_chain(k, init_position, logdensity_fn, lrd_imm):
    """Single-chain LRD warmup — Python function, NOT vmapped.

    mclmc_find_L_and_step_size contains a while_loop / FFT-based Welford step
    that causes an XLA C-level abort under jax.vmap.  Same workaround as the
    window_adaptation vmap bug.  The sampling phase (jax.lax.scan) is vmap-safe.
    """
    init_k, tune_k = jax.random.split(k)
    state = blackjax.mcmc.mclmc.init(init_position, logdensity_fn, init_k)
    kernel = make_lrd_kernel(lrd_imm)
    adapted_state, adaptation_state, _ = blackjax.mclmc_find_L_and_step_size(
        kernel,
        num_steps=N_WARMUP,
        state=state,
        rng_key=tune_k,
        logdensity_fn=logdensity_fn,
        diagonal_preconditioning=False,
    )
    return adapted_state, adaptation_state


def run_one_seed(seed: int, entry) -> dict:
    """Full LRD calibration for one seed. Returns result dict."""
    print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")

    master_key = jax.random.key(seed)
    init_key, pilot_key, run_key = jax.random.split(master_key, 3)

    # 1. Build logdensity_fn
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)
    print(f"  Position keys: {list(init_position.keys())}")

    # 2. NUTS pilot → geometry samples
    t0 = time.perf_counter()
    print(
        f"  [1] NUTS pilot (n_warmup={PILOT_N_WARMUP}, n_samples={PILOT_N_SAMPLES}) ..."
    )
    pilot_positions = run_pilot_nuts(
        logdensity_fn,
        init_position,
        pilot_key,
        n_warmup=PILOT_N_WARMUP,
        n_samples=PILOT_N_SAMPLES,
    )
    jax.block_until_ready(pilot_positions)
    print(f"      done in {time.perf_counter()-t0:.1f}s")

    # 3. Extract LRD k=8
    print(f"  [2] Extracting LRD k={K_RANK} ...")
    mean, sigma, U, lam = extract_lrd_from_samples(pilot_positions, K_RANK)
    lrd_imm = LowRankInverseMassMatrix(sigma=sigma, U=U, lam=lam)
    print(
        f"      sigma range: [{float(jnp.min(sigma)):.4f}, {float(jnp.max(sigma)):.4f}]"
    )
    print(f"      lam range:   [{float(jnp.min(lam)):.3f}, {float(jnp.max(lam)):.3f}]")

    # 4. LRD warmup — Python loop (vmap-over-find_L_and_step_size hits XLA abort)
    t1 = time.perf_counter()
    print(
        f"  [3] LRD MCLMC warmup (Python loop, {NUM_CHAINS} chains × {N_WARMUP} steps) ..."
    )
    warmup_key, sampling_key = jax.random.split(run_key)
    warmup_keys = jax.random.split(warmup_key, NUM_CHAINS)

    per_chain = [
        _warmup_one_chain(warmup_keys[i], init_position, logdensity_fn, lrd_imm)
        for i in range(NUM_CHAINS)
    ]
    adapted_states = jax.tree.map(lambda *xs: jnp.stack(xs), *[s for s, _ in per_chain])
    adaptation_states = jax.tree.map(
        lambda *xs: jnp.stack(xs), *[p for _, p in per_chain]
    )
    jax.block_until_ready(adaptation_states)

    step_sizes = np.array(adaptation_states.step_size)
    Ls = np.array(adaptation_states.L)
    step_size_mean = float(np.mean(step_sizes))
    L_mean = float(np.mean(Ls))
    print(f"      warmup done in {time.perf_counter()-t1:.1f}s")
    print(f"      step_size: {[round(x, 6) for x in step_sizes.tolist()]}")
    print(f"      L:         {[round(x, 6) for x in Ls.tolist()]}")
    print(f"      mean step_size={step_size_mean:.6f}, L={L_mean:.6f}")

    # 5. Sampling — vmapped jax.lax.scan (safe)
    print(f"  [4] Sampling ({NUM_CHAINS} chains × {N_SAMPLES} draws) ...")
    sampling_keys = jax.random.split(sampling_key, NUM_CHAINS)

    @jax.vmap
    def run_sampling(k, state, params):
        kernel = make_lrd_kernel(lrd_imm)

        def body_fn(s, rng):
            s, info = kernel(
                rng,
                s,
                logdensity_fn,
                inverse_mass_matrix=lrd_imm,
                L=params.L,
                step_size=params.step_size,
            )
            return s, (s.position, info)

        _, (positions, infos) = jax.lax.scan(
            body_fn, state, jax.random.split(k, N_SAMPLES)
        )
        return positions, infos

    samples, infos = run_sampling(sampling_keys, adapted_states, adaptation_states)
    jax.block_until_ready((samples, infos))
    print(f"      sampling done in {time.perf_counter()-t1:.1f}s total")

    # 6. Gate
    print(f"  [5] Auto-gate ...")
    gt_summaries = _load_gt_summaries()
    verdict = auto_gate(
        samples,
        infos,
        ground_truth_summaries=gt_summaries,
        n_chunks=NUM_CHAINS,
    )
    print(
        f"      verdict={verdict.verdict}  rhat={verdict.rhat_max:.5f}"
        f"  ess={verdict.min_bulk_ess:.1f}"
        f"  divs={verdict.n_divergences}"
        f"  z={verdict.max_abs_mean_z:.3f}"
        if verdict.max_abs_mean_z is not None
        else f"      verdict={verdict.verdict}  rhat={verdict.rhat_max:.5f}"
        f"  ess={verdict.min_bulk_ess:.1f}  divs={verdict.n_divergences}"
    )

    return {
        "seed": seed,
        "verdict": verdict.verdict,
        "rhat_max": float(verdict.rhat_max),
        "min_bulk_ess": float(verdict.min_bulk_ess),
        "n_divergences": int(verdict.n_divergences),
        "max_abs_mean_z": (
            float(verdict.max_abs_mean_z)
            if verdict.max_abs_mean_z is not None
            else None
        ),
        "step_size": step_size_mean,
        "L": L_mean,
        "sigma": sigma,
        "U": U,
        "lam": lam,
    }


def main():
    print("german_credit LRD-MCLMC calibration")
    print(
        f"k={K_RANK}, n_warmup={N_WARMUP}, n_samples={N_SAMPLES}, chains={NUM_CHAINS}"
    )
    print(f"seeds: {SEEDS}")

    entry = MODELS[MODEL_NAME]
    RECIPE_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in SEEDS:
        r = run_one_seed(seed, entry)
        results.append(r)

    # Multi-seed summary
    print("\n" + "=" * 60)
    print("MULTI-SEED SUMMARY")
    print("=" * 60)
    for r in results:
        z_str = (
            f"  z={r['max_abs_mean_z']:.3f}" if r["max_abs_mean_z"] is not None else ""
        )
        print(
            f"  seed={r['seed']:5d}: {r['verdict']:6s}  "
            f"rhat={r['rhat_max']:.5f}  ess={r['min_bulk_ess']:.1f}"
            f"  divs={r['n_divergences']}"
            f"  step_size={r['step_size']:.6f}  L={r['L']:.6f}" + z_str
        )

    primary = results[0]  # seed=42

    # Save IMM artifact (seed-42 geometry)
    print(f"\nSaving IMM → {IMM_PATH.name}")
    np.savez(
        str(IMM_PATH),
        sigma=np.array(primary["sigma"]),
        U=np.array(primary["U"]),
        lam=np.array(primary["lam"]),
        k=K_RANK,
        model=MODEL_NAME,
        seed=int(primary["seed"]),
        note=(
            f"LRD k={K_RANK} from NUTS pilot (n_warmup={PILOT_N_WARMUP}, "
            f"n_samples={PILOT_N_SAMPLES}); seed={primary['seed']}; "
            f"k=8 not full-rank k=26 (full-rank overfits pilot, R-hat>1.01)."
        ),
    )
    print(f"    Saved {IMM_PATH.stat().st_size} bytes.")

    # Grad eval accounting (2 grads per MCLMC step)
    mclmc_sampling_grad_evals = 2 * N_SAMPLES * NUM_CHAINS
    mclmc_warmup_grad_evals = 2 * N_WARMUP * NUM_CHAINS
    ess_per_grad = primary["min_bulk_ess"] / mclmc_sampling_grad_evals

    # Gate evidence
    gate_evidence = {
        "auto": {
            "rhat_max": primary["rhat_max"],
            "min_bulk_ess": primary["min_bulk_ess"],
            "n_divergences": primary["n_divergences"],
            "max_abs_mean_z": primary["max_abs_mean_z"],
            "verdict": primary["verdict"],
            "ess_per_grad": ess_per_grad,
            "total_grad_evals": mclmc_sampling_grad_evals,
            "seed": primary["seed"],
        },
        "override": {
            "decision": "",
            "statistician_id": "",
            "reason": "",
        },
    }

    # All seeds as attempted_configurations
    attempted_configurations = [
        {
            "seed": r["seed"],
            "k": K_RANK,
            "n_warmup": N_WARMUP,
            "n_samples": N_SAMPLES,
            "num_chains": NUM_CHAINS,
            "outcome": r["verdict"],
            "rhat_max": round(r["rhat_max"], 6),
            "min_bulk_ess": round(r["min_bulk_ess"], 2),
            "n_divergences": r["n_divergences"],
            "max_abs_mean_z": (
                round(r["max_abs_mean_z"], 4)
                if r["max_abs_mean_z"] is not None
                else None
            ),
            "ess_per_grad": round(r["min_bulk_ess"] / mclmc_sampling_grad_evals, 6),
            "step_size": round(r["step_size"], 6),
            "L": round(r["L"], 6),
        }
        for r in results
    ]

    recipe = {
        "model_name": MODEL_NAME,
        "base_method_name": "mclmc",
        "effort": "low",
        "base_method_params": {
            "step_size": primary["step_size"],
            "L": primary["L"],
            "k_rank": K_RANK,
        },
        "warmup_name": "mclmc_lrd_tuning",
        "warmup_params": {
            "n_warmup": N_WARMUP,
            "num_chains": NUM_CHAINS,
            "k_rank": K_RANK,
            "pilot_n_warmup": PILOT_N_WARMUP,
            "pilot_n_samples": PILOT_N_SAMPLES,
        },
        "headline_metric": ess_per_grad,
        "sample_quality": None,
        "calibration_budget": {
            "trials": len(SEEDS),
            "n_warmup": N_WARMUP,
            "n_samples": N_SAMPLES,
            "num_chains": NUM_CHAINS,
            "pilot_n_warmup": PILOT_N_WARMUP,
            "pilot_n_samples": PILOT_N_SAMPLES,
            "warmup_grad_evals": mclmc_warmup_grad_evals,
            "sampling_grad_evals": mclmc_sampling_grad_evals,
        },
        "difficulty": None,
        "instructions": (
            f"LRD-preconditioned MCLMC on {MODEL_NAME} (26-D logistic regression).\n"
            f"Pipeline: (1) {PILOT_N_WARMUP}-step NUTS pilot; "
            f"(2) rank-{K_RANK} SVD extraction via extract_lrd_from_samples; "
            f"(3) mclmc_find_L_and_step_size with make_lrd_kernel binding the "
            f"LowRankInverseMassMatrix.\n"
            f"k={K_RANK} (not full-rank k=26; full-rank overfits the pilot → R-hat>1.01).\n"
            f"IMM checkpoint at inverse_mass_matrix_path; load with np.load to skip re-extraction.\n"
            f"Warmup uses Python loop over chains (not vmap) — mclmc_find_L_and_step_size "
            f"contains while_loop / FFT steps that abort under jax.vmap."
        ),
        "notes": (
            f"Multi-seed hardening: seeds {SEEDS}. "
            f"k={K_RANK} is the statistician-approved rank "
            f"(full-rank k=26 was the R-hat culprit at prior attempt). "
            f"See attempted_configurations for per-seed gate evidence."
        ),
        "step_policy": None,
        "warmups": [
            {
                "name": "mclmc_lrd_tuning",
                "params": {
                    "n_warmup": N_WARMUP,
                    "num_chains": NUM_CHAINS,
                    "k_rank": K_RANK,
                    "pilot_n_warmup": PILOT_N_WARMUP,
                    "pilot_n_samples": PILOT_N_SAMPLES,
                },
            }
        ],
        "warmup_inner_kernel": None,
        "warmup_num_chains": NUM_CHAINS,
        "init_strategy": None,
        "inverse_mass_matrix_path": f"recipes/{IMM_PATH.name}",
        "workflow": "",
        "gate_evidence": gate_evidence,
        "failure_diagnosis": None,
        "attempted_configurations": attempted_configurations,
        "warmup_grad_evals": mclmc_warmup_grad_evals,
    }

    print(f"Writing recipe → {RECIPE_PATH.name}")
    with open(RECIPE_PATH, "w") as f:
        json.dump(recipe, f, indent=2)
        f.write("\n")

    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print(f"  Recipe:      {RECIPE_PATH}")
    print(f"  IMM:         {IMM_PATH}")
    print(f"  Primary seed={primary['seed']}: {primary['verdict']}")
    print(f"  rhat_max:    {primary['rhat_max']:.5f}")
    print(f"  min_ess:     {primary['min_bulk_ess']:.1f}")
    z_str = (
        f"  {primary['max_abs_mean_z']:.3f}"
        if primary["max_abs_mean_z"] is not None
        else " N/A"
    )
    print(f"  z:          {z_str}")
    print(f"  ESS/grad:    {ess_per_grad:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
