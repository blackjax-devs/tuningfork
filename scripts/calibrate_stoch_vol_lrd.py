# DEFERRED — stoch_vol LRD calibration track stopped per mission fallback (2026-06-10).
#
# Phase (c) Track 2 result: 0/3 cert seeds ERROR before sampling completed.
# Root cause: mixed-rank parameter pytree (h:(500,), phi/sigma/mu:()) triggers
# TypeError in _run_cert_seed's rhat/ESS aggregation step.
# Fix committed in blackjax feat/calibrated-emission (76e1dfd); see:
#   tuningfork/docs/phase_c_track2_failure_analysis_2026_06_09.md
#
# Status: this script is the SOLE PROVENANCE for the committed stoch_vol LRD
# artifacts until the mixed-rank pytree bug is verified fixed and the track
# is re-opened by @user ruling. DO NOT DELETE — the catalog recipe was not
# regenerated via library path this mission.
#
# When @user authorises retry:
#   1. Verify fix 76e1dfd is on main.
#   2. If stoch_vol is registered: use the standard generator CLI:
#      uv run python -m tuningfork.recipes._generate_starter \
#          --warmup mclmc_lrd_tuning --only stoch_vol \
#          --calibrate --cert-seeds <seeds> --n-warmup 3000 --n-samples 2000 --k-rank 30
#   3. If still unregistered (flat-init NCP variant): re-run this script.
#      Restart the 3-attempt cert clock from zero.
#
"""Calibrate stoch_vol LRD-MCLMC recipe (flat-init NCP experimental variant).

Protocol (statistician-approved, 2026-06-09):
  - Flat-init NCP variant: h[0] = mu + sigma*h_std[0]  (drops sigma/sqrt(1-phi^2) coupling)
  - k=30 (not k=50), n_warmup=3000, n_samples=2000, num_chains=4
  - NUTS pilot: n_warmup=1000, n_samples=1000 (single chain)
  - Seeds 42, 99 (two-seed protocol)
  - Hard stop: if R-hat stays >1.02 on both seeds → documented REVIEW boundary

Parameterization note:
  This is an EXPERIMENTAL variant, NOT the registered stoch_vol model.
  The registered model uses stationary initialization:
      h[0] = mu + (sigma / sqrt(1 - phi^2)) * h_std[0]
  This variant uses flat initialization:
      h[0] = mu + sigma * h_std[0]
  The flat-init reduces the coupling between phi and h[0] near the unit root,
  potentially improving mixing for phi close to 1. The model registry key is
  NOT "stoch_vol" — the logdensity_fn is a custom NumPyro model defined below.
  Statistician must flip NCP_VARIANT=True when verifying from checkpoint.

Emits:
  tuningfork/catalog/stoch_vol/recipes/low__mclmc_lrd__mclmc_lrd_tuning_flatinit.json
  tuningfork/catalog/stoch_vol/recipes/low__mclmc_lrd__mclmc_lrd_tuning_flatinit.imm.npz

Run from repo root:
    uv run python scripts/calibrate_stoch_vol_lrd.py
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
import numpyro
import numpyro.distributions as dist
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from numpyro.infer.util import initialize_model

from tuningfork.base_method.mclmc import (
    extract_lrd_from_samples,
    make_lrd_kernel,
    run_pilot_nuts,
)
from tuningfork.calibration.statistician_gate import auto_gate
from tuningfork.model.stoch_vol import RETURNS, T_LENGTH

# ── Hyperparameters (statistician-approved greenlit protocol) ──────────────
SEEDS = [42, 99]
K_RANK = 30
PILOT_N_WARMUP = 1000
PILOT_N_SAMPLES = 1000
N_WARMUP = 3000
N_SAMPLES = 2000
NUM_CHAINS = 4

# Hard-stop R-hat boundary (REVIEW ceiling declared if both seeds exceed this)
RHAT_HARD_STOP = 1.02

MODEL_NAME = "stoch_vol"
NCP_VARIANT_KEY = "stoch_vol_flatinit_ncp"  # explicit key for @statistician
CATALOG_DIR = REPO_ROOT / "tuningfork" / "catalog"
RECIPE_DIR = CATALOG_DIR / MODEL_NAME / "recipes"
RECIPE_PATH = RECIPE_DIR / "low__mclmc_lrd__mclmc_lrd_tuning_flatinit.json"
IMM_PATH = RECIPE_DIR / "low__mclmc_lrd__mclmc_lrd_tuning_flatinit.imm.npz"


# ── Flat-init NCP variant NumPyro model ──────────────────────────────────────
# EXPERIMENTAL — not the registered model. Uses:
#     h[0] = mu + sigma * h_std[0]                    ← flat (this file)
# vs registered stoch_vol.py:
#     h[0] = mu + (sigma / sqrt(1 - phi^2)) * h_std[0]  ← stationary
#
# Rationale: sigma/sqrt(1-phi^2) → ∞ as phi→1 (unit root), creating a
# funnel-like coupling at h[0] that is hard for any constant preconditioner
# including LRD. Flat-init replaces the stationary distribution with
# sigma*h_std[0] (essentially Normal(mu, sigma)), decoupling h[0] from phi
# near the unit root. The marginal posteriors for mu/phi/sigma and h[1:T]
# are approximately unchanged (only h[0]'s prior changes).


def stoch_vol_flatinit_model(returns: jnp.ndarray, T: int = T_LENGTH) -> None:
    """Flat-init NCP stochastic volatility (experimental variant).

    Identical to the registered stoch_vol model except:
        h[0] = mu + sigma * h_std[0]           (flat-init, this model)
    vs  h[0] = mu + (sigma/sqrt(1-phi^2)) * h_std[0]  (stationary, registered)

    This decouples h[0] from the near-unit-root phi geometry.
    """
    mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
    phi = numpyro.sample("phi", dist.Uniform(-1.0, 1.0))
    phi_01 = (phi + 1.0) / 2.0
    numpyro.factor("phi_beta44_factor", dist.Beta(4.0, 4.0).log_prob(phi_01))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(5.0))

    h_std = numpyro.sample("h_raw", dist.Normal(jnp.zeros(T), 1.0))

    def step(h_prev, h_std_t):
        h_t = mu + phi * (h_prev - mu) + sigma * h_std_t
        return h_t, h_t

    # Flat init: h[0] = mu + sigma * h_std[0]  (no 1/sqrt(1-phi^2) coupling)
    h0 = mu + sigma * h_std[0]
    _, h_rest = jax.lax.scan(step, h0, h_std[1:])
    h = jnp.concatenate([h0[None], h_rest])
    h = numpyro.deterministic("h", h)

    numpyro.sample("returns", dist.Normal(0.0, jnp.exp(h / 2.0)), obs=returns)


def _build_flatinit_logdensity(rng_key):
    """Build logdensity_fn for the flat-init NCP stoch_vol variant."""
    model_info = initialize_model(
        rng_key,
        stoch_vol_flatinit_model,
        model_args=(RETURNS,),
        model_kwargs={"T": T_LENGTH},
        dynamic_args=False,
    )
    init_position = model_info.param_info.z
    potential_fn = model_info.potential_fn

    def logdensity_fn(position):
        return -potential_fn(position)

    return init_position, logdensity_fn


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


def run_one_seed(seed: int) -> dict:
    """Full LRD calibration for one seed. Returns result dict."""
    print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")

    master_key = jax.random.key(seed)
    init_key, pilot_key, run_key = jax.random.split(master_key, 3)

    # 1. Build flat-init logdensity_fn
    init_position, logdensity_fn = _build_flatinit_logdensity(init_key)
    param_keys = list(init_position.keys())
    print(f"  Position keys: {param_keys}")
    for k, v in init_position.items():
        print(f"    {k}: shape={jnp.shape(v)}")

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

    # 3. Extract LRD k=30
    print(f"  [2] Extracting LRD k={K_RANK} ...")
    mean, sigma_imm, U, lam = extract_lrd_from_samples(pilot_positions, K_RANK)
    lrd_imm = LowRankInverseMassMatrix(sigma=sigma_imm, U=U, lam=lam)
    print(
        f"      sigma range: [{float(jnp.min(sigma_imm)):.4f}, {float(jnp.max(sigma_imm)):.4f}]"
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

    # 6. Gate — no GT summaries (flat-init changes h[0] prior vs registered model)
    print(f"  [5] Auto-gate ...")
    verdict = auto_gate(
        samples,
        infos,
        ground_truth_summaries=None,  # flat-init variant: h[0] prior differs from GT
        n_chunks=NUM_CHAINS,
    )
    print(
        f"      verdict={verdict.verdict}  rhat={verdict.rhat_max:.5f}"
        f"  ess={verdict.min_bulk_ess:.1f}  divs={verdict.n_divergences}"
    )

    return {
        "seed": seed,
        "verdict": verdict.verdict,
        "rhat_max": float(verdict.rhat_max),
        "min_bulk_ess": float(verdict.min_bulk_ess),
        "n_divergences": int(verdict.n_divergences),
        "max_abs_mean_z": None,  # no GT for flat-init variant
        "step_size": step_size_mean,
        "L": L_mean,
        "sigma_imm": sigma_imm,
        "U": U,
        "lam": lam,
    }


def main():
    print("stoch_vol LRD-MCLMC calibration (flat-init NCP experimental variant)")
    print(
        f"k={K_RANK}, n_warmup={N_WARMUP}, n_samples={N_SAMPLES}, chains={NUM_CHAINS}"
    )
    print(f"seeds: {SEEDS}")
    print(f"Hard-stop R-hat boundary: {RHAT_HARD_STOP}")
    print(f"Parameterization: {NCP_VARIANT_KEY}")

    RECIPE_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in SEEDS:
        r = run_one_seed(seed)
        results.append(r)

    # Multi-seed summary
    print("\n" + "=" * 60)
    print("MULTI-SEED SUMMARY")
    print("=" * 60)
    for r in results:
        print(
            f"  seed={r['seed']:5d}: {r['verdict']:6s}  "
            f"rhat={r['rhat_max']:.5f}  ess={r['min_bulk_ess']:.1f}"
            f"  divs={r['n_divergences']}"
            f"  step_size={r['step_size']:.6f}  L={r['L']:.6f}"
        )

    # Hard-stop check: if both seeds have R-hat > RHAT_HARD_STOP, document REVIEW boundary
    both_above_hardstop = all(r["rhat_max"] > RHAT_HARD_STOP for r in results)
    if both_above_hardstop:
        print(f"\nHARD STOP: both seeds have R-hat > {RHAT_HARD_STOP}.")
        print(
            "Documenting REVIEW boundary — funnel geometry likely limits mixing further."
        )
        print("LRD is a constant preconditioner; position-dependent curvature at phi→1")
        print("requires adaptive geometry (NCP reparam or per-step adaptation).")

    primary = results[0]  # seed=42

    # Save IMM artifact (seed-42 geometry)
    print(f"\nSaving IMM → {IMM_PATH.name}")
    np.savez(
        str(IMM_PATH),
        sigma=np.array(primary["sigma_imm"]),
        U=np.array(primary["U"]),
        lam=np.array(primary["lam"]),
        k=K_RANK,
        model=MODEL_NAME,
        ncp_variant=NCP_VARIANT_KEY,
        seed=int(primary["seed"]),
        note=(
            f"LRD k={K_RANK} from flat-init NCP NUTS pilot "
            f"(n_warmup={PILOT_N_WARMUP}, n_samples={PILOT_N_SAMPLES}); "
            f"seed={primary['seed']}. "
            f"h[0]=mu+sigma*h_std[0] (flat-init, NOT stationary). "
            f"@statistician: set NCP_VARIANT=True (stoch_vol_flatinit_ncp) "
            f"when verifying from this checkpoint."
        ),
    )
    print(f"    Saved {IMM_PATH.stat().st_size} bytes.")

    # Grad eval accounting
    mclmc_sampling_grad_evals = 2 * N_SAMPLES * NUM_CHAINS
    mclmc_warmup_grad_evals = 2 * N_WARMUP * NUM_CHAINS
    ess_per_grad = primary["min_bulk_ess"] / mclmc_sampling_grad_evals

    # Gate evidence — use primary seed; all-seeds in attempted_configurations
    gate_evidence = {
        "auto": {
            "rhat_max": primary["rhat_max"],
            "min_bulk_ess": primary["min_bulk_ess"],
            "n_divergences": primary["n_divergences"],
            "max_abs_mean_z": None,
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

    # Boundary documentation if both seeds exceed hard-stop
    failure_diagnosis = None
    if both_above_hardstop:
        failure_diagnosis = (
            f"Both seeds have R-hat > {RHAT_HARD_STOP} (seeds {[r['seed'] for r in results]}, "
            f"R-hats {[round(r['rhat_max'], 5) for r in results]}). "
            f"REVIEW is the documented ceiling for LRD-MCLMC on this target. "
            f"Root cause: position-dependent curvature near phi→1 (unit-root funnel). "
            f"LRD is a constant O(dk) preconditioner; it cannot adapt to the "
            f"sigma/sqrt(1-phi^2) blowup geometry at runtime. "
            f"Flat-init h[0]=mu+sigma*h_std[0] reduces the phi-h[0] coupling "
            f"but does not eliminate the unit-root tail geometry. "
            f"Further improvement would require per-step adaptive geometry "
            f"(e.g. Riemannian MCLMC) or a stronger prior that rules out phi_con>0.99."
        )

    attempted_configurations = [
        {
            "seed": r["seed"],
            "k": K_RANK,
            "n_warmup": N_WARMUP,
            "n_samples": N_SAMPLES,
            "num_chains": NUM_CHAINS,
            "ncp_variant": NCP_VARIANT_KEY,
            "outcome": r["verdict"],
            "rhat_max": round(r["rhat_max"], 6),
            "min_bulk_ess": round(r["min_bulk_ess"], 2),
            "n_divergences": r["n_divergences"],
            "max_abs_mean_z": None,
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
            "ncp_variant": NCP_VARIANT_KEY,
        },
        "warmup_name": "mclmc_lrd_tuning",
        "warmup_params": {
            "n_warmup": N_WARMUP,
            "num_chains": NUM_CHAINS,
            "k_rank": K_RANK,
            "pilot_n_warmup": PILOT_N_WARMUP,
            "pilot_n_samples": PILOT_N_SAMPLES,
            "ncp_variant": NCP_VARIANT_KEY,
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
            f"LRD-preconditioned MCLMC on {MODEL_NAME} (503-D AR(1) stochastic volatility).\n"
            f"PARAMETERIZATION: {NCP_VARIANT_KEY} (EXPERIMENTAL — NOT the registered stoch_vol model).\n"
            f"  Flat-init: h[0] = mu + sigma * h_std[0]  (drops sigma/sqrt(1-phi^2) coupling)\n"
            f"  vs registered: h[0] = mu + (sigma/sqrt(1-phi^2)) * h_std[0]  (stationary)\n"
            f"Pipeline: (1) {PILOT_N_WARMUP}-step flat-init NUTS pilot; "
            f"(2) rank-{K_RANK} SVD extraction via extract_lrd_from_samples; "
            f"(3) mclmc_find_L_and_step_size with make_lrd_kernel binding the "
            f"LowRankInverseMassMatrix.\n"
            f"IMM checkpoint at inverse_mass_matrix_path; load with np.load to skip re-extraction.\n"
            f"@statistician: NCP_VARIANT=True → use stoch_vol_flatinit_ncp logdensity.\n"
            f"Warmup uses Python loop over chains (not vmap) — mclmc_find_L_and_step_size "
            f"contains while_loop / FFT steps that abort under jax.vmap."
        ),
        "notes": (
            f"Multi-seed hardening: seeds {SEEDS}. "
            f"k={K_RANK} (not k=50). "
            f"Flat-init NCP variant reduces phi-h[0] coupling near unit root. "
            f"Hard-stop R-hat boundary: {RHAT_HARD_STOP}. "
            + (
                "REVIEW is the documented ceiling — see failure_diagnosis."
                if both_above_hardstop
                else "See attempted_configurations for per-seed gate evidence."
            )
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
                    "ncp_variant": NCP_VARIANT_KEY,
                },
            }
        ],
        "warmup_inner_kernel": None,
        "warmup_num_chains": NUM_CHAINS,
        "init_strategy": None,
        "inverse_mass_matrix_path": f"recipes/{IMM_PATH.name}",
        "workflow": "",
        "gate_evidence": gate_evidence,
        "failure_diagnosis": failure_diagnosis,
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
    print(f"  Variant:     {NCP_VARIANT_KEY}")
    print(f"  Primary seed={primary['seed']}: {primary['verdict']}")
    print(f"  rhat_max:    {primary['rhat_max']:.5f}")
    print(f"  min_ess:     {primary['min_bulk_ess']:.1f}")
    print(f"  ESS/grad:    {ess_per_grad:.6f}")
    if both_above_hardstop:
        print(
            f"  NOTE: REVIEW boundary documented (both seeds R-hat > {RHAT_HARD_STOP})"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
