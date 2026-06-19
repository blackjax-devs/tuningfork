"""Emit medium__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json for banana.

Params are LOCKED (cert-pinned — do NOT re-tune):
  step_size : 0.20662   (median over cert seeds 101–106; spread 0.114–0.285)
  L         : 4.95888   (= median * 24, the cert's best avg_window)
  IMM       : [8.0, 9.0] (analytic banana diagonal marginal var; Cov=0)

This script regenerates TOOLING-OWNED fields only:
  headline_metric, calibration_budget (incl. machine_info, timing),
  *_version, timestamp_utc, gate_evidence.auto (rhat, ess, div, max_abs_mean_z)

STATISTICIAN-OWNED fields (base_method_params, gate_evidence.override) are
FIXED here from the cert and must NOT be changed without a new cert.

Usage:
  cd /home/jp/blackjax-devs/tuningfork
  JAX_PLATFORM_NAME=cpu .venv/bin/python \
      experiments/mclmc_scaling/emit_banana_medium_dynamic_recipe.py
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(_REPO)
sys.path.insert(0, _REPO)

# ── Pinned / LOCKED params ──────────────────────────────────────────────────
STEP_SIZE = 0.20662
L_VAL = 4.95888
IMM_DIAG = [8.0, 9.0]
N_SAMPLES = 5000
N_WARMUP = 5000
NUM_CHAINS = 4
TUNING_SEED = 682737  # match the catalog convention seed

# Analytic banana ground truth
BANANA_MEAN = np.array([0.0, 2.0])
BANANA_STD = np.sqrt(np.array([8.0, 9.0]))

RECIPE_PATH = "tuningfork/catalog/banana/recipes/medium__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json"


# ── Collision guard ──────────────────────────────────────────────────────────
def collision_guard(path: str) -> None:
    if os.path.exists(path):
        raise RuntimeError(
            f"COLLISION: {path} already exists.\n"
            "Delete or rename the existing file before filing a new recipe."
        )
    print(f"[collision guard] OK — target path is clear")


# ── Model ────────────────────────────────────────────────────────────────────
def load_banana():
    from jax.flatten_util import ravel_pytree

    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.model._registry import MODELS

    entry = MODELS["banana"]
    init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), entry)
    flat_init, unravel = ravel_pytree(init_dict)
    d = flat_init.shape[0]
    assert d == 2, f"banana d={d}, expected 2"
    ld = lambda xf: ld_raw(unravel(xf))
    return ld, flat_init, d


# ── Warmup (for timing; adapted params are discarded) ───────────────────────
def run_warmup_for_timing(logdensity_fn, init_pos, rng_key) -> tuple[float, int]:
    """Run adjusted_mclmc_tuning; return (wall_seconds, warmup_grad_evals)."""
    import blackjax

    from tuningfork.base_method.adjusted_mclmc_dynamic import ENTRY as bentry
    from tuningfork.warmup.adjusted_mclmc_tuning import ENTRY as wentry

    print(f"  Running adjusted_mclmc_tuning ({N_WARMUP} steps × {NUM_CHAINS} chains)…")
    t0 = time.monotonic()
    _, adapted = wentry.runner(
        rng_key,
        init_pos,
        N_WARMUP,
        bentry,
        logdensity_fn=logdensity_fn,
        num_chains=NUM_CHAINS,
        target=0.9,
    )
    jax.block_until_ready(adapted)
    t_warmup = time.monotonic() - t0
    warmup_grad_evals = int(
        adapted.get("_total_tuning_steps", N_WARMUP * NUM_CHAINS * 2)
    )
    print(f"  warmup done in {t_warmup:.2f}s; grad_evals≈{warmup_grad_evals}")
    return t_warmup, warmup_grad_evals


# ── Sampling with PINNED params ──────────────────────────────────────────────
def run_sampling_pinned(logdensity_fn, init_pos, rng_key):
    """Vmap multichain sampling with LOCKED (step_size, L, IMM).

    Returns (positions_chains, infos, wall_seconds).
    positions_chains: (NUM_CHAINS, N_SAMPLES, d)
    """
    import blackjax
    from blackjax.mcmc.adjusted_mclmc_dynamic import make_random_trajectory_length_fn

    imm = jnp.asarray(IMM_DIAG, dtype=jnp.float64)
    ss = jnp.float64(STEP_SIZE)
    L = jnp.float64(L_VAL)
    avg = L / ss  # ≈ 24.0

    steps_fn = make_random_trajectory_length_fn(True)

    algo = blackjax.adjusted_mclmc_dynamic(
        logdensity_fn,
        step_size=ss,
        integration_steps_fn=steps_fn,
        integration_steps_params=(avg,),
        inverse_mass_matrix=imm,
    )

    # Per-chain init keys (adjusted_mclmc_dynamic.init needs rng_key)
    init_keys = jax.random.split(rng_key, NUM_CHAINS)

    @jax.vmap
    def _init_one(key):
        return algo.init(init_pos, rng_key=key)

    batched_state = jax.block_until_ready(_init_one(init_keys))

    # Scan over samples; vmap over chains inside each step
    step_keys = jax.random.split(jax.random.fold_in(rng_key, 42), N_SAMPLES)

    @jax.vmap
    def _step_one(state, key):
        return algo.step(key, state)

    def _scan_step(carry, scan_key):
        chain_keys = jax.random.split(scan_key, NUM_CHAINS)
        new_state, info = _step_one(carry, chain_keys)
        return new_state, (new_state.position, info)

    # One JIT compile pass (warm up XLA)
    _ = jax.block_until_ready(_scan_step(batched_state, step_keys[0]))

    t0 = time.monotonic()
    _, (positions_scan, infos_scan) = jax.lax.scan(_scan_step, batched_state, step_keys)
    jax.block_until_ready((positions_scan, infos_scan))
    t_sample = time.monotonic() - t0

    # positions_scan: (N_SAMPLES, NUM_CHAINS, d) → transpose to (NUM_CHAINS, N_SAMPLES, d)
    pos_chains = jnp.transpose(positions_scan, (1, 0, 2))

    print(f"  sampling done in {t_sample:.3f}s, shape={pos_chains.shape}")
    return np.asarray(pos_chains), infos_scan, t_sample


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(pos_chains, infos_scan):
    """Compute headline, gate_evidence.auto for (NUM_CHAINS, N_SAMPLES, d) chains."""
    import arviz as az

    from tuningfork.calibration.statistician_gate import auto_gate

    n_chains, n_samples, d = pos_chains.shape

    # num_integration_steps: (N_SAMPLES, NUM_CHAINS) → total grad evals
    nis = np.asarray(infos_scan.num_integration_steps)  # (N_SAMPLES, NUM_CHAINS)
    total_grad_evals = int(2 * int(np.sum(nis)))

    # Ground-truth summaries (analytic banana moments)
    gt_summaries = {"x": {"mean": BANANA_MEAN, "std": BANANA_STD}}

    # Gate (passes (NUM_CHAINS, N_SAMPLES, d) as {"x": ...})
    gate_verdict = auto_gate(
        {"x": pos_chains},
        infos_scan,
        ground_truth_summaries=gt_summaries,
    )

    # Headline metric
    headline = (
        gate_verdict.min_bulk_ess / total_grad_evals if total_grad_evals > 0 else None
    )

    headline_basis = {
        "total_grad_evals": total_grad_evals,
        "min_bulk_ess": float(gate_verdict.min_bulk_ess),
        "grad_count_convention": "2 × info.num_integration_steps",
        "is_lower_bound": False,
    }

    return headline, headline_basis, gate_verdict


# ── Version helpers ───────────────────────────────────────────────────────────
def _ver(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return "unknown"


def _now_utc() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    collision_guard(RECIPE_PATH)

    print("Loading banana model…")
    ld, init_pos, d = load_banana()

    from tuningfork._machine_info import get_machine_info

    machine_info = get_machine_info()
    print(
        f"Machine: {machine_info['cpu_model']}, {machine_info['cpu_count_logical']} cores, "
        f"JAX {machine_info['jax_version']}"
    )

    rng = jax.random.key(TUNING_SEED)
    warmup_key, sample_key = jax.random.split(rng)

    # 1. Warmup (for timing only — adapted params are discarded)
    t_warmup, warmup_grad_evals = run_warmup_for_timing(ld, init_pos, warmup_key)

    # 2. Sampling with PINNED params
    print("Running sampling with pinned params…")
    pos_chains, infos_scan, t_sample = run_sampling_pinned(ld, init_pos, sample_key)

    # 3. Compute metrics
    print("Computing metrics…")
    headline, headline_basis, gate_verdict = compute_metrics(pos_chains, infos_scan)

    print(f"  headline_metric  : {headline:.6f}")
    print(f"  gate verdict     : {gate_verdict.verdict}")
    print(f"  rhat_max         : {gate_verdict.rhat_max:.5f}")
    print(f"  min_bulk_ess     : {gate_verdict.min_bulk_ess:.1f}")
    print(f"  n_divergences    : {gate_verdict.n_divergences}")
    if gate_verdict.max_abs_mean_z is not None:
        print(f"  max_abs_mean_z   : {gate_verdict.max_abs_mean_z:.4f}")

    t_total = t_warmup + t_sample

    avg_implied = L_VAL / STEP_SIZE
    instructions = (
        f"**Medium-effort recipe** (Statistician-certified "
        f"`(adjusted_mclmc_tuning, adjusted_mclmc_dynamic)` on banana; "
        f"params PINNED from production cert at avg={round(avg_implied)}).\n"
        f"To use the pinned config (skip warmup at runtime):\n"
        f"  ```python\n"
        f"  kernel = blackjax.adjusted_mclmc_dynamic("
        f"logdensity_fn, **{{'step_size': {STEP_SIZE}, 'L': {L_VAL}, "
        f"'inverse_mass_matrix': {IMM_DIAG}}})\n"
        f"  ```\n"
        f"Expected `min-bulk-ESS / total_grad_evals`: {headline:.4f}.\n"
        f"Wall time: machine + Statistician investigation (see `calibration_budget`)."
    )

    gate_ev = gate_verdict.to_dict()
    gate_ev["gt_cert_coverage"] = "full_posterior"

    recipe = {
        "model_name": "banana",
        "base_method_name": "adjusted_mclmc_dynamic",
        "effort": "medium",
        "base_method_params": {
            "step_size": STEP_SIZE,
            "L": L_VAL,
            "inverse_mass_matrix": IMM_DIAG,
        },
        "headline_metric": headline,
        "sample_quality": {
            "mae_vs_reference": None,
            "q05_error": None,
            "q95_error": None,
            "std_ratio_max_dev": None,
        },
        "calibration_budget": {
            "trials": 0,
            "wall_seconds_estimate": round(t_total, 3),
            "n_warmup": N_WARMUP,
            "n_samples": N_SAMPLES,
            "num_chains": NUM_CHAINS,
            "warmup_wall_seconds": round(t_warmup, 3),
            "sampling_wall_seconds": round(t_sample, 3),
            "sampling_seconds_per_draw": round(
                t_sample / max(N_SAMPLES * NUM_CHAINS, 1), 6
            ),
            "split_source": "measured",
            "machine_info": machine_info,
            "warmup_grad_evals": warmup_grad_evals,
        },
        "difficulty": None,
        "instructions": instructions,
        "notes": (
            "medium__ banana recipe for adjusted_mclmc_dynamic (dynamic trajectory length). "
            "step_size=0.20662 PINNED from median of cert seeds 101–106 (spread 0.114–0.285); "
            "do NOT re-tune per run (warmup EEVPD step is noisy on banana's curved geometry). "
            "L=4.95888 = median_step × 24 (avg_window best point from grid search). "
            "IMM=[8,9] is the analytic diagonal marginal covariance (Cov(x1,x2)=0). "
            "The pre-existing failed__adjusted_mclmc__adjusted_mclmc_tuning.json "
            "(non-dynamic, warmup avg=2 cap) is a SEPARATE method and stays failed__. "
            "See #25 / #191 / L-override #22."
        ),
        "step_policy": None,
        "warmups": [
            {
                "name": "adjusted_mclmc_tuning",
                "params": {
                    "n_warmup": N_WARMUP,
                    "num_chains": NUM_CHAINS,
                    "target_acceptance": 0.9,
                },
            }
        ],
        "warmup_inner_kernel": None,
        "warmup_num_chains": None,
        "init_strategy": None,
        "variant_label": None,
        "inverse_mass_matrix_path": None,
        "workflow": "",
        "gate_evidence": {
            "auto": gate_ev,
            "override": {
                "decision": "PASS",
                "tier": "medium",
                "statistician_id": "stat-2026-06-19",
                "date": "2026-06-19",
                "reason": (
                    "Production-budget cert of adjusted_mclmc_dynamic at PINNED avg=24 "
                    "(L=24*step). 6/6 seeds PASS the medium gate (2mbias<0.1 & mbias_sd<0.06 "
                    "& rhat<1.01 & minESS>100/ch & div=0): med 2mbias 0.023 (max 0.050), "
                    "max mbias_sd 0.023, medESS 7530 (minESS 6165), rhat<=1.001, acc 0.928, "
                    "0 div. avg=24 is the robust plateau center (only 6/6 point; avg=18/36/54 "
                    "each nick the bias gate on the longest-step seed). Distinct from the "
                    "non-dynamic failed__adjusted_mclmc recipe (warmup avg=2), which genuinely "
                    "fails to mix and remains failed__. See #25 / #191 / L-override #22."
                ),
            },
        },
        "tuning_seed": TUNING_SEED,
        "tuningfork_version": _ver("tuningfork"),
        "blackjax_version": _ver("blackjax"),
        "jax_version": _ver("jax"),
        "timestamp_utc": _now_utc(),
        "headline_basis": headline_basis,
        "failure_diagnosis": None,
        "attempted_configurations": [],
    }

    os.makedirs(os.path.dirname(RECIPE_PATH), exist_ok=True)
    with open(RECIPE_PATH, "w") as fh:
        json.dump(recipe, fh, indent=2, default=str)
        fh.write("\n")

    print(f"\n[OK] Wrote {RECIPE_PATH}")
    print(f"  headline_metric  : {headline:.6f}")
    print(f"  gate.verdict     : {gate_verdict.verdict}")
    print(f"  rhat_max         : {gate_verdict.rhat_max:.5f}")
    print(f"  max_abs_mean_z   : {gate_verdict.max_abs_mean_z}")

    # Load-verify
    from tuningfork.catalog.inspect import load_recipe

    r = load_recipe(RECIPE_PATH)
    assert r.model_name == "banana"
    assert r.base_method_name == "adjusted_mclmc_dynamic"
    assert r.effort.value == "medium"
    assert r.base_method_params["step_size"] == STEP_SIZE
    assert r.base_method_params["L"] == L_VAL
    print(
        f"  [load verify] OK — model={r.model_name}, effort={r.effort.value}, "
        f"step_size={r.base_method_params['step_size']}, L={r.base_method_params['L']}"
    )

    return recipe


if __name__ == "__main__":
    main()
