"""W1 full-catalog re-validation — step 3 + step 4 (irt_2pl×chees resolution).

Step 3: For every catalog cell that currently PASSES and is W1-eligible
  (has GT draws + summary_v2, NOT VI, R̂/ESS/div NOT-FAIL by virtue of PASS
  verdict), compare baseline_verdict (existing gate without W1) vs
  w1_active_verdict (W1/σ two-prong gate applied).  Expected result: 0 flips.

Step 4: Re-gate the held irt_2pl×chees cell under the honest multichain GT
  (10×10k).  Old z=4.174 was under the dispersed-init-poisoned single-chain
  GT (issue #222); expected to PASS under honest GT.  Runs CHEES warmup on
  irt_2pl at nc=CHEES_IRT2PL_N_CHAINS (default 16; smoke uses 4), then
  applies the full gate including W1.

Paths
-----
A  Per-recipe draws cache exists        → load + W1 gate (seconds)
B  No cache, standard MCMC              → run_recipe_to_idata skip_warmup=True
   (dynamic_hmc / nuts / dmhmc / mhmc / hmc / ghmc / barker / rmhmc / ...)
C  No cache, MCLMC or laplace           → run_recipe_to_idata full warmup
   (mclmc / adjusted_mclmc_dynamic / laplace_*)
SK SMC or VI recipe                     → skip (W1 N/A)

Usage
-----
  JAX_PLATFORM_NAME=cpu uv run python experiments/w1_full_catalog_revalidation.py

Checkpoint: experiments/w1_full_catalog_revalidation_results.json
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CATALOG = pathlib.Path("tuningfork/catalog")
RESULTS_FILE = pathlib.Path("experiments/w1_full_catalog_revalidation_results.json")

# W1 bootstrap replicates.  Override via env: W1_B=500 for a fast sweep,
# STEP4_W1_B=5000 to keep full-precision on the irt_2pl×chees cell.
W1_B: int = int(os.environ.get("W1_B", "5000"))
STEP4_W1_B: int = int(os.environ.get("STEP4_W1_B", "5000"))
W1_ALPHA: float = 0.05
W1_SEED: int = 42

# Re-sample N for path B/C cells (skip_warmup or full warmup).
RESAMPLE_N: int = 500

# irt_2pl×chees step-4 config.
CHEES_IRT2PL_N_CHAINS: int = 16  # nc (reduced from nc=128 GPU for CPU run)
CHEES_IRT2PL_N_WARMUP: int = 500  # warmup steps
CHEES_IRT2PL_N_SAMPLES: int = 1000
CHEES_IRT2PL_SEED: int = 42

VI_METHODS = frozenset({"meanfield_vi", "fullrank_vi"})
SKIP_WARMUP_METHODS = frozenset(
    {
        "dynamic_hmc",
        "nuts",
        "dmhmc",
        "mhmc",
        "hmc",
        "ghmc",
        "rmhmc",
        "barker",
        "orbital_hmc",
    }
)
MCLMC_METHODS = frozenset({"mclmc", "adjusted_mclmc", "adjusted_mclmc_dynamic"})
LAPLACE_METHODS = frozenset(
    {"laplace_hmc", "laplace_dhmc", "laplace_mhmc", "laplace_dmhmc"}
)

_BATCH_ORDER = [
    "mvn_10",
    "logistic_synthetic",
    "banana",
    "eight_schools_ncp",
    "ill_cond_50",
    "german_credit",
    "lotka_volterra",
    "gmm_25",
    "irt_1pl",
    "irt_2pl",
    "radon",
    "stoch_vol",
    "horseshoe",
    "neals_funnel",
    "lgcp",
    "gp_regression",
]


# ---------------------------------------------------------------------------
# GT data loading
# ---------------------------------------------------------------------------


def load_gt_data(model_name: str) -> tuple[dict, dict] | None:
    """Load GT draws and summary_v2 for a model.

    Returns
    -------
    (gt_draws_per_site, gt_summary_per_site) or None if not available.

    gt_draws_per_site: {site: np.ndarray} in original (n_gt_chains, n_gt_draws[, D])
      format.  Scalar sites kept as (n_gt_chains, n_gt_draws) — compute_w1_realm
      handles the 2D expansion internally.
    gt_summary_per_site: {site: {"std": ..., "bulk_ess": ..., "tail_ess": ...}}
    """
    draws_path = CATALOG / model_name / "groundtruth_samples" / "blackjax" / "draws.npz"
    summ_path = (
        CATALOG / model_name / "groundtruth_samples" / "blackjax" / "summary_v2.json"
    )
    if not draws_path.exists() or not summ_path.exists():
        return None

    raw = np.load(draws_path)
    gt_draws_per_site: dict[str, np.ndarray] = {
        site: raw[site].astype(np.float64) for site in raw.files
    }

    summ = json.loads(summ_path.read_text())
    per_site = summ.get("per_site", {})
    gt_summary_per_site: dict[str, dict] = {
        site: {
            "std": np.asarray(stats["std"], dtype=np.float64),
            "bulk_ess": np.asarray(stats["bulk_ess"], dtype=np.float64),
            "tail_ess": np.asarray(stats["tail_ess"], dtype=np.float64),
        }
        for site, stats in per_site.items()
    }

    return gt_draws_per_site, gt_summary_per_site


# ---------------------------------------------------------------------------
# Recipe draw loading / re-sampling
# ---------------------------------------------------------------------------


def load_cached_draws(recipe_path: pathlib.Path) -> dict | None:
    """Load per-recipe cached draws from _cache/<stem>.draws.npz.

    Returns dict {site: np.ndarray(n_chains, n_draws[, D])}.
    compute_w1_realm handles 2D scalar-site arrays internally.
    Returns None if no cache file exists.
    """
    cache_path = recipe_path.parent.parent / "_cache" / f"{recipe_path.stem}.draws.npz"
    if not cache_path.exists():
        return None
    raw = np.load(cache_path)
    return {site: raw[site].astype(np.float64) for site in raw.files}


def resample_recipe_draws(
    recipe_path: pathlib.Path,
    n_samples: int,
    *,
    skip_warmup: bool = True,
) -> dict:
    """Re-sample from a committed recipe; return draws dict.

    Returns {site: np.ndarray(n_chains, n_draws[, D])}.
    Uses run_recipe_to_idata with skip_warmup=True (fast) or full warmup.
    """
    from tuningfork.catalog.inspect import load_recipe  # type: ignore
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata  # type: ignore

    recipe = load_recipe(recipe_path.absolute())
    idata = run_recipe_to_idata(
        recipe,
        n_samples=n_samples,
        skip_warmup=skip_warmup,
        _suppress_print=True,
    )
    post = idata.posterior
    # xarray posterior: (chain, draw, *event) dimensions
    return {str(v): np.asarray(post[v], dtype=np.float64) for v in post.data_vars}


# ---------------------------------------------------------------------------
# W1 gate application
# ---------------------------------------------------------------------------


def apply_w1_gate(
    draws: dict,
    gt_summary_per_site: dict,
    gt_draws_per_site: dict,
    *,
    B: int = W1_B,
    alpha: float = W1_ALPHA,
    seed: int = W1_SEED,
):
    """Run compute_w1_realm on draws vs GT.

    Parameters
    ----------
    draws
        {site: np.ndarray(n_chains, n_draws[, D])} — recipe generated samples.
    gt_summary_per_site
        {site: {std, bulk_ess, tail_ess}} — from GT summary_v2.json per_site.
    gt_draws_per_site
        {site: np.ndarray(n_gt_chains, n_gt_draws[, D])} — GT draws.
    """
    from tuningfork.calibration._gate.w1_realm import compute_w1_realm  # type: ignore

    return compute_w1_realm(
        samples=draws,
        ground_truth_summaries=gt_summary_per_site,
        gt_draws=gt_draws_per_site,
        B=B,
        alpha=alpha,
        seed=seed,
        multichain=True,
    )


# ---------------------------------------------------------------------------
# irt_2pl×chees: step-4 runner
# ---------------------------------------------------------------------------


def run_irt2pl_chees(
    *,
    num_chains: int = CHEES_IRT2PL_N_CHAINS,
    n_warmup: int = CHEES_IRT2PL_N_WARMUP,
    n_samples: int = CHEES_IRT2PL_N_SAMPLES,
    seed: int = CHEES_IRT2PL_SEED,
) -> tuple[dict, dict]:
    """Run CHEES warmup + dynamic_hmc sampling on irt_2pl.

    CHEES state post-warmup is already DynamicHMCState — no reinit needed.
    Per-chain (step_size, IMM) from adapted_params; integration_steps_fn /
    next_random_arg_fn / integration_steps_params shared across chains (not
    vmappable; captured in closure per recipe-runner design).

    Returns
    -------
    (draws, gate_stats)
        draws: {site: np.ndarray(num_chains, n_samples[, D])}
        gate_stats: {"rhat_max", "min_bulk_ess", "n_divergences"}
    """
    import blackjax
    import jax
    import jax.numpy as jnp

    from tuningfork.base_method import BASE_METHODS  # type: ignore
    from tuningfork.model import MODELS  # type: ignore
    from tuningfork.model._numpyro import build_logdensity_fn  # type: ignore
    from tuningfork.warmup import WARMUPS  # type: ignore

    print(
        f"  [step4] irt_2pl×chees: nc={num_chains}, n_warmup={n_warmup}, n_samples={n_samples}"
    )

    posterior = MODELS["irt_2pl"]
    base_method = BASE_METHODS["dynamic_hmc"]
    chees_warmup = WARMUPS["chees"]

    # Build logdensity function — signature: build_logdensity_fn(rng_key, entry)
    # → (init_position, logdensity_fn, postprocess_fn)
    rng_key = jax.random.key(seed)
    init_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    _prior_init, logdensity_fn, _postprocess_fn = build_logdensity_fn(
        init_key, posterior
    )

    # Use GT draws mean as init position (more stable than prior sample for D=144)
    gt_raw_path = CATALOG / "irt_2pl" / "groundtruth_samples" / "blackjax" / "draws.npz"
    raw_npz = np.load(gt_raw_path)
    init_pos: dict = {}
    for site in raw_npz.files:
        arr = raw_npz[site]  # (n_gt_chains, n_gt_draws) or (n_gt_chains, n_gt_draws, D)
        flat = arr.reshape(-1, *arr.shape[2:])  # (n_gt_chains*n_gt_draws, *event)
        mean_val = flat.mean(axis=0)  # () for scalars, (D,) for vectors
        init_pos[site] = jnp.asarray(mean_val, dtype=jnp.float32)

    # --- CHEES warmup ---
    print(f"  [step4] Running CHEES warmup ({n_warmup} steps, {num_chains} chains)...")
    t_warmup = time.perf_counter()
    states, adapted_params = chees_warmup.runner(
        warmup_key,
        init_pos,
        n_warmup,
        base_method,
        logdensity_fn=logdensity_fn,
        num_chains=num_chains,
    )
    jax.effects_barrier()
    warmup_elapsed = time.perf_counter() - t_warmup
    print(f"  [step4] Warmup done in {warmup_elapsed:.1f}s")

    # Extract adapted params — numeric per-chain, callables shared (closure-captured)
    _ss = adapted_params["step_size"]  # (num_chains,) scalar broadcast
    _imm = adapted_params["inverse_mass_matrix"]  # (num_chains, d)
    _integ_fn = adapted_params["integration_steps_fn"]  # callable (shared)
    _next_rng_fn = adapted_params.get("next_random_arg_fn")  # callable (shared)
    _integ_params = adapted_params.get("integration_steps_params")  # (1,) (shared)

    print(f"  [step4] Adapted: mean_step_size={float(np.mean(np.asarray(_ss))):.4f}")

    # Build per-step kwargs with shared callables (not vmappable; captured in closure)
    dhmc_kwargs: dict = {"integration_steps_fn": _integ_fn}
    if _next_rng_fn is not None:
        dhmc_kwargs["next_random_arg_fn"] = _next_rng_fn
    if _integ_params is not None:
        dhmc_kwargs["integration_steps_params"] = _integ_params

    # Sampling: vmap over chains, per-chain (step_size, IMM)
    # ChEES states are already DynamicHMCState — pass directly (no reinit).
    # Matches recipe-runner _build_vmapped_inference default path for dynamic_hmc+chees.

    def _step_one_chain(state, key, step_size, imm):
        return blackjax.dynamic_hmc(
            logdensity_fn,
            step_size=step_size,
            inverse_mass_matrix=imm,
            **dhmc_kwargs,
        ).step(key, state)

    def _sample_one_chain(init_state, chain_key, step_size, imm):
        def one_step(carry, _):
            s, k = carry
            k, subk = jax.random.split(k)
            new_s, info = _step_one_chain(s, subk, step_size, imm)
            return (new_s, k), (new_s.position, info.is_divergent)

        _, (positions, divergences) = jax.lax.scan(
            one_step, (init_state, chain_key), None, length=n_samples
        )
        return positions, divergences

    print(f"  [step4] Sampling {n_samples} steps × {num_chains} chains...")
    t_sample = time.perf_counter()

    chain_keys = jax.random.split(sample_key, num_chains)
    all_positions, all_divergences = jax.vmap(_sample_one_chain)(
        states, chain_keys, _ss, _imm
    )
    jax.effects_barrier()
    sample_elapsed = time.perf_counter() - t_sample
    print(f"  [step4] Sampling done in {sample_elapsed:.1f}s")

    # Convert positions to numpy: {site: (num_chains, n_samples, *event)}
    draws: dict[str, np.ndarray] = {}
    if isinstance(all_positions, dict):
        for site, arr in all_positions.items():
            draws[site] = np.asarray(arr, dtype=np.float64)
    else:
        draws["position"] = np.asarray(all_positions, dtype=np.float64)

    n_div = int(np.sum(np.asarray(all_divergences)))

    # Basic convergence stats for reporting (rhat, min_bulk_ess)
    gate_stats: dict = {"n_divergences": n_div}
    try:
        from blackjax.diagnostics import effective_sample_size as ess_fn
        from blackjax.diagnostics import (
            potential_scale_reduction as rhat_fn,  # type: ignore
        )

        rhat_vals, ess_vals = [], []
        for arr in draws.values():
            a = arr if arr.ndim > 2 else arr[:, :, np.newaxis]
            rhat_vals.append(float(np.nanmax(np.asarray(rhat_fn(a)))))
            ess_vals.append(float(np.nanmin(np.asarray(ess_fn(a)))))
        gate_stats["rhat_max"] = float(max(rhat_vals)) if rhat_vals else None
        gate_stats["min_bulk_ess"] = float(min(ess_vals)) if ess_vals else None
    except Exception as exc:
        print(f"  [step4] gate_stats computation failed: {exc}")
        gate_stats["rhat_max"] = None
        gate_stats["min_bulk_ess"] = None

    rhat_str = (
        f"{gate_stats.get('rhat_max'):.4f}" if gate_stats.get("rhat_max") else "n/a"
    )
    ess_str = (
        f"{gate_stats.get('min_bulk_ess'):.1f}"
        if gate_stats.get("min_bulk_ess")
        else "n/a"
    )
    print(
        f"  [step4] rhat_max={rhat_str}, min_bulk_ess={ess_str}, "
        f"n_div={gate_stats.get('n_divergences')}"
    )
    return draws, gate_stats


# ---------------------------------------------------------------------------
# Catalog W1 sweep (step 3)
# ---------------------------------------------------------------------------


def collect_eligible_cells() -> list[tuple[str, pathlib.Path, str]]:
    """Return (model, recipe_path, path_code) for all W1-eligible PASS cells.

    path_code: 'A' (cached draws), 'B' (skip_warmup), 'C' (full warmup),
               'SK' (skip — SMC / VI / no GT / unknown method).
    """
    recipe_map: dict[str, list[pathlib.Path]] = {}
    for p in CATALOG.glob("*/recipes/*.json"):
        if "failed__" in p.name:
            continue
        model = p.parent.parent.name
        recipe_map.setdefault(model, []).append(p)

    # Warmups whose adapted_params include non-serialisable callables
    # (integration_steps_fn, next_random_arg_fn) — skip_warmup=True fails for
    # these; they need a full re-warmup (path C).
    FULL_WARMUP_REQUIRED = frozenset({"chees", "meads"})

    def classify(recipe_path: pathlib.Path) -> tuple[bool, str]:
        model = recipe_path.parent.parent.name
        try:
            d = json.loads(recipe_path.read_text())
        except Exception:
            return False, "SK"

        ae = d.get("gate_evidence", {}).get("auto", {})
        if ae.get("verdict") != "PASS":
            return False, "SK"

        bm = d.get("base_method_name", "")
        if not bm:
            # SMC recipe — base_method_name absent or empty
            return True, "SK"
        if bm in VI_METHODS:
            return False, "SK"

        gt_d = CATALOG / model / "groundtruth_samples" / "blackjax" / "draws.npz"
        gt_s = CATALOG / model / "groundtruth_samples" / "blackjax" / "summary_v2.json"
        if not gt_d.exists() or not gt_s.exists():
            return True, "SK"

        cache = recipe_path.parent.parent / "_cache" / f"{recipe_path.stem}.draws.npz"
        if cache.exists():
            return True, "A"

        # Derive warmup from recipe stem: level__method__warmup[__extra]
        parts = recipe_path.stem.split("__")
        warmup_name = parts[2] if len(parts) >= 3 else ""
        if warmup_name in FULL_WARMUP_REQUIRED:
            # CHEES / MEADS adapted_params contain callables → must re-run warmup
            return True, "C"

        # Check for sidecar IMM: stored as "sidecar" in base_method_params (new
        # schema) OR top-level inverse_mass_matrix (old schema).  skip_warmup=True
        # fails because the sidecar file path is not stored in the recipe JSON.
        bmp = d.get("base_method_params", {})
        imm = bmp.get("inverse_mass_matrix") or d.get("inverse_mass_matrix")
        if imm == "sidecar":
            # Route to path C (full warmup regenerates the IMM)
            return True, "C"

        if bm in SKIP_WARMUP_METHODS:
            return True, "B"
        if bm in MCLMC_METHODS or bm in LAPLACE_METHODS:
            return True, "C"
        return True, "SK"

    seen_models = set(recipe_map.keys())
    ordered: list[tuple[str, pathlib.Path, str]] = []
    for model in _BATCH_ORDER:
        for p in sorted(recipe_map.get(model, [])):
            eligible, code = classify(p)
            if eligible:
                ordered.append((model, p, code))
    for model in sorted(seen_models - set(_BATCH_ORDER)):
        for p in sorted(recipe_map.get(model, [])):
            eligible, code = classify(p)
            if eligible:
                ordered.append((model, p, code))
    return ordered


def process_catalog_cell(
    model: str,
    recipe_path: pathlib.Path,
    path_code: str,
) -> dict:
    """Process one catalog cell; return result dict."""
    if path_code == "SK":
        return {
            "status": "SKIP",
            "path_code": "SK",
            "reason": "SMC / VI / no GT / unknown",
        }

    gt_data = load_gt_data(model)
    if gt_data is None:
        return {"status": "SKIP", "path_code": "SK", "reason": "GT data not found"}
    gt_draws_per_site, gt_summary_per_site = gt_data

    t0 = time.perf_counter()
    resample_elapsed = 0.0
    try:
        if path_code == "A":
            draws = load_cached_draws(recipe_path)
            if draws is None:
                return {
                    "status": "ERROR",
                    "path_code": "A",
                    "error": "cache file disappeared",
                }
        elif path_code == "B":
            draws = resample_recipe_draws(recipe_path, RESAMPLE_N, skip_warmup=True)
            resample_elapsed = time.perf_counter() - t0
        elif path_code == "C":
            draws = resample_recipe_draws(recipe_path, RESAMPLE_N, skip_warmup=False)
            resample_elapsed = time.perf_counter() - t0
        else:
            return {"status": "SKIP", "path_code": path_code, "reason": "unknown code"}
    except Exception as exc:
        return {
            "status": "ERROR",
            "path_code": path_code,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }

    t_w1 = time.perf_counter()
    try:
        w1_result = apply_w1_gate(draws, gt_summary_per_site, gt_draws_per_site)
    except Exception as exc:
        return {
            "status": "ERROR",
            "path_code": path_code,
            "error": f"W1 gate failed: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
    w1_elapsed = time.perf_counter() - t_w1
    total_elapsed = time.perf_counter() - t0

    recipe_data = json.loads(recipe_path.read_text())
    baseline_verdict = (
        recipe_data.get("gate_evidence", {}).get("auto", {}).get("verdict", "UNKNOWN")
    )
    flip = baseline_verdict == "PASS" and w1_result.verdict == "FAIL"

    return {
        "status": "OK",
        "path_code": path_code,
        "baseline_verdict": baseline_verdict,
        "w1_verdict": w1_result.verdict,
        "flip": flip,
        "max_w1_sigma": float(w1_result.max_w1_sigma),
        "floor_of_max": float(w1_result.floor_of_max),
        "frac_failing_dims": float(w1_result.frac_failing_dims),
        "tau_frac": (
            float(w1_result.tau_frac) if not np.isnan(w1_result.tau_frac) else None
        ),
        "max_prong_verdict": w1_result.max_prong_verdict,
        "frac_prong_verdict": w1_result.frac_prong_verdict,
        "n_dims": w1_result.n_dims,
        "n_heavy_tail_dims": w1_result.n_heavy_tail_dims,
        "resample_elapsed_s": (
            round(resample_elapsed, 1) if path_code in ("B", "C") else None
        ),
        "w1_elapsed_s": round(w1_elapsed, 1),
        "elapsed_s": round(total_elapsed, 1),
    }


# ---------------------------------------------------------------------------
# irt_2pl×chees step-4 wrapper
# ---------------------------------------------------------------------------


def process_irt2pl_chees(
    *,
    num_chains: int = CHEES_IRT2PL_N_CHAINS,
    n_warmup: int = CHEES_IRT2PL_N_WARMUP,
    n_samples: int = CHEES_IRT2PL_N_SAMPLES,
) -> dict:
    """Run irt_2pl×chees re-gate and return result dict."""
    t0 = time.perf_counter()
    try:
        draws, gate_stats = run_irt2pl_chees(
            num_chains=num_chains,
            n_warmup=n_warmup,
            n_samples=n_samples,
        )
    except Exception as exc:
        return {
            "status": "ERROR",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }

    gt_data = load_gt_data("irt_2pl")
    if gt_data is None:
        return {"status": "ERROR", "error": "irt_2pl GT data not found"}
    gt_draws_per_site, gt_summary_per_site = gt_data

    try:
        w1_result = apply_w1_gate(
            draws, gt_summary_per_site, gt_draws_per_site, B=STEP4_W1_B
        )
    except Exception as exc:
        return {
            "status": "ERROR",
            "error": f"W1 gate failed: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }

    elapsed = time.perf_counter() - t0
    return {
        "status": "OK",
        "model": "irt_2pl",
        "warmup": "chees",
        "sampler": "dynamic_hmc",
        "num_chains": num_chains,
        "n_warmup": n_warmup,
        "n_samples": n_samples,
        "rhat_max": gate_stats.get("rhat_max"),
        "min_bulk_ess": gate_stats.get("min_bulk_ess"),
        "n_divergences": gate_stats.get("n_divergences"),
        "w1_verdict": w1_result.verdict,
        "max_prong_verdict": w1_result.max_prong_verdict,
        "frac_prong_verdict": w1_result.frac_prong_verdict,
        "max_w1_sigma": float(w1_result.max_w1_sigma),
        "floor_of_max": float(w1_result.floor_of_max),
        "frac_failing_dims": float(w1_result.frac_failing_dims),
        "tau_frac": (
            float(w1_result.tau_frac) if not np.isnan(w1_result.tau_frac) else None
        ),
        "n_dims": w1_result.n_dims,
        "n_heavy_tail_dims": w1_result.n_heavy_tail_dims,
        "w1_sigma_per_dim_sample": w1_result.w1_sigma_per_dim[:20].tolist(),
        "floor_per_dim_sample": w1_result.floor_per_dim[:20].tolist(),
        "khat_per_dim_sample": w1_result.khat_per_dim[:20].tolist(),
        "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_SEP = "=" * 60  # separator used in main output blocks


def main() -> None:
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    cells = collect_eligible_cells()
    total = len(cells)
    print(f"W1 full-catalog re-validation: {total} candidate cells")
    print(f"  W1_B={W1_B} (sweep)  STEP4_W1_B={STEP4_W1_B} (irt_2pl×chees)")
    print(f"Checkpoint: {RESULTS_FILE}")

    from collections import Counter

    code_counts = Counter(code for _, _, code in cells)
    for code in ("A", "B", "C", "SK"):
        if code in code_counts:
            label = {
                "A": "cached (fast)",
                "B": "skip_warmup",
                "C": "full warmup",
                "SK": "skip",
            }[code]
            print(f"  Path {code} ({label}): {code_counts[code]}")

    results: dict = {}
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())
        done = sum(
            1
            for k, v in results.items()
            if k != "_meta" and v.get("status") in ("OK", "SKIP")
        )
        print(f"Resuming: {done}/{total + 1} already done (incl. step4)\n")

    # Persist run metadata on every invocation (overwrites previous _meta if any).
    results["_meta"] = {
        "W1_B": W1_B,
        "STEP4_W1_B": STEP4_W1_B,
        "W1_ALPHA": W1_ALPHA,
        "W1_SEED": W1_SEED,
        "RESAMPLE_N": RESAMPLE_N,
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2) + "\n")

    sweep_start = time.perf_counter()
    flips: list[str] = []

    for i, (model, recipe_path, path_code) in enumerate(cells, 1):
        key = f"{model}/{recipe_path.stem}"

        if key in results and results[key].get("status") in ("OK", "SKIP"):
            r = results[key]
            flip = r.get("flip", False)
            w1v = r.get("w1_verdict", "?")
            print(f"[{i:3d}/{total}] RESUME ({w1v}) [{path_code}]: {key}")
            if flip:
                flips.append(key)
            continue

        print(f"\n[{i:3d}/{total}] {key} [{path_code}]", flush=True)
        result = process_catalog_cell(model, recipe_path, path_code)
        results[key] = result
        RESULTS_FILE.write_text(json.dumps(results, indent=2) + "\n")
        sys.stdout.flush()

        status = result.get("status", "?")
        if status == "ERROR":
            print(f"  => ERROR: {result.get('error', '')[:120]}")
        elif status == "SKIP":
            print(f"  => SKIP ({result.get('reason', '')})")
        else:
            flip = result.get("flip", False)
            w1v = result.get("w1_verdict", "?")
            mxw = result.get("max_w1_sigma")
            fom = result.get("floor_of_max")
            elapsed = result.get("elapsed_s", 0)
            mxw_s = f"{mxw:.4f}" if mxw is not None else "nan"
            fom_s = f"{fom:.4f}" if fom is not None else "nan"
            flip_s = "  *** FLIP ***" if flip else ""
            print(
                f"  => W1={w1v} (max_w1σ={mxw_s} vs floor={fom_s}) "
                f"[{path_code}] {elapsed:.0f}s{flip_s}"
            )
            if flip:
                flips.append(key)

    # --- Step 4: irt_2pl×chees ---
    step4_key = "irt_2pl/chees"
    print(f"\n{_SEP}")
    print("STEP 4: irt_2pl×chees re-gate under honest multichain GT")
    print(_SEP)

    if step4_key in results and results[step4_key].get("status") == "OK":
        r4 = results[step4_key]
        print(
            f"RESUME step4: w1_verdict={r4.get('w1_verdict')} "
            f"max_w1σ={r4.get('max_w1_sigma'):.4f} "
            f"floor={r4.get('floor_of_max'):.4f}"
        )
    else:
        step4_result = process_irt2pl_chees()
        results[step4_key] = step4_result
        RESULTS_FILE.write_text(json.dumps(results, indent=2) + "\n")
        sys.stdout.flush()

        if step4_result.get("status") == "ERROR":
            print(f"  => ERROR: {step4_result.get('error', '')[:200]}")
        else:
            print(f"  => W1 verdict: {step4_result.get('w1_verdict')}")
            print(
                f"     max_prong: {step4_result.get('max_prong_verdict')} "
                f"(max_w1σ={step4_result.get('max_w1_sigma'):.4f} "
                f"vs floor={step4_result.get('floor_of_max'):.4f})"
            )
            print(
                f"     frac_prong: {step4_result.get('frac_prong_verdict')} "
                f"(frac={step4_result.get('frac_failing_dims'):.4f} "
                f"vs τ_frac={step4_result.get('tau_frac')})"
            )
            print(
                f"     k̂>0.7 trims: {step4_result.get('n_heavy_tail_dims')}"
                f"/{step4_result.get('n_dims')}"
            )
            print(
                f"     rhat_max={step4_result.get('rhat_max')} "
                f"min_bulk_ess={step4_result.get('min_bulk_ess')} "
                f"n_div={step4_result.get('n_divergences')}"
            )
            print(f"     wall={step4_result.get('elapsed_s'):.1f}s")

    # --- Summary ---
    total_elapsed = time.perf_counter() - sweep_start
    ok_count = sum(
        1 for k, v in results.items() if v.get("status") == "OK" and k != step4_key
    )
    skip_count = sum(1 for v in results.values() if v.get("status") == "SKIP")
    err_count = sum(1 for v in results.values() if v.get("status") == "ERROR")
    pass_w1 = sum(
        1
        for k, v in results.items()
        if v.get("status") == "OK"
        and k != step4_key
        and v.get("w1_verdict") in ("PASS", "SKIP")
    )
    fail_w1 = sum(
        1
        for k, v in results.items()
        if v.get("status") == "OK" and k != step4_key and v.get("w1_verdict") == "FAIL"
    )

    print(f"\n{_SEP}")
    print("SWEEP COMPLETE")
    print(f"Total wall: {total_elapsed / 60:.1f} min")
    print(f"  OK={ok_count}  SKIP={skip_count}  ERROR={err_count}")
    print(f"  W1-PASS={pass_w1}  W1-FAIL={fail_w1}  FLIPS={len(flips)}")

    if flips:
        print(
            f"\n*** {len(flips)} FLIP(S): currently-PASS cells newly FAILed by W1 gate ***"
        )
        for k in flips:
            r = results[k]
            print(
                f"  {k}: max_w1σ={r.get('max_w1_sigma'):.4f} "
                f"floor={r.get('floor_of_max'):.4f}"
            )
        print("STOP: Report to @tl for diagnosis. Do not remediate.")
    else:
        print("\nGO: 0 flips — no currently-PASS cell newly FAILed from W1 gate.")

    print(f"\nCheckpoint: {RESULTS_FILE}")
    sys.exit(1 if err_count > 0 else 0)


if __name__ == "__main__":
    main()
