# Copyright 2026- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Phase 3 LOW-effort recipe emission — warmup + sampling + auto-gate pipeline.

Implements the full ``warmup → sample → auto_gate → Recipe.LOW`` flow for the
wadapt-hmc-sweep Phase 3.  Each cell runs:

1. ``warmup.runner`` (single chain, ``n_warmup=2000``) to get adapted
   ``(step_size, inverse_mass_matrix)``.
2. ``base_method.factory(..., **kernel_params)`` + ``run_inference_algorithm``
   for ``n_samples=10000`` post-warmup draws.
3. ``auto_gate(samples, infos)`` to classify PASS / REVIEW / FAIL.
4. On PASS: saves ``catalog/<model>/recipes/low__<sampler>__<warmup>.json``
   and an IMM sidecar ``.imm.npz`` when the IMM has more than 50 elements.

Usage (CLI):

    JAX_PLATFORM_NAME=cpu uv run python -m tuningfork.recipes._phase3_emit \
        --model mvn_10 \
        --warmup window_adaptation_diag_imm \
        --sampler nuts

The module is **not** exposed through the public ``tuningfork.recipes``
``__init__.py``; it is an internal generator-layer script.

Phase 3 spec (from ``wadapt-hmc-sweep.md`` §8 / §10):
    - ``n_warmup=2000``, ``n_samples=10000``, ``seed=20260517``
    - ``target_acceptance`` from ``base_method`` default (default 0.8)
    - ``n_chunks=4`` (split-Rhat)
    - PASS verdict → emit LOW recipe; FAIL/REVIEW → write note to
      ``/tmp/wadapt-phase3-outcomes.md`` and exit non-zero.
"""

import dataclasses
import datetime
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np
from blackjax.util import run_inference_algorithm

from tuningfork._version import __version__ as _tuningfork_version
from tuningfork.base_method import BASE_METHODS
from tuningfork.calibration.statistician_gate import auto_gate
from tuningfork.calibration.tune import default_params_for
from tuningfork.metrics.grad_counter import total_grad_evals
from tuningfork.metrics.headline import min_bulk_ess_per_grad
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.recipes._base import Effort, Recipe
from tuningfork.recipes._instructions import render_instructions
from tuningfork.warmup import WARMUPS
from tuningfork.warmup._base import squeeze_single_chain

__all__ = ["emit_low_recipe_for_cell", "CellResult"]

# Phase 3 canonical parameters (locked in wadapt-hmc-sweep.md Decision 3)
PHASE3_N_WARMUP: int = 2000
PHASE3_N_SAMPLES: int = 10000
PHASE3_SEED: int = 20260517
PHASE3_N_CHUNKS: int = 4
PHASE3_TARGET_ACCEPTANCE: float = 0.8

# Catalog root (relative to this file: tuningfork/tuningfork/catalog/)
_CATALOG_ROOT: Path = Path(__file__).parent.parent / "catalog"

# Outcomes log for FAIL / REVIEW cells
_OUTCOMES_FILE: Path = Path("/tmp/wadapt-phase3-outcomes.md")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class CellResult:
    """Outcome of one Phase 3 emit attempt.

    Parameters
    ----------
    model_name, warmup_name, sampler_name
        Cell identity.
    verdict
        ``"PASS"``, ``"REVIEW"``, ``"FAIL"``, or ``"ERROR"``.
    recipe_path
        Path to the saved ``low__*.json`` file (only on PASS).
    imm_sidecar_path
        Path to the saved ``.imm.npz`` file (only on PASS + large IMM).
    gate_rhat_max, gate_min_ess, gate_n_div
        Auto-gate metrics (``None`` on ERROR before gate ran).
    wall_seconds
        Total wall time for warmup + sampling + gate.
    note
        One-line summary of what happened (appended to outcomes file on non-PASS).
    """

    def __init__(
        self,
        *,
        model_name: str,
        warmup_name: str,
        sampler_name: str,
        verdict: str,
        recipe_path: Path | None = None,
        imm_sidecar_path: str | None = None,
        gate_rhat_max: float | None = None,
        gate_min_ess: float | None = None,
        gate_n_div: int | None = None,
        wall_seconds: float = 0.0,
        note: str = "",
    ):
        self.model_name = model_name
        self.warmup_name = warmup_name
        self.sampler_name = sampler_name
        self.verdict = verdict
        self.recipe_path = recipe_path
        self.imm_sidecar_path = imm_sidecar_path
        self.gate_rhat_max = gate_rhat_max
        self.gate_min_ess = gate_min_ess
        self.gate_n_div = gate_n_div
        self.wall_seconds = wall_seconds
        self.note = note

    def __repr__(self) -> str:
        return (
            f"CellResult({self.model_name}/{self.warmup_name}/{self.sampler_name} "
            f"verdict={self.verdict} wall={self.wall_seconds:.1f}s)"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_blackjax_version() -> str:
    try:
        import blackjax

        return getattr(blackjax, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unavailable"


def _get_jax_version() -> str:
    try:
        return jax.__version__
    except Exception:  # noqa: BLE001
        return "unavailable"


def _now_utc_iso() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_jsonable(d: dict[str, Any]) -> dict[str, Any]:
    """Coerce JAX/numpy arrays in a flat dict to JSON-serialisable Python types."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, jax.Array):
            out[k] = np.asarray(v).tolist()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _append_outcome(model: str, warmup: str, sampler: str, message: str) -> None:
    """Append one line to the outcomes log file."""
    _OUTCOMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _OUTCOMES_FILE.open("a") as fh:
        fh.write(f"- {model} x {warmup} x {sampler}: {message}\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main emit function
# ---------------------------------------------------------------------------


def emit_low_recipe_for_cell(
    model_name: str,
    warmup_name: str,
    sampler_name: str,
    *,
    n_warmup: int = PHASE3_N_WARMUP,
    n_samples: int = PHASE3_N_SAMPLES,
    seed: int = PHASE3_SEED,
    n_chunks: int = PHASE3_N_CHUNKS,
    catalog_root: Path = _CATALOG_ROOT,
    outcomes_file: Path = _OUTCOMES_FILE,
    verbose: bool = True,
) -> CellResult:
    """Run warmup + sampling + auto-gate for one cell; emit LOW recipe on PASS.

    Parameters
    ----------
    model_name
        Registry key in ``MODELS``, e.g. ``"mvn_10"``.
    warmup_name
        Registry key in ``WARMUPS``, e.g. ``"window_adaptation_diag_imm"``.
    sampler_name
        Registry key in ``BASE_METHODS``, e.g. ``"nuts"``.
    n_warmup
        Warmup steps (default ``PHASE3_N_WARMUP`` = 2000).
    n_samples
        Post-warmup sampler steps (default ``PHASE3_N_SAMPLES`` = 10000).
    seed
        Master JAX random seed (default ``PHASE3_SEED`` = 20260517).
    n_chunks
        Split-Rhat rechunk count (default 4).
    catalog_root
        Root of the catalog directory (default: ``tuningfork/catalog/``).
    outcomes_file
        File to append FAIL / REVIEW notes to.
    verbose
        Print progress to stdout.

    Returns
    -------
    CellResult
        Outcome summary for this cell.
    """

    def _log(msg: str) -> None:
        if verbose:
            print(msg)
            sys.stdout.flush()

    _log(f"\n=== {model_name} x {warmup_name} x {sampler_name} ===")
    t_start = time.perf_counter()

    # --- Validate registry membership ---
    _registry_checks: list[tuple[str, str, str]] = [
        (model_name, "model", "MODELS"),
        (warmup_name, "warmup", "WARMUPS"),
        (sampler_name, "sampler", "BASE_METHODS"),
    ]
    for _key, _label, _reg_name in _registry_checks:
        _valid = (
            _key in MODELS
            if _reg_name == "MODELS"
            else (_key in WARMUPS if _reg_name == "WARMUPS" else _key in BASE_METHODS)
        )
        if not _valid:
            note = f"ERROR: {_label} {_key!r} not in {_reg_name} registry"
            _log(f"  {note}")
            _append_outcome(model_name, warmup_name, sampler_name, note)
            return CellResult(
                model_name=model_name,
                warmup_name=warmup_name,
                sampler_name=sampler_name,
                verdict="ERROR",
                note=note,
            )

    posterior = MODELS[model_name]
    warmup = WARMUPS[warmup_name]
    base_method = BASE_METHODS[sampler_name]

    # --- Compatibility check ---
    if not warmup.is_compatible(sampler_name):
        note = f"SKIP: {warmup_name} incompatible with {sampler_name}"
        _log(f"  {note}")
        _append_outcome(model_name, warmup_name, sampler_name, note)
        return CellResult(
            model_name=model_name,
            warmup_name=warmup_name,
            sampler_name=sampler_name,
            verdict="ERROR",
            note=note,
        )

    # --- Build logdensity ---
    master_key = jax.random.key(seed)
    init_key, warmup_key, sample_key = jax.random.split(master_key, 3)

    try:
        init_position, logdensity_fn, _model_data = build_logdensity_fn(
            init_key, posterior
        )
    except Exception as exc:
        note = f"ERROR: build_logdensity_fn failed: {type(exc).__name__}: {exc}"
        _log(f"  {note}")
        _append_outcome(model_name, warmup_name, sampler_name, note)
        return CellResult(
            model_name=model_name,
            warmup_name=warmup_name,
            sampler_name=sampler_name,
            verdict="ERROR",
            note=note,
        )

    # --- Warmup (single chain) ---
    _log(f"  Warmup ({warmup_name}, n_warmup={n_warmup})...")
    t_warmup0 = time.perf_counter()
    try:
        batched_state, batched_params = warmup.runner(
            warmup_key,
            init_position,
            n_warmup,
            base_method,
            logdensity_fn=logdensity_fn,
            num_chains=1,
        )
        adapted_state, adapted_params = squeeze_single_chain(
            batched_state, batched_params
        )
    except Exception as exc:
        note = f"FAIL warmup error: {type(exc).__name__}: {exc}"
        _log(f"  {note}")
        _append_outcome(model_name, warmup_name, sampler_name, note)
        return CellResult(
            model_name=model_name,
            warmup_name=warmup_name,
            sampler_name=sampler_name,
            verdict="FAIL",
            wall_seconds=time.perf_counter() - t_start,
            note=note,
        )
    t_warmup = time.perf_counter() - t_warmup0

    step_size_val = adapted_params.get("step_size", None)
    if step_size_val is not None:
        ss = float(np.asarray(step_size_val).ravel()[0])
        _log(f"  Warmup done in {t_warmup:.1f}s. step_size={ss:.4g}")
    else:
        _log(f"  Warmup done in {t_warmup:.1f}s.")

    # Check for NaN/Inf in adapted params
    for k, v in adapted_params.items():
        arr = np.asarray(v)
        if not np.all(np.isfinite(arr)):
            note = f"FAIL warmup produced NaN/Inf in {k}"
            _log(f"  {note}")
            _append_outcome(model_name, warmup_name, sampler_name, note)
            return CellResult(
                model_name=model_name,
                warmup_name=warmup_name,
                sampler_name=sampler_name,
                verdict="FAIL",
                wall_seconds=time.perf_counter() - t_start,
                note=note,
            )

    # --- Build kernel params: defaults merged with adapted ---
    default_params = default_params_for(base_method)
    clean_adapted = {k: v for k, v in adapted_params.items() if not k.startswith("_")}
    kernel_params: dict[str, Any] = {**default_params, **clean_adapted}

    # --- Sampling ---
    _log(f"  Sampling ({sampler_name}, n_samples={n_samples})...")
    t_sample0 = time.perf_counter()
    try:
        kernel = base_method.factory(logdensity_fn, **kernel_params)
        _, (states, infos) = run_inference_algorithm(
            rng_key=sample_key,
            inference_algorithm=kernel,
            num_steps=n_samples,
            initial_state=adapted_state,
        )
        positions = states.position  # dict {param: (n_samples, *shape)}
    except Exception as exc:
        note = f"FAIL sampler error: {type(exc).__name__}: {exc}"
        _log(f"  {note}")
        _append_outcome(model_name, warmup_name, sampler_name, note)
        return CellResult(
            model_name=model_name,
            warmup_name=warmup_name,
            sampler_name=sampler_name,
            verdict="FAIL",
            wall_seconds=time.perf_counter() - t_start,
            note=note,
        )
    t_sample = time.perf_counter() - t_sample0
    t_total = time.perf_counter() - t_start
    _log(f"  Sampling done in {t_sample:.1f}s (total {t_total:.1f}s).")

    # Check for non-finite positions
    for site, arr in positions.items():
        arr_np = np.asarray(arr)
        if not np.all(np.isfinite(arr_np)):
            note = f"FAIL sampler produced NaN/Inf in {site}"
            _log(f"  {note}")
            _append_outcome(model_name, warmup_name, sampler_name, note)
            return CellResult(
                model_name=model_name,
                warmup_name=warmup_name,
                sampler_name=sampler_name,
                verdict="FAIL",
                wall_seconds=t_total,
                note=note,
            )

    # --- Auto-gate ---
    _log("  Running auto-gate...")
    gate_verdict = auto_gate(
        positions,
        infos,
        posterior=posterior,
        n_chunks=n_chunks,
    )
    _log(
        f"  Gate: {gate_verdict.verdict}, "
        f"rhat_max={gate_verdict.rhat_max:.4f}, "
        f"min_ess={gate_verdict.min_bulk_ess:.1f}, "
        f"n_div={gate_verdict.n_divergences}"
    )

    if gate_verdict.verdict != "PASS":
        note = (
            f"{gate_verdict.verdict} "
            f"rhat={gate_verdict.rhat_max:.4f} "
            f"ess={gate_verdict.min_bulk_ess:.1f} "
            f"div={gate_verdict.n_divergences}"
        )
        _log(f"  => gate {note}")
        _append_outcome(model_name, warmup_name, sampler_name, note)
        return CellResult(
            model_name=model_name,
            warmup_name=warmup_name,
            sampler_name=sampler_name,
            verdict=gate_verdict.verdict,
            gate_rhat_max=gate_verdict.rhat_max,
            gate_min_ess=gate_verdict.min_bulk_ess,
            gate_n_div=gate_verdict.n_divergences,
            wall_seconds=t_total,
            note=note,
        )

    # --- Build headline metric ---
    mc_positions = {k: np.asarray(v)[np.newaxis, ...] for k, v in positions.items()}
    grad_evals = total_grad_evals(infos, base_method.grad_count_per_step)
    headline: float | None = None
    if grad_evals > 0:
        headline = float(min_bulk_ess_per_grad(mc_positions, grad_evals))

    # --- Build recipe ---
    _log("  Building LOW recipe...")
    jsonable_params = _to_jsonable(kernel_params)
    imm_arr: np.ndarray | None = None
    imm_raw = adapted_params.get("inverse_mass_matrix", None)
    if imm_raw is not None:
        imm_arr = np.asarray(imm_raw)
        if imm_arr.size > 50:
            jsonable_params["inverse_mass_matrix"] = "sidecar"
        else:
            jsonable_params["inverse_mass_matrix"] = imm_arr.tolist()

    tuning_seed = int(jax.random.bits(warmup_key, dtype="uint32"))

    gate_evidence = {
        "auto": gate_verdict.to_dict(),
        "override": {"reason": "", "statistician_id": "", "decision": ""},
    }

    recipe_kwargs: dict[str, Any] = dict(
        model_name=posterior.name,
        base_method_name=base_method.name,
        warmup_name=warmup.name,
        effort=Effort.LOW,
        base_method_params=jsonable_params,
        warmup_params={
            "n_warmup": n_warmup,
            "target_acceptance": PHASE3_TARGET_ACCEPTANCE,
        },
        headline_metric=headline,
        sample_quality=None,
        calibration_budget={
            "trials": 0,
            "wall_seconds_estimate": t_total,
            "n_warmup": n_warmup,
            "n_samples": n_samples,
        },
        difficulty=None,
        instructions="",
        notes="",
        tuning_seed=tuning_seed,
        tuningfork_version=_tuningfork_version,
        blackjax_version=_get_blackjax_version(),
        jax_version=_get_jax_version(),
        timestamp_utc=_now_utc_iso(),
        gate_evidence=gate_evidence,
        inverse_mass_matrix_path=None,
    )
    provisional = Recipe(**recipe_kwargs)
    recipe_kwargs["instructions"] = render_instructions(provisional)
    recipe = Recipe(**recipe_kwargs)

    # --- Save recipe ---
    recipe_path = recipe.save(catalog_root)
    _log(f"  Saved recipe: {recipe_path}")

    # --- Save IMM sidecar if needed ---
    imm_sidecar_rel: str | None = None
    if imm_arr is not None and imm_arr.size > 50:
        imm_sidecar_rel = recipe.save_imm_sidecar(catalog_root, imm_arr)
        # Rebuild recipe with sidecar path (Recipe is frozen)
        recipe = dataclasses.replace(recipe, inverse_mass_matrix_path=imm_sidecar_rel)
        recipe.save(catalog_root)
        _log(f"  Saved IMM sidecar: {imm_sidecar_rel}")

    _log(f"  PASS. headline={headline:.4g}" if headline is not None else "  PASS.")
    return CellResult(
        model_name=model_name,
        warmup_name=warmup_name,
        sampler_name=sampler_name,
        verdict="PASS",
        recipe_path=recipe_path,
        imm_sidecar_path=imm_sidecar_rel,
        gate_rhat_max=gate_verdict.rhat_max,
        gate_min_ess=gate_verdict.min_bulk_ess,
        gate_n_div=gate_verdict.n_divergences,
        wall_seconds=t_total,
        note=f"PASS rhat={gate_verdict.rhat_max:.4f} ess={gate_verdict.min_bulk_ess:.1f} div={gate_verdict.n_divergences}",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:
    """CLI: emit LOW recipe for a single (model, warmup, sampler) cell.

    Usage::

        JAX_PLATFORM_NAME=cpu uv run python -m tuningfork.recipes._phase3_emit \
            --model mvn_10 \
            --warmup window_adaptation_diag_imm \
            --sampler nuts

    Exits 0 on PASS, 1 on FAIL/REVIEW/ERROR.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Phase 3 LOW emit: warmup + sample + auto-gate for one "
            "(model, warmup, sampler) cell.  Exits 0 on PASS."
        )
    )
    parser.add_argument(
        "--model", required=True, help="Model name from MODELS registry"
    )
    parser.add_argument(
        "--warmup",
        required=True,
        help="Warmup name from WARMUPS registry",
    )
    parser.add_argument(
        "--sampler", required=True, help="Sampler name from BASE_METHODS registry"
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=PHASE3_N_WARMUP,
        help=f"Warmup steps (default {PHASE3_N_WARMUP})",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=PHASE3_N_SAMPLES,
        help=f"Post-warmup samples (default {PHASE3_N_SAMPLES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=PHASE3_SEED,
        help=f"JAX random seed (default {PHASE3_SEED})",
    )
    args = parser.parse_args()

    result = emit_low_recipe_for_cell(
        model_name=args.model,
        warmup_name=args.warmup,
        sampler_name=args.sampler,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    sys.exit(0 if result.verdict == "PASS" else 1)


if __name__ == "__main__":
    _main()
