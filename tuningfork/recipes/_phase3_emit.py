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

1. ``warmup.runner(num_chains=4)`` to get per-chain adapted
   ``(step_size, inverse_mass_matrix)`` for ``n_warmup=1000`` steps each.
2. ``jax.vmap(run_one_chain)`` over the four chains: each chain rebuilds its
   own kernel with its own adapted params and runs ``n_samples=1000``
   post-warmup draws via ``blackjax.util.run_inference_algorithm``.
3. ``auto_gate(samples, infos)`` over the ``(num_chains, n_samples, *event)``
   shape to classify PASS / REVIEW / FAIL.
4. On PASS: saves ``catalog/<model>/recipes/low__<sampler>__<warmup>.json``
   pinning chain 0's (step_size, IMM); IMM sidecar when ``imm.size > 50``.

Usage (CLI):

    JAX_PLATFORM_NAME=cpu uv run python -m tuningfork.recipes._phase3_emit \
        --model mvn_10 \
        --warmup window_adaptation_diag_imm \
        --sampler nuts

The module is **not** exposed through the public ``tuningfork.recipes``
``__init__.py``; it is an internal generator-layer script.

Phase 3 spec (per worklog/decisions/2026-05-11-phase6-visualization-diagnostics.md):
    - ``n_warmup=1000``, ``n_samples=1000``, ``num_chains=4`` (quick mode)
    - ``seed=20260517`` (master); per-chain keys split internally
    - ``target_acceptance`` from ``base_method`` default (default 0.8)
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
from blackjax.mcmc.laplace_marginal import laplace_marginal_factory
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
from tuningfork.warmup._laplace_adapter import LAPLACE_METHOD_NAMES

__all__ = ["emit_low_recipe_for_cell", "CellResult"]

# ---------------------------------------------------------------------------
# Laplace phi/theta split table — model-specific
# ---------------------------------------------------------------------------
# For laplace_* cells the recipe pipeline needs to split the joint position into
# phi (hyperparameters, the subspace the sampler operates on) and theta (latent
# variables that are analytically marginalised via the Laplace approximation).
#
# Structure: model_name → (phi_site_names, theta_site_names)
# All names must match the numpyro.sample site names in the model.
#
# Models currently in scope for laplace_* (per §6 Phase 2b eligibility):
#   eight_schools_ncp: phi=(mu, tau), theta=(theta_raw,)
#
# radon and irt_2pl are predicted MEDIUM (not LOW) — not needed here yet.
# This table is extended as Phase 4 adds more models.
_LAPLACE_PHI_THETA_SPLITS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "eight_schools_ncp": (("mu", "tau"), ("theta_raw",)),
}

# Phase 3 canonical parameters
# (4 chains x 1000 samples = "quick mode" non-groundtruth recipe protocol per
#  worklog/decisions/2026-05-11-phase6-visualization-diagnostics.md § Section 0;
#  matches auto_gate's `min_bulk_ess >= 400` calibration. Use `quick` for LOW
#  recipes; MEDIUM/HIGH should bump `n_samples` to 4000 via CLI override.)
PHASE3_N_WARMUP: int = 1000
PHASE3_N_SAMPLES: int = 1000
PHASE3_NUM_CHAINS: int = 4
PHASE3_SEED: int = 20260517
PHASE3_N_CHUNKS: int = 4  # for split-R̂; ignored when samples are multi-chain
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


def _build_laplace_components(
    model_name: str,
    full_position: dict[str, Any],
    joint_logdensity_fn: Any,
) -> tuple[dict[str, Any], Any, Any, Any] | None:
    """Build laplace pipeline components from the full joint position.

    Returns ``(phi_init, log_joint_fn, theta_init, marginal_logdensity_fn)``
    for use in warmup (marginal) and sampling (log_joint_fn + theta_init).

    Returns ``None`` if the model is not in the phi/theta split table.

    Parameters
    ----------
    model_name
        Registry key, e.g. ``"eight_schools_ncp"``.
    full_position
        Full unconstrained position dict from ``build_logdensity_fn``.
    joint_logdensity_fn
        Joint logdensity ``phi ∪ theta → float`` from ``build_logdensity_fn``.
        Used to build the factored ``log_joint_fn(theta, phi)`` for the
        laplace_* kernel.
    """
    if model_name not in _LAPLACE_PHI_THETA_SPLITS:
        return None

    phi_sites, theta_sites = _LAPLACE_PHI_THETA_SPLITS[model_name]
    phi_init = {k: full_position[k] for k in phi_sites}
    theta_init = {k: full_position[k] for k in theta_sites}

    def log_joint_fn(theta: dict[str, Any], phi: dict[str, Any]) -> Any:
        return joint_logdensity_fn({**theta, **phi})

    laplace = laplace_marginal_factory(log_joint_fn, theta_init)

    def marginal_logdensity_fn(phi: dict[str, Any]) -> Any:
        lp, _theta_star = laplace(phi)
        return lp

    return phi_init, log_joint_fn, theta_init, marginal_logdensity_fn


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
    num_chains: int = PHASE3_NUM_CHAINS,
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
        Warmup steps per chain (default ``PHASE3_N_WARMUP`` = 1000).
    n_samples
        Post-warmup sampler steps per chain (default ``PHASE3_N_SAMPLES`` = 1000).
    num_chains
        Number of independent chains run in parallel via ``jax.vmap``
        (default ``PHASE3_NUM_CHAINS`` = 4).  The non-groundtruth recipe
        protocol (per `worklog/decisions/2026-05-11-phase6-visualization-
        diagnostics.md` § Section 0) is 4 chains × 1000 quick mode.
    seed
        Master JAX random seed (default ``PHASE3_SEED`` = 20260517).
    n_chunks
        Split-Rhat rechunk count if samples come in single-chain layout;
        ignored when samples are already multi-chain (default 4).
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

    # --- Laplace-* special path ---
    # For laplace_* samplers the warmup must run on the Laplace marginal
    # logdensity over phi (not the joint). The sampling phase then uses
    # log_joint_fn + theta_init. We build both here if the sampler is laplace_*.
    is_laplace = sampler_name in LAPLACE_METHOD_NAMES
    laplace_log_joint_fn: Any = None
    laplace_theta_init: Any = None

    if is_laplace:
        laplace_result = _build_laplace_components(
            model_name, init_position, logdensity_fn
        )
        if laplace_result is None:
            note = (
                f"ERROR: laplace_* sampler requested but {model_name!r} has no "
                "phi/theta split in _LAPLACE_PHI_THETA_SPLITS — cannot build "
                "marginal logdensity. Add the split to the table in _phase3_emit.py."
            )
            _log(f"  {note}")
            _append_outcome(model_name, warmup_name, sampler_name, note)
            return CellResult(
                model_name=model_name,
                warmup_name=warmup_name,
                sampler_name=sampler_name,
                verdict="ERROR",
                note=note,
            )
        init_position, laplace_log_joint_fn, laplace_theta_init, logdensity_fn = (
            laplace_result
        )
        # init_position is now phi_init (phi-space only); logdensity_fn is the
        # marginal logdensity over phi — this is what warmup.runner will use.

    # --- Warmup (multi-chain via warmup.runner's internal vmap) ---
    _log(f"  Warmup ({warmup_name}, n_warmup={n_warmup}, num_chains={num_chains})...")
    t_warmup0 = time.perf_counter()
    try:
        batched_state, batched_params = warmup.runner(
            warmup_key,
            init_position,
            n_warmup,
            base_method,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
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

    step_size_arr = batched_params.get("step_size", None)
    if step_size_arr is not None:
        ss_np = np.asarray(step_size_arr).ravel()
        _log(
            f"  Warmup done in {t_warmup:.1f}s. "
            f"step_size per chain: min={float(ss_np.min()):.4g} "
            f"max={float(ss_np.max()):.4g}"
        )
    else:
        _log(f"  Warmup done in {t_warmup:.1f}s.")

    # Check for NaN/Inf in adapted params (per-chain)
    for k, v in batched_params.items():
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

    # --- Build shared kernel kwargs (per-chain (step_size, IMM) comes via vmap) ---
    default_params = default_params_for(base_method)
    # Defaults minus the per-chain-adapted keys
    shared_kwargs: dict[str, Any] = {
        k: v
        for k, v in default_params.items()
        if k not in ("step_size", "inverse_mass_matrix")
    }
    # `dynamic_hmc` / `dmhmc` factories expect `integration_steps_fn` (callable),
    # not the int `num_integration_steps` that the HMC-substituted warmup adapts.
    # Strip the int; blackjax's default `integration_steps_fn` (uniform L in
    # [1, 10)) takes over at the kernel.
    if sampler_name in ("dynamic_hmc", "dmhmc"):
        shared_kwargs.pop("num_integration_steps", None)
    # laplace_* factories expect `log_joint_fn` and `theta_init` as positional-style
    # kwargs but NOT `logdensity_fn` (the marginal).  Strip laplace_*-incompatible
    # defaults from shared_kwargs (laplace_* don't have any standard incompatible
    # HP defaults currently, but guard explicitly for future-proofing).
    if is_laplace:
        # log_joint_fn + theta_init are not in shared_kwargs (they come from the
        # model decomposition built above); no stripping needed.  Just ensure
        # they're present as extra kwargs for the factory call below.
        pass

    batched_step_size = batched_params["step_size"]
    batched_imm = batched_params["inverse_mass_matrix"]
    needs_dyn_reinit = sampler_name in ("dynamic_hmc", "dmhmc")

    # Capture laplace extras in closure for _run_one_chain.
    _laplace_log_joint_fn = laplace_log_joint_fn
    _laplace_theta_init = laplace_theta_init

    def _run_one_chain(rng, init_state, step_size, imm):
        if is_laplace:
            # laplace_* factory: positional-style kwargs log_joint_fn + theta_init;
            # the `logdensity_fn` arg is present for interface uniformity but unused.
            kernel = base_method.factory(
                logdensity_fn,  # unused (marginal — not joint)
                log_joint_fn=_laplace_log_joint_fn,
                theta_init=_laplace_theta_init,
                step_size=step_size,
                inverse_mass_matrix=imm,
                **shared_kwargs,
            )
            # laplace_hmc / laplace_dhmc / laplace_mhmc / laplace_dmhmc all use
            # `.init(phi_init)` which runs a cold-start L-BFGS. The warmup state
            # carries only phi positions (HMC-substituted warmup ran on phi only).
            reinit_key, run_key = jax.random.split(rng)
            init_for_run = kernel.init(init_state.position, reinit_key)
        else:
            kernel = base_method.factory(
                logdensity_fn,
                step_size=step_size,
                inverse_mass_matrix=imm,
                **shared_kwargs,
            )
            if needs_dyn_reinit:
                # DynamicHMCState extends HMCState with `random_generator_arg`;
                # warmup output (HMC-substituted) is an HMCState, so re-init from
                # the position to get the correct state structure.
                reinit_key, run_key = jax.random.split(rng)
                init_for_run = kernel.init(init_state.position, reinit_key)
            else:
                init_for_run, run_key = init_state, rng
        _, (st, inf) = run_inference_algorithm(
            rng_key=run_key,
            inference_algorithm=kernel,
            num_steps=n_samples,
            initial_state=init_for_run,
        )
        return st, inf

    # --- Sampling (multi-chain via jax.vmap) ---
    _log(
        f"  Sampling ({sampler_name}, n_samples={n_samples}, "
        f"num_chains={num_chains})..."
    )
    t_sample0 = time.perf_counter()
    try:
        chain_keys = jax.random.split(sample_key, num_chains)
        states, infos = jax.vmap(_run_one_chain)(
            chain_keys, batched_state, batched_step_size, batched_imm
        )
        positions = states.position  # dict {param: (num_chains, n_samples, *shape)}
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
    # positions is already (num_chains, n_samples, *event) — no rechunk needed.
    mc_positions = {k: np.asarray(v) for k, v in positions.items()}
    grad_evals = total_grad_evals(infos, base_method.grad_count_per_step)
    headline: float | None = None
    if grad_evals > 0:
        headline = float(min_bulk_ess_per_grad(mc_positions, grad_evals))

    # --- Build recipe ---
    # The recipe pins ONE reproducible (step_size, IMM) config — the multi-chain
    # run was the auto-gate validation, but a recipe is a single replayable
    # specification, so we pin chain 0's adapted params.  Other chains' values
    # are functionally equivalent given the deterministic seed + per-chain key.
    _log("  Building LOW recipe...")
    chain0_step_size = float(np.asarray(batched_step_size).ravel()[0])
    pinned_params: dict[str, Any] = {**shared_kwargs, "step_size": chain0_step_size}
    jsonable_params = _to_jsonable(pinned_params)

    imm_arr: np.ndarray | None = None
    imm_raw = batched_params.get("inverse_mass_matrix", None)
    if imm_raw is not None and hasattr(imm_raw, "_fields"):
        # Structured IMM (e.g., LowRankInverseMassMatrix NamedTuple).  Each
        # field is shape (num_chains, *event); pin chain 0 across all fields.
        # Always sidecar for structured IMMs.
        imm_arr = None  # sentinel — no flat array
        jsonable_params["inverse_mass_matrix"] = "sidecar"
    elif imm_raw is not None:
        imm_full = np.asarray(imm_raw)
        # Chain-0 IMM: drop the leading num_chains axis.
        imm_arr = imm_full[0] if imm_full.ndim >= 1 else imm_full
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
            "num_chains": num_chains,
            "target_acceptance": PHASE3_TARGET_ACCEPTANCE,
        },
        headline_metric=headline,
        sample_quality=None,
        calibration_budget={
            "trials": 0,
            "wall_seconds_estimate": t_total,
            "n_warmup": n_warmup,
            "n_samples": n_samples,
            "num_chains": num_chains,
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
