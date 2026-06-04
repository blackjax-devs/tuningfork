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
"""Recipe-emit pipeline — warmup + sampling + auto-gate for LOW-effort recipes.

Implements the full ``warmup → sample → auto_gate → Recipe.LOW`` flow.
Each cell runs:

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

    JAX_PLATFORM_NAME=cpu uv run python -m tuningfork.recipes._recipe_runner \
        --model mvn_10 \
        --warmup window_adaptation_diag_imm \
        --sampler nuts

The module is **not** exposed through the public ``tuningfork.recipes``
``__init__.py``; it is an internal generator-layer script.

Recipe runner spec (per the visualization-diagnostics decision record):
    - ``n_warmup=1000``, ``n_samples=1000``, ``num_chains=4`` (quick mode)
    - ``seed=20260517`` (master); per-chain keys split internally
    - ``target_acceptance`` from ``base_method`` default (default 0.8)
    - PASS verdict → emit LOW recipe; FAIL/REVIEW → write note to
      ``/tmp/recipe-runner-outcomes.md`` and exit non-zero.
"""

import dataclasses
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.base import SamplingAlgorithm as _SamplingAlgorithm
from blackjax.mcmc.laplace_marginal import laplace_marginal_factory
from blackjax.util import run_inference_algorithm as _run_inference_algorithm

from tuningfork._machine_info import get_machine_info
from tuningfork._version import __version__ as _tuningfork_version
from tuningfork.base_method import BASE_METHODS
from tuningfork.base_method._step_policy_registry import build_step_policy
from tuningfork.base_method._warmup_to_sampler_transform import transform_warmup_state
from tuningfork.calibration.statistician_gate import auto_gate
from tuningfork.calibration.tune import default_params_for, default_value_for_space
from tuningfork.metrics.grad_counter import total_grad_evals
from tuningfork.metrics.headline import min_bulk_ess_per_grad
from tuningfork.metrics.reference_compare import (
    compute_sample_quality as _compute_sample_quality,
)
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.recipes._base import Effort, Recipe, RecipeFailedError
from tuningfork.recipes._instructions import render_instructions
from tuningfork.warmup import WARMUPS
from tuningfork.warmup._laplace_adapter import LAPLACE_METHOD_NAMES

__all__ = [
    "emit_low_recipe_for_cell",
    "run_recipe_to_idata",
    "stamp_headline_from_chain_stats",
    "CellResult",
]

# ---------------------------------------------------------------------------
# MCLMC-family constants
# ---------------------------------------------------------------------------

# Samplers that accept ``L`` as a factory kwarg and whose warmup routines
# (mclmc_tuning / adjusted_mclmc_tuning) return the adapted ``L`` in
# ``batched_params["L"]``.  This key must be vmapped per-chain (not served from
# the default_params_for fallback) to ensure the warmup-adapted trajectory
# length reaches the sampling kernel.
_MCLMC_FAMILY_NAMES: frozenset[str] = frozenset(
    ("mclmc", "adjusted_mclmc", "adjusted_mclmc_dynamic")
)

# ---------------------------------------------------------------------------
# Laplace optimizer kwargs helpers
# ---------------------------------------------------------------------------

# Known optimizer kwargs for laplace_marginal_factory / minimize_lbfgs.
# Stored as flat keys in recipe.base_method_params (per-model) or per-phase
# warmup params; forwarded to laplace_marginal_factory at run + emit time.
_LAPLACE_OPTIMIZER_KWARG_NAMES: tuple[str, ...] = (
    "maxiter",
    "maxcor",
    "gtol",
    "ftol",
    "maxls",
)


def _extract_laplace_optimizer_kwargs(
    primary: dict[str, Any], fallback: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Extract laplace_marginal_factory optimizer kwargs from a params dict.

    Checks ``primary`` first; falls back to ``fallback`` (if provided) for any
    key not found in ``primary``.  Returns only keys in
    ``_LAPLACE_OPTIMIZER_KWARG_NAMES`` that are explicitly set.

    Parameters
    ----------
    primary
        Dict checked first (e.g. a per-phase ``warmup["params"]`` dict).
    fallback
        Dict used when a key is absent from ``primary``
        (e.g. ``recipe.base_method_params``).

    Returns
    -------
    dict
        Subset of ``_LAPLACE_OPTIMIZER_KWARG_NAMES`` keys that are present,
        with their values from ``primary`` or ``fallback``.
    """
    result: dict[str, Any] = {}
    for key in _LAPLACE_OPTIMIZER_KWARG_NAMES:
        if key in primary:
            result[key] = primary[key]
        elif fallback is not None and key in fallback:
            result[key] = fallback[key]
    return result


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
# Models currently in scope for laplace_* (per laplace-marginal preflight eligibility):
#   eight_schools_ncp: phi=(mu, tau), theta=(theta_raw,)
#   gp_regression:     phi=(log_lengthscale, log_kernel_scale, log_noise_scale),
#                      theta=(f_raw,)    — NCP base variable; Laplace is exact
#                      (see worklog/decisions/2026-05-24-gp-regression-laplace-factorisation.md)
#
# radon and irt_2pl are predicted MEDIUM (not LOW) — not needed here yet.
# This table is extended as more recipe sweeps add models.
_LAPLACE_PHI_THETA_SPLITS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "eight_schools_ncp": (("mu", "tau"), ("theta_raw",)),
    # NCP: p(f_raw | phi) = N(0, I) — independent of phi; Laplace is exact.
    # phi-dim=3 (log-scale hyperparams), theta-dim=200 (latent GP values).
    "gp_regression": (
        ("log_lengthscale", "log_kernel_scale", "log_noise_scale"),
        ("f_raw",),
    ),
}

# Recipe runner canonical parameters
# (4 chains x 1000 samples = "quick mode" non-groundtruth recipe protocol per
#  the visualization-diagnostics decision record;
#  matches auto_gate's `min_bulk_ess >= 400` calibration. Use `quick` for LOW
#  recipes; MEDIUM/HIGH should bump `n_samples` to 4000 via CLI override.)
RECIPE_N_WARMUP: int = 1000
RECIPE_N_SAMPLES: int = 1000
RECIPE_NUM_CHAINS: int = 4
RECIPE_SEED: int = 20260517
RECIPE_N_CHUNKS: int = 4  # for split-R̂; ignored when samples are multi-chain
RECIPE_TARGET_ACCEPTANCE: float = 0.8

# Catalog root (relative to this file: tuningfork/tuningfork/catalog/)
_CATALOG_ROOT: Path = Path(__file__).parent.parent / "catalog"

# Outcomes log for FAIL / REVIEW cells
_OUTCOMES_FILE: Path = Path("/tmp/recipe-runner-outcomes.md")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class CellResult:
    """Outcome of one recipe-runner emit attempt.

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
        warmup_grad_evals: int | None = None,
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
        # Exact warmup gradient evaluations (gate-independent — populated whenever
        # warmup completes successfully, regardless of subsequent sampling verdict).
        self.warmup_grad_evals = warmup_grad_evals

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


def _run_warmup_with_inner_kernel(
    warmup_key: Any,
    init_position: Any,
    n_warmup: int,
    logdensity_fn: Any,
    warmup_inner_kernel_name: str,
    num_chains: int,
    target_acceptance: float | None,
    is_mass_matrix_diagonal: bool = True,
) -> tuple[Any, dict[str, Any], Any]:
    """Run window_adaptation with an explicitly specified inner kernel.

    Used when ``warmup_inner_kernel`` is set and differs from the implicit
    default for the base method (e.g. NUTS warmup for HMC sampling).

    Returns ``(states, adapted_params, warmup_info)`` where ``warmup_info``
    is the stacked per-chain warmup trace info (contains NIS for NUTS kernel).
    The warmup_info has a leading ``num_chains`` axis from vmap.

    Parameters
    ----------
    warmup_key
        JAX random key.
    init_position
        Single-chain initial position (replicated internally across chains).
    n_warmup
        Number of warmup steps per chain.
    logdensity_fn
        Log-density callable.
    warmup_inner_kernel_name
        Name of the blackjax kernel to use (e.g. ``"nuts"``).
    num_chains
        Number of parallel chains.
    target_acceptance
        Target acceptance rate or None (uses 0.8 default).
    is_mass_matrix_diagonal
        True for diagonal mass matrix, False for dense.
    """
    import blackjax

    from tuningfork.warmup._base import _maybe_replicate

    kernel_factory = getattr(blackjax, warmup_inner_kernel_name)
    target = target_acceptance or 0.8

    warmup = blackjax.window_adaptation(
        kernel_factory,
        logdensity_fn,
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
        target_acceptance_rate=target,
    )

    chain_keys = jax.random.split(warmup_key, num_chains)
    init_positions = _maybe_replicate(init_position, num_chains)

    @jax.vmap
    def run_one(k: Any, x0: Any) -> tuple[Any, Any, Any]:
        (state, params), info = warmup.run(k, x0, n_warmup)
        return state, params, info

    states, adapted_params_raw, warmup_info = run_one(chain_keys, init_positions)
    adapted_params = dict(adapted_params_raw)
    return states, adapted_params, warmup_info


def _build_laplace_components(
    model_name: str,
    full_position: dict[str, Any],
    joint_logdensity_fn: Any,
    **optimizer_kwargs: Any,
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

    laplace = laplace_marginal_factory(log_joint_fn, theta_init, **optimizer_kwargs)

    def marginal_logdensity_fn(phi: dict[str, Any]) -> Any:
        lp, _theta_star = laplace(phi)
        return lp

    return phi_init, log_joint_fn, theta_init, marginal_logdensity_fn


# ---------------------------------------------------------------------------
# Init-strategy helper
# ---------------------------------------------------------------------------


def _apply_init_strategy(
    strategy: dict[str, Any],
    init_position: Any,
    rng_key: Any,
) -> Any:
    """Override ``init_position`` according to an ``init_strategy`` spec.

    Called after ``build_logdensity_fn`` (and after the laplace phi-space
    transformation, if applicable) to replace the prior-sampled starting point
    with a schematic alternative.

    Parameters
    ----------
    strategy
        A validated tagged-union dict with a ``"type"`` key.  See
        :py:func:`~tuningfork.recipes._base.validate_init_strategy` for the
        valid spec formats.
    init_position
        Starting position produced by ``build_logdensity_fn`` (or the phi-init
        from the laplace transformation).  A pytree of JAX arrays.
    rng_key
        JAX random key used when ``strategy["type"] == "uniform"``.

    Returns
    -------
    Any
        A pytree with the same structure as ``init_position``, modified per
        the strategy spec.
    """
    type_ = strategy.get("type", "prior_sample")
    if type_ == "prior_sample":
        return init_position
    elif type_ == "zero":
        return jax.tree.map(lambda x: jnp.zeros_like(x), init_position)
    elif type_ == "uniform":
        low = float(strategy["low"])
        high = float(strategy["high"])
        leaves, treedef = jax.tree_util.tree_flatten(init_position)
        keys = jax.random.split(rng_key, len(leaves))
        new_leaves = [
            jax.random.uniform(k, leaf.shape, dtype=leaf.dtype, minval=low, maxval=high)
            for k, leaf in zip(keys, leaves)
        ]
        return treedef.unflatten(new_leaves)
    else:
        # Should not reach here — validate_init_strategy catches unknown types at load.
        raise ValueError(f"Unknown init_strategy type: {type_!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# GT alignment helper
# ---------------------------------------------------------------------------


def _align_gt_keys_for_gate(
    gt_summary: dict,
    draws_dict: dict,
    is_laplace_family: bool,
    model_name: str,
) -> dict | None:
    """Build GT-alignment dict for ``auto_gate`` and ``compute_sample_quality``.

    Returns ``{param: {"mean", "std", "q05", "q95", "n_samples"}}`` for
    parameters present in both *draws_dict* and the GT summary.

    Parameters
    ----------
    gt_summary
        Parsed ``reference/summary.json`` dict with top-level keys
        ``"mean"``, ``"std"``, ``"q05"``, ``"q95"``, ``"n_samples"``.
    draws_dict
        The post-warmup ``positions`` dict
        ``{param: (num_chains, n_samples, *shape)}``.
    is_laplace_family
        When ``True``, restrict alignment to phi sites from
        ``_LAPLACE_PHI_THETA_SPLITS[model_name]``.  The laplace sampler
        only explores the phi subspace; theta is analytically marginalised
        and absent from ``draws_dict``.
    model_name
        Registry key, used to look up phi sites.

    Returns
    -------
    dict | None
        Per-param GT dict, or ``None`` if no matching keys were found.
    """
    gt_mean = gt_summary.get("mean", {})
    gt_std = gt_summary.get("std", {})
    gt_q05 = gt_summary.get("q05", {})
    gt_q95 = gt_summary.get("q95", {})
    n_samples = int(gt_summary.get("n_samples", 0)) or None

    if is_laplace_family and model_name in _LAPLACE_PHI_THETA_SPLITS:
        # Only check the phi sites — theta is not in draws_dict for laplace.
        phi_sites, _ = _LAPLACE_PHI_THETA_SPLITS[model_name]
        candidate_keys = [k for k in phi_sites if k in draws_dict and k in gt_mean]
    else:
        # Full-posterior path: align on the intersection of draws and GT.
        candidate_keys = [k for k in draws_dict if k in gt_mean]

    if not candidate_keys:
        return None

    aligned: dict = {}
    for k in candidate_keys:
        aligned[k] = {
            "mean": gt_mean[k],
            "std": gt_std[k],
            "q05": gt_q05[k],
            "q95": gt_q95[k],
        }
        if n_samples is not None:
            aligned[k]["n_samples"] = n_samples

    return aligned if aligned else None


# ---------------------------------------------------------------------------
# Warmup gradient-eval counter (M2)
# ---------------------------------------------------------------------------


def _compute_warmup_grad_evals(
    batched_params: Any,
    batched_warmup_info: Any,
    base_method: Any,
    n_warmup: int,
    num_chains: int,
) -> int | None:
    """Compute total warmup gradient evaluations (summed across all chains).

    Three sources, in priority order:

    1. ``_total_tuning_steps`` in ``batched_params`` (mclmc / adjusted_mclmc
       family): the mclmc-specific warmup runner returns this metadata key.
       It is the total number of integrator steps across all chains.

    2. ``batched_warmup_info.num_integration_steps`` — exact CUMSUM of per-step
       NIS.  For standard window adaptation, the runner now returns ``adapt_info.info``
       (NUTSInfo / HMCInfo with per-step counts) as the third element.  For
       inner-kernel warmup (``warmup_inner_kernel`` set), the warmup trace carries
       per-step NIS.  Shape is ``(num_chains, n_warmup)`` after vmap; sum gives exact
       total across all chains × warmup steps.

    3. ``None`` — warmup runner didn't return kernel info (e.g. non-window-adaptation
       warmups that don't expose per-step NIS).

    Parameters
    ----------
    batched_params
        Dict returned by ``warmup.runner``.
    batched_warmup_info
        Per-step warmup trace (only populated for inner-kernel path).
    base_method
        BaseMethod entry (used for ``grad_count_per_step`` if needed).
    n_warmup
        Warmup steps (for context).
    num_chains
        Number of chains.

    Returns
    -------
    int | None
        Total warmup gradient evaluations, or ``None`` if not available.
    """
    import numpy as np

    # Source 1: mclmc-family _total_tuning_steps
    total_tuning = batched_params.get("_total_tuning_steps") if batched_params else None
    if total_tuning is not None:
        try:
            return int(total_tuning)
        except (TypeError, ValueError):
            pass

    # Source 2: inner-kernel warmup info (has num_integration_steps per step)
    if batched_warmup_info is not None:
        try:
            nis = getattr(batched_warmup_info, "num_integration_steps", None)
            if nis is not None:
                arr = np.asarray(nis)
                # Defensive self-check: arr.shape[-1] must equal n_warmup.
                # adapt_info.info.num_integration_steps is shape (n_warmup,) per chain
                # (vmapped → (num_chains, n_warmup)) — per-step, NOT a running CUMSUM.
                # If blackjax ever changes to cumulative or partial-window NIS this
                # assert will fire, catching the silent double-count.
                assert arr.shape[-1] == n_warmup, (  # noqa: S101
                    f"NIS array last-dim {arr.shape[-1]} != n_warmup {n_warmup}; "
                    "blackjax may have changed from per-step to cumulative NIS — "
                    "CUMSUM would be incorrect"
                )
                # batched_warmup_info shape: (num_chains, n_warmup) or (n_warmup,)
                return int(np.sum(arr))
        except AssertionError as exc:
            # Shape guard fired: NIS array is not per-step length.
            # Log visibly so we notice a blackjax regression rather than silently
            # getting wge=null.
            import warnings as _warnings

            _warnings.warn(
                f"warmup_grad_evals CUMSUM guard: {exc}  — returning None",
                stacklevel=2,
            )
        except Exception:  # noqa: BLE001
            pass

    # Source 3: not available
    return None


# Main emit function
# ---------------------------------------------------------------------------


def emit_low_recipe_for_cell(
    model_name: str,
    warmup_name: str,
    sampler_name: str,
    *,
    n_warmup: int = RECIPE_N_WARMUP,
    n_samples: int = RECIPE_N_SAMPLES,
    num_chains: int = RECIPE_NUM_CHAINS,
    seed: int = RECIPE_SEED,
    n_chunks: int = RECIPE_N_CHUNKS,
    catalog_root: Path = _CATALOG_ROOT,
    outcomes_file: Path = _OUTCOMES_FILE,
    verbose: bool = True,
    target_acceptance: float | None = None,
    sampler_kwargs_override: dict[str, Any] | None = None,
    warmup_kwargs_override: dict[str, Any] | None = None,
    step_policy: dict[str, Any] | None = None,
    policy_tag: str | None = None,
    effort: Effort = Effort.LOW,
    warmup_inner_kernel: str | None = None,
    init_strategy: dict[str, Any] | None = None,
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
        Warmup steps per chain (default ``RECIPE_N_WARMUP`` = 1000).
    n_samples
        Post-warmup sampler steps per chain (default ``RECIPE_N_SAMPLES`` = 1000).
    num_chains
        Number of independent chains run in parallel via ``jax.vmap``
        (default ``RECIPE_NUM_CHAINS`` = 4).  The non-groundtruth recipe
        protocol (per `worklog/decisions/2026-05-11-phase6-visualization-
        diagnostics.md` § Section 0) is 4 chains × 1000 quick mode.
    seed
        Master JAX random seed (default ``RECIPE_SEED`` = 20260517).
    n_chunks
        Split-Rhat rechunk count if samples come in single-chain layout;
        ignored when samples are already multi-chain (default 4).
    catalog_root
        Root of the catalog directory (default: ``tuningfork/catalog/``).
    outcomes_file
        File to append FAIL / REVIEW notes to.
    verbose
        Print progress to stdout.
    target_acceptance
        Override for the warmup dual-averaging target acceptance rate.
        ``None`` (default) uses ``base_method.target_acceptance_rate`` or
        0.80.  Pass e.g. ``target_acceptance=0.99`` for curvature-sensitive
        models (banana, lotka_volterra, ill_cond_50) where the groundtruth
        already required ta=0.99.  The effective value is recorded in
        ``warmup_params["target_acceptance"]`` of the emitted recipe.
    sampler_kwargs_override
        Optional dict merged into ``shared_kwargs`` before building the
        sampling kernel.  Useful for overriding defaults that differ from
        the base_method's registered defaults, e.g.
        ``{"num_integration_steps": 20}`` to tune HMC trajectory length.
        Keys in this dict take precedence over both the default HP space and
        the per-chain adapted values (except ``step_size`` and
        ``inverse_mass_matrix``, which always come from warmup adaptation and
        are never overridden here).  Cannot serialise non-JSON-able values
        (e.g., callables); pass ``None`` for those and handle separately.
    step_policy
        Step-policy spec dict for ``dynamic_hmc`` / ``dmhmc`` cells,
        controlling the ``integration_steps_fn`` callable.  ``None``
        (default) means "use the library default" (V0: uniform integer in
        [1, 10)).  Non-None specs are passed to
        ``build_step_policy(spec)`` to construct the callable at execution
        time; the spec is also stored in ``Recipe.step_policy`` so it
        round-trips through JSON without closure capture.

        For non-``dynamic_hmc`` / non-``dmhmc`` samplers, this parameter is
        ignored.  See ``worklog/threads/d-hmc-integration-steps-fn-matrix.md``
        §5 for valid spec formats.
    policy_tag
        Optional filename tag for policy-variant MEDIUM recipes, e.g.
        ``"policy_v7-empirical-oracle"``.  When provided, the emitted recipe
        filename becomes
        ``<effort>__<sampler>__<warmup>__<policy_tag>.json`` and the
        ``effort`` parameter should be set to ``Effort.MEDIUM``.
        ``None`` (default) preserves the canonical ``<effort>__<sampler>__<warmup>.json``
        filename (backward-compatible with all existing callers).
    effort
        Effort tier for the emitted recipe (default ``Effort.LOW``).
        Pass ``Effort.MEDIUM`` together with ``policy_tag`` for MEDIUM
        policy-variant recipes.  Behaviour for other tiers is not tested.
    warmup_inner_kernel
        Optional explicit warmup inner kernel name (e.g. ``"nuts"``).
        ``None`` (default) preserves the current implicit substitute-family
        logic (``resolve_warmup_algorithm``): NUTS for laplace_*/dynamic_hmc/
        dmhmc; sampler's own kernel for all other methods.
        When set to ``"nuts"`` for a non-substitute-family sampler (e.g.
        ``sampler_name="hmc"``), the warmup runs with NUTS instead of HMC,
        capturing NIS to derive ``num_integration_steps`` via
        ``transform_warmup_state``.  This is the schema-extension inner-kernel opt-in
        path (§3 of RECIPE_SCHEMA.md).  The ``__inner_<kernel>`` filename
        modifier is appended when this differs from the implicit default (§3.5).
    init_strategy
        Schematic init spec stored in the emitted recipe and applied to the
        initial position before warmup.  ``None`` (default) uses the prior-sample
        position returned by ``build_logdensity_fn`` — the current backward-
        compatible behavior.  Non-None values are validated by
        :py:func:`~tuningfork.recipes._base.validate_init_strategy` before use.
        Valid specs: ``{"type": "prior_sample"}``, ``{"type": "zero"}``,
        ``{"type": "uniform", "low": float, "high": float}``.

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

    # --- Load GT reference summary for gate + sample_quality wiring ---
    # Loaded early (before any JAX work) so it's available at auto_gate time.
    # Missing summary.json is graceful: _gt_summary stays None, aligned_gt
    # stays None, and auto_gate / compute_sample_quality simply skip GT checks.
    _gt_summary_path = catalog_root / model_name / "reference" / "summary.json"
    _gt_summary: dict | None = None
    if _gt_summary_path.exists():
        try:
            _gt_summary = json.loads(_gt_summary_path.read_text())
        except Exception:  # noqa: BLE001
            _gt_summary = None

    # Per-model x64 requirement: auto-enable BEFORE any JAX computation.
    # Must happen before jax.random.key() below.
    if posterior.requires_x64 and not jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", True)

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
        _emit_opt_kwargs = _extract_laplace_optimizer_kwargs(
            sampler_kwargs_override or {}, default_params_for(base_method)
        )
        laplace_result = _build_laplace_components(
            model_name, init_position, logdensity_fn, **_emit_opt_kwargs
        )
        if laplace_result is None:
            note = (
                f"ERROR: laplace_* sampler requested but {model_name!r} has no "
                "phi/theta split in _LAPLACE_PHI_THETA_SPLITS — cannot build "
                "marginal logdensity. Add the split to the table in _recipe_runner.py."
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

    # --- elliptical_slice special path (B3) ---
    # blackjax.elliptical_slice expects a LIKELIHOOD-ONLY logdensity, not the
    # joint log-posterior.  We subtract the Gaussian prior analytically:
    #   loglik(x) = logposterior(x) − logprior_gaussian(x, µ, diag(Σ))
    #   logprior_gaussian = -0.5 · Σ_site Σ_dim (x_site - µ_site)² / σ²_site
    # This requires posterior.prior_mean and posterior.prior_cov_diag to be set
    # (both are per-site dicts; gp_regression has them for Phase 8B.3).
    # JAX_ENABLE_X64 is handled by the requires_x64 flag above (line ~800).
    is_elliptical_slice = sampler_name == "elliptical_slice"
    if is_elliptical_slice:
        if posterior.prior_mean is None or posterior.prior_cov_diag is None:
            note = (
                f"ERROR: elliptical_slice requested on {model_name!r} but "
                "posterior.prior_mean / prior_cov_diag are None. "
                "Set both fields on the Posterior entry."
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
        # Build per-site prior pytrees for the logprior computation.
        import jax.numpy as _jnp

        _prior_mean_pytree = {k: _jnp.array(v) for k, v in posterior.prior_mean.items()}
        _prior_cov_pytree = {
            k: _jnp.array(v) for k, v in posterior.prior_cov_diag.items()
        }
        _joint_logdensity_fn = logdensity_fn

        def loglik_fn(x: Any) -> Any:
            """Likelihood-only: logposterior(x) minus the diagonal Gaussian prior."""
            logprior = sum(
                -0.5
                * _jnp.sum(
                    (x[site] - _prior_mean_pytree[site]) ** 2 / _prior_cov_pytree[site]
                )
                for site in _prior_mean_pytree
            )
            return _joint_logdensity_fn(x) - logprior

        logdensity_fn = loglik_fn
        _log(
            "  elliptical_slice: built likelihood-only logdensity "
            f"(subtracted diagonal Gaussian prior over {len(_prior_mean_pytree)} sites)"
        )

        # Auto-initialize near the reference posterior mean when no init_strategy
        # is provided.  The GP posterior is highly concentrated (~20σ from the
        # prior draw for log_noise_scale), so cold-start from prior gives
        # rhat>>1 after 1000 steps.  _build_stationary_init_positions reads the
        # reference/summary.json and places chains at mean ± small offsets,
        # giving near-posterior initialization at zero extra computation cost.
        if init_strategy is None:
            try:
                _stationary = _build_stationary_init_positions(
                    model_name, 1, catalog_root
                )
                # _build_stationary_init_positions returns (1, ...) batched; squeeze.
                init_position = jax.tree.map(lambda x: x[0], _stationary)
                _log(
                    "  elliptical_slice: initialized from reference posterior mean "
                    "(cold-start from prior is ~20σ away for concentrated GP posteriors)"
                )
            except (FileNotFoundError, KeyError):
                # If reference summary unavailable, fall back to prior draw.
                _log(
                    "  elliptical_slice: reference summary unavailable, using prior draw"
                )

    # --- Apply init_strategy (optional override of initial position) ---
    # Applied after the laplace phi-space transformation so the override acts
    # on the same position space that the warmup kernel will operate on.
    if init_strategy is not None:
        from tuningfork.recipes._base import validate_init_strategy

        validate_init_strategy(init_strategy)
        _override_key = jax.random.fold_in(init_key, 42)
        init_position = _apply_init_strategy(
            init_strategy, init_position, _override_key
        )

    # --- Warmup (multi-chain via warmup.runner's internal vmap) ---
    # When warmup_inner_kernel is set, run with explicit kernel (captures NIS for
    # transform_warmup_state). When None, use the normal warmup.runner path
    # (backward-compat: current implicit substitute-family logic).
    _log(
        f"  Warmup ({warmup_name}, n_warmup={n_warmup}, "
        f"num_chains={num_chains}"
        + (f", inner_kernel={warmup_inner_kernel}" if warmup_inner_kernel else "")
        + ")..."
    )
    t_warmup0 = time.perf_counter()
    batched_warmup_info: Any = (
        None  # set from inner_kernel path OR standard runner 3-tuple
    )
    try:
        if warmup_inner_kernel is not None:
            # Schema extension: explicit inner kernel path — run window_adaptation with
            # the specified kernel (e.g. NUTS for HMC sampling) and capture NIS.
            batched_state, batched_params, batched_warmup_info = (
                _run_warmup_with_inner_kernel(
                    warmup_key,
                    init_position,
                    n_warmup,
                    logdensity_fn,
                    warmup_inner_kernel_name=warmup_inner_kernel,
                    num_chains=num_chains,
                    target_acceptance=target_acceptance,
                )
            )
        else:
            # Legacy path: warmup.runner handles implicit substitute-family logic.
            # window_adaptation runners return (states, params, kernel_info) as of
            # the exact-wge instrumentation; unpack defensively for backward compat.
            # Build warmup HP kwargs: declared warmup HP space defaults, then
            # override with warmup_kwargs_override (BO-found values or explicit config).
            _warmup_hp_kwargs: dict[str, Any] = {
                space.name: default_value_for_space(space)
                for space in getattr(warmup, "default_hp_space", ())
            }
            if warmup_kwargs_override:
                for _k, _v in warmup_kwargs_override.items():
                    _warmup_hp_kwargs[_k] = _v
            _warmup_result = warmup.runner(
                warmup_key,
                init_position,
                n_warmup,
                base_method,
                logdensity_fn=logdensity_fn,
                num_chains=num_chains,
                target_acceptance_rate=target_acceptance,
                # Warmup HPs (e.g. num_optimization_steps for VI warmup).
                **_warmup_hp_kwargs,
                # B2: pass posterior_entry ONLY for no_warmup so it can read
                # prior_mean/cov for elliptical_slice (extra_required_kwargs).
                # Gradient-adapted warmup runners (window_adaptation_*) forward
                # **kwargs → extra_kwargs → **warmup_kwargs → blackjax kernel,
                # which rejects unknown kwargs; posterior_entry must not leak there.
                **(
                    {} if warmup_name != "no_warmup" else {"posterior_entry": posterior}
                ),
            )
            batched_state = _warmup_result[0]
            batched_params = _warmup_result[1]
            batched_warmup_info = (
                _warmup_result[2]  # type: ignore[misc]
                if len(_warmup_result) == 3
                else None
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
    # SYNC: block until warmup compute completes before stamping the clock.
    # JAX dispatches kernels asynchronously — without this, t_warmup measures
    # dispatch latency only (potentially 100× less than actual compute time).
    jax.block_until_ready((batched_state, batched_params))
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

    # Check for NaN/Inf in adapted params (per-chain). Structured IMMs (e.g.
    # `LowRankInverseMassMatrix` NamedTuple with sigma/U/lam fields of
    # heterogeneous shapes) can't be `np.asarray`-ed as a single tensor; flatten
    # via `jax.tree.leaves` so each leaf is checked individually.
    for k, v in batched_params.items():
        leaves = jax.tree.leaves(v)
        for i, leaf in enumerate(leaves):
            arr = np.asarray(leaf)
            if not np.all(np.isfinite(arr)):
                leaf_id = k if len(leaves) == 1 else f"{k}[leaf={i}]"
                note = f"FAIL warmup produced NaN/Inf in {leaf_id}"
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

    # Compute warmup_grad_evals immediately after warmup succeeds.  Gate-independent:
    # always stamp regardless of whether the subsequent sampling pass or auto-gate
    # passes.  Fixed-HMC = CUMSUM(constant L); NUTS/dmhmc = CUMSUM(variable NIS).
    _wge = _compute_warmup_grad_evals(
        batched_params, batched_warmup_info, base_method, n_warmup, num_chains
    )

    # --- Build shared kernel kwargs (per-chain (step_size, IMM) comes via vmap) ---
    default_params = default_params_for(base_method)
    # Defaults minus the per-chain-adapted keys
    shared_kwargs: dict[str, Any] = {
        k: v
        for k, v in default_params.items()
        if k not in ("step_size", "inverse_mass_matrix")
    }

    # --- Schema extension: transform_warmup_state dispatch ---
    # When warmup_inner_kernel is set (explicit opt-in), run the resolution table:
    #   nuts → hmc/mhmc  : inject num_integration_steps = median(NIS)
    #   nuts → dynamic_hmc/dmhmc : inject step_policy = empirical(NIS)
    #   (other rows: identity — step_size + IMM only)
    # When warmup_inner_kernel is None (legacy): keep the existing implicit path
    # (build_step_policy(step_policy) for dynamic_hmc/dmhmc; no change elsewhere).
    _effective_step_policy = step_policy  # may be overwritten by transform below
    if warmup_inner_kernel is not None and batched_warmup_info is not None:
        # Explicit inner-kernel path: use transform_warmup_state resolution table.
        # Pass step_policy as override only when an explicit spec was provided by
        # the caller (prevents re-harvesting from warmup_info on recipe re-run).
        _transform_result = transform_warmup_state(
            warmup_inner_kernel,
            sampler_name,
            batched_params,
            batched_warmup_info,
            step_policy_override=step_policy if step_policy is not None else None,
        )
        # Inject transform results into shared_kwargs (excluding step_size and IMM
        # which are handled per-chain via vmap below).
        for _tk, _tv in _transform_result.items():
            if _tk not in ("step_size", "inverse_mass_matrix"):
                if _tk == "step_policy":
                    _effective_step_policy = _tv
                elif _tk == "num_integration_steps":
                    shared_kwargs["num_integration_steps"] = _tv

    # Compute step_policy to persist HERE (before any early returns) so that
    # dynamic_hmc/dmhmc with inner_nuts always carry the harvested step_policy
    # regardless of gate verdict.  For non-dynamic kernels this is always None.
    # NOTE: this must stay before the NaN-check + gate early-returns below.
    _recipe_step_policy = (
        _effective_step_policy if sampler_name in ("dynamic_hmc", "dmhmc") else None
    )

    # `dynamic_hmc` / `dmhmc` factories expect `integration_steps_fn` (callable),
    # not the int `num_integration_steps` that the HMC-substituted warmup adapts.
    # Strip the int; then inject the step_policy callable (V0 = library default
    # when step_policy=None; non-V0 from build_step_policy when spec is provided).
    if sampler_name in ("dynamic_hmc", "dmhmc"):
        shared_kwargs.pop("num_integration_steps", None)
        # Build integration_steps_fn from the (possibly transform-updated) spec.
        # V0 (spec=None): returns the same callable as blackjax's built-in default,
        # so behaviour is identical to not specifying it — we still inject explicitly
        # to make the code path consistent and testable.
        _integration_steps_fn = build_step_policy(_effective_step_policy)
        shared_kwargs["integration_steps_fn"] = _integration_steps_fn
    # Apply sampler_kwargs_override: caller-supplied values take precedence over
    # defaults.  step_size and inverse_mass_matrix are always excluded — they
    # come from warmup adaptation and must not be overridden here.
    if sampler_kwargs_override:
        for _k, _v in sampler_kwargs_override.items():
            if _k not in ("step_size", "inverse_mass_matrix"):
                shared_kwargs[_k] = _v
    # laplace_* factories expect `log_joint_fn` and `theta_init` as positional-style
    # kwargs but NOT `logdensity_fn` (the marginal).  Strip laplace_*-incompatible
    # defaults from shared_kwargs (laplace_* don't have any standard incompatible
    # HP defaults currently, but guard explicitly for future-proofing).
    if is_laplace:
        # log_joint_fn + theta_init are not in shared_kwargs (they come from the
        # model decomposition built above); no stripping needed.  Just ensure
        # they're present as extra kwargs for the factory call below.
        pass

    # Gradient-free / no-adapted-params samplers (elliptical_slice, rwm in no_warmup)
    # return empty batched_params from no_warmup — they have no step_size or IMM.
    # Inject prior_cov / prior_mean into shared_kwargs from posterior for
    # elliptical_slice (needed to rebuild the kernel at each step).
    batched_step_size = batched_params.get("step_size")
    batched_imm = batched_params.get("inverse_mass_matrix")
    is_no_adapted_params = batched_step_size is None and batched_imm is None

    if is_elliptical_slice:
        # B3 (Phase 8B.3): prior kwargs go into shared_kwargs so the factory
        # can build the kernel with the correct Gaussian prior at each step.
        from jax.flatten_util import ravel_pytree as _ravel_pytree

        _mean_pytree = {k: jnp.array(v) for k, v in posterior.prior_mean.items()}  # type: ignore[union-attr]
        _cov_pytree = {k: jnp.array(v) for k, v in posterior.prior_cov_diag.items()}  # type: ignore[union-attr]
        _mean_flat, _ = _ravel_pytree(_mean_pytree)
        _cov_flat, _ = _ravel_pytree(_cov_pytree)
        shared_kwargs["prior_mean"] = _mean_flat
        shared_kwargs["prior_cov"] = _cov_flat

    needs_dyn_reinit = sampler_name in ("dynamic_hmc", "dmhmc")

    # mclmc-family: warmup returns adapted L in batched_params["L"].  Extract it
    # and remove the default-L entry from shared_kwargs so the factory receives
    # the per-chain warmup-adapted value, not the default_params_for fallback.
    batched_L = batched_params.get("L")  # (num_chains,) array or None
    is_mclmc_family = sampler_name in _MCLMC_FAMILY_NAMES and batched_L is not None
    if is_mclmc_family:
        # L comes per-chain from batched_L; remove the static default from shared_kwargs.
        shared_kwargs.pop("L", None)

    # adjusted_mclmc_dynamic uses adjusted_mclmc_tuning which returns HMCState,
    # but the sampling kernel needs DynamicHMCState (has random_generator_arg).
    # A per-chain reinit converts HMCState → DynamicHMCState via kernel.init().
    needs_mclmc_dyn_reinit = (
        sampler_name == "adjusted_mclmc_dynamic" and is_mclmc_family
    )

    # Capture laplace extras in closure for per-chain step function.
    _laplace_log_joint_fn = laplace_log_joint_fn
    _laplace_theta_init = laplace_theta_init

    # --- Pre-scan init: handle samplers that need a different state type ---
    # Laplace and dynamic_hmc/dmhmc require re-init before the scan loop because
    # window_adaptation produces HMCState while these kernels need their own state.
    # adjusted_mclmc_dynamic needs the same treatment (adapted HMCState → DynamicHMCState).
    # Per-chain reinit is safe (vmapping .init() has no io_callback issues).
    _reinit_keys = jax.random.split(jax.random.fold_in(sample_key, 9999), num_chains)
    if is_laplace:

        def _init_one_chain_laplace(init_state, step_size, imm, reinit_key):
            kernel = base_method.factory(
                logdensity_fn,
                log_joint_fn=_laplace_log_joint_fn,
                theta_init=_laplace_theta_init,
                step_size=step_size,
                inverse_mass_matrix=imm,
                **shared_kwargs,
            )
            return kernel.init(init_state.position, reinit_key)

        _run_states = jax.vmap(_init_one_chain_laplace)(
            batched_state, batched_step_size, batched_imm, _reinit_keys
        )
    elif needs_dyn_reinit:

        def _init_one_chain_dyn(init_state, step_size, imm, reinit_key):
            kernel = base_method.factory(
                logdensity_fn,
                step_size=step_size,
                inverse_mass_matrix=imm,
                **shared_kwargs,
            )
            return kernel.init(init_state.position, reinit_key)

        _run_states = jax.vmap(_init_one_chain_dyn)(
            batched_state, batched_step_size, batched_imm, _reinit_keys
        )
    elif needs_mclmc_dyn_reinit:
        # adjusted_mclmc_dynamic: convert warmup HMCState → DynamicHMCState using
        # the per-chain adapted L (required by the factory; not in shared_kwargs).

        def _init_one_chain_mclmc_dyn(init_state, step_size, imm, L, reinit_key):
            kernel = base_method.factory(
                logdensity_fn,
                step_size=step_size,
                inverse_mass_matrix=imm,
                L=L,
                **shared_kwargs,
            )
            return kernel.init(init_state.position, reinit_key)

        _run_states = jax.vmap(_init_one_chain_mclmc_dyn)(
            batched_state, batched_step_size, batched_imm, batched_L, _reinit_keys
        )
    else:
        _run_states = batched_state

    # --- run_inference_algorithm(vmapped input) — canonical multi-chain pattern ---
    # ONE kernel is built with per-chain (step_size, imm) via vmap inside the step.
    # run_inference_algorithm is NOT vmapped; inputs are batched (num_chains, ...).
    # progress_bar=False for the runner (uses timing/logging not a terminal bar).
    if is_laplace:

        def _step_one_chain_laplace(state, key, step_size, imm):
            kernel_step = base_method.factory(
                logdensity_fn,
                log_joint_fn=_laplace_log_joint_fn,
                theta_init=_laplace_theta_init,
                step_size=step_size,
                inverse_mass_matrix=imm,
                **shared_kwargs,
            ).step
            return kernel_step(key, state)

        def _vmapped_step_laplace(rng_key, states):
            keys = jax.random.split(rng_key, num_chains)
            return jax.vmap(_step_one_chain_laplace)(
                states, keys, batched_step_size, batched_imm
            )

        _alg = _SamplingAlgorithm(lambda pos, key=None: pos, _vmapped_step_laplace)
    elif is_mclmc_family:
        # mclmc-family: vmap L alongside step_size and imm so each chain
        # receives its warmup-adapted trajectory length.  L was removed from
        # shared_kwargs above; it arrives here as batched_L (num_chains,).

        def _step_one_chain_mclmc(state, key, step_size, imm, L):
            kernel_step = base_method.factory(
                logdensity_fn,
                step_size=step_size,
                inverse_mass_matrix=imm,
                L=L,
                **shared_kwargs,
            ).step
            return kernel_step(key, state)

        def _vmapped_step_mclmc(rng_key, states):
            keys = jax.random.split(rng_key, num_chains)
            return jax.vmap(_step_one_chain_mclmc)(
                states, keys, batched_step_size, batched_imm, batched_L
            )

        _alg = _SamplingAlgorithm(lambda pos, key=None: pos, _vmapped_step_mclmc)
    elif is_no_adapted_params:
        # Gradient-free / no-adapted-params path (e.g. elliptical_slice).
        # No per-chain step_size or IMM; kernel built entirely from shared_kwargs
        # (which for elliptical_slice contains prior_mean + prior_cov injected above).

        def _step_one_chain_gf(state: Any, key: Any) -> Any:
            kernel_step = base_method.factory(logdensity_fn, **shared_kwargs).step
            return kernel_step(key, state)

        def _vmapped_step_gf(rng_key: Any, states: Any) -> Any:
            keys = jax.random.split(rng_key, num_chains)
            return jax.vmap(_step_one_chain_gf)(states, keys)

        _alg = _SamplingAlgorithm(lambda pos, key=None: pos, _vmapped_step_gf)
    else:

        def _step_one_chain(state, key, step_size, imm):
            kernel_step = base_method.factory(
                logdensity_fn,
                step_size=step_size,
                inverse_mass_matrix=imm,
                **shared_kwargs,
            ).step
            return kernel_step(key, state)

        def _vmapped_step(rng_key, states):
            keys = jax.random.split(rng_key, num_chains)
            return jax.vmap(_step_one_chain)(
                states, keys, batched_step_size, batched_imm
            )

        _alg = _SamplingAlgorithm(lambda pos, key=None: pos, _vmapped_step)

    # --- Sampling (multi-chain via run_inference_algorithm(vmapped input)) ---
    _log(
        f"  Sampling ({sampler_name}, n_samples={n_samples}, "
        f"num_chains={num_chains})..."
    )
    t_sample0 = time.perf_counter()
    try:
        _, (_states_hist, infos) = _run_inference_algorithm(
            sample_key,
            _alg,
            num_steps=n_samples,
            initial_state=_run_states,
            progress_bar=False,
        )
        # run_inference_algorithm output: (n_samples, num_chains, ...) → swap to (num_chains, n_samples, ...)
        states = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), _states_hist)
        infos = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), infos)
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
            warmup_grad_evals=_wge,
        )
    # SYNC: block until sampling compute completes before stamping the clock.
    # positions/infos are JAX futures; without sync t_sample measures dispatch only.
    jax.block_until_ready((states, infos))
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
                warmup_grad_evals=_wge,
            )

    # --- Align GT keys for auto-gate and sample_quality ---
    # For laplace-family: phi-subset alignment (only phi sites are in positions).
    # For full-posterior: intersection of positions and GT keys.
    _aligned_gt: dict | None = None
    if _gt_summary is not None:
        _aligned_gt = _align_gt_keys_for_gate(
            _gt_summary, positions, is_laplace, model_name
        )

    # --- Auto-gate ---
    # Pass step_size + num_integration_steps for the resonance check (fixed-L
    # HMC only; dynamic kernels have no fixed L, so NIS is absent from shared_kwargs).
    # Gradient-free samplers have no step_size; pass 0.0 (no resonance check applies).
    _gate_chain0_ss = (
        float(np.asarray(batched_step_size).ravel()[0])
        if batched_step_size is not None
        else 0.0
    )
    _gate_nis: int | None = shared_kwargs.get("num_integration_steps")
    # VI-sampler mode: base_method is meanfield_vi/fullrank_vi + warmup = no_warmup.
    # iid draws make rhat/ESS/div vacuous; only max_abs_mean_z gates (z<4.0 REVIEW band).
    # Per decision doc 2026-06-04-vi-sampler-pivotal-z-review-gate.md.
    _vi_sampler_mode = (
        sampler_name
        in (
            "meanfield_vi",
            "fullrank_vi",
        )
        and warmup_name == "no_warmup"
    )
    _log("  Running auto-gate...")
    gate_verdict = auto_gate(
        positions,
        infos,
        ground_truth_summaries=_aligned_gt,
        posterior=posterior,
        n_chunks=n_chunks,
        step_size=_gate_chain0_ss,
        num_integration_steps=_gate_nis,
        vi_sampler_mode=_vi_sampler_mode,
    )
    _log(
        f"  Gate: {gate_verdict.verdict}, "
        f"rhat_max={gate_verdict.rhat_max:.4f}, "
        f"min_ess={gate_verdict.min_bulk_ess:.1f}, "
        f"n_div={gate_verdict.n_divergences}"
        + (
            f", max_z={gate_verdict.max_abs_mean_z:.3f}"
            if gate_verdict.max_abs_mean_z is not None
            else ""
        )
        + (
            f", RESONANCE_WARN(L×ε={(_gate_nis or 0) * _gate_chain0_ss:.2f})"
            if gate_verdict.resonance_warning
            else ""
        )
    )

    # Convention (user decision 2026-05-30): PASS + REVIEW both get headline/recipe.
    # FAIL = catastrophic non-mixing → no meaningful ESS → early return, no recipe.
    # REVIEW = borderline GT-agreement → ESS is real and useful for review → stamp it.
    if gate_verdict.verdict == "FAIL":
        note = (
            f"FAIL "
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
            verdict="FAIL",
            gate_rhat_max=gate_verdict.rhat_max,
            gate_min_ess=gate_verdict.min_bulk_ess,
            gate_n_div=gate_verdict.n_divergences,
            wall_seconds=t_total,
            note=note,
            warmup_grad_evals=_wge,
        )
    # REVIEW: fall through to headline computation + recipe building (same as PASS).
    if gate_verdict.verdict == "REVIEW":
        _log(
            f"  => gate REVIEW (rhat={gate_verdict.rhat_max:.4f}, "
            f"ess={gate_verdict.min_bulk_ess:.1f}) — stamping headline and emitting recipe."
        )

    # --- Build headline metric + basis ---
    # positions is already (num_chains, n_samples, *event) — no rechunk needed.
    mc_positions = {k: np.asarray(v) for k, v in positions.items()}
    grad_evals = total_grad_evals(infos, base_method.grad_count_per_step)
    headline: float | None = None
    _headline_basis: dict | None = None
    if grad_evals > 0:
        headline = float(min_bulk_ess_per_grad(mc_positions, grad_evals))
        # Gap-1 (decisions/2026-05-30): capture accounting details so cross-recipe
        # comparisons are interpretable (convention varies by base_method family).
        # Back-compute min_bulk_ess = headline × grad_evals (exact, no rounding).
        _is_laplace = sampler_name in LAPLACE_METHOD_NAMES
        _headline_basis = {
            "total_grad_evals": int(grad_evals),
            "min_bulk_ess": headline * grad_evals,  # back-derived from headline
            # Complete formula text from BaseMethod.grad_count_convention (single
            # source of truth), not a truncated slice of the general notes.
            "grad_count_convention": base_method.grad_count_convention or sampler_name,
            "is_lower_bound": _is_laplace,
        }
    elif grad_evals == 0 and gate_verdict.min_bulk_ess is not None:
        # Gradient-free sampler (e.g. elliptical_slice, rwm).
        # Headline = min_bulk_ess / n_total_samples (efficiency per draw, not per grad).
        # Phase 8B.3 ratified convention: don't leave headline null/inf for gradient-free.
        _n_total = n_samples * num_chains
        _min_ess: float = gate_verdict.min_bulk_ess  # narrowed: not None above
        headline = _min_ess / _n_total
        _grad_free_convention = (
            "0 (gradient-free; headline = min_bulk_ess/n_total_samples)"
        )
        _headline_basis = {
            "total_grad_evals": 0,
            "min_bulk_ess": _min_ess,
            "grad_count_convention": _grad_free_convention,
            "is_lower_bound": False,
        }

    # --- Compute sample_quality (GT-agreement; compare draws to reference) ---
    # Uses the same aligned GT keys used by auto_gate above.
    # compute_sample_quality requires SCALAR reference stats (mean/std/q05/q95
    # must be float-coercible).  For vector parameters (e.g. mvn_10 "x" is 10-D),
    # the GT summary stores per-element arrays; we collapse those to their element-wise
    # mean, consistent with _param_metrics which collapses draws via .mean(axis=1).
    _sample_quality: dict | None = None
    if _aligned_gt is not None:
        _sq_draws = {k: mc_positions[k] for k in _aligned_gt if k in mc_positions}
        _sq_refs: dict = {}
        for _k, _v in _aligned_gt.items():
            if _k not in _sq_draws:
                continue
            _sq_refs[_k] = {}
            for _stat in ("mean", "std", "q05", "q95"):
                # Pass per-element reference stats as-is (scalar or list/array).
                # _param_metrics now handles per-element comparison correctly;
                # the old .mean() collapse caused a dimension-collapse artefact
                # (std_ratio_dev ≈ 1−1/√d for perfect draws of d-dim params).
                _sq_refs[_k][_stat] = _v[_stat]
        if _sq_draws and _sq_refs:
            try:
                _sample_quality = _compute_sample_quality(_sq_draws, _sq_refs)
            except Exception:  # noqa: BLE001
                # Non-fatal: leave sample_quality=None rather than failing the emit.
                _sample_quality = None

    # --- Build recipe ---
    # The recipe pins ONE reproducible (step_size, IMM) config — the multi-chain
    # run was the auto-gate validation, but a recipe is a single replayable
    # specification, so we pin chain 0's adapted params.  Other chains' values
    # are functionally equivalent given the deterministic seed + per-chain key.
    _log(f"  Building {effort.value.upper()} recipe...")
    # Exclude integration_steps_fn (callable; not JSON-serialisable) from the
    # pinned params — it is reconstructed at recipe-run time via step_policy spec.
    # Also exclude prior_cov / prior_mean: they are stored on the Posterior entry
    # and do not need to be pinned per-recipe (they are read at run time).
    _NON_SERIALISABLE_KEYS = {"integration_steps_fn", "prior_cov", "prior_mean"}
    pinned_params: dict[str, Any] = {
        k: v for k, v in shared_kwargs.items() if k not in _NON_SERIALISABLE_KEYS
    }
    if not is_no_adapted_params:
        # Gradient-adapted samplers: pin chain 0's adapted step_size.
        chain0_step_size = float(np.asarray(batched_step_size).ravel()[0])
        pinned_params["step_size"] = chain0_step_size
    # mclmc-family: save the warmup-adapted L (chain 0) to the recipe so that
    # recipe re-runs (recertification) use the correct trajectory length rather
    # than the default_params_for fallback.  L was removed from shared_kwargs
    # above; add it back here as a concrete scalar for JSON serialisation.
    if is_mclmc_family and batched_L is not None:
        pinned_params["L"] = float(np.asarray(batched_L).ravel()[0])
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
    # Surface the structural scope of the GT comparison so downstream readers can
    # branch on it.  Decision doc: worklog/decisions/2026-05-28-max-abs-mean-z-threshold.md §3.
    # Laplace: only phi marginals are gate-verified; theta is analytically marginalised
    # and absent from positions — the theta block is NOT certified against the GT.
    # Full-posterior: all posterior sites in positions are compared to the GT reference.
    gate_evidence["auto"]["gt_cert_coverage"] = (
        "phi_subset_only (theta marginals not gate-verified)"
        if is_laplace
        else "full_posterior"
    )

    # _recipe_step_policy is computed above (before gate early-returns) so that
    # it's available regardless of gate verdict.  See the assignment near line 897.

    # Base warmup params: n_warmup, num_chains, target_acceptance.
    # Warmup HP params (e.g. num_optimization_steps for VI warmup) are appended
    # from the warmup's default_hp_space, overridden by warmup_kwargs_override.
    _warmup_hp_params_to_record: dict[str, Any] = {
        space.name: default_value_for_space(space)
        for space in getattr(warmup, "default_hp_space", ())
    }
    if warmup_kwargs_override:
        _warmup_hp_params_to_record.update(
            {
                k: v
                for k, v in warmup_kwargs_override.items()
                if any(k == s.name for s in getattr(warmup, "default_hp_space", ()))
            }
        )
    _warmup_params_dict: dict[str, Any] = {
        "n_warmup": n_warmup,
        "num_chains": num_chains,
        "target_acceptance": (
            target_acceptance
            if target_acceptance is not None
            else (base_method.target_acceptance_rate or RECIPE_TARGET_ACCEPTANCE)
        ),
        **_warmup_hp_params_to_record,
    }

    recipe_kwargs: dict[str, Any] = dict(
        model_name=posterior.name,
        base_method_name=base_method.name,
        warmup_name=warmup.name,
        effort=effort,
        base_method_params=jsonable_params,
        warmup_params=_warmup_params_dict,
        warmups=[{"name": warmup.name, "params": _warmup_params_dict}],
        warmup_inner_kernel=warmup_inner_kernel,
        headline_metric=headline,
        headline_basis=_headline_basis,
        sample_quality=_sample_quality,
        calibration_budget={
            "trials": 0,
            "wall_seconds_estimate": t_total,
            "n_warmup": n_warmup,
            "n_samples": n_samples,
            "num_chains": num_chains,
            # Timing breakdown — measured at Python orchestration level.
            # warmup: from warmup.runner() call return; sampling: from vmap return.
            "warmup_wall_seconds": round(t_warmup, 3),
            "sampling_wall_seconds": round(t_sample, 3),
            "sampling_seconds_per_draw": round(
                t_sample / max(n_samples * num_chains, 1), 6
            ),
            "split_source": "measured",
            "machine_info": get_machine_info(),
            # warmup_grad_evals: total gradient evaluations during warmup (M2).
            # For mclmc-family: read from _total_tuning_steps in batched_params.
            # For window adaptation (diag/dense/low_rank): CUMSUM of per-step
            # num_integration_steps from adapt_info.info (exact for both HMC and NUTS).
            # For inner-kernel warmup: same CUMSUM path via warmup trace.
            "warmup_grad_evals": _compute_warmup_grad_evals(
                batched_params, batched_warmup_info, base_method, n_warmup, num_chains
            ),
        },
        difficulty=None,
        instructions="",
        notes="",
        # Store the step_policy spec so the recipe JSON is self-describing.
        # None = library default (V0); non-None = explicit spec from caller or
        # harvested by transform_warmup_state.
        step_policy=_recipe_step_policy,
        # Store init_strategy so the recipe is self-describing; applied at
        # re-run time by run_recipe_to_idata via _apply_init_strategy.
        init_strategy=init_strategy,
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

    # --- Compute filename tag for inner-kernel modifier (§3.5) ---
    # Append __inner_<kernel> when warmup_inner_kernel is explicitly set AND
    # differs from the implicit default for this base_method.
    # Implicit default: substitute-family → "nuts"; others → base_method_name.
    from tuningfork.warmup._laplace_adapter import WARMUP_SUBSTITUTE_METHOD_NAMES

    _implicit_default = (
        "nuts" if sampler_name in WARMUP_SUBSTITUTE_METHOD_NAMES else sampler_name
    )
    _inner_tag: str | None = None
    if warmup_inner_kernel is not None and warmup_inner_kernel != _implicit_default:
        _inner_tag = f"inner_{warmup_inner_kernel}"

    # Compose filename_tag: inner_tag + policy_tag (ordering per §5).
    _all_tags = [t for t in [_inner_tag, policy_tag] if t]
    _combined_tag = "__".join(_all_tags) if _all_tags else None

    # --- Save recipe ---
    recipe_path = recipe.save(catalog_root, filename_tag=_combined_tag)
    _log(f"  Saved recipe: {recipe_path}")

    # --- Save IMM sidecar if needed ---
    imm_sidecar_rel: str | None = None
    if imm_arr is not None and imm_arr.size > 50:
        imm_sidecar_rel = recipe.save_imm_sidecar(
            catalog_root, imm_arr, filename_tag=_combined_tag
        )
        # Rebuild recipe with sidecar path (Recipe is frozen)
        recipe = dataclasses.replace(recipe, inverse_mass_matrix_path=imm_sidecar_rel)
        recipe.save(catalog_root, filename_tag=_combined_tag)
        _log(f"  Saved IMM sidecar: {imm_sidecar_rel}")

    _verdict = gate_verdict.verdict  # "PASS" or "REVIEW" (FAIL returned early above)
    _hl_str = f" headline={headline:.4g}" if headline is not None else ""
    _log(f"  {_verdict}.{_hl_str}")
    return CellResult(
        model_name=model_name,
        warmup_name=warmup_name,
        sampler_name=sampler_name,
        verdict=_verdict,
        recipe_path=recipe_path,
        imm_sidecar_path=imm_sidecar_rel,
        gate_rhat_max=gate_verdict.rhat_max,
        gate_min_ess=gate_verdict.min_bulk_ess,
        gate_n_div=gate_verdict.n_divergences,
        wall_seconds=t_total,
        note=f"{_verdict} rhat={gate_verdict.rhat_max:.4f} ess={gate_verdict.min_bulk_ess:.1f} div={gate_verdict.n_divergences}",
        warmup_grad_evals=_wge,
    )


# ---------------------------------------------------------------------------
# Public helper for on-demand resampling (catalog notebook integration)
# ---------------------------------------------------------------------------


def _build_stationary_init_positions(
    model_name: str,
    num_chains: int,
    catalog_root: Path,
) -> Any:
    """Build per-chain initial positions from GT reference summary (unconstrained space).

    Reads ``mean`` and ``std`` from ``<catalog_root>/<model_name>/reference/summary.json``
    and returns a batched pytree with leading dim ``num_chains``.  Each chain is
    initialised at::

        gt_mean + offsets[i % 4] * gt_std

    where ``offsets = [+0.1, -0.1, +0.05, -0.05]``.  This places every chain near
    the posterior mean with small, diverse jitter — avoiding cold-start burn-in
    while giving enough chain diversity for R-hat to be informative.

    Parameters
    ----------
    model_name
        Model name matching a subdirectory under ``catalog_root``.
    num_chains
        Number of chains; positions shape is ``(num_chains, *event)``.
    catalog_root
        Root of the catalog directory.

    Returns
    -------
    batched_positions
        Dict of JAX arrays with leading dim ``num_chains``.
    """
    import json

    summary_path = catalog_root / model_name / "reference" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Reference summary not found at {summary_path}. "
            f"Cannot build stationary init for skip_warmup=True on model {model_name!r}."
        )
    summary = json.loads(summary_path.read_text())
    gt_mean = summary["mean"]
    gt_std = summary["std"]

    _OFFSETS = [0.1, -0.1, 0.05, -0.05]
    # Follow model precision: x64-enabled models (e.g. lotka_volterra) use float64;
    # all others use float32.  This check fires at call time, AFTER run_recipe_to_idata
    # has already called jax.config.update("jax_enable_x64", True) for x64 models.
    _dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

    def _chain_init(i: int) -> dict:
        offset = _OFFSETS[i % len(_OFFSETS)]
        return {
            k: jnp.asarray(gt_mean[k], dtype=_dtype)
            + offset * jnp.asarray(gt_std[k], dtype=_dtype)
            for k in gt_mean
        }

    chain_positions = [_chain_init(i) for i in range(num_chains)]
    # Stack into batched pytree with leading dim num_chains
    return jax.tree.map(lambda *arrs: jnp.stack(arrs, axis=0), *chain_positions)


def _reduce_and_broadcast_warmup_output(
    warmup_state: Any,
    warmup_params: dict[str, Any],
    num_warmup_chains: int,
    num_sampling_chains: int,
) -> tuple[Any, dict[str, Any]]:
    """Reduce W warmup chains to shared params, broadcast to S sampling chains.

    Used when ``warmup_num_chains[i] != num_chains`` for phase i.

    The reduce step computes the arithmetic mean across the W warmup chains:
    - ``step_size``: scalar mean of the W per-chain step sizes.
    - ``inverse_mass_matrix``: element-wise mean across W chains.

    The broadcast step replicates the shared params to S copies.
    Position broadcast uses ``position[s % W]`` so each sampling chain
    starts at one of the W warmup endpoints.

    Parameters
    ----------
    warmup_state
        Batched warmup state with leading dimension W.
    warmup_params
        Dict of batched warmup params with leading dimension W.
    num_warmup_chains
        W — number of warmup chains (leading dim of warmup_state).
    num_sampling_chains
        S — number of sampling chains to broadcast to.

    Returns
    -------
    (broadcasted_state, broadcasted_params)
        State with leading dim S; params dict with leading dim S.
    """
    # Reduce params to scalar via arithmetic mean (on-device, no host materialization).
    mean_step_size = jnp.mean(warmup_params["step_size"])
    mean_imm = jnp.mean(warmup_params["inverse_mass_matrix"], axis=0)

    # Broadcast shared params to S sampling chains.
    broad_step_size = jnp.broadcast_to(mean_step_size[None], (num_sampling_chains,))
    broad_imm = jnp.broadcast_to(
        mean_imm[None], (num_sampling_chains,) + mean_imm.shape
    )
    broadcasted_params = {
        **warmup_params,
        "step_size": broad_step_size,
        "inverse_mass_matrix": broad_imm,
    }

    # Replicate position: sampling chain s starts at warmup endpoint s % W.
    # Uses jax.tree.map + index selection — no Python loop over S.
    # Build index array [0, 1, ..., S-1] % W → shape (S,)
    indices = jnp.arange(num_sampling_chains) % num_warmup_chains
    broadcasted_state = jax.tree.map(lambda x: x[indices], warmup_state)
    return broadcasted_state, broadcasted_params


def run_recipe_to_idata(
    recipe: Recipe,
    *,
    n_samples: int | None = None,
    skip_warmup: bool = False,
    force_resample: bool = False,
    force_resample_config: dict[str, Any] | None = None,
    catalog_root: Path = _CATALOG_ROOT,
    warmup_num_chains: list[int] | None = None,
    _allow_failed_diagnostic: bool = False,
    _return_timing: bool = False,
    _suppress_print: bool = False,
) -> Any:
    """Run a LOW/MEDIUM recipe's warmup + sampling pipeline; return as InferenceData.

    On-demand resample helper for the catalog notebook: when the user picks a
    LOW/MEDIUM recipe, this re-runs the recipe's pinned warmup + sampler config
    at the protocol specified in the recipe's warmup_params (n_warmup, num_chains,
    target_acceptance). Returns InferenceData with posterior + sample_stats.

    For FAILED recipes, this raises an error (no valid config to run).
    For GROUNDTRUTH recipes, this delegates to load_idata (no re-run needed).

    Parameters
    ----------
    recipe
        A Recipe object loaded via ``load_recipe``.
    n_samples
        Override the recipe's n_samples. If None, use the recipe's
        warmup_params["n_samples"] or fall back to RECIPE_N_SAMPLES.
    skip_warmup
        When ``True``, bypass the warmup entirely and use the stored
        ``step_size`` / ``inverse_mass_matrix`` from
        ``recipe.base_method_params``.  Chain states are initialised from
        the GT-means in ``reference/summary.json`` with per-chain jitter
        (``gt_mean ± {0.1, 0.05}σ``), so the chains start near the
        posterior — no burn-in needed.

        This is the low-latency path for the catalog notebook: skip ≈10–30 s
        of warmup and go straight to sampling with the pre-tuned params.

        Restrictions:

        - ``recipe.base_method_params`` must contain ``"step_size"`` and
          ``"inverse_mass_matrix"`` (i.e. the recipe must have been emitted
          from a warmup-adaptation run, not from ``no_warmup``).
        - Laplace-marginal samplers (``laplace_*``) are not supported because
          the GT-means are in the full unconstrained space, not the phi-only
          space the laplace marginal operates on.
        - MCLMC is not supported (momentum init requires a special key path).
    force_resample
        **Deprecated.** Pass ``force_resample_config={"seed": recipe.tuning_seed}``
        instead.  When ``True``, emits a :class:`DeprecationWarning` and maps to
        ``force_resample_config={"seed": recipe.tuning_seed}`` automatically.
    force_resample_config
        Dict controlling a forced re-run with a different seed (and optionally
        different ``n_warmup`` / ``n_samples``).  Required key: ``"seed"`` (int).
        Optional keys: ``"n_warmup"`` (int), ``"n_samples"`` (int).

        Example::

            run_recipe_to_idata(recipe, force_resample_config={"seed": 42})

        ``None`` (default) re-runs with the recipe's own ``tuning_seed``
        (backward-compatible).
    catalog_root
        Root of the catalog directory.
    warmup_num_chains
        Runtime override for ``recipe.warmup_num_chains``.  When not ``None``,
        overrides the recipe-stamped value at call time.  Must satisfy the same
        validation rules: list of ints, one per warmup phase, each >= 1.

        Use ``warmup_num_chains=[1, 1]`` (or ``[1]`` for single-phase) to
        force single-chain warmup + broadcast — avoids the vmap-of-while_loop
        penalty for expensive-logprob models (e.g. gp_regression × laplace_*).
        ``None`` (default) falls back to ``recipe.warmup_num_chains`` (which is
        itself ``None`` for legacy recipes, meaning all phases use ``num_chains``).

    Returns
    -------
    arviz.InferenceData
        Posterior group + sample_stats group (if available).

    Raises
    ------
    RecipeFailedError
        If the recipe is FAILED (no gate-passing config).
    ValueError
        If the recipe model/warmup/sampler are not in the registries.
    """
    import warnings

    if force_resample:
        warnings.warn(
            "force_resample=True is deprecated; use "
            "force_resample_config={'seed': recipe.tuning_seed} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if force_resample_config is None:
            force_resample_config = {"seed": recipe.tuning_seed}

    from tuningfork.catalog.render import load_idata

    # For GROUNDTRUTH, use the standard load path
    if recipe.effort == Effort.GROUNDTRUTH:
        return load_idata(recipe, cache_dir=catalog_root)

    # FAILED recipes raise by default — no gate-passing config.
    # _allow_failed_diagnostic=True bypasses this for on-demand diagnostic
    # re-runs (catalog_explorer "Re-run failed config" path): the user explicitly
    # opts in to running the failed config to inspect the failure mode visually.
    if recipe.effort == Effort.FAILED and not _allow_failed_diagnostic:
        raise RecipeFailedError(recipe)

    # Validate registry membership
    if recipe.model_name not in MODELS:
        raise ValueError(f"Model {recipe.model_name!r} not in MODELS registry")
    if recipe.warmup_name not in WARMUPS:
        raise ValueError(f"Warmup {recipe.warmup_name!r} not in WARMUPS registry")
    if recipe.base_method_name not in BASE_METHODS:
        raise ValueError(
            f"Base method {recipe.base_method_name!r} not in BASE_METHODS registry"
        )

    posterior = MODELS[recipe.model_name]
    warmup = WARMUPS[recipe.warmup_name]
    base_method = BASE_METHODS[recipe.base_method_name]

    # Per-model x64 requirement: auto-enable BEFORE any JAX computation.
    # Must happen before jax.random.key() below. Analogous to how the cert
    # pipeline enforces x64 (certify_reference.py:609).
    if posterior.requires_x64 and not jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", True)

    # Compatibility check
    if not warmup.is_compatible(recipe.base_method_name):
        raise ValueError(
            f"Warmup {recipe.warmup_name!r} incompatible with sampler "
            f"{recipe.base_method_name!r}"
        )

    # Extract protocol from recipe
    n_warmup = int(recipe.warmup_params.get("n_warmup", RECIPE_N_WARMUP))
    num_chains = int(recipe.warmup_params.get("num_chains", RECIPE_NUM_CHAINS))
    target_acceptance = recipe.warmup_params.get("target_acceptance", None)

    # Resolve warmup_num_chains: call-time override wins over recipe-stamped value.
    # None means "use num_chains for every phase" (current vmap'd behavior).
    from tuningfork.recipes._base import validate_warmup_num_chains

    _wnc_effective: list[int] | None = (
        warmup_num_chains if warmup_num_chains is not None else recipe.warmup_num_chains
    )
    if _wnc_effective is not None:
        _n_phases = max(len(recipe.warmups), 1)
        validate_warmup_num_chains(_wnc_effective, _n_phases)

    if n_samples is None:
        # Prefer calibration_budget["n_samples"] (the validated config stamp)
        # over warmup_params (which is derived from Phase-1 warmup params and
        # typically has no n_samples key for multi-phase HIGH recipes).
        n_samples = int(
            recipe.calibration_budget.get("n_samples")
            or recipe.warmup_params.get("n_samples", RECIPE_N_SAMPLES)
        )

    # Wall-time gate: start clock before any JAX compilation / warmup work.
    _t0_idata = time.perf_counter()

    # Use recipe's tuning_seed (or fallback to RECIPE_SEED if 0); allow
    # force_resample_config to override seed (and optionally n_warmup / n_samples).
    seed = recipe.tuning_seed if recipe.tuning_seed != 0 else RECIPE_SEED
    if force_resample_config is not None:
        seed = int(force_resample_config["seed"])
        if "n_warmup" in force_resample_config:
            n_warmup = int(force_resample_config["n_warmup"])
        if "n_samples" in force_resample_config:
            n_samples = int(force_resample_config["n_samples"])

    # Build logdensity and initial position
    init_key, warmup_key, sample_key = jax.random.split(jax.random.key(seed), 3)

    init_position, logdensity_fn, _model_data = build_logdensity_fn(init_key, posterior)

    # Handle laplace_* special case
    is_laplace = recipe.base_method_name in LAPLACE_METHOD_NAMES
    laplace_log_joint_fn: Any = None
    laplace_theta_init: Any = None

    if is_laplace:
        _run_opt_kwargs = _extract_laplace_optimizer_kwargs(recipe.base_method_params)
        laplace_result = _build_laplace_components(
            recipe.model_name, init_position, logdensity_fn, **_run_opt_kwargs
        )
        if laplace_result is None:
            raise ValueError(
                f"laplace_* sampler {recipe.base_method_name!r} requested but "
                f"model {recipe.model_name!r} has no phi/theta split in "
                "_LAPLACE_PHI_THETA_SPLITS"
            )
        init_position, laplace_log_joint_fn, laplace_theta_init, logdensity_fn = (
            laplace_result
        )

    # Apply init_strategy from the recipe (optional override of initial position).
    # Applied after the laplace phi-space transformation so the override acts on
    # the same position space that the warmup kernel will operate on.
    if recipe.init_strategy is not None:
        _override_key = jax.random.fold_in(init_key, 42)
        init_position = _apply_init_strategy(
            recipe.init_strategy, init_position, _override_key
        )

    # Validate skip_warmup constraints upfront (before touching JAX / warmup).
    if skip_warmup:
        if is_laplace:
            raise ValueError(
                "skip_warmup=True is not supported for laplace_* samplers. "
                "The GT-means in reference/summary.json are in full unconstrained "
                "space, not the phi-only space the laplace marginal operates on."
            )
        if base_method.name == "mclmc":
            raise ValueError(
                "skip_warmup=True is not supported for MCLMC: momentum init "
                "requires a special key path not handled here."
            )
        if "step_size" not in recipe.base_method_params:
            raise ValueError(
                "skip_warmup=True requires recipe.base_method_params to contain "
                "'step_size'. This recipe was likely emitted from no_warmup."
            )
        if "inverse_mass_matrix" not in recipe.base_method_params:
            raise ValueError(
                "skip_warmup=True requires recipe.base_method_params to contain "
                "'inverse_mass_matrix'. This recipe was likely emitted from no_warmup."
            )

    # Run warmup — multi-phase (recipe.warmups > 1) or single-phase.
    #
    # Multi-phase warmup (e.g. the gp_regression HIGH laplace recipe):
    #   Phase 1: diag IMM warmup with lower maxiter LaplaceMarginal
    #   Phase 2: dense IMM warmup with higher maxiter LaplaceMarginal;
    #            initial_step_size seeded from Phase 1's adapted step_size.
    #   Final phase's (step_size, IMM) are used for sampling.
    #
    # Single-phase: existing logic (warmup_inner_kernel or warmup.runner).
    _recipe_warmup_info: Any = None
    _is_multiphase = len(recipe.warmups) > 1
    _t_warmup_start = time.perf_counter()

    if skip_warmup:
        # Bypass warmup entirely: build stationary init from GT-means, use
        # stored step_size / IMM from recipe.base_method_params.
        # recipe.base_method_params["step_size"] is a Python float from JSON — no JAX conversion needed.
        _stored_ss = float(recipe.base_method_params["step_size"])

        # Bug fix: for D>50 models the recipe stores inverse_mass_matrix="sidecar"
        # (a string path token) — jnp.asarray("sidecar") raises.  Load the actual
        # array via load_imm_sidecar when the value is not array-like.
        _imm_raw = recipe.base_method_params["inverse_mass_matrix"]
        if _imm_raw == "sidecar":
            _stored_imm = recipe.load_imm_sidecar(catalog_root)
            if _stored_imm is None:
                raise ValueError(
                    f"skip_warmup=True: recipe for {recipe.model_name!r} has "
                    "inverse_mass_matrix='sidecar' but inverse_mass_matrix_path "
                    "is not set — cannot load sidecar."
                )
        else:
            _stored_imm = jnp.asarray(_imm_raw)

        # Build stationary positions: GT-means + per-chain jitter from reference/summary.json
        _stationary_positions = _build_stationary_init_positions(
            recipe.model_name, num_chains, catalog_root
        )

        # Extra kwargs beyond step_size/IMM that the factory requires (e.g.
        # num_integration_steps for hmc/mhmc).  Mirror the non-skip-warmup
        # recipe_params injection (lines ~1889-1894) so that hmc/mhmc kernels
        # receive all required positional kwargs at init time.
        _skip_extra_kwargs: dict[str, Any] = {
            k: v
            for k, v in recipe.base_method_params.items()
            if k not in ("step_size", "inverse_mass_matrix", "integration_steps_fn")
        }

        # Build kernel (step_size/IMM don't affect .init; only used to instantiate)
        _skip_init_kernel = base_method.factory(
            logdensity_fn,
            step_size=_stored_ss,
            inverse_mass_matrix=_stored_imm,
            **_skip_extra_kwargs,
        )

        # adjusted_mclmc_dynamic.init requires a random_generator_arg (rng_key).
        # All other samplers init from position only.
        if recipe.base_method_name == "adjusted_mclmc_dynamic":
            _skip_init_keys = jax.random.split(
                jax.random.fold_in(init_key, 7777), num_chains
            )

            @jax.vmap
            def _init_one_skip_dyn(pos: Any, key: Any) -> Any:
                return _skip_init_kernel.init(pos, rng_key=key)

            batched_state = _init_one_skip_dyn(_stationary_positions, _skip_init_keys)
        else:

            @jax.vmap
            def _init_one_skip(pos: Any) -> Any:
                return _skip_init_kernel.init(pos)

            batched_state = _init_one_skip(_stationary_positions)

        # Replicate stored params to (num_chains, ...) to match warmup output shape
        batched_params = {
            "step_size": jnp.full((num_chains,), _stored_ss),
            "inverse_mass_matrix": jnp.broadcast_to(
                _stored_imm[None], (num_chains,) + _stored_imm.shape
            ),
        }
    elif _is_multiphase and is_laplace:
        # Multi-phase laplace warmup: loop phases, threading adapted params.
        # Each phase uses a separate LaplaceMarginal with phase-specific maxiter
        # and its own target_acceptance / is_mass_matrix_diagonal / n_warmup.
        import blackjax as _bj

        from tuningfork.warmup._base import _maybe_replicate

        _phase_inner = recipe.warmup_inner_kernel or "laplace_hmc"
        _phase_kernel_factory = getattr(_bj, _phase_inner)

        _prev_state: Any = None  # W-shaped state (W = phase's warmup chain count)
        _prev_params_mp: dict[str, Any] = {}
        _prev_n_warmup = n_warmup  # fallback when recipe.warmups is sparse
        _prev_phase_W: int = 0  # track previous phase's W for cross-phase init

        for _phase_idx, _phase in enumerate(recipe.warmups):
            _phase_params = _phase["params"]
            _phase_n_warmup = int(_phase_params.get("n_warmup", _prev_n_warmup))
            _phase_target = float(
                _phase_params.get(
                    "target_acceptance",
                    _phase_params.get(
                        "target_acceptance_rate", target_acceptance or 0.8
                    ),
                )
            )
            _phase_is_dense = "dense" in _phase["name"]

            # Per-phase warmup chain count W (or num_chains if not set).
            _phase_W = (
                _wnc_effective[_phase_idx] if _wnc_effective is not None else num_chains
            )

            # Build phase-specific LaplaceMarginal with this phase's optimizer kwargs.
            # Phase params take precedence; recipe.base_method_params is the fallback.
            # Pass directly to window_adaptation as logdensity_fn: LaplaceMarginal
            # returns (lp, theta_star) which satisfies the has_aux=True contract
            # expected by laplace_hmc.init(phi, laplace).
            _phase_opt_kwargs = _extract_laplace_optimizer_kwargs(
                _phase_params, recipe.base_method_params
            )
            _phase_laplace = laplace_marginal_factory(
                laplace_log_joint_fn, laplace_theta_init, **_phase_opt_kwargs
            )

            # initial_step_size: seed Phase 2+ dual-averaging from Phase 1 result.
            # Keep as a 0-d JAX scalar — no host materialization here. The prior
            # `float(np.asarray(_prev_ss).mean())` used the buffer protocol on
            # the *vmap'd* warmup output and deadlocked for gp_regression ×
            # laplace_mhmc × multi-phase warmup (2026-05-28). `jnp.mean` stays
            # on-device; window_adaptation accepts the 0-d JAX scalar at the
            # use site (its `float` annotation is structural, not enforced).
            _initial_step_size: Any | None = None
            if _phase_idx > 0 and _phase_params.get("initial_step_size_from_phase1"):
                _prev_ss = _prev_params_mp.get("step_size")
                if _prev_ss is not None:
                    # If prev phase had W > 1, step_size is batched; mean stays
                    # on-device as 0-d JAX scalar.
                    _initial_step_size = jnp.mean(_prev_ss)

            # Use fold_in per phase to avoid key correlation between phases.
            _phase_key = jax.random.fold_in(warmup_key, _phase_idx)
            _chain_keys = jax.random.split(_phase_key, _phase_W)

            # Init positions for this phase (W warmup chains).
            # NOTE: _prev_state always has _prev_phase_W leading dim (W-shaped,
            # NOT broadcast to num_chains). Broadcasting happens only at the
            # final output stage, after all phases complete.
            if _phase_idx == 0:
                _init_pos_batch = _maybe_replicate(init_position, _phase_W)
            else:
                # Carry forward from previous phase's W-shaped state.
                # Use position[s % prev_W] to handle W changes between phases.
                if _phase_W == _prev_phase_W:
                    _init_pos_batch = _prev_state.position  # type: ignore[union-attr]
                else:
                    _indices = jnp.arange(_phase_W) % _prev_phase_W
                    _init_pos_batch = jax.tree.map(
                        lambda x: x[_indices], _prev_state.position  # type: ignore[union-attr]
                    )

            # Build window_adaptation for this phase.
            # initial_step_size is a named window_adaptation param; num_integration_steps
            # and any other kernel-specific params go to **extra_parameters inside WA.
            _wa_kwargs: dict[str, Any] = {}
            if _initial_step_size is not None:
                _wa_kwargs["initial_step_size"] = _initial_step_size
            _phase_nis = _phase_params.get("num_integration_steps")
            if _phase_nis is not None:
                _wa_kwargs["num_integration_steps"] = int(_phase_nis)
            _warmup_phase = _bj.window_adaptation(
                _phase_kernel_factory,
                _phase_laplace,
                is_mass_matrix_diagonal=not _phase_is_dense,
                target_acceptance_rate=_phase_target,
                **_wa_kwargs,
            )

            # Dispatch: W == num_chains → vmap over num_chains (existing behavior).
            # W != num_chains → vmap over W, then reduce+broadcast to num_chains.
            _n_steps_phase = _phase_n_warmup  # capture for closure

            if _phase_W == num_chains:
                # Current behavior: vmap warmup over all num_chains.
                @jax.vmap
                def _run_one_warmup_phase(
                    k: Any, x0: Any
                ) -> tuple[Any, Any]:  # noqa: B023
                    (st, pr), _ = _warmup_phase.run(k, x0, _n_steps_phase)
                    return st, pr

                _phase_states, _phase_params_raw = _run_one_warmup_phase(
                    _chain_keys, _init_pos_batch
                )
                # Store W-shaped state (W == num_chains, so no difference here).
                _prev_state = _phase_states
                _prev_params_mp = dict(_phase_params_raw)
            else:
                # W != num_chains: vmap over W warmup chains.
                # Store W-shaped state for cross-phase position carry-over.
                # Broadcast to num_chains happens only at final output (below).
                @jax.vmap
                def _run_W_warmup_phase(
                    k: Any, x0: Any
                ) -> tuple[Any, Any]:  # noqa: B023
                    (st, pr), _ = _warmup_phase.run(k, x0, _n_steps_phase)
                    return st, pr

                _w_states, _w_params_raw = _run_W_warmup_phase(
                    _chain_keys, _init_pos_batch
                )
                # Keep W-shaped state for next phase's init positions.
                _prev_state = _w_states
                _prev_params_mp = dict(_w_params_raw)

            _prev_n_warmup = _phase_n_warmup
            _prev_phase_W = _phase_W

        # After all phases: reduce+broadcast if last phase used W != num_chains.
        _last_phase_W = _prev_phase_W
        if _last_phase_W != num_chains:
            batched_state, batched_params = _reduce_and_broadcast_warmup_output(
                _prev_state, _prev_params_mp, _last_phase_W, num_chains
            )
        else:
            batched_state = _prev_state
            batched_params = _prev_params_mp
    elif recipe.warmup_inner_kernel is not None:
        # Single-phase: resolve W from warmup_num_chains[0] or num_chains.
        _single_W = _wnc_effective[0] if _wnc_effective is not None else num_chains
        _raw_state_ik, _raw_params_ik, _recipe_warmup_info = (
            _run_warmup_with_inner_kernel(
                warmup_key,
                init_position,
                n_warmup,
                logdensity_fn,
                warmup_inner_kernel_name=recipe.warmup_inner_kernel,
                num_chains=_single_W,
                target_acceptance=target_acceptance,
            )
        )
        if _single_W != num_chains:
            batched_state, batched_params = _reduce_and_broadcast_warmup_output(
                _raw_state_ik, _raw_params_ik, _single_W, num_chains
            )
        else:
            batched_state, batched_params = _raw_state_ik, _raw_params_ik
    else:
        # Single-phase standard warmup. Resolve W from warmup_num_chains[0].
        _single_W = _wnc_effective[0] if _wnc_effective is not None else num_chains
        # Read warmup HP params from recipe.warmup_params (e.g. num_optimization_steps
        # for VI warmup); fall back to warmup HP defaults when not recorded.
        _recipe_warmup_hp: dict[str, Any] = {
            space.name: recipe.warmup_params.get(
                space.name, default_value_for_space(space)
            )
            for space in getattr(warmup, "default_hp_space", ())
        }
        _std_result = warmup.runner(
            warmup_key,
            init_position,
            n_warmup,
            base_method,
            logdensity_fn=logdensity_fn,
            num_chains=_single_W,
            target_acceptance_rate=target_acceptance,
            **_recipe_warmup_hp,
        )
        _raw_state_std, _raw_params_std = _std_result[0], _std_result[1]
        if _single_W != num_chains:
            batched_state, batched_params = _reduce_and_broadcast_warmup_output(
                _raw_state_std, _raw_params_std, _single_W, num_chains
            )
        else:
            batched_state, batched_params = _raw_state_std, _raw_params_std

    if not skip_warmup:
        # SYNC: block until warmup compute completes before stamping the clock.
        # Without this, _t_warmup measures dispatch latency only, not actual
        # wall time (JAX async dispatch; same hazard as certify_reference.py:642).
        jax.block_until_ready((batched_state, batched_params))
    _t_warmup = 0.0 if skip_warmup else (time.perf_counter() - _t_warmup_start)

    # Build shared kernel kwargs
    default_params = default_params_for(base_method)
    shared_kwargs: dict[str, Any] = {
        k: v
        for k, v in default_params.items()
        if k not in ("step_size", "inverse_mass_matrix")
    }

    # Schema extension: transform_warmup_state for explicit inner kernel recipes.
    if recipe.warmup_inner_kernel is not None and _recipe_warmup_info is not None:
        _rtransform = transform_warmup_state(
            recipe.warmup_inner_kernel,
            recipe.base_method_name,
            batched_params,
            _recipe_warmup_info,
            step_policy_override=recipe.step_policy,  # use pinned spec from recipe
        )
        for _rtk, _rtv in _rtransform.items():
            if _rtk not in ("step_size", "inverse_mass_matrix"):
                if _rtk == "step_policy":
                    pass  # handled below via build_step_policy(recipe.step_policy)
                elif _rtk == "num_integration_steps":
                    shared_kwargs["num_integration_steps"] = _rtv

    if recipe.base_method_name in ("dynamic_hmc", "dmhmc"):
        shared_kwargs.pop("num_integration_steps", None)
        # Wire the step_policy callable from the recipe's stored spec.
        # None = V0 library default; non-None = reconstructed from spec.
        shared_kwargs["integration_steps_fn"] = build_step_policy(recipe.step_policy)

    # Inject recipe's base_method_params (overrides defaults)
    recipe_params = dict(recipe.base_method_params)
    # Exclude step_size, IMM, and integration_steps_fn (callable; not in recipe params).
    for k in list(recipe_params.keys()):
        if k not in ("step_size", "inverse_mass_matrix", "integration_steps_fn"):
            shared_kwargs[k] = recipe_params[k]

    batched_step_size = batched_params["step_size"]
    batched_imm = batched_params["inverse_mass_matrix"]
    needs_dyn_reinit = recipe.base_method_name in ("dynamic_hmc", "dmhmc")
    # adjusted_mclmc_dynamic: L is in shared_kwargs (from recipe_params above);
    # needs reinit from HMCState → DynamicHMCState using the stored scalar L.
    # Unlike the emit path, L here is a Python float (not a vmapped array), so the
    # factory call uses **shared_kwargs which already contains the correct stored L.
    needs_mclmc_dyn_reinit_r = recipe.base_method_name == "adjusted_mclmc_dynamic"

    _laplace_log_joint_fn = laplace_log_joint_fn
    _laplace_theta_init = laplace_theta_init

    # --- Pre-scan init: handle samplers that need a different state type ---
    # Laplace and dynamic_hmc/dmhmc require re-init before the scan loop because
    # window_adaptation produces HMCState while these kernels need their own state.
    # adjusted_mclmc_dynamic needs the same treatment (HMCState → DynamicHMCState).
    _reinit_keys_r = jax.random.split(jax.random.fold_in(sample_key, 9999), num_chains)
    if is_laplace:

        def _init_one_chain_laplace_r(init_state, step_size, imm, reinit_key):
            kernel = base_method.factory(
                logdensity_fn,
                log_joint_fn=_laplace_log_joint_fn,
                theta_init=_laplace_theta_init,
                step_size=step_size,
                inverse_mass_matrix=imm,
                **shared_kwargs,
            )
            return kernel.init(init_state.position, reinit_key)

        _run_states_r = jax.vmap(_init_one_chain_laplace_r)(
            batched_state, batched_step_size, batched_imm, _reinit_keys_r
        )
    elif needs_dyn_reinit:

        def _init_one_chain_dyn_r(init_state, step_size, imm, reinit_key):
            kernel = base_method.factory(
                logdensity_fn,
                step_size=step_size,
                inverse_mass_matrix=imm,
                **shared_kwargs,
            )
            return kernel.init(init_state.position, reinit_key)

        _run_states_r = jax.vmap(_init_one_chain_dyn_r)(
            batched_state, batched_step_size, batched_imm, _reinit_keys_r
        )
    elif needs_mclmc_dyn_reinit_r:
        # adjusted_mclmc_dynamic: L is in shared_kwargs (from recipe_params).
        # The factory call includes L; kernel.init converts HMCState→DynamicHMCState.

        def _init_one_chain_mclmc_dyn_r(init_state, step_size, imm, reinit_key):
            kernel = base_method.factory(
                logdensity_fn,
                step_size=step_size,
                inverse_mass_matrix=imm,
                **shared_kwargs,
            )
            return kernel.init(init_state.position, reinit_key)

        _run_states_r = jax.vmap(_init_one_chain_mclmc_dyn_r)(
            batched_state, batched_step_size, batched_imm, _reinit_keys_r
        )
    else:
        _run_states_r = batched_state

    # --- run_inference_algorithm(vmapped input) — canonical multi-chain pattern ---
    # progress_bar=False for the runner (uses timing/logging not a terminal bar).
    if is_laplace:

        def _step_one_chain_laplace_r(state, key, step_size, imm):
            kernel_step = base_method.factory(
                logdensity_fn,
                log_joint_fn=_laplace_log_joint_fn,
                theta_init=_laplace_theta_init,
                step_size=step_size,
                inverse_mass_matrix=imm,
                **shared_kwargs,
            ).step
            return kernel_step(key, state)

        def _vmapped_step_laplace_r(rng_key, states):
            keys = jax.random.split(rng_key, num_chains)
            return jax.vmap(_step_one_chain_laplace_r)(
                states, keys, batched_step_size, batched_imm
            )

        _alg_r = _SamplingAlgorithm(lambda pos, key=None: pos, _vmapped_step_laplace_r)
    else:

        def _step_one_chain_r(state, key, step_size, imm):
            kernel_step = base_method.factory(
                logdensity_fn,
                step_size=step_size,
                inverse_mass_matrix=imm,
                **shared_kwargs,
            ).step
            return kernel_step(key, state)

        def _vmapped_step_r(rng_key, states):
            keys = jax.random.split(rng_key, num_chains)
            return jax.vmap(_step_one_chain_r)(
                states, keys, batched_step_size, batched_imm
            )

        _alg_r = _SamplingAlgorithm(lambda pos, key=None: pos, _vmapped_step_r)

    # Run sampling via run_inference_algorithm(vmapped input)
    _t_sample_start = time.perf_counter()
    _, (_states_hist_r, infos) = _run_inference_algorithm(
        sample_key,
        _alg_r,
        num_steps=n_samples,
        initial_state=_run_states_r,
        progress_bar=False,
    )
    # run_inference_algorithm output: (n_samples, num_chains, ...) → swap to (num_chains, n_samples, ...)
    states = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), _states_hist_r)
    # NOTE: we intentionally do NOT do `jax.tree.map(jnp.swapaxes, infos)`
    # over the full info pytree here. The samplers' info NamedTuples (notably
    # ``LaplaceHMCInfo``) carry large unused fields — ``momentum``, ``proposal``
    # (full IntegratorState), ``lbfgs_*`` — that materialize-then-drop wasted
    # both compute and buffer-pool capacity, and at production scale (1000 ×
    # 4 chains, gp_regression × laplace_mhmc × dense_imm) triggered a buffer
    # mutex deadlock in the post-scan ``np.asarray`` loop (recert v1/v3 hang,
    # 2026-05-28). Extract field-by-field below — only the four scalar fields
    # we actually persist get swapaxes'd + materialized.
    # SYNC: block until sampling compute completes before stamping the clock.
    # states.position is still a JAX future at this point; without sync
    # _t_sample measures dispatch only (can be 10×+ underestimate).
    jax.block_until_ready(states)
    _t_sample = time.perf_counter() - _t_sample_start
    positions = states.position  # shape: (num_chains, n_samples, *event_shape)

    # Convert to InferenceData via the catalog helper
    from tuningfork.catalog.diagnostics import samples_to_idata

    # positions is already in multi-chain format (num_chains, n_samples, *event)
    # Convert to dict of arrays
    positions_dict = {k: np.asarray(v) for k, v in positions.items()}

    # Field-selective extraction from infos: swapaxes + materialize only the
    # four scalar fields we persist. Functionally identical to the previous
    # "swap whole tree then read these four" behavior, minus the wasted work.
    chain_stats = {}
    for _fld in (
        "is_divergent",
        "energy",
        "acceptance_rate",
        "num_integration_steps",
        "lbfgs_iter_num",
        "lbfgs_hit_maxiter",  # laplace-family: flag for truncated θ* solves
    ):
        if hasattr(infos, _fld):
            chain_stats[_fld] = np.asarray(jnp.swapaxes(getattr(infos, _fld), 0, 1))

    idata_result = samples_to_idata(
        positions_dict,
        is_multichain=True,
        chain_stats=chain_stats if chain_stats else None,
        n_chunks=1,  # Already in multi-chain format
    )

    _wall_idata = time.perf_counter() - _t0_idata
    if not _suppress_print:
        print(
            f"[run_recipe_to_idata] wall_seconds={_wall_idata:.1f}"
            f"  warmup={_t_warmup:.1f}s  sampling={_t_sample:.1f}s"
            f"  n_samples={n_samples}  num_chains={num_chains}"
            f"  recipe={recipe.model_name}/{recipe.effort.value}__{recipe.base_method_name}"
        )
    if _return_timing:
        return idata_result, _t_warmup, _t_sample
    return idata_result


# ---------------------------------------------------------------------------
# Performance-block recovery utility (for MEDIUM/step-policy recipes)
# ---------------------------------------------------------------------------


def stamp_headline_from_chain_stats(
    recipe: "Recipe",
    base_method: Any,
    catalog_root: Path = _CATALOG_ROOT,
) -> "Recipe":
    """Stamp headline_metric + headline_basis on a recipe with null headline.

    For MEDIUM/step-policy recipes created via non-``emit_low_recipe_for_cell``
    paths (e.g. ``Recipe.from_warmup_only`` + manual gate patching), the
    performance block is not automatically computed.  This function recovers it
    deterministically from already-available data:

    - ``min_bulk_ess`` from ``recipe.gate_evidence["auto"]["min_bulk_ess"]``
    - ``total_grad_evals`` from cached ``catalog/<model>/_cache/chain_stats.npz``
      via ``base_method.grad_count_per_step``

    No re-sampling is required.  If chain_stats are unavailable, ``headline_metric``
    is left as-is (null) and a warning is printed.

    Parameters
    ----------
    recipe
        Recipe with populated ``gate_evidence.auto.min_bulk_ess`` but null
        ``headline_metric``.  If ``headline_metric`` is already non-null this
        is a no-op (returns recipe unchanged).
    base_method
        BaseMethod registry entry for the recipe's sampler.  Provides
        ``grad_count_per_step`` and ``grad_count_convention``.
    catalog_root
        Root of the catalog directory (used to locate chain_stats cache).

    Returns
    -------
    Recipe
        Updated recipe (frozen; returns a new object via ``dataclasses.replace``).
    """
    import dataclasses
    import warnings

    from tuningfork.warmup._laplace_adapter import (
        LAPLACE_METHOD_NAMES as _LAPLACE_NAMES,
    )

    if recipe.headline_metric is not None:
        return recipe  # already stamped; no-op

    # Convention (user decision 2026-05-30): PASS + REVIEW both expose headline.
    # FAIL = catastrophic non-mixing → no meaningful ESS → refuse to stamp.
    verdict = recipe.gate_evidence.get("auto", {}).get("verdict")
    if verdict == "FAIL":
        raise ValueError(
            f"stamp_headline_from_chain_stats: refusing to stamp headline on a "
            f"FAIL recipe ({recipe.model_name}/{recipe.base_method_name}). "
            "FAIL recipes have no meaningful ESS and correctly carry null headline_metric. "
            "Only PASS and REVIEW recipes may be stamped."
        )

    min_bulk_ess = recipe.gate_evidence.get("auto", {}).get("min_bulk_ess")
    if min_bulk_ess is None:
        warnings.warn(
            f"stamp_headline_from_chain_stats: gate_evidence.auto.min_bulk_ess is None "
            f"for {recipe.model_name}/{recipe.base_method_name}; cannot recover headline.",
            stacklevel=2,
        )
        return recipe

    # Load chain_stats from cache
    chain_stats_path = catalog_root / recipe.model_name / "_cache" / "chain_stats.npz"
    if not chain_stats_path.exists():
        warnings.warn(
            f"stamp_headline_from_chain_stats: chain_stats cache not found at "
            f"{chain_stats_path}; cannot recover total_grad_evals.",
            stacklevel=2,
        )
        return recipe

    import jax.numpy as jnp
    import numpy as np

    from tuningfork.metrics.grad_counter import total_grad_evals as _total_grad_evals

    stats_data = np.load(str(chain_stats_path))
    # Reconstruct a minimal infos-like object from chain_stats so that
    # grad_count_per_step can access the needed fields via attribute access.
    # Must be a NamedTuple (not a plain class instance) so that jax.vmap can
    # traverse it as a pytree (plain class instances are not JAX pytrees).
    from collections import namedtuple

    _fields = list(stats_data.files)
    _ChainStatsProxy = namedtuple("_ChainStatsProxy", _fields)  # type: ignore[misc]
    proxy = _ChainStatsProxy(**{k: jnp.asarray(stats_data[k]) for k in _fields})
    try:
        grad_evals = _total_grad_evals(proxy, base_method.grad_count_per_step)
        if grad_evals <= 0:
            warnings.warn(
                f"stamp_headline_from_chain_stats: grad_evals={grad_evals} ≤ 0 "
                f"from chain_stats; using min_bulk_ess as headline (denominator=1).",
                stacklevel=2,
            )
            return recipe
    except Exception as exc:
        warnings.warn(
            f"stamp_headline_from_chain_stats: failed to compute total_grad_evals: {exc}; "
            f"falling back to gate_evidence.min_bulk_ess / n_steps approximation.",
            stacklevel=2,
        )
        return recipe

    _headline = float(min_bulk_ess) / float(grad_evals)
    _is_laplace = recipe.base_method_name in _LAPLACE_NAMES
    _basis = {
        "total_grad_evals": int(grad_evals),
        "min_bulk_ess": float(min_bulk_ess),
        "grad_count_convention": base_method.grad_count_convention
        or recipe.base_method_name,
        "is_lower_bound": _is_laplace,
    }

    return dataclasses.replace(recipe, headline_metric=_headline, headline_basis=_basis)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:
    """CLI: emit LOW recipe for a single (model, warmup, sampler) cell.

    Usage::

        JAX_PLATFORM_NAME=cpu uv run python -m tuningfork.recipes._recipe_runner \
            --model mvn_10 \
            --warmup window_adaptation_diag_imm \
            --sampler nuts

    Exits 0 on PASS, 1 on FAIL/REVIEW/ERROR.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Recipe runner LOW emit: warmup + sample + auto-gate for one "
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
        default=RECIPE_N_WARMUP,
        help=f"Warmup steps (default {RECIPE_N_WARMUP})",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=RECIPE_N_SAMPLES,
        help=f"Post-warmup samples (default {RECIPE_N_SAMPLES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RECIPE_SEED,
        help=f"JAX random seed (default {RECIPE_SEED})",
    )
    parser.add_argument(
        "--target-acceptance",
        type=float,
        default=None,
        help=(
            "Override warmup dual-averaging target acceptance rate "
            "(default: use base_method.target_acceptance_rate or 0.80)"
        ),
    )
    parser.add_argument(
        "--num-integration-steps",
        type=int,
        default=None,
        help=(
            "Override num_integration_steps for HMC/mhmc via sampler_kwargs_override "
            "(ignored for NUTS/dynamic_hmc/dmhmc)"
        ),
    )
    args = parser.parse_args()

    sampler_kwargs_override: dict[str, Any] | None = None
    if args.num_integration_steps is not None:
        sampler_kwargs_override = {"num_integration_steps": args.num_integration_steps}

    result = emit_low_recipe_for_cell(
        model_name=args.model,
        warmup_name=args.warmup,
        sampler_name=args.sampler,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        seed=args.seed,
        target_acceptance=args.target_acceptance,
        sampler_kwargs_override=sampler_kwargs_override,
    )
    sys.exit(0 if result.verdict == "PASS" else 1)


if __name__ == "__main__":
    _main()
