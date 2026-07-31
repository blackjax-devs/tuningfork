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
"""Recipe dataclass and Effort enum.

A Recipe is a pinned ``(model, warmup, sampler)`` configuration with
hyperparameters, gate verdict, and provenance metadata.  Effort tiers measure
human + machine wall time to produce a gate-passing recipe; the Statistician
escalates LOW → MEDIUM → HIGH via the TL when the auto-gate fails.  See the
``Effort`` enum docstring for the per-tier semantics.

CI consumes a Recipe by reading the pinned ``base_method_params`` (and the
``inverse_mass_matrix_path`` sidecar if present) and running the BlackJAX
kernel directly.  What differs across tiers is the production effort, not the
consumption pattern.

Constructor helpers:

  - ``Recipe.from_default_config(posterior, base_method)`` — placeholder
    Recipe stamped with default sampler params; no MCMC run.
  - ``Recipe.from_warmup_only(posterior, base_method, warmup, ...)`` —
    runs the warmup, captures the adapted ``(step_size, IMM)``, returns a
    Recipe with the adapted params.
  - ``Recipe.from_tuning_result(tuning_result, ...)`` — wraps a BO tuning
    outcome (best params + difficulty profile) into a Recipe.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import jax

    from tuningfork.base_method._base import BaseMethod
    from tuningfork.model._base import Posterior

__all__ = [
    "Effort",
    "FailureDiagnosis",
    "SplitSource",
    "AttemptedConfig",
    "Recipe",
    "RecipeFailedError",
    "validate_warmup_num_chains",
]


class Effort(str, Enum):
    """Calibration effort tier — measures human + machine wall time to produce
    a recipe that the Statistician auto-gate approves.

    LOW    — ``_generate_starter`` runs the *conventional* ``(warmup, sampler)``
             pairing for the cell with all BlackJAX library defaults; the
             ``NATURAL_WARMUP_FOR_SAMPLER`` map in ``_generate_starter.py``
             defines the conventional pairing per sampler.  The Statistician
             auto-gate (``tuningfork.calibration.statistician_gate``) evaluates
             the resulting samples on R̂ / bulk-ESS / divergence count and
             against the reference where available (``max_abs_mean_z``,
             ``sample_quality``).  Recipe commits at LOW iff the gate passes
             (or the Statistician overrides REVIEW to APPROVE).
             Wall time: machine only (warmup + sampling on the run host).

    MEDIUM — Statistician investigation.  Two branches lead here:

             (a) **LOW gate failed**.  Manual workarounds — change random seed,
                 investigate "obvious bugs" (chain not moving, NaNs), try
                 alternate initialisations (default is from the prior; try
                 ``uniform(-1, 1)`` Stan-style, zero, or model-specific values).
             (b) **Unconventional pairing exploration**.  The cell pairs a
                 sampler with a *technically-possible-but-unconventional* warmup
                 outside its ``NATURAL_WARMUP_FOR_SAMPLER`` mapping (e.g.,
                 ``window_adaptation_diag_imm`` + ``mala``, ``window_adaptation_diag_imm`` + ``rmhmc``,
                 ``pathfinder`` + ``hmc``).  These are not in the LOW emit set;
                 the Statistician explores them deliberately to learn whether the
                 unconventional pairing is worth recommending.

             Both branches re-run warmup + sampler + auto_gate; the intervention
             is recorded in ``Recipe.notes``.
             Wall time: LOW + Statistician investigation.

    HIGH   — Both LOW and MEDIUM failed: the conventional pairing with defaults
             doesn't pass, and Statistician workarounds + alternative pairings
             don't recover it.  The Statistician brings in a gold-standard
             reference — compares the failing run against NUTS + window_adaptation
             output (step_size, inverse_mass_matrix), runs BO over warmup
             hyperparameters (BO is used primarily for warmup HPs; optional on
             sampler HPs), and injects model-specific parameters into either the
             warmup or the sampler.  The Statistician writes up the full Bayesian-
             workflow journey in ``Recipe.workflow``.
             Wall time: MEDIUM + extra Statistician work + BO compute.
             When the HIGH cell consumes groundtruth samples for reference comparison,
             ``wall_seconds_estimate`` MUST = ``groundtruth_wall + extra_engineering_wall``
             (i.e., include the upstream groundtruth generation cost). The convention
             applies from future specialised-sampler work onward.

    GROUNDTRUTH — Long-NUTS reference run (1×100k samples, 10-chunk split-R̂
                  certification). Not a recommendation; not part of the
                  LOW→MEDIUM→HIGH escalation ladder. One per NUTS-path model.
                  Wall time: dominated by long single chain (~5–15 min/model on CPU).
                  The cached draws under ``reference/<model>/draws.npz`` are the
                  canonical samples; the recipe pins the protocol for re-running.

    FAILED    — A hard direction to land. The Statistician's HIGH-effort
                investigation walked one or more forking paths without producing
                a gate-passing config; the recipe records every attempt + diagnosis
                in ``attempted_configurations`` so future agents can pick up
                directions not yet tried. See ``FailureDiagnosis`` for the
                canonical failure buckets.

    **Tier transition discipline.**  LOW → MEDIUM and MEDIUM → HIGH transitions
    start with the Statistician communicating findings to the TL; the TL
    evaluates and makes the escalation call.  This prevents the auto-gate from
    silently churning through every cell at HIGH — only cells the TL explicitly
    escalates get the expensive treatment.

    **Recipe count per cell.**  Conceptually a ``(model, warmup, sampler)`` cell
    has a SINGLE recipe at the lowest tier that passed.  Edge cases that warrant
    multiple recipes per cell:

      - LOW passes but is unstable across seeds → a more reliable MEDIUM recipe
        is also kept.
      - Extra effort yields meaningfully better ESS/grad → both LOW and HIGH are
        kept so the user can choose by ESS-per-grad budget.

    **CI consumption.**  CI reads the pinned ``base_method_params`` (and
    ``inverse_mass_matrix_path`` sidecar if present) directly from the recipe and
    runs the BlackJAX kernel without re-running warmup.  This "sampler-only at
    runtime" consumption pattern applies to recipes at *any* tier — what makes
    HIGH special is the production effort, not the consumption.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    GROUNDTRUTH = "groundtruth"
    FAILED = "failed"


class SplitSource(str, Enum):
    """How the ``warmup_wall_seconds`` / ``sampling_wall_seconds`` split was obtained.

    MEASURED   — both phases were timed by recipe certification or generated
                 execution. The most trustworthy source.
    MANUAL     — a human set the values by hand (e.g., timing from an external
                 run log, or a retrospective estimate).
    ANALYTIC_NA — analytic-path model: no warmup or sampling phase exists
                  (draws are drawn directly from the closed-form posterior);
                  timing fields are ``None`` by definition.
    """

    MEASURED = "measured"
    MANUAL = "manual"
    ANALYTIC_NA = "analytic_na"


class FailureDiagnosis(str, Enum):
    """Categorization of why a recipe FAILED (no gate-passing config found).

    OUT_OF_SCOPE          — Sampler conceptually wrong for this model class.
    REQUIRES_ALT_SAMPLER  — Requires a kernel not in the v1 inventory.
    REQUIRES_MODEL_CHANGE — Model parameterization needs work upstream.
    TRIVIAL_FIX_DEFERRED  — Known fix not yet landed (deferred work).
    HARD_DIRECTION        — Tried multiple forking paths; none cleared the gate.
                            Default for most Statistician-closed failures.
    """

    OUT_OF_SCOPE = "out_of_scope"
    REQUIRES_ALT_SAMPLER = "requires_alt_sampler"
    REQUIRES_MODEL_CHANGE = "requires_model_change"
    TRIVIAL_FIX_DEFERRED = "trivial_fix_deferred"
    HARD_DIRECTION = "hard_direction"


@dataclass(frozen=True)
class AttemptedConfig:
    """One forking-path branch the Statistician walked down for a FAILED recipe.

    Each FAILED Recipe's ``attempted_configurations`` is a list of these,
    forming the full forking-path log of HP combinations tried.
    """

    base_method_params: dict
    warmup_params: dict
    seed: int
    gate_verdict: dict  # contains: verdict, rhat_max, min_bulk_ess, n_divergences
    wall_seconds: float
    note: str  # one-line "why I tried this and what I saw"


class RecipeFailedError(RuntimeError):
    """Raised when a consumer tries to run a FAILED recipe.

    Carries failure_diagnosis + a pointer to attempted_configurations
    in the message so the consumer can decide whether to retry with
    a different forking path.
    """

    def __init__(self, recipe: Recipe):
        diagnosis = recipe.failure_diagnosis
        diagnosis_text = (
            diagnosis.value if isinstance(diagnosis, FailureDiagnosis) else diagnosis
        )
        super().__init__(
            f"Recipe {recipe.model_name}/{recipe.base_method_name}/"
            f"{recipe.warmup_name} is FAILED ({diagnosis_text or 'no_diagnosis'}). "
            f"See workflow + attempted_configurations for the forking-path log."
        )
        self.recipe = recipe


# ── init_strategy validation ──────────────────────────────────────────────────

_VALID_INIT_STRATEGY_TYPES: frozenset[str] = frozenset(
    {
        "prior_sample",
        "zero",
        "uniform",
        "zero_perchain",
        "uniform_perchain",
        "reference_summary",
    }
)


def validate_init_strategy(v: dict[str, Any] | None) -> None:
    """Validate an ``init_strategy`` dict at load time.

    Parameters
    ----------
    v
        ``None`` (default — backward-compatible, behaves as ``"prior_sample"``)
        or a tagged-union dict with a ``"type"`` key.

    Raises
    ------
    ValueError
        If ``v`` is not ``None``, not a ``dict``, uses an unknown ``"type"``,
        or is a ``"uniform"`` / ``"uniform_perchain"`` spec with ``low >= high``
        or missing bounds.
    """
    if v is None:
        return
    if not isinstance(v, dict):
        raise ValueError(
            f"init_strategy must be a dict or None; got {type(v).__name__!r}"
        )
    type_ = v.get("type")
    if type_ not in _VALID_INIT_STRATEGY_TYPES:
        raise ValueError(
            f"init_strategy type {type_!r} not recognised. "
            f"Valid types: {sorted(_VALID_INIT_STRATEGY_TYPES)!r}"
        )
    if type_ in ("uniform", "uniform_perchain"):
        if "low" not in v or "high" not in v:
            raise ValueError(
                f"init_strategy type='{type_}' requires both 'low' and 'high' keys"
            )
        low, high = float(v["low"]), float(v["high"])
        if low >= high:
            raise ValueError(
                f"init_strategy type='{type_}' requires low < high; "
                f"got low={v['low']!r}, high={v['high']!r}"
            )
    if type_ == "zero_perchain":
        # Jitter scale is optional; default is 0.5
        if "jitter" in v:
            jitter = float(v["jitter"])
            if jitter < 0:
                raise ValueError(
                    f"init_strategy type='zero_perchain' jitter must be >= 0; "
                    f"got jitter={v['jitter']!r}"
                )
    if type_ == "reference_summary":
        import math
        import re

        def _numeric_tree(value: Any, label: str) -> tuple[int, ...]:
            if isinstance(value, bool):
                raise ValueError(f"reference_summary {label} contains boolean")
            if isinstance(value, (int, float)):
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(
                        f"reference_summary {label} contains non-finite value"
                    )
                return ()
            if isinstance(value, list):
                shapes = [_numeric_tree(item, label) for item in value]
                if shapes and any(shape != shapes[0] for shape in shapes[1:]):
                    raise ValueError(f"reference_summary {label} has ragged shape")
                return (len(value),) + (shapes[0] if shapes else ())
            raise ValueError(
                f"reference_summary {label} must contain only numeric lists"
            )

        required = {"mean", "std", "offsets", "source_path", "source_sha256"}
        missing = required.difference(v)
        if missing:
            raise ValueError(
                "init_strategy type='reference_summary' missing keys: "
                f"{sorted(missing)!r}"
            )
        if not isinstance(v["mean"], dict) or not isinstance(v["std"], dict):
            raise ValueError(
                "reference_summary mean and std must be JSON object mappings"
            )
        if set(v["mean"]) != set(v["std"]):
            raise ValueError("reference_summary mean/std keys must match")
        offsets = v["offsets"]
        if not isinstance(offsets, list) or not offsets:
            raise ValueError("reference_summary offsets must be a non-empty list")
        if any(isinstance(x, bool) for x in offsets):
            raise ValueError("reference_summary offsets must not contain booleans")
        try:
            values = [float(x) for x in offsets]
        except (TypeError, ValueError) as exc:
            raise ValueError("reference_summary offsets must be numeric") from exc

        if not all(math.isfinite(x) for x in values):
            raise ValueError("reference_summary offsets must be finite")
        if not isinstance(v["source_path"], str) or not v["source_path"]:
            raise ValueError("reference_summary source_path must be a non-empty string")
        if not isinstance(v["source_sha256"], str) or len(v["source_sha256"]) != 64:
            raise ValueError(
                "reference_summary source_sha256 must be a SHA-256 hex digest"
            )
        try:
            int(v["source_sha256"], 16)
        except ValueError as exc:
            raise ValueError(
                "reference_summary source_sha256 must be hexadecimal"
            ) from exc
        mean_shapes: dict[str, tuple[int, ...]] = {}
        for name in ("mean", "std"):
            for key, values in v[name].items():
                mean_shape = _numeric_tree(values, f"{name}[{key!r}]")
                if name == "std":

                    def _check_nonnegative(item: Any) -> None:
                        if isinstance(item, list):
                            for child in item:
                                _check_nonnegative(child)
                        elif float(item) < 0:
                            raise ValueError(
                                f"reference_summary std[{key!r}] must be non-negative"
                            )

                    _check_nonnegative(values)
                if name == "mean":
                    mean_shapes[key] = mean_shape
                elif mean_shapes.get(key) != mean_shape:
                    raise ValueError(
                        f"reference_summary mean/std[{key!r}] shapes must match"
                    )
        if not re.fullmatch(r"[0-9a-f]{64}", v["source_sha256"]):
            raise ValueError(
                "reference_summary source_sha256 must be lowercase hexadecimal"
            )


def validate_warmup_num_chains(v: list[int] | None, n_phases: int) -> None:
    """Validate a ``warmup_num_chains`` list at load / recipe-construction time.

    Parameters
    ----------
    v
        ``None`` (default — all phases use sampling ``num_chains``)
        or a list of ints, one per warmup phase.
    n_phases
        Number of warmup phases in ``Recipe.warmups`` (``len(recipe.warmups)``).
        For single-phase recipes this is 1; for multi-phase recipes it is the
        length of the ``warmups`` list.

    Raises
    ------
    ValueError
        If ``v`` is not ``None``, not a list, contains non-positive values, or
        has a length that does not match ``n_phases``.
    """
    if v is None:
        return
    if not isinstance(v, list):
        raise ValueError(
            f"warmup_num_chains must be a list[int] or None; "
            f"got {type(v).__name__!r}"
        )
    if len(v) != n_phases:
        raise ValueError(
            f"warmup_num_chains has {len(v)} entries but recipe has "
            f"{n_phases} warmup phase(s); lengths must match"
        )
    for i, w in enumerate(v):
        if not isinstance(w, int):
            raise ValueError(
                f"warmup_num_chains[{i}] must be an int; got {type(w).__name__!r}"
            )
        if w < 1:
            raise ValueError(f"warmup_num_chains[{i}] must be >= 1; got {w!r}")


@dataclass(frozen=True)
class Recipe:
    """A pinned (model, base_method, warmup) configuration with provenance.

    Stored as JSON at ``catalog/<model>/recipes/<effort>__<method>__<warmup>.json``
    (or ``catalog/<model>/groundtruth.json`` for ``effort=GROUNDTRUTH``).
    Loaded via ``Recipe.load(path)``; saved via ``recipe.save(root)``.

    Parameters
    ----------
    model_name
        Registry name of the posterior, e.g. ``"mvn_10"``.
    base_method_name
        Registry name of the algorithm, e.g. ``"nuts"``.
    warmup_name
        Name of the warmup procedure registered in WARMUPS.  The choice of
        warmup is a property of the cell (model, warmup, sampler), not the
        effort tier.  Conventional cells pair a sampler with its natural
        warmup (window_adaptation_diag_imm for nuts/hmc/mala/barker; mclmc_tuning for mclmc;
        meads for ghmc; chees for dynamic_hmc; no_warmup for gradient-free /
        specialised samplers) — see ``NATURAL_WARMUP_FOR_SAMPLER`` in
        ``_generate_starter.py``.
    effort
        Calibration effort level (``Effort.LOW``, ``Effort.MEDIUM``,
        or ``Effort.HIGH``); see the ``Effort`` enum docstring for the
        canonical gate-driven escalation semantics.
    base_method_params
        Pinned hyperparameter dict for the sampler kernel, e.g.
        ``{"step_size": 0.031, "num_integration_steps": 64}``.
    warmup_params
        Hyperparameters used for the warmup procedure at recipe-build time
        (e.g., ``n_warmup``, ``target_acceptance``).  Non-empty at every
        effort tier — LOW always runs warmup with library defaults;
        MEDIUM/HIGH may tune these values.
    headline_metric
        ``min_bulk_ess_per_grad`` at the pinned ``(warmup_params,
        base_method_params)``.  Filled at every effort tier that produces a
        gate-passing recipe (LOW, MEDIUM, HIGH all run warmup + sampler at
        recipe-build time).  ``None`` means "not yet measured" — used only
        for in-flight scaffolding stubs.  The bulk-ESS is the rank-normalised
        split-chain estimator (``blackjax.diagnostics.ess_bulk``); see
        ``catalog/RECIPE_SCHEMA.md`` §4.5 and ``headline_basis`` below.
    sample_quality
        Optional dict of quality metrics vs. reference draws
        (``{"mae_vs_reference": ..., "q05_error": ...}``); filled by the
        recipe-emission pipeline once ``tuningfork/metrics/reference_compare.py``
        is wired.
    calibration_budget
        Cost summary: ``{"trials": int, "wall_seconds_estimate": float, ...}``.
        ``trials > 0`` only for HIGH (where BO ran); ``wall_seconds_estimate``
        is filled at every tier (LOW captures warmup + sampler wall time).

        Optional timing-breakdown fields (all ``None`` for legacy recipes):

        ``warmup_wall_seconds`` : float | None
            Wall seconds for the warmup phase (between compiled calls, at Python
            orchestration level).  Set by the runner when ``split_source="measured"``.
        ``sampling_wall_seconds`` : float | None
            Wall seconds for the sampling phase.  Set by the runner when
            ``split_source="measured"``.
        ``sampling_seconds_per_draw`` : float | None
            ``sampling_wall_seconds / (n_samples * num_chains)``.  Normalised
            per-draw cost useful for cross-model comparisons.
        ``split_source`` : str | None
            How the split was obtained — ``"measured"``, ``"manual"``, or
            ``"analytic_na"``.  See ``SplitSource`` enum for semantics.
        ``machine_info`` : dict | None
            Hardware + software snapshot at recipe-write time (CPU model, core
            count, OS, JAX/BlackJAX versions, x64 flag, GPU if visible).
            Written by ``get_machine_info()`` from ``tuningfork._machine_info``.
    difficulty
        Serialised ``TuningDifficulty.asdict()`` or ``None``.  Only meaningful
        for HIGH recipes (the only tier with a BO study).
    instructions
        Auto-templated user-facing prose (rendered by ``_instructions.py``).
    notes
        Statistician-authored note when MEDIUM workaround was applied
        (seed change, init change, "obvious bug" fix, or reason for
        exploring an unconventional pairing).  Empty string for LOW.
    tuning_seed
        Random seed used during recipe-build time MCMC.  ``0`` for legacy
        ``from_default_config`` no-MCMC stubs; nonzero for any tier that
        actually ran warmup + sampler.
    tuningfork_version
        ``tuningfork.__version__`` at generation time.
    blackjax_version
        ``blackjax.__version__`` at generation time.
    jax_version
        ``jax.__version__`` at generation time.
    timestamp_utc
        ISO-8601 UTC timestamp when the recipe was generated.
    """

    # ---- Identity (the 4-axis index) ----
    model_name: str
    base_method_name: str
    # Name of the warmup procedure registered in WARMUPS.  The choice of warmup
    # is a property of the *cell* (model, warmup, sampler), not the effort tier:
    # every tier uses whichever warmup the cell specifies.  Conventional cells
    # pair a sampler with its natural warmup (window_adaptation_diag_imm for nuts/hmc/mala/barker;
    # mclmc_tuning for mclmc; meads for ghmc; chees for dynamic_hmc; no_warmup
    # for gradient-free / specialised samplers).  Unconventional but
    # technically-possible cells (e.g., window_adaptation_diag_imm + rmhmc, window_adaptation_diag_imm + mala)
    # are explored under MEDIUM effort.
    warmup_name: str
    effort: Effort

    # ---- Pinned config ----
    base_method_params: dict[str, Any]
    # Warmup hyperparameters used at recipe-build time (e.g., n_warmup,
    # target_acceptance).  Non-empty at every effort tier — LOW always runs
    # warmup with library defaults; MEDIUM/HIGH may tune these values.
    warmup_params: dict[str, Any]

    # ---- Performance ----
    # min_bulk_ess_per_grad measured at the pinned (warmup_params,
    # base_method_params).  Filled at every effort tier that produces a
    # gate-passing recipe (LOW, MEDIUM, HIGH all run MCMC at recipe-build
    # time).  None means "not yet measured".
    headline_metric: float | None
    # sample_quality is filled by the recipe-emit pipeline once
    # `tuningfork/metrics/reference_compare.py` is wired.
    sample_quality: dict[str, float] | None

    # ---- Calibration cost (production effort summary) ----
    calibration_budget: dict[
        str, Any
    ]  # {"trials": int, "wall_seconds_estimate": float, ...}
    # For HIGH recipes that consume GROUNDTRUTH samples upstream, include the
    # upstream groundtruth wall time in `wall_seconds_estimate`.

    # ---- Difficulty profile (only meaningful for HIGH; None for LOW/MEDIUM) ----
    difficulty: dict[str, Any] | None  # serialized TuningDifficulty.asdict() or None

    # ---- User-facing prose ----
    instructions: str
    notes: str = ""
    # headline_basis records the accounting details behind headline_metric so that
    # cross-recipe comparisons are interpretable (Gap-1, decisions/2026-05-30).
    # None when headline_metric is None (e.g., failed recipes or scaffolding stubs).
    # Keys: total_grad_evals, min_bulk_ess, ess_estimator, min_bulk_ess_classic_legacy,
    # estimator_ratio, grad_count_convention, is_lower_bound.  ess_estimator is the
    # provenance stamp — a basis that merely reproduces headline_metric is
    # self-consistent under ANY estimator, so consistency alone cannot audit it.
    # Full field semantics: catalog/RECIPE_SCHEMA.md §4.5.
    headline_basis: dict[str, Any] | None = None  # optional; added 2026-05-30

    # ---- Callable-injection policy ----
    # Callable-injection policy for samplers that accept a distribution over
    # integration steps (``dynamic_hmc``, ``dmhmc``).  Stored as a plain JSON
    # dict so the recipe is fully serialisable; the registry at
    # ``tuningfork.base_method._step_policy_registry.build_step_policy`` reconstructs
    # the callable at execution time.
    #
    # ``None`` (default) means "use the library default": for ``dynamic_hmc`` /
    # ``dmhmc`` that is ``lambda key: jax.random.randint(key, (), 1, 10)``
    # (V0, uniform integer in [1, 10)).  Existing recipes without this field
    # round-trip cleanly — ``Recipe.load`` falls back to ``None`` via
    # ``d.setdefault("step_policy", None)``.
    #
    # Path A (parametric) examples::
    #
    #   {"kind": "uniform_int", "low": 1, "high": 10}   # V0 — library default
    #   {"kind": "uniform_int", "low": 50, "high": 200}  # V2 — long trajectory
    #
    # Path B (empirical)::
    #
    #   {"kind": "empirical", "values": [...], "weights": [...]}  # V7 NUTS-harvested
    #
    # See also ``tuningfork.base_method._step_policy_registry.build_step_policy``.
    step_policy: dict[str, Any] | None = None

    # ---- Warmup sequence (schema extension for warmups list) ----
    # Ordered list of warmup stages; each stage is a dict with "name" and "params"
    # keys.  Replaces the legacy ``warmup_name`` / ``warmup_params`` flat fields
    # in the JSON serialisation (§2.4: immediate deprecation on schema-add,
    # 2026-05-21).  ``Recipe.save`` emits only ``warmups``; ``Recipe.load``
    # accepts EITHER the new ``warmups`` list OR legacy ``warmup_name`` /
    # ``warmup_params`` flat fields so that on-disk legacy recipes
    # continue to load without regen.
    #
    # Single-stage example (current default)::
    #
    #   [{"name": "window_adaptation_diag_imm",
    #     "params": {"n_warmup": 1000, "num_chains": 4, "target_acceptance": 0.8}}]
    #
    # Multi-stage example (future use; §2.2)::
    #
    #   [{"name": "pathfinder",                  "params": {...}},
    #    {"name": "window_adaptation_diag_imm",  "params": {...}}]
    warmups: list[dict[str, Any]] = field(default_factory=list)

    # ---- Warmup inner kernel (schema extension for warmup_inner_kernel; §3) ----
    # When ``None`` (default), the runner resolves the warmup kernel via
    # ``resolve_warmup_algorithm(base_method)`` — the current implicit
    # substitute-family logic (NUTS for laplace_*/dynamic_hmc/dmhmc; the
    # sampler itself for all other methods).
    #
    # When set explicitly (e.g. ``"nuts"``), the specified kernel is used for
    # all window-adaptation warmup stages, overriding the implicit default.
    # This enables opt-in NUTS warmup for non-substitute-family samplers (e.g.
    # ``hmc + inner_nuts`` where NUTS's tree-based trajectory adapts (step_size,
    # IMM) more robustly on some geometries).
    #
    # See RECIPE_SCHEMA.md §3 and ``_warmup_to_sampler_transform.py`` for the
    # resolution-table semantics.
    warmup_inner_kernel: str | None = None

    # ---- Per-phase warmup chain count (schema extension) ----
    # One int per warmup phase: how many independent chains to warm up.
    # ``None`` → backward-compat default (all phases use sampling ``num_chains``,
    # i.e. the current vmap'd behavior).
    #
    # When set, ``len(warmup_num_chains)`` must equal ``len(warmups)`` (or 1
    # for single-phase warmups).  Each entry W must satisfy ``W >= 1``; W > S
    # is allowed and uses the same reduce-then-broadcast path as W < S.
    #
    # Dispatch semantics per phase i with W = warmup_num_chains[i], S = num_chains:
    #
    #   W == S  → current vmap'd warmup; per-chain adapted params (no reduce/broadcast)
    #   W != S  → (i) vmap warmup over W chains;
    #             (ii) reduce W params to 1 via arithmetic mean;
    #             (iii) broadcast params to S;
    #             (iv) replicate position via position[s % W] for s in [0, S)
    #
    # Use W=1 (single-chain warmup + broadcast) to avoid the vmap-of-while_loop
    # worst-case-iteration penalty for expensive-logprob models such as
    # gp_regression × laplace_mhmc (7× slower under vmap; deadlock at scale).
    warmup_num_chains: list[int] | None = None

    # ---- Init strategy (schema extension) ----
    # Tagged-union dict specifying how the initial position for warmup + sampling
    # is drawn.  ``None`` (default) preserves backward-compatible behavior — the
    # runner calls ``build_logdensity_fn`` which samples from the prior.
    #
    # Valid specs::
    #
    #   None                                              # default: prior sample
    #   {"type": "prior_sample"}                          # explicit prior sample
    #   {"type": "zero"}                                  # all-zero init
    #   {"type": "uniform", "low": -1.0, "high": 1.0}    # uniform in [low, high]
    #
    # Validated at :py:meth:`Recipe.load` time via :py:func:`validate_init_strategy`
    # so unknown types / malformed specs raise immediately.
    # Applied at execution time by ``_apply_init_strategy`` in ``_recipe_runner.py``.
    # ``low`` / ``high`` are site-agnostic (all parameters share the same bounds);
    # per-site bounds are deferred to a future schema extension.
    init_strategy: dict[str, Any] | None = None

    # ---- Variant label (filename-stem method slot) ----
    # When multiple recipes share the same (model, base_method) pair but differ
    # in preconditioner geometry (e.g., diagonal vs. LRD mclmc), ``variant_label``
    # replaces ``base_method_name`` in the filename stem to avoid collision.
    #
    # ``None`` (default): stem uses ``base_method_name`` — backward-compat.
    # ``"mclmc_lrd"``: stem becomes ``<effort>__mclmc_lrd__<warmup_name>``.
    #
    # Does NOT change ``base_method_name`` (the registry key).
    # Absent from older recipes → ``load()`` calls ``setdefault("variant_label", None)``.
    variant_label: str | None = None

    inverse_mass_matrix_path: str | None = None
    # Path (relative to the recipe JSON's directory) to a .npz sidecar holding the
    # adapted inverse mass matrix when it's too large to inline (e.g., diagonal IMM
    # > ~50 entries, or any dense IMM). HIGH recipes for high-dim models populate
    # this; LOW/MEDIUM typically leave it None.
    # For multichain GT recipes (gt_schema_version="gt_v2_multichain"), this is always
    # None — per-chain IMMs are adapted at runtime, not stored.

    # ---- Multichain GT schema discriminator (schema extension 2026-07-15) ----
    # Absent / None → LEGACY single-chain GT or non-GT recipe (backward-compat).
    # "gt_v2_multichain" → migrated multichain groundtruth (10×10k draws, per-chain
    # window_adaptation).  Allows consumers to distinguish protocol at the JSON level
    # without a filesystem lookup to groundtruth_samples/blackjax/summary_v2.json.
    gt_schema_version: str | None = None

    # Relative path (catalog-root-relative) to the authoritative summary_v2.json for
    # this multichain GT recipe.  None for legacy recipes.  Allows a consumer to
    # navigate from recipe → diagnostics without hard-coding the path derivation rule.
    summary_v2_path: str | None = None

    workflow: str = ""
    # Long-form Bayesian-workflow narrative (markdown). HIGH recipes populate this
    # with the journey: gold-standard comparison findings, BO trial summary,
    # model-specific param choices. LOW/MEDIUM typically leave empty (use `notes`
    # instead for the shorter MEDIUM workaround text).

    gate_evidence: dict = field(
        default_factory=lambda: {
            "auto": {
                "rhat_max": None,
                "min_bulk_ess": None,
                "n_divergences": None,
                "max_abs_mean_z": None,  # None if no ground truth available
                "verdict": "NOT_RUN",  # "PASS" | "REVIEW" | "FAIL" | "NOT_RUN"
                "margins": {},  # auto_gate per-threshold proximity info
            },
            "override": {
                "reason": "",
                "statistician_id": "",
                "decision": "",  # "" | "APPROVE" | "REJECT" | "ESCALATE"
            },
        }
    )
    # Gate provenance. `auto` is populated by
    # `tuningfork.calibration.statistician_gate.auto_gate(samples, ...)`.
    # `override` is populated by the Statistician agent when it manually overrides
    # the auto-gate verdict; empty fields mean no override.
    # Schema kept as plain dict (matches existing calibration_budget /
    # base_method_params / warmup_params convention).

    # ---- Provenance ----
    tuning_seed: int = 0
    tuningfork_version: str = "0.0.0.dev0"
    blackjax_version: str = ""
    jax_version: str = ""
    timestamp_utc: str = ""

    # ---- FAILED recipe fields ----
    failure_diagnosis: FailureDiagnosis | str | None = None
    attempted_configurations: list[Any] = field(default_factory=list)

    # Top-level annotations from newer or locally-extended recipe schemas.
    # This is intentionally private: it is not part of the ordinary Python or
    # persisted schema, but is carried through load/save for lossless I/O.
    _extra_fields: dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    # ── persistence ──────────────────────────────────────────────────────────

    def to_dict(self, *, include_legacy_warmup_fields: bool = False) -> dict[str, Any]:
        """Return a canonical, independent mapping for recipe serialization.

        By default the consolidated ``warmups`` schema is emitted.  The CLI
        compatibility path can request the legacy flat warmup keys explicitly.
        """
        # ``asdict`` recursively copies nested dataclasses (including typed
        # AttemptedConfig entries), so this method never mutates the recipe.
        raw = asdict(self)

        def _serialize(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {key: _serialize(item) for key, item in value.items()}
            if isinstance(value, list):
                return [_serialize(item) for item in value]
            if isinstance(value, tuple) and hasattr(value, "_fields"):
                # Preserve namedtuple-backed structured values such as
                # LowRankInverseMassMatrix for save()'s sidecar detection.
                return type(value)(*(_serialize(item) for item in value))
            if isinstance(value, tuple):
                return [_serialize(item) for item in value]
            return value

        d = _serialize(raw)
        extras = d.pop("_extra_fields", {})
        if not include_legacy_warmup_fields:
            d.pop("warmup_name", None)
            d.pop("warmup_params", None)
        # Merge extensions only after canonical fields are established, so a
        # future schema promotion cannot be silently overwritten.
        for key, value in extras.items():
            if key in d:
                raise ValueError(
                    f"Cannot serialize extension field {key!r}: "
                    "it collides with a canonical Recipe field"
                )
            d[key] = value
        return d

    def catalog_stem(self, *, filename_tag: str | None = None) -> str:
        """Return the canonical filename stem for this recipe and its artifacts."""
        if self.effort == Effort.GROUNDTRUTH:
            return "groundtruth"
        name = self.variant_label or self.base_method_name
        baked_from = (self.calibration_budget or {}).get("baked_from", {})
        baked_warmup = (
            baked_from.get("warmup_name", "") if isinstance(baked_from, dict) else ""
        )
        warmup = self.warmup_name or baked_warmup
        stem = f"{self.effort.value}__{name}__{warmup}"
        return f"{stem}__{filename_tag}" if filename_tag else stem

    def save(
        self,
        root: Path,
        *,
        filename_tag: str | None = None,
        imm_sidecar: str | bool = "auto",
    ) -> Path:
        """Write the recipe to its canonical location under ``root``.

        Per the catalog layout (post-R2, 2026-05-17):

        - GROUNDTRUTH recipes go to ``<root>/<model_name>/groundtruth.json``
          (no filename suffix — there's exactly one groundtruth path per model).
        - All other efforts (LOW / MEDIUM / HIGH / FAILED) go to
          ``<root>/<model_name>/recipes/<effort>__<base_method>__<warmup>.json``
          or, when ``filename_tag`` is supplied:
          ``<root>/<model_name>/recipes/<effort>__<base_method>__<warmup>__<tag>.json``.
          This is used for policy-variant MEDIUM recipes (e.g.
          ``medium__dynamic_hmc__window_adaptation_diag_imm__policy_v7-empirical-oracle.json``).

        Parameters
        ----------
        root
            Catalog root directory (e.g., ``tuningfork/catalog/``).
        filename_tag
            Optional tag appended to the recipe filename stem, e.g.
            ``"policy_v7-empirical-oracle"``.  Ignored for GROUNDTRUTH recipes.
            ``None`` (default) preserves the canonical ``<effort>__<method>__<warmup>.json``
            filename.

        Returns
        -------
        Path
            The path of the written JSON file.
        """
        model_dir = Path(root) / self.model_name
        if self.effort == Effort.GROUNDTRUTH:
            target_dir = model_dir
        else:
            target_dir = model_dir / "recipes"
        filename = f"{self.catalog_stem(filename_tag=filename_tag)}.json"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        d = self.to_dict()
        # Auto-write LRD IMM sidecar when imm_sidecar="auto" and inverse_mass_matrix
        # is a LowRankInverseMassMatrix namedtuple (not JSON-serialisable inline).
        if imm_sidecar == "auto" or imm_sidecar is True:
            _imm_in_params = d.get("base_method_params", {}).get("inverse_mass_matrix")
            _is_lrd = (
                _imm_in_params is not None
                and hasattr(_imm_in_params, "_fields")
                and "sigma" in getattr(_imm_in_params, "_fields", ())
            )
            if _is_lrd:
                _sidecar_rel = self.save_imm_sidecar(
                    root,
                    _imm_in_params,
                    filename_tag=filename_tag,
                    model=self.model_name,
                )
                d["base_method_params"].pop("inverse_mass_matrix", None)
                d["inverse_mass_matrix_path"] = _sidecar_rel
                # M1 fix: Recipe is frozen=True so we cannot assign self.x = v,
                # but the in-memory object must stay consistent with the on-disk
                # artifact.  object.__setattr__ bypasses the frozen guard — this
                # is the standard pattern for "finalising" a frozen dataclass from
                # inside one of its own methods.
                object.__setattr__(self, "inverse_mass_matrix_path", _sidecar_rel)
                object.__setattr__(
                    self,
                    "base_method_params",
                    {
                        k: v
                        for k, v in self.base_method_params.items()
                        if k != "inverse_mass_matrix"
                    },
                )
        target.write_text(json.dumps(d, indent=2, default=str) + "\n")
        return target

    @classmethod
    def load(cls, path: Path) -> Recipe:
        """Load a ``Recipe`` from a JSON file written by ``save``.

        Parameters
        ----------
        path
            Path to the JSON file.

        Returns
        -------
        Recipe
            The deserialized recipe.
        """
        d = json.loads(Path(path).read_text())
        # Defensive: missing "effort" key should raise a clear error rather than KeyError.
        # SMC recipes do not carry "effort" and should never reach Recipe.load() --
        # use load_recipe() from tuningfork.catalog.inspect which dispatches to SMCRecipe.load().
        if "effort" not in d:
            raise ValueError(
                f"Recipe at {path} is missing the 'effort' key. "
                "If this is an SMC recipe (smc__*.json), use load_recipe() or SMCRecipe.load() "
                "instead of Recipe.load() — SMC recipes do not carry the MCMC 'effort' field."
            )
        d["effort"] = Effort(d["effort"])
        # Deserialize failure_diagnosis if present (backward compat: missing key defaults to None).
        # Recognized values remain typed; free-text historical diagnoses remain
        # strings so their original annotation survives a round-trip.
        if "failure_diagnosis" in d and d["failure_diagnosis"] is not None:
            try:
                d["failure_diagnosis"] = FailureDiagnosis(d["failure_diagnosis"])
            except ValueError:
                pass
        # Deserialize attempted_configurations if present (backward compat: missing key defaults to []).
        # Only the exact canonical shape is typed.  Historical/noncanonical
        # entries remain raw JSON values so load/save is lossless.
        _ac_known = {f.name for f in fields(AttemptedConfig)}
        if "attempted_configurations" in d and d["attempted_configurations"]:
            _parsed_acs = []
            for ac in d["attempted_configurations"]:
                if isinstance(ac, dict) and set(ac) == _ac_known:
                    _parsed_acs.append(AttemptedConfig(**ac))
                else:
                    _parsed_acs.append(ac)
            d["attempted_configurations"] = _parsed_acs
        else:
            # Ensure default if key missing
            d.setdefault("attempted_configurations", [])
        # Backward compat: step_policy absent in recipes written before schema wiring.
        # None means "library default" (V0).
        d.setdefault("step_policy", None)
        # Schema-extension backward-compat (§2.4): accept EITHER the new ``warmups`` list
        # OR the legacy ``warmup_name`` / ``warmup_params`` flat fields.
        #
        # Case 1 — New format (warmups list present, no flat fields):
        #   Derive ``warmup_name`` / ``warmup_params`` from ``warmups[0]`` so the
        #   Python dataclass fields are populated correctly.
        #
        # Case 2 — Legacy format (flat fields present, no warmups list):
        #   Construct ``warmups = [{"name": warmup_name, "params": warmup_params}]``
        #   so new code that reads ``recipe.warmups`` works correctly.
        if "warmups" in d and d["warmups"]:
            # New format: derive flat fields from first warmup stage.
            first = d["warmups"][0]
            d.setdefault("warmup_name", first.get("name", ""))
            d.setdefault("warmup_params", first.get("params", {}))
        else:
            # Legacy format: construct warmups list from flat fields.
            d.setdefault("warmups", [])
            if "warmup_name" in d:
                d["warmups"] = [
                    {"name": d["warmup_name"], "params": d.get("warmup_params", {})}
                ]
            d.setdefault("warmup_name", "")
            d.setdefault("warmup_params", {})
        # Schema extension: warmup_inner_kernel absent in pre-extension recipes.
        d.setdefault("warmup_inner_kernel", None)
        # Schema extension: init_strategy absent in pre-extension recipes.
        # None = backward-compat default (prior_sample behavior).
        d.setdefault("init_strategy", None)
        validate_init_strategy(d["init_strategy"])
        # Schema extension: variant_label absent in pre-extension recipes.
        d.setdefault("variant_label", None)
        # Schema extension: warmup_num_chains absent in pre-extension recipes.
        # None = backward-compat default (all phases use sampling num_chains,
        # i.e. the current vmap'd behavior).
        d.setdefault("warmup_num_chains", None)
        _n_phases = max(len(d.get("warmups") or []), 1)
        validate_warmup_num_chains(d["warmup_num_chains"], _n_phases)
        # Schema extension: timing breakdown absent in pre-extension recipes.
        # calibration_budget is a free-form dict; back-fill None for absent keys
        # so callers can rely on .get() returning None on legacy recipes.
        if "calibration_budget" in d and isinstance(d["calibration_budget"], dict):
            for _k in (
                "warmup_wall_seconds",
                "sampling_wall_seconds",
                "sampling_seconds_per_draw",
                "split_source",
                "machine_info",
            ):
                d["calibration_budget"].setdefault(_k, None)
        # Backward-compat: calibration_budget / difficulty / instructions absent in
        # recipes that were emitted before the full schema stamp (e.g. via manual
        # triage or before the emit-path fix that moved these before early-returns).
        # Defaults: minimal calibration_budget stub; None difficulty; empty instructions.
        d.setdefault("calibration_budget", {"trials": 0, "wall_seconds_estimate": 0.0})
        d.setdefault("difficulty", None)
        d.setdefault("instructions", "")
        # Backward-compat: headline_basis absent in recipes emitted before Gap-1 (2026-05-30).
        d.setdefault("headline_basis", None)
        # Schema extension 2026-07-15: multichain GT discriminator + summary_v2 pointer.
        # None = legacy single-chain GT or non-GT recipe (backward-compat).
        d.setdefault("gt_schema_version", None)
        d.setdefault("summary_v2_path", None)
        # Keep unknown top-level keys privately for lossless load/save.  They are
        # deliberately excluded from the regular dataclass schema on disk.
        # Private implementation fields are not part of the on-disk schema;
        # a JSON key with such a name is therefore still an unknown annotation.
        _known_fields = {f.name for f in fields(cls) if not f.name.startswith("_")}
        extras = {k: v for k, v in d.items() if k not in _known_fields}
        d = {k: v for k, v in d.items() if k in _known_fields}
        d["_extra_fields"] = extras
        return cls(**d)

    # ── constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_default_config(
        cls,
        posterior: Posterior,
        base_method: BaseMethod,
        *,
        tuningfork_version: str = "0.0.0.dev0",
    ) -> Recipe:
        """Build a placeholder LOW Recipe stamped with default sampler params; no MCMC runs.

        Useful as a scaffolding stub (e.g., for tests, or to pre-allocate a Recipe
        before measurement).  ``headline_metric``, ``sample_quality``, and
        ``gate_evidence.auto`` all remain at their not-yet-measured defaults;
        provenance fields (blackjax_version, jax_version, timestamp_utc) are
        populated at call time.  A gate-passing LOW recipe — one where warmup +
        sampler ran and the Statistician auto-gate approved — is produced by the
        recipe-emit pipeline, not by this constructor.

        Parameters
        ----------
        posterior
            The target posterior describing the benchmark model.
        base_method
            The sampling algorithm whose default HP space seeds the recipe.
        tuningfork_version
            Version string to embed in provenance; defaults to ``"0.0.0.dev0"``.

        Returns
        -------
        Recipe
            A frozen ``Recipe`` with ``effort=Effort.LOW``, ``warmup_name="no_warmup"``,
            ``headline_metric=None``, and ``base_method_params`` from
            ``default_params_for(base_method)``.
        """
        from tuningfork.calibration.tune import default_params_for
        from tuningfork.recipes._instructions import render_instructions

        params = default_params_for(base_method)
        recipe_kwargs: dict[str, Any] = dict(
            model_name=posterior.name,
            base_method_name=base_method.name,
            warmup_name=(
                "mclmc_tuning"
                if base_method.name
                in {"mclmc", "adjusted_mclmc", "adjusted_mclmc_dynamic"}
                else "no_warmup"
            ),
            effort=Effort.LOW,
            base_method_params=params,
            warmup_params={},
            warmups=[
                {
                    "name": (
                        "mclmc_tuning"
                        if base_method.name
                        in {"mclmc", "adjusted_mclmc", "adjusted_mclmc_dynamic"}
                        else "no_warmup"
                    ),
                    "params": {},
                }
            ],
            # headline_metric is None when no MCMC has been run; may be filled later
            headline_metric=None,
            sample_quality=None,
            calibration_budget={"trials": 0, "wall_seconds_estimate": 0.0},
            difficulty=None,
            instructions="",  # rendered below after provisional construction
            notes="",
            tuning_seed=0,
            tuningfork_version=tuningfork_version,
            blackjax_version=_get_blackjax_version(),
            jax_version=_get_jax_version(),
            timestamp_utc=_now_utc_iso(),
        )
        # Build provisional recipe to render instructions, then rebuild with prose.
        # Two-step construction avoids threading `recipe` into render_instructions
        # before the object exists.
        provisional = cls(**recipe_kwargs)
        recipe_kwargs["instructions"] = render_instructions(provisional)
        return cls(**recipe_kwargs)

    @classmethod
    def from_warmup_only(
        cls,
        posterior: Posterior,
        base_method: BaseMethod,
        warmup: Any,  # Warmup; imported inline to avoid circular dep
        *,
        n_warmup: int = 1000,
        rng_key: Any,  # jax.Array
        tuningfork_version: str = "0.0.0.dev0",
        effort: Effort = Effort.MEDIUM,
        headline_metric: float | None = None,
        bake_warmup: bool = False,
        attempted_configurations: list | None = None,
        notes: str = "",
        variant_label: str | None = None,
        init_strategy: dict | None = None,
        **warmup_kwargs: Any,
    ) -> Recipe:
        """Build a Recipe by running ONLY the warmup (no post-warmup sampler chain).

        Captures the warmup-adapted ``(step_size, inverse_mass_matrix, ...)``
        values into a Recipe with ``effort=Effort.MEDIUM``.  Calibration cost is
        the warmup wall-clock time; ``calibration_budget`` records both
        ``n_warmup`` and ``wall_seconds_estimate``.

        Calling ``from_warmup_only(..., WARMUPS["no_warmup"])`` returns a Recipe
        with ``effort=Effort.MEDIUM`` and ``warmup_name="no_warmup"`` —
        semantically distinct from a LOW placeholder produced by
        ``from_default_config``, which never goes through this warmup path.

        Parameters
        ----------
        posterior
            The target posterior describing the benchmark model.
        base_method
            The sampling algorithm whose HP space seeds the recipe.
        warmup
            A ``Warmup`` instance from ``tuningfork.warmup.WARMUPS``.
        n_warmup
            Number of warmup adaptation steps.
        rng_key
            JAX random key for both model initialization and warmup.
        tuningfork_version
            Version string to embed in provenance; defaults to ``"0.0.0.dev0"``.
        **warmup_kwargs
            Extra keyword arguments forwarded verbatim to ``warmup.runner``
            (e.g. ``k_rank=40`` for ``mclmc_lrd_tuning``).

        Returns
        -------
        Recipe
            A frozen ``Recipe`` with ``effort=Effort.MEDIUM``, populated
            ``base_method_params`` (defaults merged with warmup-adapted values),
            and ``calibration_budget`` recording the wall-clock time.

        Raises
        ------
        ValueError
            If the warmup is not compatible with the base_method.
        """
        import time

        import jax

        from tuningfork.calibration.tune import default_params_for
        from tuningfork.model._numpyro import build_logdensity_fn
        from tuningfork.recipes._instructions import render_instructions

        if not warmup.is_compatible(base_method.name):
            raise ValueError(
                f"warmup {warmup.name!r} is not compatible with base_method "
                f"{base_method.name!r}; "
                f"compatible_methods = {warmup.compatible_methods}"
            )

        # Validate init_strategy before doing any work.
        validate_init_strategy(init_strategy)

        init_key, warmup_key = jax.random.split(rng_key, 2)
        init_position, logdensity_fn, _ = build_logdensity_fn(init_key, posterior)

        # Apply init_strategy override (e.g. zero-init, uniform jitter).
        if init_strategy is not None:
            from tuningfork.recipes._recipe_runner import _apply_init_strategy

            init_position = _apply_init_strategy(init_strategy, init_position, init_key)

        # MEDIUM recipes are single-chain by design: they capture one chain's
        # adapted (step_size, IMM, ...) for downstream sampling.  Multi-chain
        # execution happens at recipe-run time, not at recipe-build time.
        # Pass num_chains=1 + squeeze the leading dim out of the result, mirroring
        # the BO tuning-trial pattern.
        from tuningfork.warmup._base import squeeze_single_chain

        t0 = time.perf_counter()
        _base_warmup_result = warmup.runner(
            warmup_key,
            init_position,
            n_warmup,
            base_method,
            logdensity_fn=logdensity_fn,
            num_chains=1,
            **warmup_kwargs,
        )
        batched_state, batched_params = _base_warmup_result[0], _base_warmup_result[1]
        # SYNC: block until warmup compute completes before stamping wall time.
        # Without this, elapsed measures dispatch latency only, not actual compute.
        jax.block_until_ready((batched_state, batched_params))
        _state, adapted_params = squeeze_single_chain(batched_state, batched_params)
        elapsed = time.perf_counter() - t0

        # Thread underscore-prefixed metadata (e.g. "_total_tuning_steps" from
        # mclmc_tuning) into calibration_budget; strip them from recipe params.
        metadata_keys = {k: v for k, v in adapted_params.items() if k.startswith("_")}
        clean_adapted = {
            k: v for k, v in adapted_params.items() if not k.startswith("_")
        }

        # Merge defaults with adapted (adapted wins) for a complete config.
        base_params = {**default_params_for(base_method), **clean_adapted}
        base_params = _to_jsonable(base_params)

        # Coerce metadata values to JSON-safe types before storing.
        metadata_jsonable = {
            k: (int(v) if isinstance(v, (int, float)) else v)
            for k, v in metadata_keys.items()
        }

        calibration_budget: dict[str, Any] = {
            "trials": 0,
            "wall_seconds_estimate": elapsed,
            "n_warmup": n_warmup,
            **metadata_jsonable,
        }
        if attempted_configurations is not None:
            calibration_budget["seed_evidence"] = attempted_configurations

        # Extract a stable int seed from the rng_key.  In JAX 0.10.0 the key is
        # a typed-key Array; jax.random.bits() is the portable extraction path.
        tuning_seed = int(jax.random.bits(rng_key, dtype="uint32"))

        _warmup_params_dict = {"n_warmup": n_warmup}

        # bake_warmup: blank out warmup fields (runner-skip hint); provenance
        # preserved under calibration_budget["baked_from"].
        if bake_warmup:
            _effective_warmup_name = ""
            _effective_warmups: list[dict[str, Any]] = []
            calibration_budget["baked_from"] = {
                "warmup_name": warmup.name,
                "n_warmup": n_warmup,
                "tuning_seed": tuning_seed,
            }
        else:
            _effective_warmup_name = warmup.name
            _effective_warmups = [{"name": warmup.name, "params": _warmup_params_dict}]

        recipe_kwargs: dict[str, Any] = dict(
            model_name=posterior.name,
            base_method_name=base_method.name,
            warmup_name=_effective_warmup_name,
            effort=effort,
            base_method_params=base_params,
            warmup_params=_warmup_params_dict,
            warmups=_effective_warmups,
            headline_metric=headline_metric,
            sample_quality=None,
            calibration_budget=calibration_budget,
            difficulty=None,
            instructions="",  # rendered below after provisional construction
            notes=notes,
            variant_label=variant_label,
            init_strategy=init_strategy,
            tuning_seed=tuning_seed,
            tuningfork_version=tuningfork_version,
            blackjax_version=_get_blackjax_version(),
            jax_version=_get_jax_version(),
            timestamp_utc=_now_utc_iso(),
        )
        provisional = cls(**recipe_kwargs)
        recipe_kwargs["instructions"] = render_instructions(provisional)
        return cls(**recipe_kwargs)

    @classmethod
    def from_tuning_result(
        cls,
        tuning_result: Any,  # TuningResult; imported inline to avoid circular dep
        *,
        posterior: Posterior,
        base_method: BaseMethod,
        warmup: Any,  # Warmup; imported inline
        tuningfork_version: str = "0.0.0.dev0",
    ) -> Recipe:
        """Build a HIGH Recipe by wrapping a BO tuning outcome.

        The ``TuningResult`` already carries ``best_params``, ``best_score``,
        and ``difficulty``; this constructor stamps provenance, serialises the
        difficulty profile, and renders the instructions prose.

        Parameters
        ----------
        tuning_result
            A ``TuningResult`` from ``tuningfork.calibration.tune.tune_algorithm``.
        posterior
            The target posterior (used for model_name provenance).
        base_method
            The sampling algorithm (used for base_method_name provenance).
        warmup
            A ``Warmup`` instance recording which warmup ran during the BO study.
        tuningfork_version
            Version string to embed in provenance; defaults to ``"0.0.0.dev0"``.

        Returns
        -------
        Recipe
            A frozen ``Recipe`` with ``effort=Effort.HIGH``, ``headline_metric``
            set to ``tuning_result.best_score``, and ``difficulty`` populated from
            ``tuning_result.difficulty``.
        """
        from dataclasses import asdict as _asdict

        from tuningfork.recipes._instructions import render_instructions

        # difficulty is a TuningDifficulty frozen dataclass; serialize to dict
        # for JSON persistence.  dataclasses.asdict produces Python primitives
        # (float, bool, int) — verified to round-trip cleanly through json.dumps.
        difficulty_dict = (
            _asdict(tuning_result.difficulty)
            if tuning_result.difficulty is not None
            else None
        )

        base_params = _to_jsonable(tuning_result.best_params)

        calibration_budget: dict[str, Any] = {
            "trials": tuning_result.n_trials_completed,
            "wall_seconds_estimate": (
                tuning_result.difficulty.wall_seconds_to_best
                if tuning_result.difficulty is not None
                else 0.0
            ),
            "n_seeds": tuning_result.n_seeds,
        }

        recipe_kwargs: dict[str, Any] = dict(
            model_name=tuning_result.posterior_name,
            base_method_name=tuning_result.base_method_name,
            warmup_name=warmup.name,
            effort=Effort.HIGH,
            base_method_params=base_params,
            warmup_params={},
            warmups=[{"name": warmup.name, "params": {}}],
            headline_metric=float(tuning_result.best_score),
            sample_quality=None,
            calibration_budget=calibration_budget,
            difficulty=difficulty_dict,
            instructions="",  # rendered below after provisional construction
            notes="",
            tuning_seed=0,
            tuningfork_version=tuningfork_version,
            blackjax_version=_get_blackjax_version(),
            jax_version=_get_jax_version(),
            timestamp_utc=_now_utc_iso(),
        )
        provisional = cls(**recipe_kwargs)
        recipe_kwargs["instructions"] = render_instructions(provisional)
        return cls(**recipe_kwargs)

    @classmethod
    def from_groundtruth_run(
        cls,
        posterior: Posterior,
        *,
        cert: Any,  # CertificationResult; imported inline to avoid circular dep
        adaptation: Any,  # AdaptationParams; imported inline
        wall_seconds: float,
        tuning_seed: int,
        n_warmup: int,
        n_samples: int,
        n_chunks: int,
        target_acceptance: float,
        max_num_doublings: int = 10,
        tuningfork_version: str = "0.0.0.dev0",
    ) -> Recipe:
        """Build a GROUNDTRUTH Recipe from a long-NUTS reference-certification run.

        The cached draws under ``reference/<model>/draws.npz`` are the canonical
        samples; this Recipe pins the protocol so the diagnostics notebook (or any
        future re-run) can reproduce it. ``gate_evidence.auto.verdict`` is ``"PASS"``
        by construction — the cert gate (split-R̂ ≤ 1.01, min-chunk bulk-ESS ≥ 400,
        n_divergences == 0, E-BFMI ≥ 0.3) is strictly tighter than the auto-gate
        PASS band.

        ``max_abs_mean_z`` is ``None`` because groundtruth IS the reference — there
        is no upstream ground truth to compare against.

        ``headline_metric`` is ``None`` — groundtruth wall is dominated by sampling,
        not by warmup; ``wall_seconds`` belongs in ``calibration_budget``.

        Parameters
        ----------
        posterior
            The target posterior describing the benchmark model.
        cert
            ``CertificationResult`` from ``certify_reference_nuts``.
        adaptation
            ``AdaptationParams`` from ``certify_reference_nuts``.
        wall_seconds
            Total wall-clock time (warmup + sampling) for the reference run.
        tuning_seed
            Random seed used for the reference run.
        n_warmup
            Number of warmup steps.
        n_samples
            Number of post-warmup samples.
        n_chunks
            Number of chunks used for split-R̂ certification.
        target_acceptance
            Target acceptance rate used during warmup.
        tuningfork_version
            Version string to embed in provenance; defaults to ``"0.0.0.dev0"``.

        Returns
        -------
        Recipe
            A frozen ``Recipe`` with ``effort=Effort.GROUNDTRUTH``, ``base_method_name="nuts"``,
            ``warmup_name="window_adaptation_diag_imm"``, and gate evidence pre-populated from ``cert``.

        Notes
        -----
        IMM sidecar: if ``adaptation.inverse_mass_matrix.size > 50``, the caller
        (orchestrator) is responsible for calling ``recipe.save_imm_sidecar()`` after
        construction, then using ``dataclasses.replace(recipe, inverse_mass_matrix_path=...)``
        to attach the sidecar path — Recipe is frozen so this classmethod cannot do it
        inline.  When the IMM is small enough to inline, it is stored as a list in
        ``base_method_params["inverse_mass_matrix"]``; otherwise the sentinel string
        ``"sidecar"`` is stored and the caller replaces it after writing the sidecar.
        """
        import numpy as np

        from tuningfork.recipes._instructions import render_instructions

        imm = adaptation.inverse_mass_matrix
        imm_np = np.asarray(imm)
        if imm_np.size > 50:
            # Large IMM: store sentinel; caller writes sidecar + uses dataclasses.replace
            imm_value: Any = "sidecar"
        else:
            imm_value = imm_np.tolist()

        base_method_params: dict[str, Any] = {
            "step_size": float(adaptation.step_size),
            "inverse_mass_matrix": imm_value,
        }

        warmup_params: dict[str, Any] = {
            "n_warmup": n_warmup,
            "n_chunks": n_chunks,
            "target_acceptance": target_acceptance,
            "max_num_doublings": max_num_doublings,
        }

        gate_evidence: dict[str, Any] = {
            "auto": {
                "rhat_max": float(cert.split_rhat_max),
                "min_bulk_ess": float(cert.min_chunk_bulk_ess),
                "n_divergences": int(cert.num_divergences),
                "max_abs_mean_z": None,  # groundtruth IS the reference
                "verdict": "PASS",
                "margins": {},
            },
            "override": {
                "reason": "",
                "statistician_id": "",
                "decision": "",
            },
        }

        calibration_budget: dict[str, Any] = {
            "trials": 0,
            "wall_seconds_estimate": wall_seconds,
            "n_warmup": n_warmup,
            "n_samples": n_samples,
        }

        recipe_kwargs: dict[str, Any] = dict(
            model_name=posterior.name,
            base_method_name="nuts",
            warmup_name="window_adaptation_diag_imm",
            effort=Effort.GROUNDTRUTH,
            base_method_params=base_method_params,
            warmup_params=warmup_params,
            warmups=[{"name": "window_adaptation_diag_imm", "params": warmup_params}],
            headline_metric=None,
            sample_quality=None,
            calibration_budget=calibration_budget,
            difficulty=None,
            instructions="",  # rendered below after provisional construction
            notes="",
            tuning_seed=tuning_seed,
            tuningfork_version=tuningfork_version,
            blackjax_version=_get_blackjax_version(),
            jax_version=_get_jax_version(),
            timestamp_utc=_now_utc_iso(),
            gate_evidence=gate_evidence,
        )
        provisional = cls(**recipe_kwargs)
        recipe_kwargs["instructions"] = render_instructions(provisional)
        return cls(**recipe_kwargs)

    def load_cached_samples(
        self,
        *,
        cache_dir: Path | None = None,
    ) -> dict[str, Any] | None:
        """Return cached samples for this recipe's model, or None on cache miss.

        For GROUNDTRUTH recipes, returns the long-NUTS reference draws if the
        reference cache for ``self.model_name`` exists and is valid. For other
        recipe tiers, currently returns None (extensible — future HIGH-effort
        recipes with long-run samplers can opt in by populating the cache).

        Useful for the diagnostics notebook: load-or-run pattern avoids redundant
        multi-minute chain re-runs when the cache is already populated.

        Parameters
        ----------
        cache_dir
            Override the cache directory (default: standard reference cache).

        Returns
        -------
        dict[str, jax.Array] or None
            Draws dict on cache hit, None on miss or non-GROUNDTRUTH recipe.
        """
        if self.effort != Effort.GROUNDTRUTH:
            return None
        from tuningfork._cache_io import try_load_cached_draws
        from tuningfork.model import MODELS

        if self.model_name not in MODELS:
            return None
        entry = MODELS[self.model_name]
        return try_load_cached_draws(entry, cache_dir=cache_dir)

    # ── IMM sidecar helpers ───────────────────────────────────────────────────

    def save_imm_sidecar(
        self,
        root: Path,
        imm: jax.Array,
        *,
        filename_tag: str | None = None,
        model: str | None = None,
        seed: int | None = None,
        note: str = "",
    ) -> str:
        """Save an inverse mass matrix as a .npz sidecar next to this recipe.

        Returns the path STRING (relative to ``root``) to embed in
        ``inverse_mass_matrix_path``. Use this from a HIGH-effort emit function
        to persist a non-scalar IMM:

            recipe = Recipe(..., inverse_mass_matrix_path=None, ...)
            sidecar_path = recipe.save_imm_sidecar(STARTER_ROOT, adapted_imm)
            # Then dataclasses.replace(recipe, inverse_mass_matrix_path=sidecar_path)
            # since Recipe is frozen.

        Per the catalog layout (post-R2):

        - For GROUNDTRUTH recipes: ``<root>/<model>/groundtruth.imm.npz``
        - For other efforts: ``<root>/<model>/recipes/<effort>__<sampler>__<warmup>.imm.npz``
          or with ``filename_tag``:
          ``<root>/<model>/recipes/<effort>__<sampler>__<warmup>__<tag>.imm.npz``

        ``LowRankInverseMassMatrix`` namedtuples are auto-detected and saved as
        structured keys (sigma/U/lam/k) instead of a flat ``"imm"`` array.
        The ``model``, ``seed``, and ``note`` kwargs add metadata fields to LRD
        sidecars only (ignored for flat-array IMMs).

        Parameters
        ----------
        root
            Catalog root directory (e.g., ``tuningfork/catalog/``).
        imm
            The inverse mass matrix to persist.  Either a plain array (saved
            under key ``"imm"``) or a ``LowRankInverseMassMatrix`` namedtuple
            (saved as structured keys sigma/U/lam/k).
        filename_tag
            Optional tag appended to the sidecar filename stem (must match the
            tag passed to ``save()`` for the corresponding recipe JSON).
        model
            Model name metadata for LRD sidecars (e.g. ``"ill_cond_50"``).
        seed
            Tuning seed metadata for LRD sidecars.
        note
            Free-text provenance note for LRD sidecars.

        Returns
        -------
        str
            Path relative to ``root`` that should be stored in
            ``inverse_mass_matrix_path``.
        """
        import numpy as np

        model_dir = root / self.model_name
        if self.effort == Effort.GROUNDTRUTH:
            sidecar_dir = model_dir
        else:
            sidecar_dir = model_dir / "recipes"
        filename = f"{self.catalog_stem(filename_tag=filename_tag)}.imm.npz"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = sidecar_dir / filename

        # Detect LowRankInverseMassMatrix (namedtuple with sigma/U/lam fields).
        _is_lrd = hasattr(imm, "_fields") and "sigma" in getattr(imm, "_fields", ())
        if _is_lrd:
            _save_kwargs: dict[str, Any] = {
                "sigma": np.asarray(imm.sigma),
                "U": np.asarray(imm.U),
                "lam": np.asarray(imm.lam),
                "k": int(np.asarray(imm.U).shape[1]),
            }
            if model is not None:
                _save_kwargs["model"] = str(model)
            if seed is not None:
                _save_kwargs["seed"] = int(seed)
            if note:
                _save_kwargs["note"] = str(note)
            np.savez_compressed(sidecar_path, **_save_kwargs)
        else:
            np.savez_compressed(sidecar_path, imm=np.asarray(imm))

        return str(sidecar_path.relative_to(root))

    def load_imm_sidecar(self, root: Path) -> jax.Array | None:
        """Load the IMM sidecar if ``inverse_mass_matrix_path`` is set, else None.

        Parameters
        ----------
        root
            Directory relative to which ``inverse_mass_matrix_path`` is resolved.

        Returns
        -------
        jax.Array or None
            The inverse mass matrix as a JAX array, or ``None`` if
            ``inverse_mass_matrix_path`` is unset.
        """
        if self.inverse_mass_matrix_path is None:
            return None
        import jax.numpy as jnp
        import numpy as np

        sidecar_path = root / self.inverse_mass_matrix_path
        with np.load(sidecar_path, allow_pickle=False) as data:
            if "sigma" in data and "U" in data and "lam" in data:
                # LRD structured format — reconstruct LowRankInverseMassMatrix namedtuple.
                from blackjax.mcmc.metrics import LowRankInverseMassMatrix

                return LowRankInverseMassMatrix(
                    sigma=jnp.asarray(data["sigma"]),
                    U=jnp.asarray(data["U"]),
                    lam=jnp.asarray(data["lam"]),
                )
            else:
                # Legacy flat format.
                return jnp.asarray(data["imm"])

    def normalize_pinned_replay(self) -> Recipe:
        """Return the canonical no-warmup form of a baked replay recipe.

        Baked recipes historically blanked their warmup fields.  Keep that
        legacy identity in ``calibration_budget["baked_from"]`` while exposing
        a real zero-step stage to execution-plan consumers.  Existing
        provenance keys are never overwritten, making this transformation
        idempotent and lossless for evidence and unknown schema fields.
        """
        import copy

        budget = copy.deepcopy(self.calibration_budget or {})
        provenance = budget.setdefault("baked_from", {})
        if not isinstance(provenance, dict):
            provenance = {"legacy": provenance}
            budget["baked_from"] = provenance
        provenance.setdefault("warmup_name", self.warmup_name)
        provenance.setdefault("warmup_params", copy.deepcopy(self.warmup_params))
        provenance.setdefault("warmups", copy.deepcopy(self.warmups))
        provenance.setdefault(
            "warmup_num_chains", copy.deepcopy(self.warmup_num_chains)
        )
        provenance.setdefault("init_strategy", copy.deepcopy(self.init_strategy))
        return replace(
            self,
            warmup_name="no_warmup",
            warmup_params={},
            warmups=[{"name": "no_warmup", "params": {}}],
            warmup_num_chains=None,
            calibration_budget=budget,
        )

    def is_failed(self) -> bool:
        """Return True iff this recipe is FAILED (no gate-passing config found).

        Returns
        -------
        bool
            True if ``self.effort == Effort.FAILED``, False otherwise.
        """
        return self.effort == Effort.FAILED


# ── private helpers ───────────────────────────────────────────────────────────


def _to_jsonable(d: dict[str, Any]) -> dict[str, Any]:
    """Coerce non-JSON-serializable values in a flat dict to Python types.

    Converts ``jax.Array`` values (including scalars and 1-D vectors such as
    ``inverse_mass_matrix``) to Python scalars or lists via
    ``numpy.asarray(...).tolist()``.  Other types are passed through unchanged.

    Parameters
    ----------
    d
        A flat dict; values may be ``jax.Array`` or plain Python types.

    Returns
    -------
    dict[str, Any]
        A new dict with the same keys; ``jax.Array`` values replaced by
        Python lists or scalars.
    """
    import numpy as np

    try:
        import jax

        def _coerce(v: Any) -> Any:
            if isinstance(v, jax.Array):
                return np.asarray(v).tolist()
            return v

    except ImportError:

        def _coerce(v: Any) -> Any:
            return v

    return {k: _coerce(v) for k, v in d.items()}


def _get_blackjax_version() -> str:
    """Return the installed blackjax version string, or 'unavailable'."""
    try:
        import blackjax

        return getattr(blackjax, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unavailable"


def _get_jax_version() -> str:
    """Return the installed jax version string, or 'unavailable'."""
    try:
        import jax

        return jax.__version__
    except Exception:  # noqa: BLE001
        return "unavailable"


def _now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
