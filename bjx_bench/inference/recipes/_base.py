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
  - ``Recipe.from_tuning_result(tuning_result, ...)`` — wraps a Tier-B BO
    outcome (best params + difficulty profile) into a Recipe.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import jax

    from bjx_bench.inference.base_method._base import BaseMethod
    from bjx_bench.model._base import Posterior

__all__ = ["Effort", "Recipe"]


class Effort(str, Enum):
    """Calibration effort tier — measures human + machine wall time to produce
    a recipe that the Statistician auto-gate approves.

    LOW    — ``_generate_starter`` runs the *conventional* ``(warmup, sampler)``
             pairing for the cell with all BlackJAX library defaults; the
             ``NATURAL_WARMUP_FOR_SAMPLER`` map in ``_generate_starter.py``
             defines the conventional pairing per sampler.  The Statistician
             auto-gate (``bjx_bench.calibration.statistician_gate``) evaluates
             the resulting samples on R̂ / bulk-ESS / divergence count and
             against the Tier-A reference where available (``max_abs_mean_z``,
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
                 ``stan_window`` + ``mala``, ``stan_window`` + ``rmhmc``,
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


@dataclass(frozen=True)
class Recipe:
    """A pinned (model, base_method, warmup) configuration with provenance.

    Stored as JSON at ``inference/recipes/starter/<model>/<effort>__<method>__<warmup>.json``.
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
        warmup (stan_window for nuts/hmc/mala/barker; mclmc_tuning for mclmc;
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
        for in-flight scaffolding stubs.
    sample_quality
        Optional dict of quality metrics vs. reference draws
        (``{"mae_vs_reference": ..., "q05_error": ...}``); filled by the
        recipe-emission pipeline once ``bjx_bench/metrics/reference_compare.py``
        is wired.
    calibration_budget
        Cost summary: ``{"trials": int, "wall_seconds_estimate": float, ...}``.
        ``trials > 0`` only for HIGH (where BO ran); ``wall_seconds_estimate``
        is filled at every tier (LOW captures warmup + sampler wall time).
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
    bjx_bench_version
        ``bjx_bench.__version__`` at generation time.
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
    # pair a sampler with its natural warmup (stan_window for nuts/hmc/mala/barker;
    # mclmc_tuning for mclmc; meads for ghmc; chees for dynamic_hmc; no_warmup
    # for gradient-free / specialised samplers).  Unconventional but
    # technically-possible cells (e.g., stan_window + rmhmc, stan_window + mala)
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
    # `bjx_bench/metrics/reference_compare.py` is wired.
    sample_quality: dict[str, float] | None

    # ---- Calibration cost (production effort summary) ----
    calibration_budget: dict[
        str, Any
    ]  # {"trials": int, "wall_seconds_estimate": float, ...}

    # ---- Difficulty profile (only meaningful for HIGH; None for LOW/MEDIUM) ----
    difficulty: dict[str, Any] | None  # serialized TuningDifficulty.asdict() or None

    # ---- User-facing prose ----
    instructions: str
    notes: str = ""

    inverse_mass_matrix_path: str | None = None
    # Path (relative to the recipe JSON's directory) to a .npz sidecar holding the
    # adapted inverse mass matrix when it's too large to inline (e.g., diagonal IMM
    # > ~50 entries, or any dense IMM). HIGH recipes for high-dim models populate
    # this; LOW/MEDIUM typically leave it None.

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
    # `bjx_bench.calibration.statistician_gate.auto_gate(samples, ...)`.
    # `override` is populated by the Statistician agent when it manually overrides
    # the auto-gate verdict; empty fields mean no override.
    # Schema kept as plain dict (matches existing calibration_budget /
    # base_method_params / warmup_params convention).

    # ---- Provenance ----
    tuning_seed: int = 0
    bjx_bench_version: str = "0.0.0.dev0"
    blackjax_version: str = ""
    jax_version: str = ""
    timestamp_utc: str = ""

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, root: Path) -> Path:
        """Write to ``<root>/<model_name>/<effort>__<base_method>__<warmup>.json``.

        Parameters
        ----------
        root
            Directory under which the per-model subdirectory is created.

        Returns
        -------
        Path
            The path of the written JSON file.
        """
        target_dir = Path(root) / self.model_name
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{self.effort.value}__{self.base_method_name}__{self.warmup_name}.json"
        )
        target = target_dir / filename
        d = asdict(self)
        # asdict recurses; enum values become their raw value via the Enum's __repr__
        # but we need the string value, not "Effort.LOW" — override explicitly.
        d["effort"] = self.effort.value
        target.write_text(json.dumps(d, indent=2, default=str))
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
        d["effort"] = Effort(d["effort"])
        return cls(**d)

    # ── constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_default_config(
        cls,
        posterior: Posterior,
        base_method: BaseMethod,
        *,
        bjx_bench_version: str = "0.0.0.dev0",
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
        bjx_bench_version
            Version string to embed in provenance; defaults to ``"0.0.0.dev0"``.

        Returns
        -------
        Recipe
            A frozen ``Recipe`` with ``effort=Effort.LOW``, ``warmup_name="no_warmup"``,
            ``headline_metric=None``, and ``base_method_params`` from
            ``default_params_for(base_method)``.
        """
        from bjx_bench.calibration.tier_b import default_params_for
        from bjx_bench.inference.recipes._instructions import render_instructions

        params = default_params_for(base_method)
        recipe_kwargs: dict[str, Any] = dict(
            model_name=posterior.name,
            base_method_name=base_method.name,
            warmup_name="no_warmup",
            effort=Effort.LOW,
            base_method_params=params,
            warmup_params={},
            # headline_metric is None when no MCMC has been run; may be filled later
            headline_metric=None,
            sample_quality=None,
            calibration_budget={"trials": 0, "wall_seconds_estimate": 0.0},
            difficulty=None,
            instructions="",  # rendered below after provisional construction
            notes="",
            tuning_seed=0,
            bjx_bench_version=bjx_bench_version,
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
        bjx_bench_version: str = "0.0.0.dev0",
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
            A ``Warmup`` instance from ``bjx_bench.inference.warmup.WARMUPS``.
        n_warmup
            Number of warmup adaptation steps.
        rng_key
            JAX random key for both model initialization and warmup.
        bjx_bench_version
            Version string to embed in provenance; defaults to ``"0.0.0.dev0"``.

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

        from bjx_bench.calibration.tier_b import default_params_for
        from bjx_bench.inference.recipes._instructions import render_instructions
        from bjx_bench.model._numpyro import build_logdensity_fn

        if not warmup.is_compatible(base_method.name):
            raise ValueError(
                f"warmup {warmup.name!r} is not compatible with base_method "
                f"{base_method.name!r}; "
                f"compatible_methods = {warmup.compatible_methods}"
            )

        init_key, warmup_key = jax.random.split(rng_key, 2)
        init_position, logdensity_fn, _ = build_logdensity_fn(init_key, posterior)

        # MEDIUM recipes are single-chain by design: they capture one chain's
        # adapted (step_size, IMM, ...) for downstream sampling.  Multi-chain
        # execution happens at recipe-run time, not at recipe-build time.
        # Pass num_chains=1 + squeeze the leading dim out of the result, mirroring
        # the Tier-B BO-trial pattern.
        from bjx_bench.inference.warmup._base import squeeze_single_chain

        t0 = time.perf_counter()
        batched_state, batched_params = warmup.runner(
            warmup_key,
            init_position,
            n_warmup,
            base_method,
            logdensity_fn=logdensity_fn,
            num_chains=1,
        )
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

        # Extract a stable int seed from the rng_key.  In JAX 0.10.0 the key is
        # a typed-key Array; jax.random.bits() is the portable extraction path.
        tuning_seed = int(jax.random.bits(rng_key, dtype="uint32"))

        recipe_kwargs: dict[str, Any] = dict(
            model_name=posterior.name,
            base_method_name=base_method.name,
            warmup_name=warmup.name,
            effort=Effort.MEDIUM,
            base_method_params=base_params,
            warmup_params={"n_warmup": n_warmup},
            # headline_metric is None for MEDIUM: no post-warmup samples taken;
            # May be filled later via a measurement run.
            headline_metric=None,
            sample_quality=None,
            calibration_budget=calibration_budget,
            difficulty=None,
            instructions="",  # rendered below after provisional construction
            notes="",
            tuning_seed=tuning_seed,
            bjx_bench_version=bjx_bench_version,
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
        bjx_bench_version: str = "0.0.0.dev0",
    ) -> Recipe:
        """Build a HIGH Recipe by wrapping a Tier-B BO outcome.

        The ``TuningResult`` already carries ``best_params``, ``best_score``,
        and ``difficulty``; this constructor stamps provenance, serialises the
        difficulty profile, and renders the instructions prose.

        Parameters
        ----------
        tuning_result
            A ``TuningResult`` from ``bjx_bench.calibration.tier_b.tune_algorithm``.
        posterior
            The target posterior (used for model_name provenance).
        base_method
            The sampling algorithm (used for base_method_name provenance).
        warmup
            A ``Warmup`` instance recording which warmup ran during the BO study.
        bjx_bench_version
            Version string to embed in provenance; defaults to ``"0.0.0.dev0"``.

        Returns
        -------
        Recipe
            A frozen ``Recipe`` with ``effort=Effort.HIGH``, ``headline_metric``
            set to ``tuning_result.best_score``, and ``difficulty`` populated from
            ``tuning_result.difficulty``.
        """
        from dataclasses import asdict as _asdict

        from bjx_bench.inference.recipes._instructions import render_instructions

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
            headline_metric=float(tuning_result.best_score),
            sample_quality=None,
            calibration_budget=calibration_budget,
            difficulty=difficulty_dict,
            instructions="",  # rendered below after provisional construction
            notes="",
            tuning_seed=0,
            bjx_bench_version=bjx_bench_version,
            blackjax_version=_get_blackjax_version(),
            jax_version=_get_jax_version(),
            timestamp_utc=_now_utc_iso(),
        )
        provisional = cls(**recipe_kwargs)
        recipe_kwargs["instructions"] = render_instructions(provisional)
        return cls(**recipe_kwargs)

    # ── IMM sidecar helpers ───────────────────────────────────────────────────

    def save_imm_sidecar(self, root: Path, imm: jax.Array) -> str:
        """Save an inverse mass matrix as a .npz sidecar next to this recipe.

        Returns the path STRING (relative to ``root``) to embed in
        ``inverse_mass_matrix_path``. Use this from a HIGH-effort emit function
        to persist a non-scalar IMM:

            recipe = Recipe(..., inverse_mass_matrix_path=None, ...)
            sidecar_path = recipe.save_imm_sidecar(STARTER_ROOT, adapted_imm)
            # Then dataclasses.replace(recipe, inverse_mass_matrix_path=sidecar_path)
            # since Recipe is frozen.

        The sidecar lives at
        ``<root>/<model>/<effort>__<sampler>__<warmup>.imm.npz``
        so each recipe's IMM is self-contained.

        Parameters
        ----------
        root
            Directory under which the per-model subdirectory is created.
        imm
            The inverse mass matrix to persist (any shape; saved under key
            ``"imm"`` in the compressed npz).

        Returns
        -------
        str
            Path relative to ``root`` that should be stored in
            ``inverse_mass_matrix_path``.
        """
        import numpy as np

        sidecar_dir = root / self.model_name
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{self.effort.value}__{self.base_method_name}__{self.warmup_name}.imm.npz"
        )
        sidecar_path = sidecar_dir / filename
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
        with np.load(sidecar_path) as data:
            return jnp.asarray(data["imm"])


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
