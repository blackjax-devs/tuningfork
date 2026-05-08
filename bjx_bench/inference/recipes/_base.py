"""Recipe dataclass and Effort enum.

A Recipe is a pinned (model, base_method, warmup) configuration with
optimal hyperparameters and provenance metadata. Recipes serve three
user personas (per PLAN_bjx_bench_API_phase2.md §"Tuning Difficulty Metric"):

  Effort.LOW    — one-off analysis; default config; zero calibration.
  Effort.MEDIUM — standard analysis; warmup-only adaptation; ~1 minute.
  Effort.HIGH   — production / repeat runs; full Tier-B BO; ~30+ minutes.

Recipes are emitted by:
  - Recipe.from_default_config(posterior, base_method) — LOW; zero MCMC
  - Recipe.from_warmup_only(...) — MEDIUM; runs warmup, captures adapted params
  - Recipe.from_tuning_result(result) — HIGH; from a TuningResult

This commit (3 of 4) implements the dataclass + LOW path. MEDIUM and HIGH
constructors are stubs (NotImplementedError pointing to the follow-up spawn);
this lets us lock the schema and emit the 6 LOW recipes for the 3 starter
models × 2 algorithms while deferring the compute-heavy generators.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bjx_bench.inference.base_method._base import BaseMethod
    from bjx_bench.model._base import Posterior

__all__ = ["Effort", "Recipe"]


class Effort(str, Enum):
    """Calibration effort axis. Maps to user personas:

    LOW    — one-off analysis; default config; zero calibration time.
    MEDIUM — standard analysis; warmup-only adaptation; ~1 minute.
    HIGH   — production / repeated runs; full Tier-B BO tuning; ~30 min+.
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
        Name of the warmup procedure used: ``"no_warmup"`` for LOW;
        ``"stan_window"`` / ``"mclmc_tuning"`` for MEDIUM/HIGH.
    effort
        Calibration effort level (``Effort.LOW``, ``Effort.MEDIUM``,
        or ``Effort.HIGH``).
    base_method_params
        Pinned hyperparameter dict for the sampler kernel, e.g.
        ``{"step_size": 0.031, "num_integration_steps": 64}``.
    warmup_params
        Pinned hyperparameter dict for the warmup procedure; ``{}`` for LOW.
    headline_metric
        ``min_bulk_ess_per_grad`` at these params.  ``None`` for LOW since
        no MCMC was run; Phase 6 may fill via measurement runs.
    sample_quality
        Optional dict of quality metrics vs. reference draws
        (``{"mae_vs_reference": ..., "q05_error": ...}``).  ``None``
        until Phase 6 wires reference comparison.
    calibration_budget
        Cost summary: ``{"trials": 0, "wall_seconds_estimate": 0.0}`` for LOW;
        filled from actual timing for MEDIUM/HIGH.
    difficulty
        Serialised ``TuningDifficulty.asdict()`` or ``None``.  Only meaningful
        for HIGH recipes.
    instructions
        Auto-templated user-facing prose (rendered by ``_instructions.py``).
    notes
        Optional human override or caveats.
    tuning_seed
        Random seed used during calibration; 0 for LOW (no calibration).
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
    warmup_name: (
        str  # "no_warmup" for LOW; "stan_window" / "mclmc_tuning" for MEDIUM/HIGH
    )
    effort: Effort

    # ---- Pinned config ----
    base_method_params: dict[str, Any]
    warmup_params: dict[str, Any]  # {} for LOW (no warmup)

    # ---- Performance ----
    # headline_metric is None for LOW since no MCMC was run
    headline_metric: float | None
    # sample_quality is None until Phase 6 wires reference comparison
    sample_quality: dict[str, float] | None

    # ---- Calibration cost (the persona filter) ----
    calibration_budget: dict[
        str, Any
    ]  # {"trials": int, "wall_seconds_estimate": float}

    # ---- Difficulty profile (only meaningful for HIGH; None for LOW/MEDIUM) ----
    difficulty: dict[str, Any] | None  # serialized TuningDifficulty.asdict() or None

    # ---- User-facing prose ----
    instructions: str
    notes: str = ""

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
        """Build a LOW-effort Recipe by stamping default_params_for(base_method).

        No MCMC runs. The recipe is the zero-calibration "just use this" config.
        Provenance fields (blackjax_version, jax_version, timestamp_utc) are
        populated at call time.

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
            # headline_metric is None for LOW: no MCMC was run; Phase 6 may fill
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
    def from_warmup_only(cls, *args: Any, **kwargs: Any) -> Recipe:
        """Build a MEDIUM-effort recipe by running warmup adaptation.

        Raises
        ------
        NotImplementedError
            MEDIUM-effort recipes require running warmup; deferred to a
            follow-up spawn after Phase 2.5 commit 3 lands.
        """
        raise NotImplementedError(
            "MEDIUM-effort recipes require running warmup; deferred to a follow-up "
            "spawn after Phase 2.5 commit 3 lands."
        )

    @classmethod
    def from_tuning_result(cls, *args: Any, **kwargs: Any) -> Recipe:
        """Build a HIGH-effort recipe from a Tier-B TuningResult.

        Raises
        ------
        NotImplementedError
            HIGH-effort recipes require running Tier-B BO; deferred to a
            follow-up spawn after Phase 2.5 commit 3 lands.
        """
        raise NotImplementedError(
            "HIGH-effort recipes require running Tier-B BO; deferred to a follow-up "
            "spawn after Phase 2.5 commit 3 lands."
        )


# ── private helpers ───────────────────────────────────────────────────────────


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
