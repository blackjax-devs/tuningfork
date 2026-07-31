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
"""SMCRecipe dataclass — sister to Recipe for Sequential Monte Carlo results.

SMC algorithms have a fundamentally different execution profile from MCMC:
  - No warmup phase (particle initialisation replaces warmup).
  - No step_size / inverse_mass_matrix adapted by a warmup runner.
  - Particles (not chains) are the unit of parallelism.
  - Gate metrics differ: particle-ESS, max_abs_mean_z with SE=std/√particle_ess,
    mode_coverage for gmm_25.  rhat, bulk-ESS, n_divergences are n/a.
  - Headline metric = particle_ess / total_grad_evals (HMC) or
    / total_likelihood_evals (RWM).

Using a separate dataclass (NOT extending Recipe) avoids polluting the MCMC
Recipe schema with 6+ inapplicable fields and the branching that would follow
in save(), load(), from_default_config(), emit_script(), and every Recipe
consumer.  SMCRecipe and Recipe share only the ``headline_metric`` field for
cross-algorithm leaderboard comparisons.

Catalog layout:  ``catalog/<model>/recipes/smc__<method>__<inner>.json``
"""

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["SMCRecipe"]


def _now_utc_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_tuningfork_version() -> str:
    try:
        from tuningfork._version import __version__

        return __version__
    except ImportError:
        return "0.0.0.dev0"


def _get_blackjax_version() -> str:
    try:
        import blackjax

        return getattr(blackjax, "__version__", "unknown")
    except ImportError:
        return "unknown"


def _get_jax_version() -> str:
    try:
        import jax

        return jax.__version__
    except ImportError:
        return "unknown"


@dataclass(frozen=True)
class SMCRecipe:
    """A pinned SMC configuration + cert evidence.

    Stored as JSON at
    ``catalog/<model>/recipes/smc__<smc_method>__<inner_method>.json``.

    Parameters
    ----------
    model_name
        Model registry key (e.g. ``"logistic_synthetic"``).
    smc_method_name
        SMC_METHODS key (e.g. ``"inner_kernel_tuning"``).
    inner_method_name
        BASE_METHODS key for the inner MCMC kernel (e.g. ``"hmc"``).
    num_particles
        Number of SMC particles.
    max_steps
        Maximum number of SMC tempering steps (for adaptive variants the
        while-loop exits earlier when λ=1 is reached).
    smc_params
        SMC-level hyperparameters (target_ess, num_mcmc_steps, ...).
    inner_params_init
        Initial inner-kernel parameters before tempering starts
        (step_size, inverse_mass_matrix).  ``None`` if no per-step tuning.
    inner_params_final
        Inner-kernel parameters at λ=1 after the run.  Populated only
        on cert PASS; ``None`` before the run or on FAIL.
    parameter_update_strategy
        W6 registry key for the ``mcmc_parameter_update_fn`` (e.g.
        ``"step_size_and_imm_from_particles"``).  ``"none"`` for no tuning.
    parameter_update_strategy_kwargs
        Declarative kwargs materialized by codegen (for example,
        ``{"target_acceptance": 0.65}``).
    headline_metric
        ``particle_ess / total_grad_evals`` (HMC) or
        ``/ total_likelihood_evals`` (RWM).  ``None`` before cert.
    gate_evidence
        Cert gate results dict.  Keys:
        ``"auto"`` (verdict, max_abs_mean_z, particle_ess,
        particle_ess_fraction, mode_coverage_fraction),
        ``"override"`` (for statistician sign-off).
    calibration_budget
        Wall-time breakdown and particle/step counts.
    notes
        Human-readable rationale (e.g. for FAILED honest-null cells).
    tuningfork_version, blackjax_version, jax_version, timestamp_utc
        Provenance metadata.
    """

    # ---- identity ----
    model_name: str
    smc_method_name: str
    inner_method_name: str

    # ---- SMC configuration ----
    num_particles: int
    max_steps: int
    seed: int = 20260517
    smc_params: dict[str, Any] = field(default_factory=dict)

    # ---- inner-kernel tuning (W6) ----
    inner_params_init: dict[str, Any] | None = None
    inner_params_final: dict[str, Any] | None = None
    parameter_update_strategy: str = "none"
    parameter_update_strategy_kwargs: dict[str, Any] = field(default_factory=dict)

    # ---- cert results ----
    headline_metric: float | None = None
    gate_evidence: dict[str, Any] = field(
        default_factory=lambda: {
            "auto": {
                "verdict": "NOT_RUN",
                "max_abs_mean_z": None,
                "particle_ess": None,
                "particle_ess_fraction": None,
                "mode_coverage_fraction": None,
            },
            "override": {"reason": "", "statistician_id": "", "decision": ""},
        }
    )
    calibration_budget: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    workflow: str = ""
    failure_diagnosis: str | None = None
    attempted_configurations: list[Any] = field(default_factory=list)

    # ---- provenance ----
    tuningfork_version: str = field(default_factory=_get_tuningfork_version)
    blackjax_version: str = field(default_factory=_get_blackjax_version)
    jax_version: str = field(default_factory=_get_jax_version)
    timestamp_utc: str = field(default_factory=_now_utc_iso)

    # Unknown top-level annotations are carried privately for lossless
    # load/save, while remaining outside the ordinary persisted schema.
    _extra_fields: dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Reject malformed plans before they can reach generated execution."""
        for name in ("model_name", "smc_method_name", "inner_method_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TypeError(f"{name} must be a non-empty string")
        for name in ("num_particles", "max_steps", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if name == "seed" and value < 0:
                raise ValueError("seed must be non-negative")
            if name != "seed" and value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "smc_params",
            "parameter_update_strategy_kwargs",
            "gate_evidence",
            "calibration_budget",
        ):
            if not isinstance(getattr(self, name), dict):
                raise TypeError(f"{name} must be a mapping")
        for name in ("inner_params_init", "inner_params_final"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, dict):
                raise TypeError(f"{name} must be a mapping or None")
        for name in (
            "parameter_update_strategy",
            "notes",
            "workflow",
            "tuningfork_version",
            "blackjax_version",
            "jax_version",
            "timestamp_utc",
        ):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
        if self.headline_metric is not None:
            if isinstance(self.headline_metric, bool) or not isinstance(
                self.headline_metric, (int, float)
            ):
                raise TypeError("headline_metric must be a number or None")
            if isinstance(self.headline_metric, float) and not math.isfinite(
                self.headline_metric
            ):
                raise ValueError("headline_metric must be finite")
        if not isinstance(self._extra_fields, dict):
            raise TypeError("_extra_fields must be a mapping")
        if self.failure_diagnosis is not None and not isinstance(
            self.failure_diagnosis, str
        ):
            raise TypeError("failure_diagnosis must be a string or None")
        if not isinstance(self.attempted_configurations, list):
            raise TypeError("attempted_configurations must be a list")

    # ---- derived ----
    @property
    def verdict(self) -> str:
        """Shortcut: gate_evidence['auto']['verdict']."""
        return self.gate_evidence.get("auto", {}).get("verdict", "NOT_RUN")

    def save(self, root: Path, *, filename_tag: str | None = None) -> Path:
        """Write the recipe to its canonical catalog location.

        Canonical path:
            ``<root>/<model_name>/recipes/smc__<smc_method>__<inner_method>.json``
        or, when ``filename_tag`` is supplied:
            ``<root>/<model_name>/recipes/smc__<smc_method>__<inner_method>__<tag>.json``

        Parameters
        ----------
        root
            Catalog root directory.
        filename_tag
            Optional extra tag (e.g. ``"n2000"``).

        Returns
        -------
        Path
            Absolute path of the written JSON file.
        """
        target_dir = Path(root) / self.model_name / "recipes"
        target_dir.mkdir(parents=True, exist_ok=True)

        stem = f"smc__{self.smc_method_name}__{self.inner_method_name}"
        if filename_tag:
            stem = f"{stem}__{filename_tag}"
        target = target_dir / f"{stem}.json"

        d = self.to_dict()
        # Serialize completely before touching the destination.  In particular,
        # ``allow_nan=False`` prevents Python's non-standard NaN/Infinity output.
        payload = json.dumps(d, indent=2, allow_nan=False) + "\n"
        fd, temporary = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        return target

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-safe dict."""
        import dataclasses

        d = dataclasses.asdict(self)
        extras = d.pop("_extra_fields", {})
        # Refuse to overwrite a canonical field if an extension is malformed.
        for key, value in extras.items():
            if key in d:
                raise ValueError(
                    f"Cannot serialize extension field {key!r}: "
                    "it collides with a canonical SMCRecipe field"
                )
            d[key] = value
        return d

    # Kept for callers that used the former private helper.
    def _to_dict(self) -> dict[str, Any]:
        return self.to_dict()

    @classmethod
    def load(cls, path: Path) -> "SMCRecipe":
        """Load an SMCRecipe from a JSON file written by ``save``.

        Tolerates older recipes missing the ``seed`` field (added 2026-06-03).
        """
        import dataclasses

        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict):
            raise TypeError("SMCRecipe JSON must contain an object")
        known = {f.name for f in dataclasses.fields(cls) if not f.name.startswith("_")}
        extras = {k: v for k, v in raw.items() if k not in known}
        values = {k: v for k, v in raw.items() if k in known}
        values["_extra_fields"] = extras
        return cls(**values)

    @classmethod
    def from_default_config(
        cls,
        model_name: str,
        smc_method_name: str,
        inner_method_name: str,
        *,
        num_particles: int = 1000,
        max_steps: int = 100,
        seed: int = 20260517,
        smc_params: dict[str, Any] | None = None,
        inner_params_init: dict[str, Any] | None = None,
        parameter_update_strategy: str = "none",
        parameter_update_strategy_kwargs: dict[str, Any] | None = None,
    ) -> "SMCRecipe":
        """Build an uncertified SMCRecipe with default/supplied configuration.

        The returned recipe has ``gate_evidence.auto.verdict = "NOT_RUN"``
        and ``headline_metric = None``; these are populated by
        ``emit_smc_recipe_for_cell`` after the actual run.

        Parameters
        ----------
        model_name
            Model registry key.
        smc_method_name
            SMC_METHODS key.
        inner_method_name
            BASE_METHODS key for the inner kernel.
        num_particles
            Number of SMC particles.
        max_steps
            Maximum tempering steps.
        smc_params
            SMC-level HPs (target_ess, num_mcmc_steps, ...).  Defaults to
            ``{"target_ess": 0.5, "num_mcmc_steps": 10}``.
        inner_params_init
            Initial inner-kernel params before tempering.  ``None`` when
            ``parameter_update_strategy == "none"``.
        parameter_update_strategy
            W6 registry key for the update function.
        parameter_update_strategy_kwargs
            Extra declarative kwargs for the generated update strategy.
        """
        from tuningfork.smc import SMC_METHODS  # inline to avoid circular dep

        smc_entry = SMC_METHODS[smc_method_name]

        # Default SMC-level HPs from the method's HP space.
        from tuningfork.calibration.tune import default_value_for_space

        default_smc_params: dict[str, Any] = {
            space.name: default_value_for_space(space)
            for space in smc_entry.default_hp_space
        }
        if smc_params:
            default_smc_params.update(smc_params)

        return cls(
            model_name=model_name,
            smc_method_name=smc_method_name,
            inner_method_name=inner_method_name,
            num_particles=num_particles,
            max_steps=max_steps,
            seed=seed,
            smc_params=default_smc_params,
            inner_params_init=inner_params_init,
            parameter_update_strategy=parameter_update_strategy,
            parameter_update_strategy_kwargs=parameter_update_strategy_kwargs or {},
        )
