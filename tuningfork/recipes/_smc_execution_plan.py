"""Resolution of an :class:`SMCRecipe` into an executable plan.

SMC particles are not chains or samples.  They therefore have a deliberately
small family-specific configuration while retaining the generic plan envelope
used by manifests and launchers.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._base_smc import SMCRecipe
from ._execution_plan import (
    ExecutionPlan,
    _freeze,
    _thaw,
    canonical_json,
    execution_config_hash,
)


@dataclass(frozen=True)
class SMCExecutionConfiguration:
    """Immutable, JSON-safe executable configuration for one SMC recipe."""

    execution_family: str
    model_name: str
    smc_method_name: str
    inner_method_name: str
    smc_params: Mapping[str, Any]
    inner_params_init: Mapping[str, Any] | None
    parameter_update_strategy: str
    parameter_update_strategy_kwargs: Mapping[str, Any]
    num_particles: int
    max_steps: int
    seed: int
    requires_x64: bool

    def __post_init__(self) -> None:
        if self.execution_family != "smc":
            raise ValueError("execution_family must be 'smc'")
        for name in (
            "model_name",
            "smc_method_name",
            "inner_method_name",
            "parameter_update_strategy",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise TypeError(f"{name} must be a non-empty string")
        for name in ("num_particles", "max_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        for name in ("smc_params", "parameter_update_strategy_kwargs"):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"{name} must be a mapping")
        if self.inner_params_init is not None and not isinstance(
            self.inner_params_init, Mapping
        ):
            raise TypeError("inner_params_init must be a mapping or None")
        if not isinstance(self.requires_x64, bool):
            raise TypeError("requires_x64 must be a boolean")
        # Validate all nested values, including non-finite floats and exotic
        # objects, at the plan boundary rather than during source rendering.
        canonical_json(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_family": self.execution_family,
            "model_name": self.model_name,
            "smc_method_name": self.smc_method_name,
            "inner_method_name": self.inner_method_name,
            "smc_params": _thaw(self.smc_params),
            "inner_params_init": _thaw(self.inner_params_init),
            "parameter_update_strategy": self.parameter_update_strategy,
            "parameter_update_strategy_kwargs": _thaw(
                self.parameter_update_strategy_kwargs
            ),
            "num_particles": self.num_particles,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "requires_x64": self.requires_x64,
        }

    @property
    def config_hash(self) -> str:
        return execution_config_hash(self.as_dict())


def resolve_smc_execution_plan(
    recipe: Any,
) -> ExecutionPlan[SMCExecutionConfiguration]:
    """Resolve and validate an ``SMCRecipe`` before code generation."""
    from tuningfork.base_method import BASE_METHODS
    from tuningfork.calibration.tune import default_value_for_space
    from tuningfork.model import MODELS
    from tuningfork.smc import SMC_METHODS
    from tuningfork.smc.parameter_update_registry import PARAMETER_UPDATE_STRATEGIES

    if not isinstance(recipe, SMCRecipe):
        raise TypeError("recipe must be an SMCRecipe")
    try:
        smc_name = recipe.smc_method_name
        smc_entry = SMC_METHODS[smc_name]
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f"unsupported SMC method: {getattr(recipe, 'smc_method_name', None)!r}"
        ) from exc
    try:
        inner_name = recipe.inner_method_name
        if inner_name is None or inner_name == "":
            inner_name = smc_entry.default_inner_method
        inner_entry = BASE_METHODS[inner_name]
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f"unsupported inner method: {getattr(recipe, 'inner_method_name', None)!r}"
        ) from exc
    if inner_name not in smc_entry.compatible_inner_methods:
        raise ValueError(
            f"inner method {inner_name!r} is incompatible with SMC method {smc_name!r}"
        )
    if getattr(inner_entry, "family", None) != "mcmc":
        raise ValueError(
            f"inner method {inner_name!r} is not an executable MCMC method"
        )
    try:
        model = MODELS[recipe.model_name]
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f"unsupported model: {getattr(recipe, 'model_name', None)!r}"
        ) from exc
    smc_params = {
        space.name: default_value_for_space(space)
        for space in smc_entry.default_hp_space
    }
    smc_params.update(copy.deepcopy(recipe.smc_params))
    inner_params = copy.deepcopy(recipe.inner_params_init or {})
    if inner_name == "rwm":
        sigma_space = next(
            space for space in inner_entry.default_hp_space if space.name == "sigma"
        )
        inner_params.setdefault("sigma", default_value_for_space(sigma_space))
    elif inner_name == "hmc":
        inner_params.setdefault("step_size", 0.1)
        inner_params.setdefault("inverse_mass_matrix", [1.0] * model.dim)
        smc_params.setdefault("num_integration_steps", 10)
    strategy_name = recipe.parameter_update_strategy
    strategy = PARAMETER_UPDATE_STRATEGIES.get(strategy_name)
    if strategy is None:
        raise ValueError(f"unsupported parameter update strategy: {strategy_name!r}")
    strategy_kwargs = copy.deepcopy(recipe.parameter_update_strategy_kwargs)
    if strategy.allowed_kwargs == frozenset({"target_acceptance"}):
        strategy_kwargs.setdefault("target_acceptance", 0.65)
    unknown_kwargs = set(strategy_kwargs) - strategy.allowed_kwargs
    if unknown_kwargs:
        raise ValueError(
            f"parameter update strategy {strategy_name!r} does not allow kwargs: "
            f"{sorted(unknown_kwargs)!r}"
        )
    if "target_acceptance" in strategy_kwargs:
        target = strategy_kwargs["target_acceptance"]
        if (
            isinstance(target, bool)
            or not isinstance(target, (int, float))
            or not math.isfinite(float(target))
            or not 0.0 < float(target) < 1.0
        ):
            raise ValueError(
                "target_acceptance must be finite and satisfy 0 < target_acceptance < 1"
            )
    if smc_name != "inner_kernel_tuning" and strategy_name != "none":
        raise ValueError(
            "non-'none' parameter update strategies require inner_kernel_tuning"
        )
    config = SMCExecutionConfiguration(
        execution_family="smc",
        model_name=recipe.model_name,
        smc_method_name=smc_name,
        inner_method_name=inner_name,
        smc_params=_freeze(smc_params),
        inner_params_init=_freeze(inner_params),
        parameter_update_strategy=strategy_name,
        parameter_update_strategy_kwargs=_freeze(strategy_kwargs),
        num_particles=recipe.num_particles,
        max_steps=recipe.max_steps,
        seed=recipe.seed,
        requires_x64=bool(model.requires_x64),
    )
    filename = f"{recipe.model_name}__smc__{smc_name}__{inner_name}.draws.npz"
    ref = f"{recipe.model_name}/smc__{smc_name}__{inner_name}"
    return ExecutionPlan.build(config, ref, filename)


build_smc_execution_plan = resolve_smc_execution_plan
resolve_smc_plan = resolve_smc_execution_plan

__all__ = [
    "SMCExecutionConfiguration",
    "resolve_smc_execution_plan",
    "build_smc_execution_plan",
    "resolve_smc_plan",
]
