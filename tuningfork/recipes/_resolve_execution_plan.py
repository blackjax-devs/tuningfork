"""Pure resolution of Recipe intent into a normalized execution plan."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ._execution_plan import (
    ExecutableConfigurationSnapshot,
    ExecutionOverrides,
    ExecutionPlan,
    WarmupStagePlan,
    _freeze,
)

if TYPE_CHECKING:
    from ._base import Recipe


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


def _seed(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}")
    return value


def _stages(recipe: Recipe) -> list[dict[str, Any]]:
    stages = getattr(recipe, "warmups", None) or []
    # A one-entry ``warmups`` list is the schema's compatibility representation
    # of the legacy flat fields.  The emitter still reads those flat fields, so
    # they remain authoritative until an explicitly ordered multi-phase list is
    # supplied.
    if len(stages) <= 1:
        stages = [
            {
                "name": recipe.warmup_name,
                "params": getattr(recipe, "warmup_params", {}) or {},
            }
        ]
    out = []
    for i, stage in enumerate(stages):
        if not isinstance(stage, Mapping) or not isinstance(stage.get("name"), str):
            raise ValueError(
                f"warmups[{i}] must contain a string name and params mapping"
            )
        params = stage.get("params", {})
        if not isinstance(params, Mapping):
            raise ValueError(f"warmups[{i}].params must be a mapping")
        out.append({"name": stage["name"], "params": copy.deepcopy(dict(params))})
    return out


def resolve_execution_plan(
    recipe: Recipe, overrides: ExecutionOverrides | None = None
) -> ExecutionPlan:
    """Resolve defaults and overrides, rejecting ambiguous executable values."""
    ov = overrides or ExecutionOverrides()
    if not isinstance(ov, ExecutionOverrides):
        raise TypeError("overrides must be an ExecutionOverrides instance")
    stages = _stages(recipe)
    nphases = len(stages)
    budget = getattr(recipe, "calibration_budget", {}) or {}
    wp = getattr(recipe, "warmup_params", {}) or {}
    samples = _positive_int(
        "num_samples",
        (
            ov.num_samples
            if ov.num_samples is not None
            else budget.get("n_samples") or 1000
        ),
    )
    chains = _positive_int(
        "num_chains",
        (
            ov.num_chains
            if ov.num_chains is not None
            else wp.get("num_chains", budget.get("num_chains", 1))
        ),
    )
    tuning_seed = _seed("tuning_seed", getattr(recipe, "tuning_seed", 0))
    seed = ov.sampler_seed if ov.sampler_seed is not None else tuning_seed + 1
    _seed("sampler_seed", seed)
    reinit_seed = ov.reinit_seed if ov.reinit_seed is not None else tuning_seed + 999
    _seed("reinit_seed", reinit_seed)
    progress = False if ov.progress_bar is None else ov.progress_bar
    if not isinstance(progress, bool):
        raise ValueError("progress_bar must be a bool or None")
    raw_warmup = ov.num_warmup
    if raw_warmup is None:
        counts = [s["params"].get("n_warmup", 1000) for s in stages]
    elif isinstance(raw_warmup, int) and not isinstance(raw_warmup, bool):
        counts = (
            [raw_warmup]
            if nphases == 1
            else (_raise("num_warmup int is only valid for single-phase recipes"))
        )
    elif isinstance(raw_warmup, (list, tuple)):
        if len(raw_warmup) != nphases:
            raise ValueError(
                f"num_warmup has {len(raw_warmup)} entries; expected {nphases}"
            )
        counts = list(raw_warmup)
    else:
        raise ValueError("num_warmup must be an int, list[int], tuple[int], or None")
    counts = [_positive_int(f"num_warmup[{i}]", c) for i, c in enumerate(counts)]
    raw_w = (
        ov.warmup_num_chains
        if ov.warmup_num_chains is not None
        else getattr(recipe, "warmup_num_chains", None)
    )
    if raw_w is None:
        ws = [chains] * nphases
    elif isinstance(raw_w, (list, tuple)):
        ws = list(raw_w)
    else:
        raise ValueError("warmup_num_chains must be a list[int], tuple[int], or None")
    if len(ws) != nphases:
        raise ValueError(f"warmup_num_chains has {len(ws)} entries; expected {nphases}")
    ws = [_positive_int(f"warmup_num_chains[{i}]", w) for i, w in enumerate(ws)]
    plans = tuple(
        WarmupStagePlan(s["name"], s["params"], counts[i], ws[i])
        for i, s in enumerate(stages)
    )
    try:
        from tuningfork.model import MODELS

        requires_x64 = bool(MODELS[recipe.model_name].requires_x64)
    except (ImportError, KeyError, AttributeError) as exc:
        raise ValueError(
            f"cannot resolve model precision for {recipe.model_name!r}"
        ) from exc
    config = ExecutableConfigurationSnapshot(
        model_name=recipe.model_name,
        base_method_name=recipe.base_method_name,
        warmup_name=recipe.warmup_name,
        base_method_params=_freeze(copy.deepcopy(dict(recipe.base_method_params))),
        warmup_params=_freeze(copy.deepcopy(dict(wp))),
        warmup_stages=tuple(
            WarmupStagePlan(s.name, _freeze(s.params), s.num_warmup, s.num_chains)
            for s in plans
        ),
        warmup_inner_kernel=copy.deepcopy(getattr(recipe, "warmup_inner_kernel", None)),
        init_strategy=_freeze(copy.deepcopy(getattr(recipe, "init_strategy", None))),
        step_policy=_freeze(copy.deepcopy(getattr(recipe, "step_policy", None))),
        tuning_seed=tuning_seed,
        sampler_seed=seed,
        reinit_seed=reinit_seed,
        num_samples=samples,
        num_chains=chains,
        progress_bar=progress,
        requires_x64=requires_x64,
    )
    variant = getattr(recipe, "variant_label", None) or recipe.base_method_name
    effort = getattr(
        getattr(recipe, "effort", None), "value", getattr(recipe, "effort", "recipe")
    )
    filename = f"{recipe.model_name}__{recipe.base_method_name}__{recipe.warmup_name}.draws.npz"
    ref = f"{recipe.model_name}/{effort}__{variant}__{recipe.warmup_name}"
    return ExecutionPlan.build(config, ref, filename)


def _raise(message: str):
    raise ValueError(message)


__all__ = ["resolve_execution_plan"]
