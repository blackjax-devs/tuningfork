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


# Canonical recipe-runner protocol: 4 chains × 1000 draws for quick mode.
_DEFAULT_NUM_CHAINS = 4


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


def _non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}")
    return value


def _seed(name: str, value: Any) -> int:
    return _non_negative_int(name, value)


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
    # Legacy baked recipes blanked warmup_name/warmups.  Normalize once at the
    # plan boundary so stage counts, manifests, and emitted source agree.
    budget = getattr(recipe, "calibration_budget", {}) or {}
    if (
        getattr(recipe, "warmup_name", None) == ""
        and isinstance(budget, Mapping)
        and isinstance(budget.get("baked_from"), Mapping)
        and hasattr(recipe, "normalize_pinned_replay")
    ):
        recipe = recipe.normalize_pinned_replay()
    ov = overrides or ExecutionOverrides()
    if not isinstance(ov, ExecutionOverrides):
        raise TypeError("overrides must be an ExecutionOverrides instance")
    stages = _stages(recipe)
    nphases = len(stages)
    budget = getattr(recipe, "calibration_budget", {}) or {}
    wp = getattr(recipe, "warmup_params", {}) or {}
    effort = getattr(recipe, "effort", None)
    legacy_chain_default = (
        1 if getattr(effort, "value", effort) == "groundtruth" else _DEFAULT_NUM_CHAINS
    )
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
            else wp.get("num_chains", budget.get("num_chains", legacy_chain_default))
        ),
    )
    tuning_seed = _seed(
        "tuning_seed",
        (
            ov.tuning_seed
            if ov.tuning_seed is not None
            else getattr(recipe, "tuning_seed", 0)
        ),
    )
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
                f"num_warmup list length {len(raw_warmup)} does not match "
                f"the number of warmup phases ({nphases})"
            )
        counts = list(raw_warmup)
    else:
        raise ValueError("num_warmup must be an int, list[int], tuple[int], or None")
    # ``no_warmup`` is a true zero-step execution.  Normalize both omitted and
    # caller-supplied counts so the plan cannot accidentally render a warmup.
    normalized_counts: list[int] = []
    for i, (stage, count) in enumerate(zip(stages, counts)):
        if stage["name"] == "no_warmup":
            if raw_warmup is not None:
                _non_negative_int(f"num_warmup[{i}]", count)
            normalized_counts.append(0)
        else:
            normalized_counts.append(_positive_int(f"num_warmup[{i}]", count))
    counts = normalized_counts
    raw_w = (
        ov.warmup_num_chains
        if ov.warmup_num_chains is not None
        else getattr(recipe, "warmup_num_chains", None)
    )
    if raw_w is None:
        # The schema's omitted topology means one warmup per sampling chain.
        ws = [chains] * nphases
    elif isinstance(raw_w, (list, tuple)):
        ws = list(raw_w)
    else:
        raise ValueError("warmup_num_chains must be a list[int], tuple[int], or None")
    if len(ws) != nphases:
        raise ValueError(f"warmup_num_chains has {len(ws)} entries; expected {nphases}")
    ws = [_positive_int(f"warmup_num_chains[{i}]", w) for i, w in enumerate(ws)]
    # No-warmup has no warmup topology; canonicalize its stage count so W does
    # not affect the executable plan or emitter dispatch.
    ws = [chains if stage["name"] == "no_warmup" else w for stage, w in zip(stages, ws)]

    # Emitters currently implement only a small set of warmup chain topologies.
    # Reject the rest before any source rendering, rather than silently choosing
    # the single-chain path.
    is_laplace = recipe.base_method_name.startswith("laplace_")
    window_names = {
        "window_adaptation_diag_imm",
        "window_adaptation_dense_imm",
        "window_adaptation_low_rank_imm",
    }
    if nphases > 1:
        phase_names = tuple(stage["name"] for stage in stages)
        expected_laplace_phases = (
            "window_adaptation_diag_imm",
            "window_adaptation_dense_imm",
        )
        if not is_laplace or phase_names != expected_laplace_phases:
            raise NotImplementedError(
                "multi-phase code generation currently requires a Laplace recipe "
                "with exactly diagonal then dense window-adaptation phases; got "
                f"{phase_names!r}"
            )
        for i, (stage, w) in enumerate(zip(stages, ws)):
            if w != 1:
                raise NotImplementedError(
                    f"warmup chain topology W={w}, S={chains} for stage "
                    f"{i} ({stage['name']!r}) is not supported by code generation"
                )
    elif stages[0]["name"] != "no_warmup":
        w = ws[0]
        supported = (
            w in {1, chains} if stages[0]["name"] in window_names else w == chains
        )
        if not supported:
            raise NotImplementedError(
                f"warmup chain topology W={w}, S={chains} for stage "
                f"0 ({stages[0]['name']!r}) is not supported by code generation"
            )

    init_strategy = getattr(recipe, "init_strategy", None)
    init_kind = (
        init_strategy.get("type") if isinstance(init_strategy, Mapping) else None
    )
    if init_kind in {"uniform_perchain", "zero_perchain", "reference_summary"}:
        single_phase = nphases == 1 and ws[0] == chains
        # Adaptation can consume per-chain initial positions directly.  A
        # normalized pinned replay has no adaptation stage, but still needs
        # the same pre-batched (one row per sampling chain) contract.
        valid_topology = single_phase and (
            stages[0]["name"] in window_names
            or (init_kind == "reference_summary" and stages[0]["name"] == "no_warmup")
        )
        if not valid_topology:
            raise ValueError(
                f"init_strategy type={init_kind!r} requires a single-phase "
                "window-adaptation or no-warmup topology with W=S; got "
                f"stages={tuple(stage['name'] for stage in stages)!r}, "
                f"W={tuple(ws)!r}, S={chains}"
            )

    step_policy = getattr(recipe, "step_policy", None)
    if step_policy is not None and recipe.base_method_name not in {
        "dynamic_hmc",
        "dmhmc",
    }:
        raise ValueError(
            "step_policy is only executable for dynamic_hmc and dmhmc recipes; "
            f"got {recipe.base_method_name!r}"
        )
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
        init_strategy=_freeze(copy.deepcopy(init_strategy)),
        step_policy=_freeze(copy.deepcopy(step_policy)),
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
