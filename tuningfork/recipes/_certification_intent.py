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
"""Pure declarative builder for certification recipes."""

from __future__ import annotations

import datetime as _datetime
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import blackjax
import jax
import numpy as np

import tuningfork
from tuningfork.base_method import (
    BASE_METHODS,
    default_params_for,
    default_value_for_space,
)
from tuningfork.model import MODELS
from tuningfork.recipes._base import Effort, Recipe, validate_init_strategy
from tuningfork.recipes._instructions import render_instructions
from tuningfork.warmup import WARMUPS

_DYNAMIC = {"dynamic_hmc", "dmhmc"}
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


def _json_value(value: Any) -> Any:
    """Convert array-like values to strict JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, tuple):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("JSON objects require string keys")
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_value(v) for v in value]
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("JSON values must be finite")
        return value
    # JAX arrays expose tolist through the NumPy conversion protocol.
    try:
        return _json_value(np.asarray(value))
    except Exception as exc:
        raise TypeError(
            f"value is not JSON-compatible: {type(value).__name__}"
        ) from exc


def _safe_tag(tag: str) -> str:
    if not isinstance(tag, str) or not tag or not _SAFE_COMPONENT.fullmatch(tag):
        raise ValueError(f"unsafe filename tag component: {tag!r}")
    return tag


@dataclass(frozen=True)
class CertificationIntent:
    recipe: Recipe
    filename_tag: str | None
    recipe_path: Path


def build_certification_intent(
    model_name: str,
    warmup_name: str,
    sampler_name: str,
    *,
    n_warmup: int,
    n_samples: int,
    num_chains: int,
    seed: int,
    catalog_root: str | Path,
    target_acceptance: float | None = None,
    sampler_kwargs_override: dict[str, Any] | None = None,
    warmup_kwargs_override: dict[str, Any] | None = None,
    step_policy: dict[str, Any] | None = None,
    policy_tag: str | None = None,
    variant_label: str | None = None,
    effort: Effort | str = Effort.LOW,
    warmup_inner_kernel: str | None = None,
    init_strategy: dict[str, Any] | None = None,
) -> CertificationIntent:
    """Construct a validated, non-executing certification recipe intent."""
    if model_name not in MODELS:
        raise ValueError(f"unknown model: {model_name!r}")
    if sampler_name not in BASE_METHODS:
        raise ValueError(f"unknown sampler: {sampler_name!r}")
    if warmup_name not in WARMUPS:
        raise ValueError(f"unknown warmup: {warmup_name!r}")
    if not WARMUPS[warmup_name].is_compatible(sampler_name):
        raise ValueError(
            f"warmup {warmup_name!r} is incompatible with {sampler_name!r}"
        )
    if warmup_inner_kernel is not None and warmup_inner_kernel not in BASE_METHODS:
        raise ValueError(f"unknown warmup inner kernel: {warmup_inner_kernel!r}")
    if any(
        isinstance(v, bool) or not isinstance(v, int)
        for v in (n_warmup, n_samples, num_chains, seed)
    ):
        raise ValueError("budgets and seed must be integers")
    if n_warmup < 0 or n_samples <= 0 or num_chains <= 0 or seed < 0:
        raise ValueError(
            "budgets must be nonnegative/positive and seed must be nonnegative"
        )
    if warmup_name == "no_warmup":
        n_warmup = 0
    validate_init_strategy(init_strategy)
    if target_acceptance is not None and (
        isinstance(target_acceptance, bool)
        or not math.isfinite(float(target_acceptance))
        or not 0.0 < float(target_acceptance) < 1.0
    ):
        raise ValueError("target_acceptance must satisfy 0 < target_acceptance < 1")
    if warmup_inner_kernel is not None and not warmup_name.startswith(
        "window_adaptation_"
    ):
        raise ValueError("warmup_inner_kernel requires a window-adaptation warmup")
    if step_policy is not None and sampler_name not in _DYNAMIC:
        raise ValueError("step_policy is only valid for dynamic_hmc and dmhmc")
    try:
        effort = Effort(effort)
    except ValueError as exc:
        raise ValueError(f"invalid effort: {effort!r}") from exc
    if effort in (Effort.GROUNDTRUTH, Effort.FAILED):
        raise ValueError(
            "certification intents cannot use groundtruth or failed effort"
        )

    sampler = BASE_METHODS[sampler_name]
    warmup = WARMUPS[warmup_name]
    base_params = default_params_for(sampler)
    if sampler_kwargs_override:
        base_params.update(sampler_kwargs_override)
    base_params = _json_value(base_params)

    target = target_acceptance
    if target is None:
        target = sampler.target_acceptance_rate or 0.8
    warmup_params: dict[str, Any] = {
        "n_warmup": n_warmup,
        "num_chains": num_chains,
        "target_acceptance": target,
    }
    for space in warmup.default_hp_space:
        warmup_params.setdefault(space.name, default_value_for_space(space))
    if warmup_kwargs_override:
        warmup_params.update(warmup_kwargs_override)
    warmup_params = _json_value(warmup_params)

    effective_policy = step_policy
    if (
        sampler_name in _DYNAMIC
        and warmup_inner_kernel is not None
        and step_policy is None
    ):
        effective_policy = {"kind": "warmup_empirical"}
    effective_policy = (
        _json_value(effective_policy) if effective_policy is not None else None
    )

    implicit_inner = (
        "nuts"
        if sampler_name in {"dynamic_hmc", "dmhmc"}
        or sampler_name.startswith("laplace_")
        else sampler_name
    )
    tags: list[str] = []
    if warmup_inner_kernel is not None and warmup_inner_kernel != implicit_inner:
        tags.append(_safe_tag(f"inner_{warmup_inner_kernel}"))
    if policy_tag == "":
        raise ValueError("policy_tag must be non-empty when supplied")
    if policy_tag:
        tags.append(_safe_tag(policy_tag))
    if variant_label is not None:
        variant_label = _safe_tag(variant_label)
    filename_tag = "__".join(tags) if tags else None

    budget = {
        "trials": 0,
        "n_warmup": n_warmup,
        "n_samples": n_samples,
        "num_chains": num_chains,
    }
    recipe = Recipe(
        model_name=model_name,
        base_method_name=sampler_name,
        warmup_name=warmup_name,
        effort=effort,
        base_method_params=base_params,
        warmup_params=warmup_params,
        headline_metric=None,
        sample_quality=None,
        calibration_budget=budget,
        difficulty=None,
        instructions="",
        step_policy=effective_policy,
        warmups=[{"name": warmup_name, "params": warmup_params}],
        warmup_inner_kernel=warmup_inner_kernel,
        warmup_num_chains=[num_chains],
        init_strategy=init_strategy,
        variant_label=variant_label,
        tuning_seed=seed,
        timestamp_utc=_datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        tuningfork_version=tuningfork.__version__,
        blackjax_version=blackjax.__version__,
        jax_version=jax.__version__,
    )
    recipe = replace(recipe, instructions=render_instructions(recipe))
    root = Path(catalog_root)
    path = (
        root
        / model_name
        / "recipes"
        / f"{recipe.catalog_stem(filename_tag=filename_tag)}.json"
    )
    return CertificationIntent(recipe, filename_tag, path)


__all__ = ["CertificationIntent", "build_certification_intent"]
