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

"""Certification of already-generated draws (without launching sampling)."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from tuningfork.base_method import BASE_METHODS
from tuningfork.calibration.statistician_gate import auto_gate
from tuningfork.metrics.headline import build_headline_basis
from tuningfork.metrics.reference_compare import compute_sample_quality
from tuningfork.model import MODELS
from tuningfork.recipes._base import Recipe
from tuningfork.recipes._execution_telemetry import ExecutionTelemetry
from tuningfork.recipes._generated_evaluator import (
    GeneratedRunData,
    chain0_geometry,
    sampling_grad_evals,
)
from tuningfork.recipes._ground_truth_reference import (
    GroundTruthReference,
    align_ground_truth,
)
from tuningfork.recipes._warmup_protocol import LAPLACE_METHOD_NAMES


@dataclass(frozen=True)
class GeneratedEvaluation:
    gate_evidence: dict[str, Any]
    headline_metric: float
    headline_basis: dict[str, Any]
    sample_quality: dict[str, float]
    sampling_grad_evals: int
    warmup_grad_evals: int | None
    pinned_base_method_params: dict[str, Any]
    resolved_step_policy: Mapping[str, Any] | None
    metrics: dict[str, Any]
    gt_cert_coverage: str


def _json_safe(value: Any, *, preserve_namedtuple: bool = False) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value in generated certification")
        return value
    if isinstance(value, np.ndarray):
        return [_json_safe(x) for x in value.tolist()]
    if hasattr(value, "_fields"):
        if preserve_namedtuple:
            return type(value)(
                *(
                    _json_safe(getattr(value, field), preserve_namedtuple=True)
                    for field in value._fields
                )
            )
        return [_json_safe(getattr(value, f)) for f in value._fields]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON mappings must have string keys")
        return {
            key: _json_safe(item, preserve_namedtuple=preserve_namedtuple)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, preserve_namedtuple=preserve_namedtuple) for v in value]
    try:
        return _json_safe(np.asarray(value))
    except Exception as exc:
        raise TypeError(f"cannot JSON-normalize {type(value).__name__}") from exc


def evaluate_generated_run(
    recipe: Recipe,
    run_data: GeneratedRunData,
    telemetry: ExecutionTelemetry,
    reference: GroundTruthReference,
    *,
    n_chunks: int,
    allowed_sites: tuple[str, ...] | None = None,
) -> GeneratedEvaluation:
    """Evaluate generated output using typed evidence and no sampler execution."""
    if not isinstance(recipe, Recipe):
        raise TypeError("recipe must be a Recipe")
    if not isinstance(run_data, GeneratedRunData):
        raise TypeError("run_data must be GeneratedRunData")
    if not isinstance(telemetry, ExecutionTelemetry):
        raise TypeError("telemetry must be ExecutionTelemetry")
    if not isinstance(reference, GroundTruthReference):
        raise TypeError("reference must be GroundTruthReference")
    if recipe.model_name != reference.model_name:
        raise ValueError("recipe model does not match ground-truth reference")
    if recipe.base_method_name != run_data.base_method_name:
        raise ValueError("recipe method does not match generated run method")
    if recipe.model_name not in MODELS or recipe.base_method_name not in BASE_METHODS:
        raise ValueError("recipe model or method is not registered")
    if not isinstance(n_chunks, int) or n_chunks <= 0:
        raise ValueError("n_chunks must be positive")

    is_laplace = recipe.base_method_name in LAPLACE_METHOD_NAMES
    sites = allowed_sites if is_laplace else None
    aligned = align_ground_truth(reference, run_data.positions, allowed_sites=sites)
    if is_laplace and allowed_sites is None:
        raise ValueError("Laplace certification requires allowed_sites")
    coverage = (
        "phi_subset_only (theta marginals not gate-verified)"
        if is_laplace
        else "full_posterior"
    )

    geometry = chain0_geometry(telemetry)
    params = _json_safe(copy.deepcopy(recipe.base_method_params))
    method = BASE_METHODS[recipe.base_method_name]
    if geometry.geometry is None:
        if method.needs_mass_matrix:
            raise ValueError("required sampler geometry is unavailable")
    else:
        for key, value in geometry.geometry.items():
            params[key] = _json_safe(value, preserve_namedtuple=True)

    step_size = 0.0
    if geometry.geometry and "step_size" in geometry.geometry:
        step_size = float(np.asarray(geometry.geometry["step_size"]).reshape(-1)[0])
    fixed_steps = telemetry.fixed.get("num_integration_steps")
    vi_mode = (
        recipe.base_method_name in {"meanfield_vi", "fullrank_vi"}
        and recipe.warmup_name == "no_warmup"
    )
    gate = auto_gate(
        {k: np.asarray(v) for k, v in run_data.positions.items()},
        run_data.infos,
        ground_truth_summaries=aligned,
        posterior=MODELS[recipe.model_name],
        n_chunks=n_chunks,
        step_size=step_size,
        num_integration_steps=fixed_steps,
        vi_sampler_mode=vi_mode,
        multichain=True,
    )
    gate_dict = _json_safe(gate.to_dict())
    gate_evidence = {
        "auto": gate_dict,
        "override": {"reason": "", "statistician_id": "", "decision": ""},
    }

    grad_evals = sampling_grad_evals(run_data)
    total_draws = run_data.num_chains * run_data.num_samples
    denominator = grad_evals if grad_evals > 0 else total_draws
    convention = method.grad_count_convention or (
        "0 (gradient-free; headline = min_bulk_ess/n_total_samples)"
    )
    headline, basis = build_headline_basis(
        {k: np.asarray(v) for k, v in run_data.positions.items()},
        denominator=denominator,
        total_grad_evals=grad_evals,
        grad_count_convention=convention,
        is_lower_bound=is_laplace,
    )
    quality = compute_sample_quality(
        {k: np.asarray(run_data.positions[k]) for k in aligned},
        {k: {s: aligned[k][s] for s in ("mean", "std", "q05", "q95")} for k in aligned},
    )
    basis = _json_safe(basis)
    quality = _json_safe(quality)
    metrics = {
        "gate": gate_dict,
        "headline": headline,
        "headline_denominator": denominator,
        "headline_denominator_condition": (
            "sampling_grad_evals" if grad_evals > 0 else "total_draws (gradient-free)"
        ),
        "sample_quality": quality,
        "warmup_grad_evals": telemetry.warmup_grad_evals,
        "sampling_grad_evals": grad_evals,
        "grad_count_convention": convention,
        "sampling_grad_evals_is_lower_bound": is_laplace,
        "n_draws": total_draws,
    }
    return GeneratedEvaluation(
        gate_evidence=gate_evidence,
        headline_metric=float(headline),
        headline_basis=basis,
        sample_quality=quality,
        sampling_grad_evals=grad_evals,
        warmup_grad_evals=telemetry.warmup_grad_evals,
        pinned_base_method_params=params,
        resolved_step_policy=_json_safe(telemetry.resolved_step_policy),
        metrics=_json_safe(metrics),
        gt_cert_coverage=coverage,
    )


__all__ = ["GeneratedEvaluation", "evaluate_generated_run"]
