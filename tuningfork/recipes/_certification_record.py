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

"""Lossless records used when certifying generated recipe programs."""

from __future__ import annotations

import copy
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tuningfork.recipes._attempt_evidence import (
    ATTEMPT_SCHEMA,
    append_attempt,
    build_attempt_record,
)
from tuningfork.recipes._base import Recipe
from tuningfork.recipes._launcher import LaunchResult

_VERDICTS = frozenset({"NOT_RUN", "PASS", "REVIEW", "FAIL", "ERROR"})
_LEGACY_VIEW_ENCODING_SCHEMA = "tuningfork.legacy-current-view.v1"
_LEGACY_NONFINITE_TAG = "\u0000tuningfork_legacy_current_view_nonfinite_float"
_INTENT_FIELDS = frozenset(
    {
        "model_name",
        "base_method_name",
        "base_method_params",
        "warmup_name",
        "warmup_params",
        "warmups",
        "warmup_inner_kernel",
        "warmup_num_chains",
        "init_strategy",
        "variant_label",
        "inverse_mass_matrix_path",
        "step_policy",
        "tuning_seed",
        "calibration_budget",
    }
)


def _path(value: Path | None) -> str | None:
    return None if value is None else str(value)


def launch_evidence(result: LaunchResult) -> dict[str, Any]:
    """Extract strict, non-redundant evidence from a verified launch result."""
    if not isinstance(result, LaunchResult):
        raise TypeError("result must be a LaunchResult")
    evidence: dict[str, Any] = {
        "receipt": copy.deepcopy(result.receipt.as_dict()),
        "run_dir": _path(result.run_dir),
        "source_path": _path(result.source_path),
        "stdout_path": _path(result.stdout_path),
        "stderr_path": _path(result.stderr_path),
        "artifact_path": _path(result.artifact_path),
        "telemetry_path": _path(result.telemetry_path),
        "receipt_path": _path(result.receipt_path),
        "telemetry": None,
        "timings": None,
    }
    if result.telemetry is not None:
        evidence["telemetry"] = copy.deepcopy(result.telemetry.as_dict())
    if result.timings is not None:
        evidence["timings"] = {
            "warmup_seconds": result.timings.warmup_seconds,
            "sampling_seconds": result.timings.sampling_seconds,
            "total_seconds": result.timings.total_seconds,
        }
    return evidence


def _intent_snapshot(recipe: Recipe) -> dict[str, Any]:
    raw = recipe.to_dict(include_legacy_warmup_fields=True)
    return {
        key: copy.deepcopy(value) for key, value in raw.items() if key in _INTENT_FIELDS
    }


def _legacy_current_view(recipe: Recipe) -> dict[str, Any]:
    """Return the complete serialized pre-refactor view, excluding attempts."""
    view = recipe.to_dict(include_legacy_warmup_fields=True)
    view.pop("attempted_configurations", None)
    return copy.deepcopy(view)


def _encode_legacy_value(value: Any) -> tuple[Any, bool]:
    """Make historical non-finite floats safe for the strict attempt envelope.

    The NUL-prefixed key is reserved by the encoding metadata below, so tagged
    objects cannot be confused with ordinary recipe values by a reader that
    understands this versioned representation.
    """
    if isinstance(value, float):
        if value != value:
            return {_LEGACY_NONFINITE_TAG: "nan"}, True
        if value == float("inf"):
            return {_LEGACY_NONFINITE_TAG: "+inf"}, True
        if value == float("-inf"):
            return {_LEGACY_NONFINITE_TAG: "-inf"}, True
        return value, False
    if isinstance(value, Mapping):
        encoded: dict[str, Any] = {}
        changed = False
        for key, item in value.items():
            encoded_item, item_changed = _encode_legacy_value(item)
            encoded[key] = encoded_item
            changed = changed or item_changed
        return encoded, changed
    if isinstance(value, list):
        encoded_items = []
        changed = False
        for item in value:
            encoded_item, item_changed = _encode_legacy_value(item)
            encoded_items.append(encoded_item)
            changed = changed or item_changed
        return encoded_items, changed
    return value, False


def _legacy_verdict(recipe: Recipe) -> str:
    gate = recipe.gate_evidence
    auto = gate.get("auto") if isinstance(gate, Mapping) else None
    value = auto.get("verdict") if isinstance(auto, Mapping) else None
    return value if value in _VERDICTS else "NOT_RUN"


def import_legacy_current_view(
    recipe: Recipe, *, ground_truth: Mapping[str, Any] | None
) -> Recipe:
    """Append one pre-refactor snapshot, preserving all existing raw attempts."""
    rich_attempts = [
        item
        for item in recipe.attempted_configurations
        if isinstance(item, Mapping) and item.get("schema") == ATTEMPT_SCHEMA
    ]
    if any(item.get("attempt_id") == "legacy-current-view" for item in rich_attempts):
        return recipe
    selected = (
        recipe.calibration_budget.get("selected_attempt_id")
        if isinstance(recipe.calibration_budget, Mapping)
        else None
    )
    if isinstance(selected, str) and any(
        item.get("attempt_id") == selected for item in rich_attempts
    ):
        # The materialized view explicitly points at retained evidence.
        return recipe
    if (
        any(
            isinstance(item, Mapping) and item.get("schema") == ATTEMPT_SCHEMA
            for item in recipe.attempted_configurations
        )
        and selected is not None
    ):
        raise ValueError("selected_attempt_id does not identify a retained attempt")
    verdict = _legacy_verdict(recipe)
    lifecycle = (
        "CURATED"
        if verdict in {"PASS", "FAIL"}
        else "EVALUATED" if verdict == "REVIEW" else "DRAFT"
    )
    diagnosis = recipe.failure_diagnosis
    failure = {
        "failure_diagnosis": getattr(diagnosis, "value", diagnosis),
        "workflow": recipe.workflow,
        "notes": recipe.notes,
    }
    legacy_view, has_nonfinite = _encode_legacy_value(_legacy_current_view(recipe))
    metrics = {
        "headline_metric": recipe.headline_metric,
        "sample_quality": copy.deepcopy(recipe.sample_quality),
        "calibration_budget": copy.deepcopy(recipe.calibration_budget),
        "legacy_current_view": legacy_view,
    }
    metrics, metrics_nonfinite = _encode_legacy_value(metrics)
    has_nonfinite = has_nonfinite or metrics_nonfinite
    if has_nonfinite:
        metrics["legacy_current_view_encoding"] = {
            "schema": _LEGACY_VIEW_ENCODING_SCHEMA,
            "kind": "strict-json-tagged-nonfinite-float",
            "tag_key": _LEGACY_NONFINITE_TAG,
            "values": ["nan", "+inf", "-inf"],
        }
    intent_snapshot, intent_nonfinite = _encode_legacy_value(_intent_snapshot(recipe))
    ground_truth_snapshot, ground_truth_nonfinite = _encode_legacy_value(
        copy.deepcopy(ground_truth)
    )
    gate_snapshot, gate_nonfinite = _encode_legacy_value(
        copy.deepcopy(recipe.gate_evidence)
    )
    failure_snapshot, failure_nonfinite = _encode_legacy_value(failure)
    if (
        intent_nonfinite
        or ground_truth_nonfinite
        or gate_nonfinite
        or failure_nonfinite
    ) and "legacy_current_view_encoding" not in metrics:
        metrics["legacy_current_view_encoding"] = {
            "schema": _LEGACY_VIEW_ENCODING_SCHEMA,
            "kind": "strict-json-tagged-nonfinite-float",
            "tag_key": _LEGACY_NONFINITE_TAG,
            "values": ["nan", "+inf", "-inf"],
        }
    attempt = build_attempt_record(
        attempt_id="legacy-current-view",
        rationale="Snapshot of the pre-refactor current recipe view",
        lifecycle_stage=lifecycle,
        automatic_verdict=verdict,
        intent_snapshot=intent_snapshot,
        execution=None,
        ground_truth=ground_truth_snapshot,
        measurement_conditions={},
        metrics=metrics,
        gate_evidence=gate_snapshot,
        failure_evidence=failure_snapshot,
    )
    return append_attempt(recipe, attempt)


def append_certification_attempt(
    base_recipe: Recipe,
    intent_recipe: Recipe,
    *,
    result: LaunchResult | None,
    ground_truth: Mapping[str, Any] | None,
    lifecycle_stage: str,
    automatic_verdict: str,
    rationale: str,
    measurement_conditions: Mapping[str, Any],
    metrics: Mapping[str, Any] | None,
    gate_evidence: Mapping[str, Any] | None,
    failure_evidence: Mapping[str, Any] | None,
    recipe_updates: Mapping[str, Any] | None,
) -> tuple[Recipe, str]:
    """Build and append one generated-certification attempt."""
    attempt_id = (
        result.receipt.run_id if result is not None else f"attempt-{uuid.uuid4().hex}"
    )
    attempt = build_attempt_record(
        attempt_id=attempt_id,
        rationale=rationale,
        lifecycle_stage=lifecycle_stage,
        automatic_verdict=automatic_verdict,
        intent_snapshot=_intent_snapshot(intent_recipe),
        execution=launch_evidence(result) if result is not None else None,
        ground_truth=copy.deepcopy(ground_truth),
        measurement_conditions=copy.deepcopy(dict(measurement_conditions)),
        metrics=copy.deepcopy(metrics),
        gate_evidence=copy.deepcopy(gate_evidence),
        failure_evidence=copy.deepcopy(failure_evidence),
    )
    updates = {} if recipe_updates is None else copy.deepcopy(dict(recipe_updates))
    return append_attempt(base_recipe, attempt, **updates), attempt_id


__all__ = [
    "append_certification_attempt",
    "import_legacy_current_view",
    "launch_evidence",
]
