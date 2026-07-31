# Copyright 2026- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Lossless certification records for SMC recipes."""

from __future__ import annotations

import copy
import uuid
from collections.abc import Mapping
from typing import Any, cast

from tuningfork.recipes._attempt_evidence import (
    ATTEMPT_SCHEMA,
    append_attempt,
    build_attempt_record,
)
from tuningfork.recipes._base_smc import SMCRecipe
from tuningfork.recipes._certification_record import launch_evidence
from tuningfork.recipes._legacy_evidence import (
    encode_legacy_value,
    legacy_encoding_metadata,
)
from tuningfork.recipes._launcher import LaunchResult

_VERDICTS = frozenset({"NOT_RUN", "PASS", "REVIEW", "FAIL", "ERROR"})
_INTENT_FIELDS = frozenset(
    {
        "model_name",
        "smc_method_name",
        "inner_method_name",
        "num_particles",
        "max_steps",
        "seed",
        "smc_params",
        "inner_params_init",
        "parameter_update_strategy",
        "parameter_update_strategy_kwargs",
    }
)


def _intent_snapshot(recipe: SMCRecipe) -> dict[str, Any]:
    raw = recipe.to_dict()
    return {
        key: copy.deepcopy(value) for key, value in raw.items() if key in _INTENT_FIELDS
    }


def _legacy_verdict(recipe: SMCRecipe) -> str:
    auto = recipe.gate_evidence.get("auto")
    value = auto.get("verdict") if isinstance(auto, Mapping) else None
    return value if value in _VERDICTS else "NOT_RUN"


def import_legacy_current_view(
    recipe: SMCRecipe, *, ground_truth: Mapping[str, Any] | None
) -> SMCRecipe:
    """Append an idempotent snapshot of an older SMC recipe's current view."""
    attempts = recipe.attempted_configurations
    if any(
        isinstance(item, Mapping)
        and item.get("schema") == ATTEMPT_SCHEMA
        and item.get("attempt_id") == "legacy-current-view"
        for item in attempts
    ):
        return recipe
    rich_attempts = [
        item
        for item in attempts
        if isinstance(item, Mapping) and item.get("schema") == ATTEMPT_SCHEMA
    ]
    selected = recipe.calibration_budget.get("selected_attempt_id")
    if isinstance(selected, str) and any(
        item.get("attempt_id") == selected for item in rich_attempts
    ):
        return recipe
    if rich_attempts and selected is not None:
        raise ValueError("selected_attempt_id does not identify a retained attempt")
    verdict = _legacy_verdict(recipe)
    lifecycle = (
        "CURATED"
        if verdict in {"PASS", "FAIL"}
        else ("EVALUATED" if verdict == "REVIEW" else "DRAFT")
    )
    view = recipe.to_dict()
    view.pop("attempted_configurations", None)
    legacy_view, view_nonfinite = encode_legacy_value(view)
    metrics: dict[str, Any] = {
        "headline_metric": recipe.headline_metric,
        "calibration_budget": copy.deepcopy(recipe.calibration_budget),
        "legacy_current_view": legacy_view,
    }
    metrics, metrics_nonfinite = encode_legacy_value(metrics)
    has_nonfinite = view_nonfinite or metrics_nonfinite
    if has_nonfinite:
        metrics["legacy_current_view_encoding"] = legacy_encoding_metadata()
    intent, intent_nonfinite = encode_legacy_value(_intent_snapshot(recipe))
    gt, gt_nonfinite = encode_legacy_value(copy.deepcopy(ground_truth))
    gate, gate_nonfinite = encode_legacy_value(copy.deepcopy(recipe.gate_evidence))
    failure = {
        "failure_diagnosis": recipe.failure_diagnosis,
        "workflow": recipe.workflow,
        "notes": recipe.notes,
    }
    failure, failure_nonfinite = encode_legacy_value(failure)
    if (
        intent_nonfinite or gt_nonfinite or gate_nonfinite or failure_nonfinite
    ) and "legacy_current_view_encoding" not in metrics:
        metrics["legacy_current_view_encoding"] = legacy_encoding_metadata()
    attempt = build_attempt_record(
        attempt_id="legacy-current-view",
        rationale="Snapshot of the pre-refactor current SMC recipe view",
        lifecycle_stage=lifecycle,
        automatic_verdict=verdict,
        intent_snapshot=intent,
        execution=None,
        ground_truth=gt,
        measurement_conditions={},
        metrics=metrics,
        gate_evidence=gate,
        failure_evidence=failure,
    )
    return cast(SMCRecipe, append_attempt(cast(Any, recipe), attempt))


def append_smc_certification_attempt(
    base_recipe: SMCRecipe,
    intent_recipe: SMCRecipe,
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
) -> tuple[SMCRecipe, str]:
    """Build and append one validated SMC certification attempt."""
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
    return (
        cast(SMCRecipe, append_attempt(cast(Any, base_recipe), attempt, **updates)),
        attempt_id,
    )


__all__ = ["append_smc_certification_attempt", "import_legacy_current_view"]
