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
"""Lossless, validated records of recipe-generation attempts.

The attempt envelope is intentionally explicit so failed and partially-run
attempts remain useful evidence without requiring the sampler runner here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from tuningfork.recipes._base import Recipe

ATTEMPT_SCHEMA = "tuningfork.recipe-attempt.v1"
_INTENT_HASH_DOMAIN = ATTEMPT_SCHEMA + "\0intent_snapshot\0"
_LIFECYCLE = frozenset({"DRAFT", "GENERATED", "SAMPLED", "EVALUATED", "CURATED"})
_VERDICTS = frozenset({"NOT_RUN", "PASS", "REVIEW", "FAIL", "ERROR"})
_FIELDS = (
    "schema",
    "attempt_id",
    "rationale",
    "lifecycle_stage",
    "automatic_verdict",
    "intent_snapshot",
    "intent_snapshot_sha256",
    "execution",
    "ground_truth",
    "measurement_conditions",
    "metrics",
    "gate_evidence",
    "failure_evidence",
    "recorded_at",
)


def _check_json(value: Any, path: str = "value") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} has a non-string key: {key!r}")
            _check_json(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_json(item, f"{path}[{index}]")
        return
    raise TypeError(f"{path} is not JSON-safe: {type(value).__name__}")


def _canonical_json(value: Any) -> bytes:
    _check_json(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _timestamp(value: str | None) -> str:
    if value is None:
        value = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if not isinstance(value, str):
        raise TypeError("recorded_at must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("recorded_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("recorded_at must include timezone information")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_attempt_record(
    *,
    attempt_id: str,
    rationale: str,
    lifecycle_stage: str,
    automatic_verdict: str,
    intent_snapshot: Mapping[str, Any],
    execution: Mapping[str, Any] | None,
    ground_truth: Mapping[str, Any] | None,
    measurement_conditions: Mapping[str, Any],
    metrics: Mapping[str, Any] | None,
    gate_evidence: Mapping[str, Any] | None,
    failure_evidence: Mapping[str, Any] | None,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Build and validate one immutable-by-convention attempt envelope."""
    if not isinstance(attempt_id, str) or not attempt_id.strip():
        raise ValueError("attempt_id must be nonempty")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be nonempty")
    if lifecycle_stage not in _LIFECYCLE:
        raise ValueError(f"invalid lifecycle_stage: {lifecycle_stage!r}")
    if automatic_verdict not in _VERDICTS:
        raise ValueError(f"invalid automatic_verdict: {automatic_verdict!r}")
    if not isinstance(intent_snapshot, Mapping):
        raise TypeError("intent_snapshot must be a mapping")
    if not isinstance(measurement_conditions, Mapping):
        raise TypeError("measurement_conditions must be a mapping")
    for name, value in {
        "execution": execution,
        "ground_truth": ground_truth,
        "metrics": metrics,
        "gate_evidence": gate_evidence,
        "failure_evidence": failure_evidence,
    }.items():
        if value is not None and not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping or None")

    intent = copy.deepcopy(dict(intent_snapshot))
    canonical = _canonical_json(intent)
    digest = hashlib.sha256(_INTENT_HASH_DOMAIN.encode() + canonical).hexdigest()
    record = {
        "schema": ATTEMPT_SCHEMA,
        "attempt_id": attempt_id,
        "rationale": rationale,
        "lifecycle_stage": lifecycle_stage,
        "automatic_verdict": automatic_verdict,
        "intent_snapshot": intent,
        "intent_snapshot_sha256": digest,
        "execution": copy.deepcopy(execution),
        "ground_truth": copy.deepcopy(ground_truth),
        "measurement_conditions": copy.deepcopy(dict(measurement_conditions)),
        "metrics": copy.deepcopy(metrics),
        "gate_evidence": copy.deepcopy(gate_evidence),
        "failure_evidence": copy.deepcopy(failure_evidence),
        "recorded_at": _timestamp(recorded_at),
    }
    _check_json(record)
    return record


def _validate_envelope(attempt: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(attempt, Mapping) or set(attempt) != set(_FIELDS):
        raise ValueError("attempt is not a tuningfork.recipe-attempt.v1 envelope")
    copied = copy.deepcopy(dict(attempt))
    if copied.get("schema") != ATTEMPT_SCHEMA:
        raise ValueError("unexpected attempt schema")
    _check_json(copied)
    expected = build_attempt_record(
        attempt_id=copied["attempt_id"],
        rationale=copied["rationale"],
        lifecycle_stage=copied["lifecycle_stage"],
        automatic_verdict=copied["automatic_verdict"],
        intent_snapshot=copied["intent_snapshot"],
        execution=copied["execution"],
        ground_truth=copied["ground_truth"],
        measurement_conditions=copied["measurement_conditions"],
        metrics=copied["metrics"],
        gate_evidence=copied["gate_evidence"],
        failure_evidence=copied["failure_evidence"],
        recorded_at=copied["recorded_at"],
    )
    if copied != expected:
        raise ValueError("attempt envelope is noncanonical or has been modified")
    return copied


def append_attempt(
    recipe: Recipe, attempt: Mapping[str, Any], **recipe_updates: Any
) -> Recipe:
    """Return a recipe with a validated, deeply copied attempt appended."""
    validated = _validate_envelope(attempt)
    return replace(
        recipe,
        attempted_configurations=[*recipe.attempted_configurations, validated],
        **recipe_updates,
    )


__all__ = ["ATTEMPT_SCHEMA", "append_attempt", "build_attempt_record"]
