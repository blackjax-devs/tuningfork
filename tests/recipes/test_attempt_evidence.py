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

from pathlib import Path

import pytest

import tuningfork.recipes._base as recipe_base
from tuningfork.recipes._attempt_evidence import append_attempt, build_attempt_record
from tuningfork.recipes._base import Effort, Recipe

pytestmark = pytest.mark.fast


def _attempt(**changes):
    args = dict(
        attempt_id="a1",
        rationale="baseline",
        lifecycle_stage="GENERATED",
        automatic_verdict="NOT_RUN",
        intent_snapshot={"sampler": "hmc", "params": {"step_size": 0.1}},
        execution=None,
        ground_truth=None,
        measurement_conditions={"chains": 1},
        metrics=None,
        gate_evidence=None,
        failure_evidence=None,
        recorded_at="2026-01-01T00:00:00Z",
    )
    args.update(changes)
    return build_attempt_record(**args)


def _recipe(attempted_configurations=None):
    return Recipe(
        model_name="mvn_2",
        base_method_name="hmc",
        warmup_name="no_warmup",
        effort=Effort.LOW,
        base_method_params={},
        warmup_params={},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        attempted_configurations=attempted_configurations or [],
    )


def test_intent_hash_is_deterministic_and_changes_with_intent():
    first = _attempt()
    second = _attempt(intent_snapshot={"params": {"step_size": 0.1}, "sampler": "hmc"})
    changed = _attempt(intent_snapshot={"sampler": "hmc", "params": {"step_size": 0.2}})
    assert first["intent_snapshot_sha256"] == second["intent_snapshot_sha256"]
    assert first["intent_snapshot_sha256"] != changed["intent_snapshot_sha256"]


@pytest.mark.parametrize(
    "field,value",
    [("lifecycle_stage", "BAD"), ("automatic_verdict", "BAD")],
)
def test_invalid_state_rejected(field, value):
    with pytest.raises(ValueError):
        _attempt(**{field: value})


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        _attempt(intent_snapshot={"x": float("nan")})


def test_append_order_and_no_mutation():
    old = {"legacy": {"keep": True}}
    recipe = _recipe([old])
    attempt = _attempt()
    updated = append_attempt(recipe, attempt)
    assert recipe.attempted_configurations == [old]
    assert updated.attempted_configurations == [old, attempt]
    assert updated.attempted_configurations is not recipe.attempted_configurations
    attempt["intent_snapshot"]["sampler"] = "changed"
    assert updated.attempted_configurations[1]["intent_snapshot"]["sampler"] == "hmc"


def test_recipe_save_failure_preserves_previous_document(tmp_path, monkeypatch):
    recipe = _recipe([{"legacy": {"failure": "keep"}}])
    path = recipe.save(tmp_path)
    original = path.read_bytes()

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(recipe_base.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        append_attempt(recipe, _attempt()).save(tmp_path)

    assert path.read_bytes() == original
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))
