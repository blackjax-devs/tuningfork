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
from types import SimpleNamespace

import pytest

from tuningfork.recipes._base import Effort, Recipe
from tuningfork.recipes._certification_record import (
    append_certification_attempt,
    import_legacy_current_view,
    launch_evidence,
)
from tuningfork.recipes._launcher import ExecutionTimings, LaunchResult

pytestmark = pytest.mark.fast


def _recipe(**kwargs):
    values = dict(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="no_warmup",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1},
        warmup_params={"n_warmup": 0},
        headline_metric=1.0,
        sample_quality={"ess": 10.0},
        calibration_budget={"trials": 1},
        difficulty=None,
        instructions="history",
        notes="note",
        workflow="workflow",
        gate_evidence={"auto": {"verdict": "PASS"}},
    )
    values.update(kwargs)
    return Recipe(**values)


def test_import_legacy_appends_snapshot_and_is_idempotent():
    recipe = _recipe(
        attempted_configurations=[{"legacy": {"x": 1}}],
        warmup_num_chains=[1, 2],
        inverse_mass_matrix_path="mvn_10/dense.imm.npz",
        variant_label="hmc_variant",
    )
    migrated = import_legacy_current_view(recipe, ground_truth={"id": "gt"})
    assert migrated.attempted_configurations[0] == {"legacy": {"x": 1}}
    snapshot = migrated.attempted_configurations[-1]
    assert snapshot["automatic_verdict"] == "PASS"
    assert "attempted_configurations" not in snapshot["intent_snapshot"]
    assert snapshot["intent_snapshot"]["warmup_num_chains"] == [1, 2]
    assert snapshot["intent_snapshot"]["inverse_mass_matrix_path"] == (
        "mvn_10/dense.imm.npz"
    )
    assert snapshot["intent_snapshot"]["variant_label"] == "hmc_variant"
    assert snapshot["execution"] is None
    assert import_legacy_current_view(migrated, ground_truth=None) is migrated


def test_append_without_result_uses_unique_id_and_copies_intent():
    base = _recipe()
    intent = _recipe(instructions="do not include", workflow="old")
    updated, attempt_id = append_certification_attempt(
        base,
        intent,
        result=None,
        ground_truth=None,
        lifecycle_stage="DRAFT",
        automatic_verdict="NOT_RUN",
        rationale="record intent",
        measurement_conditions={"n": 1},
        metrics=None,
        gate_evidence=None,
        failure_evidence=None,
        recipe_updates={},
    )
    assert attempt_id.startswith("attempt-")
    assert updated.attempted_configurations[-1]["execution"] is None
    assert "instructions" not in updated.attempted_configurations[-1]["intent_snapshot"]
    assert "workflow" not in updated.attempted_configurations[-1]["intent_snapshot"]


def test_typed_launch_result_keeps_receipt_paths_and_child_timing():
    result = LaunchResult(
        run_dir=Path("/tmp/run"),
        source_path=Path("/tmp/run/program.py"),
        stdout_path=Path("/tmp/run/stdout.log"),
        stderr_path=Path("/tmp/run/stderr.log"),
        artifact_path=Path("/tmp/run/draws.npz"),
        receipt_path=Path("/tmp/run/receipt.json"),
        returncode=0,
        timed_out=False,
        source_sha256="a" * 64,
        artifact_sha256="b" * 64,
        telemetry_path=Path("/tmp/run/telemetry.json"),
        telemetry_sha256="c" * 64,
        telemetry=SimpleNamespace(as_dict=lambda: {"schema": "telemetry"}),
        manifest=SimpleNamespace(as_dict=lambda: {"manifest": "nested-only"}),
        receipt=SimpleNamespace(
            run_id="run-123", as_dict=lambda: {"run_id": "run-123", "manifest": {}}
        ),
        timings=ExecutionTimings(1.0, 2.0, 3.0),
    )
    evidence = launch_evidence(result)
    assert evidence["receipt"]["run_id"] == "run-123"
    assert "manifest" not in evidence
    assert evidence["telemetry"]["schema"] == "telemetry"
    assert evidence["timings"]["total_seconds"] == 3.0
    updated, attempt_id = append_certification_attempt(
        _recipe(),
        _recipe(),
        result=result,
        ground_truth=None,
        lifecycle_stage="SAMPLED",
        automatic_verdict="NOT_RUN",
        rationale="record generated run",
        measurement_conditions={},
        metrics=None,
        gate_evidence=None,
        failure_evidence=None,
        recipe_updates={},
    )
    assert attempt_id == "run-123"
    assert updated.attempted_configurations[-1]["attempt_id"] == "run-123"


def test_launch_evidence_keeps_unknown_values_null():
    result = LaunchResult(
        run_dir=Path("/tmp/run"),
        source_path=Path("/tmp/run/program.py"),
        stdout_path=Path("/tmp/run/stdout.log"),
        stderr_path=Path("/tmp/run/stderr.log"),
        artifact_path=None,
        receipt_path=Path("/tmp/run/receipt.json"),
        returncode=None,
        timed_out=True,
        source_sha256="a" * 64,
        artifact_sha256=None,
        telemetry_path=None,
        telemetry_sha256=None,
        telemetry=None,
        manifest=SimpleNamespace(),
        receipt=SimpleNamespace(as_dict=lambda: {"status": "failed"}),
        timings=None,
    )
    evidence = launch_evidence(result)
    assert evidence["artifact_path"] is None
    assert evidence["telemetry"] is None
    assert evidence["timings"] is None


def test_unknown_legacy_attempts_are_preserved_verbatim():
    legacy = {"legacy": {"unknown": [1, None, "value"]}}
    recipe = _recipe(attempted_configurations=[legacy])
    migrated = import_legacy_current_view(recipe, ground_truth=None)
    assert migrated.attempted_configurations[0] == legacy


def test_legacy_snapshot_preserves_full_materialized_view_and_attempt_order():
    first = {"legacy": {"unknown": [1, None, "value"]}}
    recipe = _recipe(
        attempted_configurations=[first],
        headline_basis={"ess_estimator": "bulk"},
        tuningfork_version="1.2.3",
        blackjax_version="9.8.7",
        jax_version="0.0.1",
        timestamp_utc="2026-07-31T10:00:00+00:00",
        gt_schema_version="gt_v2_multichain",
        summary_v2_path="mvn_10/reference/summary_v2.json",
        _extra_fields={"future_provenance": {"opaque": None}},
    )
    migrated = import_legacy_current_view(recipe, ground_truth=None)

    assert migrated.attempted_configurations[0] == first
    expected_view = recipe.to_dict(include_legacy_warmup_fields=True)
    expected_view.pop("attempted_configurations")
    assert (
        migrated.attempted_configurations[-1]["metrics"]["legacy_current_view"]
        == expected_view
    )
    view = migrated.attempted_configurations[-1]["metrics"]["legacy_current_view"]
    assert "attempted_configurations" not in view
    assert view["headline_basis"] == {"ess_estimator": "bulk"}
    assert view["tuningfork_version"] == "1.2.3"
    assert view["gt_schema_version"] == "gt_v2_multichain"
    assert view["future_provenance"] == {"opaque": None}


def test_legacy_snapshot_tags_nonfinite_values_with_encoding_metadata():
    recipe = _recipe(
        _extra_fields={
            "historical_metrics": {
                "positive": float("inf"),
                "negative": float("-inf"),
                "not_a_number": float("nan"),
                "nested": [{"value": 3.5}],
            }
        }
    )

    migrated = import_legacy_current_view(recipe, ground_truth=None)
    metrics = migrated.attempted_configurations[-1]["metrics"]
    view = metrics["legacy_current_view"]
    encoding = metrics["legacy_current_view_encoding"]

    assert view["historical_metrics"] == {
        "positive": {"\u0000tuningfork_legacy_current_view_nonfinite_float": "+inf"},
        "negative": {"\u0000tuningfork_legacy_current_view_nonfinite_float": "-inf"},
        "not_a_number": {"\u0000tuningfork_legacy_current_view_nonfinite_float": "nan"},
        "nested": [{"value": 3.5}],
    }
    assert encoding == {
        "schema": "tuningfork.legacy-current-view.v1",
        "kind": "strict-json-tagged-nonfinite-float",
        "tag_key": "\u0000tuningfork_legacy_current_view_nonfinite_float",
        "values": ["nan", "+inf", "-inf"],
    }


def test_new_certification_attempt_rejects_nonfinite_evidence():
    with pytest.raises(ValueError, match="non-finite"):
        append_certification_attempt(
            _recipe(),
            _recipe(),
            result=None,
            ground_truth=None,
            lifecycle_stage="DRAFT",
            automatic_verdict="NOT_RUN",
            rationale="reject non-finite evidence",
            measurement_conditions={},
            metrics={"score": float("nan")},
            gate_evidence=None,
            failure_evidence=None,
            recipe_updates={},
        )
