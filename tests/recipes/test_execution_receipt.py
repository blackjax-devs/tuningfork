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

"""Tests for versioned generated-execution receipts."""

import hashlib
from types import SimpleNamespace

import pytest

from tuningfork.recipes._execution_manifest import (
    LEGACY_GENERATOR_CONTRACT,
    LEGACY_MANIFEST_VERSION,
    ExecutionManifest,
)
from tuningfork.recipes._execution_plan import (
    ExecutionOverrides,
    canonical_json,
    legacy_execution_plan_hash,
)
from tuningfork.recipes._execution_receipt import (
    LEGACY_RECEIPT_VERSION,
    RECEIPT_VERSION,
    ExecutionReceipt,
)
from tuningfork.recipes._resolve_execution_plan import resolve_execution_plan

pytestmark = pytest.mark.fast


def _manifest():
    recipe = SimpleNamespace(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="window_adaptation_diag_imm",
        effort="low",
        base_method_params={"step_size": 0.1},
        warmup_params={"n_warmup": 10},
        warmups=[{"name": "window_adaptation_diag_imm", "params": {"n_warmup": 10}}],
        calibration_budget={"n_samples": 20, "num_chains": 2},
        tuning_seed=4,
        warmup_inner_kernel=None,
        init_strategy=None,
        step_policy=None,
        variant_label=None,
    )
    plan = resolve_execution_plan(recipe, ExecutionOverrides(num_samples=20))
    return ExecutionManifest.from_plan(plan, generator_version="2026.07")


def _receipt(**overrides):
    fields = dict(
        status="success",
        run_id="run-1",
        started_at="2026-07-30T10:00:00Z",
        finished_at="2026-07-30T10:01:00Z",
        manifest=_manifest(),
        source_sha256="a" * 64,
        program_path="runs/run-1/program.py",
        stdout_path="runs/run-1/stdout",
        stdout_sha256="c" * 64,
        stderr_path="runs/run-1/stderr",
        stderr_sha256="d" * 64,
        artifact_path="runs/run-1/mvn_10__hmc__window_adaptation_diag_imm.draws.npz",
        artifact_sha256="b" * 64,
        telemetry_path="runs/run-1/mvn_10__hmc__window_adaptation_diag_imm.telemetry.json",
        telemetry_sha256="e" * 64,
        return_code=0,
        timed_out=False,
        command=("python", "program.py"),
        environment={"python": "3.12"},
    )
    fields.update(overrides)
    return ExecutionReceipt.create(**fields)


def test_receipt_is_canonical_immutable_and_round_trips():
    receipt = _receipt(reference_identity={"sha": "x"})
    assert receipt.receipt_version == RECEIPT_VERSION
    assert receipt.to_json() == ExecutionReceipt.from_dict(receipt.as_dict()).to_json()
    with pytest.raises(TypeError):
        receipt.environment["python"] = "other"  # type: ignore[index]


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "running"),
        ("source_sha256", "bad"),
        ("program_path", "../x"),
        ("program_path", "a/./b"),
        ("program_path", "a//b"),
        ("program_path", "C:/program.py"),
        ("program_path", "a/"),
    ],
)
def test_receipt_rejects_invalid_fields(field, value):
    with pytest.raises((TypeError, ValueError)):
        _receipt(**{field: value})


def test_receipt_rejects_tampering():
    data = _receipt().as_dict()
    data["artifact_path"] = "runs/run-1/other.npz"
    with pytest.raises(ValueError):
        ExecutionReceipt.from_dict(data)


def test_v2_binds_telemetry_path_and_digest():
    data = _receipt().as_dict()
    assert data["receipt_version"] == "tuningfork.execution-receipt.v2"
    data["telemetry_path"] = "runs/run-1/other.telemetry.json"
    with pytest.raises(ValueError):
        ExecutionReceipt.from_dict(data)


def test_v2_create_rejects_declared_basename_mismatches():
    with pytest.raises(ValueError):
        _receipt(artifact_path="runs/run-1/other.draws.npz")
    with pytest.raises(ValueError):
        _receipt(telemetry_path="runs/run-1/other.telemetry.json")


def test_legacy_v1_round_trips_exactly():
    data = _receipt().as_dict()
    data.pop("telemetry_path")
    data.pop("telemetry_sha256")
    data["receipt_version"] = LEGACY_RECEIPT_VERSION
    manifest = dict(data["manifest"])
    normalized = dict(manifest["normalized_plan"])
    normalized.pop("telemetry_artifact_filename")
    manifest["manifest_version"] = LEGACY_MANIFEST_VERSION
    manifest["generator_contract"] = LEGACY_GENERATOR_CONTRACT
    manifest["normalized_plan"] = normalized
    manifest["plan_hash"] = legacy_execution_plan_hash(
        manifest["executable_config"], normalized["artifact_filename"]
    )
    data["manifest"] = manifest
    payload = dict(data)
    payload.pop("payload_hash")
    data["payload_hash"] = hashlib.sha256(
        (LEGACY_RECEIPT_VERSION + "\0" + canonical_json(payload)).encode()
    ).hexdigest()
    loaded = ExecutionReceipt.from_dict(data)
    assert loaded.as_dict() == data
    data["artifact_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        ExecutionReceipt.from_dict(data)


def test_failed_receipt_requires_failure_signal():
    with pytest.raises(ValueError):
        _receipt(status="failed", return_code=1)
    failed = _receipt(status="failed", return_code=1, error="program exited")
    assert failed.error == "program exited"


def test_timed_out_is_canonical_and_only_valid_on_failed_receipts():
    failed = _receipt(
        status="failed", return_code=None, timed_out=True, error="timed out"
    )
    assert ExecutionReceipt.from_dict(failed.as_dict()).timed_out is True
    with pytest.raises(ValueError):
        _receipt(timed_out=True)
    with pytest.raises(ValueError):
        _receipt(
            status="failed",
            return_code=-9,
            timed_out=True,
            error="timed out",
        )
    with pytest.raises(TypeError):
        _receipt(timed_out=1)


def test_receipt_rejects_naive_or_reversed_timestamps():
    with pytest.raises(ValueError):
        _receipt(started_at="2026-07-30T10:00:00")
    with pytest.raises(ValueError):
        _receipt(started_at="2026-07-30T10:02:00Z", finished_at="2026-07-30T10:01:00Z")


def test_receipt_normalizes_equivalent_timestamps_to_utc():
    receipt = _receipt(
        started_at="2026-07-30T12:00:00+02:00",
        finished_at="2026-07-30T12:01:00+02:00",
    )
    assert receipt.started_at == "2026-07-30T10:00:00Z"
    assert receipt.finished_at == "2026-07-30T10:01:00Z"
