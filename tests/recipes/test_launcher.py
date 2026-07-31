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

"""Integration checks for fail-closed execution of generated programs."""

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from tuningfork.catalog import emit_script
from tuningfork.catalog._rerun_inference import _artifact_to_idata
from tuningfork.recipes import _launcher as launcher_module
from tuningfork.recipes._base import Effort, Recipe
from tuningfork.recipes._execution_manifest import ExecutionManifest
from tuningfork.recipes._execution_plan import ExecutionOverrides
from tuningfork.recipes._execution_receipt import ExecutionReceipt
from tuningfork.recipes._generated_evaluator import load_generated_artifact
from tuningfork.recipes._launcher import (
    ExecutionTimings,
    GeneratedProgramError,
    _parse_timings,
    launch_generated_program,
)
from tuningfork.recipes._resolve_execution_plan import resolve_execution_plan

pytestmark = pytest.mark.fast


def test_parse_timings_accepts_one_valid_sentinel() -> None:
    timings = _parse_timings(
        b'TUNINGFORK_TIMINGS {"sampling_seconds":2,"total_seconds":5,"warmup_seconds":3}\n'
    )
    assert timings == ExecutionTimings(3.0, 2.0, 5.0)


@pytest.mark.parametrize(
    "stdout",
    [
        b'TUNINGFORK_TIMINGS {"sampling_seconds":2,"total_seconds":5,"warmup_seconds":3}\n'
        b'TUNINGFORK_TIMINGS {"sampling_seconds":2,"total_seconds":5,"warmup_seconds":3}\n',
        b"TUNINGFORK_TIMINGS not-json\n",
        b'TUNINGFORK_TIMINGS {"sampling_seconds":-1,"total_seconds":5,"warmup_seconds":3}\n',
        b'TUNINGFORK_TIMINGS {"sampling_seconds":4,"total_seconds":5,"warmup_seconds":3}\n',
        b'TUNINGFORK_TIMINGS {"sampling_seconds":2,"total_seconds":5}\n',
        b'TUNINGFORK_TIMINGS {"extra":0,"sampling_seconds":2,"total_seconds":5,"warmup_seconds":3}\n',
    ],
)
def test_parse_timings_rejects_invalid_sentinel(stdout: bytes) -> None:
    with pytest.raises(ValueError):
        _parse_timings(stdout)


def test_parse_timings_is_optional_for_hand_authored_sources() -> None:
    assert _parse_timings(b"DONE\n") is None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> ExecutionManifest:
    recipe = SimpleNamespace(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="window_adaptation_diag_imm",
        effort="low",
        base_method_params={"step_size": 0.1},
        warmup_params={"n_warmup": 1},
        warmups=[{"name": "window_adaptation_diag_imm", "params": {"n_warmup": 1}}],
        calibration_budget={"n_samples": 2, "num_chains": 1},
        tuning_seed=4,
        warmup_inner_kernel=None,
        init_strategy=None,
        step_policy=None,
        variant_label=None,
        warmup_num_chains=None,
        gate_evidence={},
    )
    plan = resolve_execution_plan(
        cast(Any, recipe), ExecutionOverrides(num_samples=2, num_chains=1)
    )
    return ExecutionManifest.from_plan(plan, generator_version="test")


def _source(manifest: ExecutionManifest, body: str) -> str:
    telemetry = {
        "schema": "tuningfork.generated-run-telemetry.v1",
        "plan_hash": manifest.plan_hash,
        "executable_config_hash": manifest.executable_config_hash,
        "draws_artifact": manifest.normalized_plan["artifact_filename"],
        "geometry": {},
        "geometry_source": "unavailable",
        "geometry_scope": None,
        "geometry_unavailable_reason": "test",
        "fixed": {},
        "timing_seconds": {"warmup": 0, "sampling": 0, "total": 0},
        "warmup_grad_evals": None,
        "warmup_grad_evals_reason": "test",
    }
    return (
        f"EXECUTION_MANIFEST_JSON = {manifest.to_json()!r}\n"
        "import numpy as np\n"
        f"ARTIFACT = {manifest.normalized_plan['artifact_filename']!r}\n"
        f"TELEMETRY = {manifest.normalized_plan['telemetry_artifact_filename']!r}\n"
        f"open(TELEMETRY, 'w').write({json.dumps(telemetry, sort_keys=True)!r})\n"
        f"{body}\n"
    )


def _receipt(exc: GeneratedProgramError) -> ExecutionReceipt:
    assert exc.receipt_path is not None and exc.receipt_path.exists()
    return ExecutionReceipt.from_dict(json.loads(exc.receipt_path.read_text()))


def test_success_persists_source_logs_artifact_and_verified_receipt(tmp_path):
    manifest = _manifest()
    source = _source(
        manifest,
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1)))\nprint('DONE')",
    )
    first = launch_generated_program(source, tmp_path)
    second = launch_generated_program(source, tmp_path)
    assert first.run_dir != second.run_dir
    for result in (first, second):
        assert result.source_path.exists()
        assert result.stdout_path.read_text() == "DONE\n"
        assert result.stderr_path.read_text() == ""
        assert result.artifact_path is not None and result.artifact_path.exists()
        assert result.receipt_path is not None
        receipt = ExecutionReceipt.from_dict(
            json.loads(result.receipt_path.read_text())
        )
        assert receipt.status == "success"
        assert receipt.payload_hash == result.receipt.payload_hash
        assert receipt.source_sha256 == _sha256(result.source_path)
        assert receipt.stdout_sha256 == _sha256(result.stdout_path)
        assert receipt.stderr_sha256 == _sha256(result.stderr_path)
        assert receipt.artifact_sha256 == _sha256(result.artifact_path)
        assert result.telemetry_path is not None and result.telemetry_path.exists()
        assert result.telemetry_sha256 == _sha256(result.telemetry_path)
        assert result.telemetry is not None
        assert receipt.telemetry_sha256 == result.telemetry_sha256


@pytest.mark.parametrize(
    "body",
    [
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1)))\nraise SystemExit(3)",
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1)))\nprint('NOT_DONE')",
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1), dtype=object))\nprint('DONE')",
        "np.savez(ARTIFACT, position=np.full((1, 2, 1), np.nan))\nprint('DONE')",
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1), dtype=complex))\nprint('DONE')",
        "np.savez(ARTIFACT, position=np.full((1, 2, 1), 'bad'))\nprint('DONE')",
        "np.savez(ARTIFACT, position=np.zeros((1, 3, 1)))\nprint('DONE')",
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1)))\nnp.savez('extra.draws.npz', position=np.zeros((1, 2, 1)))\nprint('DONE')",
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1)))\nopen('extra.bin', 'wb').write(b'extra')\nprint('DONE')",
    ],
)
def test_contract_failures_persist_verified_failed_receipt(tmp_path, body):
    with pytest.raises(GeneratedProgramError) as caught:
        launch_generated_program(_source(_manifest(), body), tmp_path)
    receipt = _receipt(caught.value)
    assert receipt.status == "failed"
    assert receipt.error
    assert caught.value.result is not None
    assert caught.value.result.stdout_path.exists()
    assert caught.value.result.stderr_path.exists()


def test_timeout_preserves_logs_and_verified_failed_receipt(tmp_path):
    manifest = _manifest()
    source = (
        f"EXECUTION_MANIFEST_JSON = {manifest.to_json()!r}\n"
        "import sys, time\n"
        "print('BEFORE_TIMEOUT', flush=True)\n"
        "print('WARN_TIMEOUT', file=sys.stderr, flush=True)\n"
        "time.sleep(2)\n"
    )
    with pytest.raises(GeneratedProgramError) as caught:
        launch_generated_program(source, tmp_path, timeout=0.5)
    exc = caught.value
    receipt = _receipt(exc)
    assert exc.result is not None and exc.result.timed_out
    assert exc.result.stdout_path.read_text() == "BEFORE_TIMEOUT\n"
    assert exc.result.stderr_path.read_text() == "WARN_TIMEOUT\n"
    assert receipt.status == "failed"
    assert receipt.return_code is None
    assert receipt.timed_out is True
    assert "timed out" in (receipt.error or "")


def test_missing_python_executable_preserves_logs_and_verified_failed_receipt(
    tmp_path,
):
    with pytest.raises(GeneratedProgramError) as caught:
        launch_generated_program(
            _source(_manifest(), "print('NEVER_RUN')"),
            tmp_path,
            python_executable="/definitely/missing/python",
        )
    exc = caught.value
    receipt = _receipt(exc)
    assert exc.result is not None
    assert exc.result.stdout_path.read_text() == ""
    assert exc.result.stderr_path.read_text() == ""
    assert receipt.status == "failed"
    assert receipt.return_code is None
    assert "could not start" in (receipt.error or "")


def test_archive_with_only_sampler_stats_is_rejected_with_failed_receipt(tmp_path):
    source = _source(
        _manifest(),
        "np.savez(ARTIFACT, _ss_energy=np.zeros((1, 2)))\nprint('DONE')",
    )
    with pytest.raises(GeneratedProgramError) as caught:
        launch_generated_program(source, tmp_path)
    receipt = _receipt(caught.value)
    assert receipt.status == "failed"
    assert "no position arrays" in (receipt.error or "")


@pytest.mark.parametrize("telemetry_body", ["{}", '{\\"plan_hash\\":\\"wrong\\"}'])
def test_missing_or_malformed_telemetry_is_preserved(tmp_path, telemetry_body):
    source = _source(
        _manifest(),
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1)))\nopen(TELEMETRY, 'w').write("
        + repr(telemetry_body)
        + ")\nprint('DONE')",
    )
    with pytest.raises(GeneratedProgramError) as caught:
        launch_generated_program(source, tmp_path)
    result = caught.value.result
    assert result is not None and result.telemetry_path is not None
    assert result.telemetry_path.read_text() == telemetry_body
    assert result.telemetry_sha256 == _sha256(result.telemetry_path)


def test_cross_bound_telemetry_rejected(tmp_path):
    manifest = _manifest()
    source = _source(
        manifest,
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1)))\nopen(TELEMETRY, 'w').write(open(TELEMETRY).read().replace("
        + repr(manifest.plan_hash)
        + ", '0'*64))\nprint('DONE')",
    )
    with pytest.raises(GeneratedProgramError, match="invalid execution telemetry"):
        launch_generated_program(source, tmp_path)


def test_telemetry_canonical_collision_preserved(tmp_path):
    source = _source(
        _manifest(),
        "from pathlib import Path\nPath('../'+TELEMETRY).write_text('child collision')\nnp.savez(ARTIFACT, position=np.zeros((1, 2, 1)))\nprint('DONE')",
    )
    with pytest.raises(GeneratedProgramError) as caught:
        launch_generated_program(source, tmp_path)
    result = caught.value.result
    assert result is not None
    assert list(result.run_dir.glob("*.json.child-*"))


@pytest.mark.parametrize("name", ["../escape.npz", "a/b.npz", "same.npz"])
def test_unsafe_or_duplicate_declared_names_rejected_before_run(tmp_path, name):
    data = _manifest().as_dict()
    data["normalized_plan"]["artifact_filename"] = name
    data["normalized_plan"]["telemetry_artifact_filename"] = name
    source = f"EXECUTION_MANIFEST_JSON = {json.dumps(data)!r}\n"
    with pytest.raises(GeneratedProgramError):
        launch_generated_program(source, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_legacy_v1_manifest_rejected_before_run(tmp_path):
    data = _manifest().as_dict()
    data["manifest_version"] = "tuningfork.execution-manifest.v1"
    data["generator_contract"] = "tuningfork.execution-plan.v1"
    data["normalized_plan"].pop("telemetry_artifact_filename")
    from tuningfork.recipes._execution_plan import legacy_execution_plan_hash

    data["plan_hash"] = legacy_execution_plan_hash(
        data["executable_config"], data["normalized_plan"]["artifact_filename"]
    )
    with pytest.raises(GeneratedProgramError, match="legacy"):
        launch_generated_program(
            f"EXECUTION_MANIFEST_JSON = {json.dumps(data)!r}\n", tmp_path
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "archive_expr, message",
    [
        (
            "position=np.zeros((1, 2, 1)), _ss_energy=np.zeros(())",
            "statistic '_ss_energy'",
        ),
        (
            "position=np.zeros((1, 2, 1)), _ss_energy=np.zeros((1, 3))",
            "expected (1, 2)",
        ),
        (
            "position=np.zeros((1, 2, 1)), other=np.zeros((2, 2, 1))",
            "position array 'other'",
        ),
    ],
)
def test_malformed_artifact_shapes_are_rejected_before_done_receipt(
    tmp_path, archive_expr, message
):
    source = _source(
        _manifest(),
        f"np.savez(ARTIFACT, {archive_expr})\nprint('DONE')",
    )
    with pytest.raises(GeneratedProgramError) as caught:
        launch_generated_program(source, tmp_path)
    receipt = _receipt(caught.value)
    assert receipt.status == "failed"
    assert message in (receipt.error or "")


def test_sample_stat_semantics_are_deferred_to_the_evaluator(tmp_path):
    source = _source(
        _manifest(),
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1)), "
        "_ss_unknown=np.zeros((1, 2)))\nprint('DONE')",
    )

    result = launch_generated_program(source, tmp_path)

    assert result.receipt.status == "success"
    assert result.artifact_path is not None
    with pytest.raises(ValueError, match="generated contract"):
        load_generated_artifact(result.artifact_path, result.manifest)


def test_environment_override_values_are_not_recorded_in_receipt(tmp_path):
    secret = "sentinel-secret-value"
    source = _source(
        _manifest(),
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1)))\nprint('DONE')",
    )
    result = launch_generated_program(source, tmp_path, env={"TF_SECRET": secret})
    receipt_json = result.receipt_path.read_text()
    assert secret not in receipt_json
    persisted = ExecutionReceipt.from_dict(json.loads(receipt_json))
    assert tuple(persisted.environment["environment_override_keys"]) == ("TF_SECRET",)


def test_child_cannot_remove_launcher_evidence_write_access(tmp_path):
    source = _source(
        _manifest(),
        "import os\n"
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1)))\n"
        "os.chmod('..', 0)\n"
        "os.chmod('.', 0)\n"
        "print('DONE')",
    )
    result = launch_generated_program(source, tmp_path)
    assert result.receipt.status == "success"
    assert result.stdout_path.read_text() == "DONE\n"
    assert result.receipt_path.exists()


def test_child_source_modification_is_rejected_and_original_is_preserved(tmp_path):
    source = _source(
        _manifest(),
        "from pathlib import Path\n"
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1)))\n"
        "Path(__file__).chmod(0o600)\n"
        "Path(__file__).write_text('# modified')\n"
        "print('DONE')",
    )
    with pytest.raises(GeneratedProgramError) as caught:
        launch_generated_program(source, tmp_path)
    assert caught.value.result is not None
    assert "modified its source" in (caught.value.result.receipt.error or "")
    assert caught.value.result.source_path.read_text() == source


def test_child_canonical_source_collision_is_preserved_and_rejected(tmp_path):
    source = _source(
        _manifest(),
        "from pathlib import Path\n"
        "Path('../program.py').write_text('child source')\n"
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1)))\n"
        "print('DONE')",
    )
    with pytest.raises(GeneratedProgramError) as caught:
        launch_generated_program(source, tmp_path)
    result = caught.value.result
    assert result is not None
    assert result.source_path.read_text() == source
    observed = list(result.run_dir.glob("program.py.child-*"))
    assert len(observed) == 1
    assert observed[0].read_text() == "child source"
    assert "source collision" in (result.receipt.error or "")


def test_child_canonical_log_collision_preserves_captured_output(tmp_path):
    source = _source(
        _manifest(),
        "from pathlib import Path\n"
        "Path('../stdout.log').write_text('child log')\n"
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1)))\n"
        "print('DONE')",
    )
    with pytest.raises(GeneratedProgramError) as caught:
        launch_generated_program(source, tmp_path)
    result = caught.value.result
    assert result is not None
    assert result.stdout_path.read_text() == "DONE\n"
    observed = list(result.run_dir.glob("stdout.log.child-*"))
    assert len(observed) == 1
    assert observed[0].read_text() == "child log"
    assert result.receipt.stdout_sha256 == _sha256(result.stdout_path)


def test_child_fixed_receipt_temp_symlink_cannot_overwrite_target(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("untouched")
    source = _source(
        _manifest(),
        "import os\n"
        f"os.symlink({str(outside)!r}, '../execution_receipt.json.tmp')\n"
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1)))\n"
        "print('DONE')",
    )
    with pytest.raises(GeneratedProgramError) as caught:
        launch_generated_program(source, tmp_path)
    result = caught.value.result
    assert result is not None
    assert outside.read_text() == "untouched"
    assert (result.run_dir / "execution_receipt.json.tmp").is_symlink()
    assert result.receipt_path.exists()


def test_successful_parent_with_live_process_group_descendant_is_rejected(tmp_path):
    descendant = (
        "import time; time.sleep(0.5); " "open('late.bin', 'wb').write(b'late')"
    )
    source = _source(
        _manifest(),
        "import subprocess, sys\n"
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}])\n"
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1)))\n"
        "print('DONE')",
    )
    with pytest.raises(GeneratedProgramError) as caught:
        launch_generated_program(source, tmp_path)
    result = caught.value.result
    assert result is not None
    time.sleep(0.7)
    assert not (result.run_dir / "work" / "late.bin").exists()
    assert "left process-group descendants" in (result.receipt.error or "")


def test_timeout_terminates_descendants_before_receipt_is_written(tmp_path):
    manifest = _manifest()
    descendant = (
        "import time; time.sleep(0.5); " "open('late.draws.npz', 'wb').write(b'late')"
    )
    source = (
        f"EXECUTION_MANIFEST_JSON = {manifest.to_json()!r}\n"
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}])\n"
        "time.sleep(5)\n"
    )
    with pytest.raises(GeneratedProgramError) as caught:
        launch_generated_program(source, tmp_path, timeout=0.3)
    assert caught.value.result is not None
    time.sleep(0.7)
    assert not (caught.value.result.run_dir / "work" / "late.draws.npz").exists()


def test_receipt_write_falls_back_and_marks_attempt_failed(tmp_path, monkeypatch):
    write_receipt = launcher_module._write_receipt
    calls = 0

    def fail_first_write(receipt, path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("run directory is not writable")
        write_receipt(receipt, path)

    monkeypatch.setattr(launcher_module, "_write_receipt", fail_first_write)
    source = _source(
        _manifest(),
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1)))\nprint('DONE')",
    )
    with pytest.raises(GeneratedProgramError) as caught:
        launch_generated_program(source, tmp_path)
    receipt = _receipt(caught.value)
    assert receipt.status == "failed"
    assert "receipt persistence failed" in (receipt.error or "")
    assert caught.value.receipt_path.parent == tmp_path


def test_invalid_manifest_fails_before_creating_run(tmp_path):
    bad = _manifest().as_dict()
    bad["plan_hash"] = "0" * 64
    source = f"EXECUTION_MANIFEST_JSON = {json.dumps(bad)!r}\n"
    with pytest.raises(GeneratedProgramError):
        launch_generated_program(source, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_reference_identity_is_persisted_as_opaque_receipt_provenance(tmp_path):
    identity = {
        "lfs_oid": "oid-sha256:abc123",
        "content_sha256": "f" * 64,
        "protocol": "reference-v1",
    }
    source = _source(
        _manifest(),
        "np.savez(ARTIFACT, position=np.zeros((1, 2, 1)))\nprint('DONE')",
    )
    result = launch_generated_program(source, tmp_path, reference_identity=identity)
    assert result.receipt is not None
    assert dict(result.receipt.reference_identity) == identity
    persisted = ExecutionReceipt.from_dict(json.loads(result.receipt_path.read_text()))
    assert dict(persisted.reference_identity) == identity
    with pytest.raises(TypeError):
        persisted.reference_identity["protocol"] = "tampered"  # type: ignore[index]


@pytest.mark.e2e
def test_launcher_executes_an_emitted_dynamic_hmc_program(tmp_path):
    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="dynamic_hmc",
        warmup_name="no_warmup",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.5},
        warmup_params={},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )
    source = emit_script(recipe, num_samples=3, num_chains=1)
    result = launch_generated_program(
        source,
        tmp_path,
        timeout=180,
        env={"JAX_PLATFORM_NAME": "cpu"},
    )
    assert result.receipt.status == "success"
    assert result.manifest.plan_hash == result.receipt.manifest.plan_hash
    assert result.artifact_path is not None
    with np.load(result.artifact_path, allow_pickle=False) as archive:
        assert archive["x"].shape[:2] == (1, 3)
    idata = _artifact_to_idata(result.artifact_path)
    assert idata.posterior["x"].shape[:2] == (1, 3)
    assert "diverging" in idata.sample_stats
