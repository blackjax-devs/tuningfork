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

"""Fail-closed execution of generated recipe programs."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ._execution_manifest import ExecutionManifest
from ._execution_plan import canonical_json
from ._execution_receipt import ExecutionReceipt
from ._execution_telemetry import ExecutionTelemetry
from ._sample_stats import SAMPLE_STAT_PREFIX

_PROGRAM_FILENAME = "program.py"
_STDOUT_FILENAME = "stdout.log"
_STDERR_FILENAME = "stderr.log"
_RECEIPT_FILENAME = "execution_receipt.json"
_WORK_DIRECTORY = "work"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_TIMINGS_SENTINEL = "TUNINGFORK_TIMINGS "


@dataclass(frozen=True)
class ExecutionTimings:
    """Validated wall-clock timings reported by a generated program."""

    warmup_seconds: float
    sampling_seconds: float
    total_seconds: float


def _parse_timings(stdout: bytes) -> ExecutionTimings | None:
    """Parse the optional machine-readable timing sentinel from stdout."""
    matches = [
        line[len(_TIMINGS_SENTINEL) :]
        for line in stdout.decode("utf-8", errors="replace").splitlines()
        if line.startswith(_TIMINGS_SENTINEL)
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("stdout contains duplicate timing sentinels")
    try:
        payload = json.loads(matches[0])
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("timing sentinel is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("timing sentinel payload must be an object")
    required_fields = {"warmup_seconds", "sampling_seconds", "total_seconds"}
    if set(payload) != required_fields:
        raise ValueError(
            "timing sentinel fields must be exactly " f"{sorted(required_fields)!r}"
        )
    values: list[float] = []
    for name in ("warmup_seconds", "sampling_seconds", "total_seconds"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"timing sentinel field {name!r} must be numeric")
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"timing sentinel field {name!r} must be finite and non-negative"
            )
        values.append(value)
    warmup, sampling, total = values
    if total < warmup + sampling:
        raise ValueError("timing sentinel total_seconds is less than its components")
    return ExecutionTimings(warmup, sampling, total)


@dataclass(frozen=True)
class LaunchResult:
    """Verified files and receipt from one generated-program attempt."""

    run_dir: Path
    source_path: Path
    stdout_path: Path
    stderr_path: Path
    artifact_path: Path | None
    receipt_path: Path
    returncode: int | None
    timed_out: bool
    source_sha256: str
    artifact_sha256: str | None
    telemetry_path: Path | None
    telemetry_sha256: str | None
    telemetry: ExecutionTelemetry | None
    manifest: ExecutionManifest
    receipt: ExecutionReceipt
    timings: ExecutionTimings | None


class GeneratedProgramError(RuntimeError):
    """A generated program failed after preserving its execution evidence."""

    def __init__(self, message: str, result: LaunchResult | None = None):
        super().__init__(message)
        self.result = result
        self.receipt_path = None if result is None else result.receipt_path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_exclusive(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _preserve_collision(path: Path) -> Path | None:
    if not os.path.lexists(path):
        return None
    observed = path.with_name(f"{path.name}.child-{uuid.uuid4().hex}")
    os.replace(path, observed)
    return observed


def _repository_state() -> dict[str, Any]:
    try:
        revision_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        revision = None
    else:
        revision = revision_result.stdout.strip() or None
    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=_REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        dirty = None
    else:
        dirty = bool(status_result.stdout)
    return {"sha": revision, "dirty": dirty}


def _environment(
    python_executable: str, environment_override_keys: tuple[str, ...]
) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("tuningfork", "blackjax", "jax", "jaxlib", "numpy", "numpyro"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "launcher_python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "child_python_executable": python_executable,
        "package_versions_scope": "launcher_process",
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine() or None,
            "processor": platform.processor() or None,
            "node": platform.node() or None,
        },
        "packages": packages,
        "repository": _repository_state(),
        # Values may contain credentials. Only their names are provenance.
        "environment_override_keys": list(environment_override_keys),
    }


def _binds_manifest(node: ast.stmt) -> bool:
    if isinstance(node, ast.Assign):
        return any(
            isinstance(target, ast.Name) and target.id == "EXECUTION_MANIFEST_JSON"
            for target in node.targets
        )
    if isinstance(node, ast.AnnAssign):
        return (
            isinstance(node.target, ast.Name)
            and node.target.id == "EXECUTION_MANIFEST_JSON"
        )
    if isinstance(node, ast.AugAssign):
        return (
            isinstance(node.target, ast.Name)
            and node.target.id == "EXECUTION_MANIFEST_JSON"
        )
    return False


def _manifest_from_source(source: str) -> ExecutionManifest:
    try:
        tree = ast.parse(source, filename=_PROGRAM_FILENAME)
    except SyntaxError as exc:
        raise GeneratedProgramError(f"invalid generated source: {exc}") from exc
    bindings = [node for node in tree.body if _binds_manifest(node)]
    if len(bindings) != 1:
        raise GeneratedProgramError(
            "source must contain exactly one EXECUTION_MANIFEST_JSON binding"
        )
    assignment = bindings[0]
    if (
        not isinstance(assignment, ast.Assign)
        or len(assignment.targets) != 1
        or not isinstance(assignment.targets[0], ast.Name)
    ):
        raise GeneratedProgramError(
            "EXECUTION_MANIFEST_JSON must be a standalone literal assignment"
        )
    try:
        value = ast.literal_eval(assignment.value)
    except (SyntaxError, TypeError, ValueError) as exc:
        raise GeneratedProgramError(
            "EXECUTION_MANIFEST_JSON must be a literal JSON string"
        ) from exc
    if not isinstance(value, str):
        raise GeneratedProgramError(
            "EXECUTION_MANIFEST_JSON must be a literal JSON string"
        )
    try:
        return ExecutionManifest.from_dict(json.loads(value))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise GeneratedProgramError(f"invalid execution manifest: {exc}") from exc


def _safe_basename(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or value in {".", ".."}
    ):
        raise GeneratedProgramError(f"manifest {label} must be a safe basename")
    return value


def _artifact_basename(manifest: ExecutionManifest) -> str:
    """Backward-compatible helper for validating the declared draws name."""
    return _safe_basename(
        manifest.normalized_plan["artifact_filename"], "artifact_filename"
    )


def _stop_process_group(process: subprocess.Popen[Any]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            process.kill()
    else:
        process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _append_error(error: str | None, detail: str) -> str:
    return detail if error is None else f"{error}; {detail}"


def _validate_artifact(path: Path, manifest: ExecutionManifest) -> str:
    artifact_sha256 = _sha256_file(path)
    with np.load(path, allow_pickle=False) as archive:
        files = list(archive.files)
        if not files:
            raise ValueError("draws archive is empty")
        position_names = [
            name for name in files if not name.startswith(SAMPLE_STAT_PREFIX)
        ]
        if not position_names:
            raise ValueError("draws archive has no position arrays")
        expected_shape = (
            int(manifest.executable_config["num_chains"]),
            int(manifest.executable_config["num_samples"]),
        )
        for name in files:
            array = archive[name]
            if array.dtype.hasobject or array.size == 0:
                raise ValueError(f"artifact array {name!r} is empty or object-typed")
            if not (
                np.issubdtype(array.dtype, np.bool_)
                or (
                    np.issubdtype(array.dtype, np.number)
                    and not np.issubdtype(array.dtype, np.complexfloating)
                )
            ):
                raise ValueError(f"artifact array {name!r} is not real numeric")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"artifact array {name!r} contains non-finite values")
            if array.ndim < 2 or array.shape[:2] != expected_shape:
                kind = (
                    "statistic"
                    if name.startswith(SAMPLE_STAT_PREFIX)
                    else "position array"
                )
                raise ValueError(
                    f"{kind} {name!r} has leading shape {array.shape[:2]!r}; "
                    f"expected {expected_shape!r}"
                )
            if name == SAMPLE_STAT_PREFIX:
                raise ValueError("statistic name is empty")
    return artifact_sha256


def _write_receipt(receipt: ExecutionReceipt, path: Path) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(receipt.to_json())
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-linking is an atomic no-overwrite publication. It cannot follow
        # a child-created destination symlink as write_text()/replace() could.
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _finish_attempt(
    *,
    run_dir: Path,
    source_path: Path,
    stdout_path: Path,
    stdout_sha256: str,
    stderr_path: Path,
    stderr_sha256: str,
    artifact_path: Path | None,
    expected_artifact_name: str,
    returncode: int | None,
    timed_out: bool,
    source_sha256: str,
    artifact_sha256: str | None,
    telemetry_path: Path | None,
    telemetry_sha256: str | None,
    telemetry: ExecutionTelemetry | None,
    manifest: ExecutionManifest,
    started_at: str,
    command: tuple[str, ...],
    environment: Mapping[str, Any],
    reference_identity: Mapping[str, Any] | None,
    error: str | None,
    timings: ExecutionTimings | None,
) -> LaunchResult:
    def build_receipt(receipt_error: str | None) -> ExecutionReceipt:
        return ExecutionReceipt.create(
            status="success" if receipt_error is None else "failed",
            run_id=run_dir.name,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            manifest=manifest,
            source_sha256=source_sha256,
            program_path=_PROGRAM_FILENAME,
            stdout_path=_STDOUT_FILENAME,
            stdout_sha256=stdout_sha256,
            stderr_path=_STDERR_FILENAME,
            stderr_sha256=stderr_sha256,
            artifact_path=expected_artifact_name,
            artifact_sha256=artifact_sha256,
            telemetry_path=(None if telemetry_path is None else telemetry_path.name),
            telemetry_sha256=telemetry_sha256,
            return_code=returncode,
            timed_out=timed_out,
            command=command,
            environment=environment,
            reference_identity=reference_identity,
            error=receipt_error,
        )

    receipt = build_receipt(error)
    receipt_path = run_dir / _RECEIPT_FILENAME
    try:
        _write_receipt(receipt, receipt_path)
    except OSError as primary_exc:
        persistence_error = _append_error(
            error, f"run-local receipt persistence failed: {primary_exc}"
        )
        receipt = build_receipt(persistence_error)
        receipt_path = run_dir.parent / f"{run_dir.name}.{_RECEIPT_FILENAME}"
        try:
            _write_receipt(receipt, receipt_path)
        except OSError as fallback_exc:
            raise GeneratedProgramError(
                "execution finished but no receipt location was writable: "
                f"{primary_exc}; fallback: {fallback_exc}"
            ) from fallback_exc
    return LaunchResult(
        run_dir=run_dir,
        source_path=source_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        artifact_path=artifact_path,
        receipt_path=receipt_path,
        returncode=returncode,
        timed_out=timed_out,
        source_sha256=source_sha256,
        artifact_sha256=artifact_sha256,
        telemetry_path=telemetry_path,
        telemetry_sha256=telemetry_sha256,
        telemetry=telemetry,
        manifest=manifest,
        receipt=receipt,
        timings=timings,
    )


def launch_generated_program(
    source: str,
    run_root: Path,
    *,
    timeout: float | None = None,
    python_executable: str = sys.executable,
    env: Mapping[str, str] | None = None,
    reference_identity: Mapping[str, Any] | None = None,
) -> LaunchResult:
    """Validate, execute, and receipt one generated program.

    Source and manifest validation happen before a run directory is created.
    Once execution begins, every failure preserves the source, logs, any
    produced artifact hash, and a verified failed receipt.
    """
    if not isinstance(source, str):
        raise GeneratedProgramError("source must be a string")
    if not isinstance(python_executable, str) or not python_executable:
        raise GeneratedProgramError("python_executable must be a non-empty string")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise GeneratedProgramError("timeout must be a positive number or None")
    if env is not None and (
        not isinstance(env, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in env.items()
        )
    ):
        raise GeneratedProgramError("env must be a string-to-string mapping or None")
    if reference_identity is not None:
        if not isinstance(reference_identity, Mapping):
            raise GeneratedProgramError("reference_identity must be a mapping or None")
        try:
            canonical_json(reference_identity)
        except (TypeError, ValueError) as exc:
            raise GeneratedProgramError(f"invalid reference_identity: {exc}") from exc

    manifest = _manifest_from_source(source)
    if manifest.manifest_version != "tuningfork.execution-manifest.v2":
        raise GeneratedProgramError("legacy execution manifest v1 is not executable")
    expected_artifact_name = _safe_basename(
        manifest.normalized_plan["artifact_filename"], "artifact_filename"
    )
    expected_telemetry_name = _safe_basename(
        manifest.normalized_plan["telemetry_artifact_filename"],
        "telemetry_artifact_filename",
    )
    reserved = {
        _PROGRAM_FILENAME,
        _STDOUT_FILENAME,
        _STDERR_FILENAME,
        _RECEIPT_FILENAME,
        _WORK_DIRECTORY,
    }
    if (
        expected_artifact_name == expected_telemetry_name
        or expected_artifact_name in reserved
        or expected_telemetry_name in reserved
    ):
        raise GeneratedProgramError(
            "manifest artifact names collide with reserved names"
        )

    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"{manifest.plan_hash[:12]}-", dir=root))
    work_dir = run_dir / _WORK_DIRECTORY
    work_dir.mkdir()
    run_dir_stat_before = run_dir.stat()
    source_bytes = source.encode("utf-8")
    work_source_path = work_dir / _PROGRAM_FILENAME
    work_source_path.write_bytes(source_bytes)
    work_source_path.chmod(0o400)
    source_sha256 = _sha256_bytes(source_bytes)
    stdout_path = run_dir / _STDOUT_FILENAME
    stderr_path = run_dir / _STDERR_FILENAME
    work_artifact_path = work_dir / expected_artifact_name
    expected_artifact_path = run_dir / expected_artifact_name
    work_telemetry_path = work_dir / expected_telemetry_name
    expected_telemetry_path = run_dir / expected_telemetry_name
    command = (python_executable, _PROGRAM_FILENAME)
    environment = _environment(
        python_executable, tuple(sorted(() if env is None else env))
    )
    environment["child_working_directory"] = _WORK_DIRECTORY
    started_at = datetime.now(timezone.utc).isoformat()

    child_env = os.environ.copy()
    if env is not None:
        child_env.update(env)
    child_env["PYTHONUNBUFFERED"] = "1"

    returncode: int | None = None
    timed_out = False
    error: str | None = None
    timings: ExecutionTimings | None = None
    telemetry: ExecutionTelemetry | None = None
    telemetry_sha256: str | None = None
    with (
        tempfile.TemporaryFile() as stdout_stream,
        tempfile.TemporaryFile() as stderr_stream,
    ):
        try:
            process = subprocess.Popen(
                list(command),
                cwd=work_dir,
                env=child_env,
                stdout=stdout_stream,
                stderr=stderr_stream,
                start_new_session=os.name == "posix",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            error = f"generated program could not start: {type(exc).__name__}: {exc}"
        else:
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _stop_process_group(process)
                error = "generated program timed out"
            else:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except OSError:
                        pass
                    else:
                        error = _append_error(
                            error, "generated program left process-group descendants"
                        )
        stdout_stream.flush()
        stderr_stream.flush()
        stdout_stream.seek(0)
        stdout_bytes = stdout_stream.read()
        stderr_stream.seek(0)
        stderr_bytes = stderr_stream.read()

    try:
        timings = _parse_timings(stdout_bytes)
    except ValueError as exc:
        error = _append_error(error, str(exc))

    source_path = run_dir / _PROGRAM_FILENAME
    for directory, label in (
        (run_dir, "run directory"),
        (work_dir, "working directory"),
    ):
        try:
            directory.chmod(0o700)
        except OSError as exc:
            error = _append_error(error, f"{label} access could not be restored: {exc}")

    try:
        run_dir_stat_after = run_dir.stat()
    except OSError as exc:
        error = _append_error(error, f"run directory could not be verified: {exc}")
    else:
        if run_dir_stat_after.st_mtime_ns != run_dir_stat_before.st_mtime_ns:
            error = _append_error(
                error, "child modified canonical run-directory entries"
            )

    collision_paths = (
        (source_path, "source"),
        (stdout_path, "stdout"),
        (stderr_path, "stderr"),
        (expected_artifact_path, "draws artifact"),
        (expected_telemetry_path, "telemetry"),
        (run_dir / _RECEIPT_FILENAME, "receipt"),
    )
    for path, label in collision_paths:
        try:
            observed = _preserve_collision(path)
        except OSError as exc:
            error = _append_error(
                error, f"child-created canonical {label} could not be preserved: {exc}"
            )
        else:
            if observed is not None:
                error = _append_error(
                    error,
                    f"child created a canonical {label} collision "
                    f"(preserved as {observed.name})",
                )

    for path, value, label in (
        (source_path, source_bytes, "source"),
        (stdout_path, stdout_bytes, "stdout"),
        (stderr_path, stderr_bytes, "stderr"),
    ):
        try:
            _write_bytes_exclusive(path, value)
        except OSError as exc:
            error = _append_error(
                error, f"canonical {label} evidence could not be persisted: {exc}"
            )

    stdout_sha256 = _sha256_bytes(stdout_bytes)
    stderr_sha256 = _sha256_bytes(stderr_bytes)
    for path, expected_sha256, label in (
        (source_path, source_sha256, "source"),
        (stdout_path, stdout_sha256, "stdout"),
        (stderr_path, stderr_sha256, "stderr"),
    ):
        if path.is_symlink() or not path.is_file():
            error = _append_error(error, f"canonical {label} evidence is not regular")
            continue
        try:
            persisted_sha256 = _sha256_file(path)
        except OSError as exc:
            error = _append_error(
                error, f"canonical {label} evidence could not be verified: {exc}"
            )
        else:
            if persisted_sha256 != expected_sha256:
                error = _append_error(
                    error, f"canonical {label} evidence digest does not match"
                )

    if work_source_path.is_symlink() or not work_source_path.is_file():
        error = _append_error(error, "executed source is not a regular file")
    else:
        try:
            persisted_source_sha256 = _sha256_file(work_source_path)
        except OSError as exc:
            error = _append_error(
                error, f"executed source could not be verified: {exc}"
            )
        else:
            if persisted_source_sha256 != source_sha256:
                error = _append_error(
                    error, "generated program modified its source file"
                )

    artifact_path: Path | None = None
    artifact_sha256: str | None = None
    if work_artifact_path.is_symlink():
        error = _append_error(
            error, "manifest-declared draws artifact is a symbolic link"
        )
    elif work_artifact_path.is_file():
        try:
            artifact_sha256 = _sha256_file(work_artifact_path)
        except OSError as exc:
            error = _append_error(error, f"draws artifact could not be hashed: {exc}")
        try:
            os.link(work_artifact_path, expected_artifact_path)
        except OSError as exc:
            error = _append_error(
                error, f"draws artifact could not be preserved: {exc}"
            )
        else:
            artifact_path = expected_artifact_path
            try:
                work_artifact_path.unlink()
            except OSError as exc:
                error = _append_error(
                    error, f"working draws artifact could not be retired: {exc}"
                )
    elif os.path.lexists(work_artifact_path):
        error = _append_error(
            error, "manifest-declared draws artifact is not a regular file"
        )

    telemetry_path: Path | None = None
    if work_telemetry_path.is_symlink():
        error = _append_error(error, "manifest-declared telemetry is a symbolic link")
    elif work_telemetry_path.is_file():
        try:
            telemetry_sha256 = _sha256_file(work_telemetry_path)
            os.link(work_telemetry_path, expected_telemetry_path)
            telemetry_path = expected_telemetry_path
            work_telemetry_path.unlink()
        except OSError as exc:
            error = _append_error(error, f"telemetry could not be preserved: {exc}")
    elif os.path.lexists(work_telemetry_path):
        error = _append_error(
            error, "manifest-declared telemetry is not a regular file"
        )

    if returncode is not None and returncode != 0:
        error = _append_error(
            error, f"generated program exited with return code {returncode}"
        )
    done_count = (
        stdout_bytes.decode("utf-8", errors="replace").splitlines().count("DONE")
    )
    if done_count != 1:
        error = _append_error(error, "stdout must contain exactly one DONE line")

    try:
        unexpected_work_entries = sorted(
            path.name for path in work_dir.iterdir() if path.name != _PROGRAM_FILENAME
        )
    except OSError as exc:
        error = _append_error(error, f"working directory could not be inspected: {exc}")
    else:
        if unexpected_work_entries:
            error = _append_error(
                error,
                "working directory contains unexpected entries: "
                + ", ".join(unexpected_work_entries),
            )

    allowed_top_level = {
        _PROGRAM_FILENAME,
        _STDOUT_FILENAME,
        _STDERR_FILENAME,
        expected_artifact_name,
        expected_telemetry_name,
    }
    try:
        unexpected_top_level_entries = sorted(
            path.name
            for path in run_dir.iterdir()
            if path.name != _WORK_DIRECTORY and path.name not in allowed_top_level
        )
    except OSError as exc:
        error = _append_error(error, f"run directory could not be inspected: {exc}")
    else:
        if unexpected_top_level_entries:
            error = _append_error(
                error,
                "run directory contains unexpected child-created entries: "
                + ", ".join(unexpected_top_level_entries),
            )

    if artifact_path is None:
        error = _append_error(error, "manifest-declared draws artifact is missing")
    else:
        try:
            validated_artifact_sha256 = _validate_artifact(
                expected_artifact_path, manifest
            )
        except (OSError, ValueError) as exc:
            error = _append_error(error, f"invalid draws artifact: {exc}")
        else:
            if (
                artifact_sha256 is not None
                and validated_artifact_sha256 != artifact_sha256
            ):
                error = _append_error(
                    error, "draws artifact changed while it was being preserved"
                )
            artifact_sha256 = validated_artifact_sha256

    if telemetry_path is None:
        error = _append_error(error, "manifest-declared telemetry is missing")
    else:
        try:
            telemetry = ExecutionTelemetry.read_path(telemetry_path, manifest)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            error = _append_error(error, f"invalid execution telemetry: {exc}")

    result = _finish_attempt(
        run_dir=run_dir,
        source_path=source_path,
        stdout_path=stdout_path,
        stdout_sha256=stdout_sha256,
        stderr_path=stderr_path,
        stderr_sha256=stderr_sha256,
        artifact_path=artifact_path,
        expected_artifact_name=expected_artifact_name,
        returncode=returncode,
        timed_out=timed_out,
        source_sha256=source_sha256,
        artifact_sha256=artifact_sha256,
        manifest=manifest,
        started_at=started_at,
        command=command,
        environment=environment,
        reference_identity=reference_identity,
        error=error,
        timings=timings,
        telemetry_path=telemetry_path,
        telemetry_sha256=telemetry_sha256,
        telemetry=telemetry,
    )
    if result.receipt.status == "failed":
        raise GeneratedProgramError(
            result.receipt.error or "generated program failed", result
        )
    return result


__all__ = [
    "ExecutionTimings",
    "GeneratedProgramError",
    "LaunchResult",
    "launch_generated_program",
]
