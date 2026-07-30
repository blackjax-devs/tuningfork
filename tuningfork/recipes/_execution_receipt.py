"""Immutable, hash-addressed receipts for generated program executions."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from ._execution_manifest import ExecutionManifest
from ._execution_plan import _freeze, _thaw, canonical_json

RECEIPT_VERSION = "tuningfork.execution-receipt.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATUSES = {"success", "failed"}
_PAYLOAD_DOMAIN = RECEIPT_VERSION + "\0"


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha(name: str, value: Any, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    value = _text(name, value)
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _relative_path(name: str, value: Any) -> str:
    value = _text(name, value)
    if "\\" in value or "\x00" in value:
        raise ValueError(f"{name} must be a safe relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).drive
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} must be a safe relative path")
    return value


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    canonical_json(value)
    return _freeze(dict(value))


def _timestamp(name: str, value: Any) -> str:
    value = _text(name, value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ExecutionReceipt:
    """Versioned execution result with a canonical tamper-evident payload."""

    receipt_version: str
    status: str
    run_id: str
    started_at: str
    finished_at: str
    manifest: ExecutionManifest
    source_sha256: str
    program_path: str
    stdout_path: str
    stdout_sha256: str
    stderr_path: str
    stderr_sha256: str
    artifact_path: str
    artifact_sha256: str | None
    return_code: int | None
    timed_out: bool
    command: tuple[str, ...]
    environment: Mapping[str, Any]
    reference_identity: Mapping[str, Any] | None
    error: str | None
    payload_hash: str

    @classmethod
    def create(
        cls,
        *,
        status: str,
        run_id: str,
        started_at: str,
        finished_at: str,
        manifest: ExecutionManifest,
        source_sha256: str,
        program_path: str,
        stdout_path: str,
        stdout_sha256: str,
        stderr_path: str,
        stderr_sha256: str,
        artifact_path: str,
        artifact_sha256: str | None = None,
        return_code: int | None = None,
        timed_out: bool = False,
        command: Sequence[str] = (),
        environment: Mapping[str, Any] | None = None,
        reference_identity: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> ExecutionReceipt:
        if status not in _STATUSES:
            raise ValueError("status must be 'success' or 'failed'")
        if not isinstance(manifest, ExecutionManifest):
            raise TypeError("manifest must be an ExecutionManifest")
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
            raise TypeError("command must be a sequence of strings")
        command_tuple = tuple(_text("command item", item) for item in command)
        if not command_tuple:
            raise ValueError("command must not be empty")
        if return_code is not None and (
            isinstance(return_code, bool) or not isinstance(return_code, int)
        ):
            raise TypeError("return_code must be an integer or None")
        if not isinstance(timed_out, bool):
            raise TypeError("timed_out must be a bool")
        if status == "success" and timed_out:
            raise ValueError("successful receipts cannot be timed out")
        if timed_out and return_code is not None:
            raise ValueError("timed-out receipts must not have a return code")
        if status == "success" and (
            return_code != 0 or error is not None or artifact_sha256 is None
        ):
            raise ValueError(
                "successful receipts require return_code=0, artifact_sha256, and no error"
            )
        if status == "failed" and error is None:
            raise ValueError("failed receipts require an error")
        validated_error = None if error is None else _text("error", error)
        validated_run_id = _text("run_id", run_id)
        validated_started_at = _timestamp("started_at", started_at)
        validated_finished_at = _timestamp("finished_at", finished_at)
        validated_source_sha256 = _sha("source_sha256", source_sha256)
        if validated_source_sha256 is None:
            raise AssertionError("required source digest validation returned None")
        validated_program_path = _relative_path("program_path", program_path)
        validated_stdout_path = _relative_path("stdout_path", stdout_path)
        validated_stdout_sha256 = _sha("stdout_sha256", stdout_sha256)
        if validated_stdout_sha256 is None:
            raise AssertionError("required stdout digest validation returned None")
        validated_stderr_path = _relative_path("stderr_path", stderr_path)
        validated_stderr_sha256 = _sha("stderr_sha256", stderr_sha256)
        if validated_stderr_sha256 is None:
            raise AssertionError("required stderr digest validation returned None")
        validated_artifact_path = _relative_path("artifact_path", artifact_path)
        validated_artifact_sha256 = _sha(
            "artifact_sha256", artifact_sha256, optional=True
        )
        validated_environment = _mapping(
            "environment", {} if environment is None else environment
        )
        validated_reference_identity = (
            None
            if reference_identity is None
            else _mapping("reference_identity", reference_identity)
        )
        started = datetime.fromisoformat(validated_started_at.replace("Z", "+00:00"))
        finished = datetime.fromisoformat(validated_finished_at.replace("Z", "+00:00"))
        if finished < started:
            raise ValueError("finished_at must not precede started_at")
        fields = {
            "receipt_version": RECEIPT_VERSION,
            "status": status,
            "run_id": validated_run_id,
            "started_at": validated_started_at,
            "finished_at": validated_finished_at,
            "manifest": manifest,
            "source_sha256": validated_source_sha256,
            "program_path": validated_program_path,
            "stdout_path": validated_stdout_path,
            "stdout_sha256": validated_stdout_sha256,
            "stderr_path": validated_stderr_path,
            "stderr_sha256": validated_stderr_sha256,
            "artifact_path": validated_artifact_path,
            "artifact_sha256": validated_artifact_sha256,
            "return_code": return_code,
            "timed_out": timed_out,
            "command": command_tuple,
            "environment": validated_environment,
            "reference_identity": validated_reference_identity,
            "error": validated_error,
        }
        payload_hash = hashlib.sha256(
            (_PAYLOAD_DOMAIN + canonical_json(cls._payload_dict(fields))).encode()
        ).hexdigest()
        return cls(
            receipt_version=RECEIPT_VERSION,
            status=status,
            run_id=validated_run_id,
            started_at=validated_started_at,
            finished_at=validated_finished_at,
            manifest=manifest,
            source_sha256=validated_source_sha256,
            program_path=validated_program_path,
            stdout_path=validated_stdout_path,
            stdout_sha256=validated_stdout_sha256,
            stderr_path=validated_stderr_path,
            stderr_sha256=validated_stderr_sha256,
            artifact_path=validated_artifact_path,
            artifact_sha256=validated_artifact_sha256,
            return_code=return_code,
            timed_out=timed_out,
            command=command_tuple,
            environment=validated_environment,
            reference_identity=validated_reference_identity,
            error=validated_error,
            payload_hash=payload_hash,
        )

    @staticmethod
    def _payload_dict(fields: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "receipt_version": fields["receipt_version"],
            "status": fields["status"],
            "run_id": fields["run_id"],
            "started_at": fields["started_at"],
            "finished_at": fields["finished_at"],
            "manifest": fields["manifest"].as_dict(),
            "source_sha256": fields["source_sha256"],
            "program_path": fields["program_path"],
            "stdout_path": fields["stdout_path"],
            "stdout_sha256": fields["stdout_sha256"],
            "stderr_path": fields["stderr_path"],
            "stderr_sha256": fields["stderr_sha256"],
            "artifact_path": fields["artifact_path"],
            "artifact_sha256": fields["artifact_sha256"],
            "return_code": fields["return_code"],
            "timed_out": fields["timed_out"],
            "command": list(fields["command"]),
            "environment": _thaw(fields["environment"]),
            "reference_identity": (
                None
                if fields["reference_identity"] is None
                else _thaw(fields["reference_identity"])
            ),
            "error": fields["error"],
        }

    def as_dict(self) -> dict[str, Any]:
        data = self._payload_dict(self.__dict__)
        data["payload_hash"] = self.payload_hash
        return data

    def to_json(self) -> str:
        return canonical_json(self.as_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutionReceipt:
        if not isinstance(data, Mapping):
            raise TypeError("receipt data must be a mapping")
        required = set(cls.__dataclass_fields__)
        if set(data) != required:
            raise ValueError("receipt has unsupported or missing fields")
        canonical_json(data)
        if data["receipt_version"] != RECEIPT_VERSION:
            raise ValueError("unsupported receipt_version")
        manifest = ExecutionManifest.from_dict(data["manifest"])
        obj = cls.create(
            status=data["status"],
            run_id=data["run_id"],
            started_at=data["started_at"],
            finished_at=data["finished_at"],
            manifest=manifest,
            source_sha256=data["source_sha256"],
            program_path=data["program_path"],
            stdout_path=data["stdout_path"],
            stdout_sha256=data["stdout_sha256"],
            stderr_path=data["stderr_path"],
            stderr_sha256=data["stderr_sha256"],
            artifact_path=data["artifact_path"],
            artifact_sha256=data["artifact_sha256"],
            return_code=data["return_code"],
            timed_out=data["timed_out"],
            command=data["command"],
            environment=data["environment"],
            reference_identity=data["reference_identity"],
            error=data["error"],
        )
        if data["payload_hash"] != obj.payload_hash:
            raise ValueError("payload_hash does not match receipt payload")
        return obj


__all__ = ["RECEIPT_VERSION", "ExecutionReceipt"]
