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
"""Typed telemetry emitted by generated SMC programs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tuningfork.recipes._execution_manifest import ExecutionManifest

SMC_TELEMETRY_SCHEMA = "tuningfork.generated-smc-telemetry.v1"
_FIELDS = {
    "schema",
    "plan_hash",
    "executable_config_hash",
    "draws_artifact",
    "num_particles",
    "num_smc_steps",
    "lambda_final",
    "timing_seconds",
}
_TIMING_FIELDS = {"initialization", "sampling", "total"}


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"SMC telemetry {field} must be a positive integer")
    return value


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"SMC telemetry {field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"SMC telemetry {field} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class SMCExecutionTelemetry:
    schema: str
    plan_hash: str
    executable_config_hash: str
    draws_artifact: str
    num_particles: int
    num_smc_steps: int
    lambda_final: float
    timing_seconds: Mapping[str, float]

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], manifest: ExecutionManifest
    ) -> SMCExecutionTelemetry:
        if not isinstance(data, Mapping) or set(data) != _FIELDS:
            raise ValueError("SMC telemetry has unknown or missing fields")
        if data["schema"] != SMC_TELEMETRY_SCHEMA:
            raise ValueError("unsupported SMC telemetry schema")
        if (
            data["plan_hash"] != manifest.plan_hash
            or data["executable_config_hash"] != manifest.executable_config_hash
        ):
            raise ValueError("SMC telemetry hash does not match manifest")
        expected_draws = manifest.normalized_plan["artifact_filename"]
        if data["draws_artifact"] != expected_draws:
            raise ValueError("SMC telemetry draws_artifact does not match manifest")
        particles = _positive_int(data["num_particles"], "num_particles")
        if particles != manifest.executable_config["num_particles"]:
            raise ValueError("SMC telemetry particle count does not match manifest")
        steps = _positive_int(data["num_smc_steps"], "num_smc_steps")
        lambda_final = _finite_nonnegative(data["lambda_final"], "lambda_final")
        if lambda_final > 1:
            raise ValueError("SMC telemetry lambda_final must be within [0, 1]")
        timings = data["timing_seconds"]
        if not isinstance(timings, Mapping) or set(timings) != _TIMING_FIELDS:
            raise ValueError("SMC telemetry timing_seconds has invalid fields")
        normalized_timings = {
            name: _finite_nonnegative(timings[name], f"timing_seconds.{name}")
            for name in sorted(_TIMING_FIELDS)
        }
        if (
            normalized_timings["total"] + 1e-12
            < normalized_timings["initialization"] + normalized_timings["sampling"]
        ):
            raise ValueError("SMC telemetry total timing is less than its components")
        return cls(
            SMC_TELEMETRY_SCHEMA,
            manifest.plan_hash,
            manifest.executable_config_hash,
            str(expected_draws),
            particles,
            steps,
            lambda_final,
            MappingProxyType(normalized_timings),
        )

    @classmethod
    def read_path(
        cls, path: Path, manifest: ExecutionManifest
    ) -> SMCExecutionTelemetry:
        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate SMC telemetry field: {key}")
                result[key] = value
            return result

        data = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
        return cls.from_dict(data, manifest)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_hash": self.plan_hash,
            "executable_config_hash": self.executable_config_hash,
            "draws_artifact": self.draws_artifact,
            "num_particles": self.num_particles,
            "num_smc_steps": self.num_smc_steps,
            "lambda_final": self.lambda_final,
            "timing_seconds": dict(self.timing_seconds),
        }


__all__ = ["SMC_TELEMETRY_SCHEMA", "SMCExecutionTelemetry"]
