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

"""Typed, canonical execution-plan values used by recipe code generation."""

from __future__ import annotations

import hashlib
import json
import ntpath
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Generic, Protocol, TypeVar

EXECUTION_CONFIG_HASH_DOMAIN = "tuningfork.execution-config.v1\0"
EXECUTION_PLAN_HASH_DOMAIN = "tuningfork.execution-plan.v2\0"
LEGACY_EXECUTION_PLAN_HASH_DOMAIN = "tuningfork.execution-plan.v1\0"


def _v2_artifact_basename(name: str, field: str) -> str:
    """Validate a v2 artifact name as a portable, simple basename."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"{field} must be a non-empty string")
    if "\x00" in name or "/" in name or "\\" in name:
        raise ValueError(f"{field} must be a simple basename")
    if name in {".", ".."} or ntpath.splitdrive(name)[0]:
        raise ValueError(f"{field} must be a simple basename")
    return name


def _default_telemetry_artifact_filename(artifact_filename: str) -> str:
    if not isinstance(artifact_filename, str) or not artifact_filename.endswith(
        ".draws.npz"
    ):
        raise ValueError(
            "cannot derive telemetry_artifact_filename unless artifact_filename "
            "ends with '.draws.npz'"
        )
    return artifact_filename.removesuffix(".draws.npz") + ".telemetry.json"


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("execution configuration cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("execution configuration mapping keys must be strings")
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON-safe")


def canonical_json(value: Any) -> str:
    """Return strict, deterministic JSON for an executable value."""
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(prefix: str, value: Any) -> str:
    return hashlib.sha256((prefix + canonical_json(value)).encode("utf-8")).hexdigest()


def execution_config_hash(config: Mapping[str, Any]) -> str:
    """Digest the canonical executable configuration representation."""
    return _digest(EXECUTION_CONFIG_HASH_DOMAIN, config)


def execution_plan_hash(
    config: Mapping[str, Any],
    artifact_filename: str,
    telemetry_artifact_filename: str | None = None,
) -> str:
    """Digest plan identity; ``recipe_ref`` is presentation metadata and excluded."""
    artifact_filename = _v2_artifact_basename(artifact_filename, "artifact_filename")
    if telemetry_artifact_filename is None:
        telemetry_artifact_filename = _default_telemetry_artifact_filename(
            artifact_filename
        )
    else:
        telemetry_artifact_filename = _v2_artifact_basename(
            telemetry_artifact_filename, "telemetry_artifact_filename"
        )
    if artifact_filename == telemetry_artifact_filename:
        raise ValueError(
            "artifact_filename and telemetry_artifact_filename must differ"
        )
    return _digest(
        EXECUTION_PLAN_HASH_DOMAIN,
        {
            "config": dict(config),
            "artifact_filename": artifact_filename,
            "telemetry_artifact_filename": telemetry_artifact_filename,
        },
    )


def legacy_execution_plan_hash(
    config: Mapping[str, Any], artifact_filename: str
) -> str:
    """Digest the legacy v1 plan identity (without telemetry)."""
    return _digest(
        LEGACY_EXECUTION_PLAN_HASH_DOMAIN,
        {"config": dict(config), "artifact_filename": artifact_filename},
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class ExecutionOverrides:
    tuning_seed: int | None = None
    sampler_seed: int | None = None
    num_samples: int | None = None
    num_chains: int | None = None
    progress_bar: bool | None = None
    num_warmup: int | tuple[int, ...] | list[int] | None = None
    warmup_num_chains: tuple[int, ...] | list[int] | None = None
    reinit_seed: int | None = None


@dataclass(frozen=True)
class WarmupStagePlan:
    name: str
    params: Mapping[str, Any]
    num_warmup: int
    num_chains: int


@dataclass(frozen=True)
class ExecutableConfigurationSnapshot:
    model_name: str
    base_method_name: str
    warmup_name: str
    base_method_params: Mapping[str, Any]
    warmup_params: Mapping[str, Any]
    warmup_stages: tuple[WarmupStagePlan, ...]
    warmup_inner_kernel: str | None
    init_strategy: Mapping[str, Any] | None
    step_policy: Mapping[str, Any] | None
    tuning_seed: int
    sampler_seed: int
    reinit_seed: int
    num_samples: int
    num_chains: int
    progress_bar: bool
    requires_x64: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "base_method_name": self.base_method_name,
            "warmup_name": self.warmup_name,
            "base_method_params": _thaw(self.base_method_params),
            "warmup_params": _thaw(self.warmup_params),
            "warmup_stages": [
                {
                    "name": stage.name,
                    "params": _thaw(stage.params),
                    "num_warmup": stage.num_warmup,
                    "num_chains": stage.num_chains,
                }
                for stage in self.warmup_stages
            ],
            "warmup_inner_kernel": self.warmup_inner_kernel,
            "init_strategy": _thaw(self.init_strategy),
            "step_policy": _thaw(self.step_policy),
            "tuning_seed": self.tuning_seed,
            "sampler_seed": self.sampler_seed,
            "reinit_seed": self.reinit_seed,
            "num_samples": self.num_samples,
            "num_chains": self.num_chains,
            "progress_bar": self.progress_bar,
            "requires_x64": self.requires_x64,
        }

    @property
    def config_hash(self) -> str:
        return execution_config_hash(self.as_dict())


class ExecutionConfiguration(Protocol):
    """Structural contract shared by family-specific execution configs."""

    def as_dict(self) -> dict[str, Any]:
        """Return the strict JSON representation used for plan identity."""
        raise NotImplementedError

    @property
    def config_hash(self) -> str:
        """Return the canonical executable-configuration digest."""
        raise NotImplementedError


ConfigurationT = TypeVar("ConfigurationT", bound=ExecutionConfiguration)


@dataclass(frozen=True)
class ExecutionPlan(Generic[ConfigurationT]):
    config: ConfigurationT
    recipe_ref: str
    artifact_filename: str
    telemetry_artifact_filename: str
    plan_hash: str

    @property
    def executable_config_hash(self) -> str:
        return self.config.config_hash

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "recipe_ref": self.recipe_ref,
            "artifact_filename": self.artifact_filename,
            "telemetry_artifact_filename": self.telemetry_artifact_filename,
        }

    @classmethod
    def build(
        cls,
        config: ConfigurationT,
        recipe_ref: str,
        artifact_filename: str,
        telemetry_artifact_filename: str | None = None,
    ) -> ExecutionPlan[ConfigurationT]:
        artifact_filename = _v2_artifact_basename(
            artifact_filename, "artifact_filename"
        )
        if telemetry_artifact_filename is None:
            telemetry_artifact_filename = _default_telemetry_artifact_filename(
                artifact_filename
            )
        else:
            telemetry_artifact_filename = _v2_artifact_basename(
                telemetry_artifact_filename, "telemetry_artifact_filename"
            )
        if artifact_filename == telemetry_artifact_filename:
            raise ValueError(
                "artifact_filename and telemetry_artifact_filename must differ"
            )
        plan_hash = execution_plan_hash(
            config.as_dict(), artifact_filename, telemetry_artifact_filename
        )
        return cls(
            config,
            recipe_ref,
            artifact_filename,
            telemetry_artifact_filename,
            plan_hash,
        )


__all__ = [
    "ExecutionOverrides",
    "WarmupStagePlan",
    "ExecutableConfigurationSnapshot",
    "ExecutionConfiguration",
    "ExecutionPlan",
    "canonical_json",
    "execution_config_hash",
    "execution_plan_hash",
    "legacy_execution_plan_hash",
]
