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

"""Strict, immutable telemetry emitted by generated executions."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ._execution_manifest import ExecutionManifest
from ._execution_plan import canonical_json

LEGACY_TELEMETRY_SCHEMA = "tuningfork.generated-run-telemetry.v1"
TELEMETRY_SCHEMA = "tuningfork.generated-run-telemetry.v2"
_GEOMETRY_FIELDS = frozenset(
    {"step_size", "inverse_mass_matrix", "L", "alpha", "delta"}
)
_SCALAR_GEOMETRY_FIELDS = frozenset({"step_size", "L", "alpha", "delta"})
_POSITIVE_GEOMETRY_FIELDS = frozenset({"step_size", "L"})
_LEGACY_FIELDS = frozenset(
    {
        "schema",
        "plan_hash",
        "executable_config_hash",
        "draws_artifact",
        "geometry",
        "geometry_source",
        "geometry_scope",
        "geometry_unavailable_reason",
        "fixed",
        "timing_seconds",
        "warmup_grad_evals",
        "warmup_grad_evals_reason",
    }
)
_FIELDS = _LEGACY_FIELDS | {"resolved_step_policy"}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _json_value(value: Any, *, path: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise TypeError(f"{path} mapping keys must be strings")
        return {k: _json_value(v, path=f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [_json_value(v, path=f"{path}[{i}]") for i, v in enumerate(value)]
    raise TypeError(f"{path} is not JSON-safe")


def _numeric_shape(value: Any, *, path: str) -> tuple[int, ...]:
    """Validate a rectangular finite-real JSON array and return its shape."""
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return ()
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a finite real scalar or non-empty array")
    child_shapes = [_numeric_shape(child, path=f"{path}[]") for child in value]
    if any(shape != child_shapes[0] for shape in child_shapes[1:]):
        raise ValueError(f"{path} must be a rectangular array")
    return (len(value), *child_shapes[0])


def _numeric_leaves(value: Any):
    if isinstance(value, list):
        for child in value:
            yield from _numeric_leaves(child)
    else:
        yield value


def _validate_low_rank(
    marker: Mapping[str, Any], *, expected_chains: int | None = None
) -> None:
    if set(marker) != {"type", "sigma", "U", "lam"}:
        raise ValueError("low-rank geometry marker has unsupported fields")
    if marker["type"] != "low_rank_inverse_mass_matrix":
        raise ValueError("unsupported geometry marker type")
    sigma = marker["sigma"]
    U, lam = marker["U"], marker["lam"]
    if not isinstance(sigma, list) or not sigma:
        raise ValueError("low-rank sigma must be a non-empty vector")
    batched = isinstance(sigma[0], list)
    if batched:
        if (
            not all(isinstance(x, list) and x for x in sigma)
            or not isinstance(U, list)
            or not isinstance(lam, list)
        ):
            raise ValueError("batched low-rank marker has invalid shapes")
        if len(U) != len(sigma) or len(lam) != len(sigma):
            raise ValueError("batched low-rank marker chains are misaligned")
        if expected_chains is not None and len(sigma) != expected_chains:
            raise ValueError(
                "batched low-rank marker chain count does not match manifest"
            )
        for s, u, l in zip(sigma, U, lam):
            _validate_low_rank({"type": marker["type"], "sigma": s, "U": u, "lam": l})
        return
    if not all(
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(x)
        and x > 0
        for x in sigma
    ):
        raise ValueError("low-rank sigma must be finite and positive")
    if not isinstance(U, list) or not isinstance(lam, list) or not U or not lam:
        raise ValueError("low-rank U and lam must be non-empty arrays")
    if not all(isinstance(row, list) for row in U):
        raise ValueError("low-rank U must be a matrix")
    rows, rank = len(U), len(U[0])
    if len(sigma) != rows or rank > rows:
        raise ValueError("low-rank sigma dimension or rank exceeds U dimension")
    if rank != len(lam) or rank == 0 or any(len(row) != rank for row in U):
        raise ValueError("low-rank U/lam shapes do not match")
    if any(
        not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x)
        for row in U
        for x in row
    ):
        raise ValueError("low-rank U must be finite numeric")
    if any(
        not isinstance(x, (int, float))
        or isinstance(x, bool)
        or not math.isfinite(x)
        or x <= 0
        for x in lam
    ):
        raise ValueError("low-rank lam must be finite and positive")
    for i in range(rank):
        for j in range(rank):
            dot = sum(U[r][i] * U[r][j] for r in range(rows))
            if not math.isclose(
                dot, 1.0 if i == j else 0.0, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise ValueError("low-rank U columns must be orthonormal")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"resolved_step_policy {name} must be a positive integer")
    return value


def _validate_resolved_step_policy(value: Any) -> dict[str, Any] | None:
    """Validate the concrete integration-step distribution used by a run."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("resolved_step_policy must be a mapping or null")
    kind = value.get("kind")
    if kind == "uniform_int":
        if set(value) != {"kind", "low", "high"}:
            raise ValueError("uniform_int resolved_step_policy has invalid fields")
        low = _positive_int("low", value["low"])
        high = _positive_int("high", value["high"])
        if low >= high:
            raise ValueError("uniform_int resolved_step_policy requires low < high")
    elif kind == "empirical":
        if set(value) != {"kind", "values", "weights"}:
            raise ValueError("empirical resolved_step_policy has invalid fields")
        values, weights = value["values"], value["weights"]
        if (
            not isinstance(values, list)
            or not isinstance(weights, list)
            or not values
            or len(values) != len(weights)
        ):
            raise ValueError(
                "empirical resolved_step_policy requires equal non-empty arrays"
            )
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in values
        ) or values != sorted(set(values)):
            raise ValueError(
                "empirical resolved_step_policy values must be sorted positive integers"
            )
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or item < 0
            for item in weights
        ) or not math.isclose(sum(weights), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                "empirical resolved_step_policy weights must be normalized"
            )
    elif kind == "poisson":
        if set(value) != {"kind", "lam", "low", "high"}:
            raise ValueError("poisson resolved_step_policy has invalid fields")
        lam = value["lam"]
        if (
            isinstance(lam, bool)
            or not isinstance(lam, (int, float))
            or not math.isfinite(lam)
            or lam < 0
        ):
            raise ValueError(
                "poisson resolved_step_policy lam must be finite and nonnegative"
            )
        low = _positive_int("low", value["low"])
        high = value["high"]
        if high is not None and _positive_int("high", high) <= low:
            raise ValueError("poisson resolved_step_policy requires low < high")
    elif kind == "log_uniform_int":
        if set(value) != {"kind", "low", "high"}:
            raise ValueError("log_uniform_int resolved_step_policy has invalid fields")
        low = _positive_int("low", value["low"])
        high = _positive_int("high", value["high"])
        if low >= high:
            raise ValueError("log_uniform_int resolved_step_policy requires low < high")
    elif kind == "pow2_choice":
        if set(value) != {"kind", "options"}:
            raise ValueError("pow2_choice resolved_step_policy has invalid fields")
        options = value["options"]
        if (
            not isinstance(options, list)
            or not options
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item <= 0
                or item & (item - 1)
                for item in options
            )
        ):
            raise ValueError(
                "pow2_choice resolved_step_policy requires positive powers of two"
            )
    else:
        raise ValueError(f"unsupported resolved_step_policy kind: {kind!r}")
    return value


@dataclass(frozen=True)
class ExecutionTelemetry:
    schema: str
    plan_hash: str
    executable_config_hash: str
    draws_artifact: str
    geometry: Mapping[str, Any]
    geometry_source: str
    geometry_scope: str | None
    geometry_unavailable_reason: str | None
    fixed: Mapping[str, Any]
    timing_seconds: Mapping[str, float]
    warmup_grad_evals: int | None
    warmup_grad_evals_reason: str
    resolved_step_policy: Mapping[str, Any] | None

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], manifest: ExecutionManifest
    ) -> ExecutionTelemetry:
        if not isinstance(manifest, ExecutionManifest):
            raise TypeError("manifest must be an ExecutionManifest")
        if not isinstance(data, Mapping):
            raise TypeError("telemetry data must be a mapping")
        schema = data.get("schema")
        expected_fields = (
            _LEGACY_FIELDS if schema == LEGACY_TELEMETRY_SCHEMA else _FIELDS
        )
        if set(data) != expected_fields:
            raise ValueError("telemetry has unknown or missing fields")
        normalized = _json_value(dict(data))
        if normalized["schema"] not in {
            LEGACY_TELEMETRY_SCHEMA,
            TELEMETRY_SCHEMA,
        }:
            raise ValueError("unsupported telemetry schema")
        if (
            normalized["plan_hash"] != manifest.plan_hash
            or normalized["executable_config_hash"] != manifest.executable_config_hash
        ):
            raise ValueError("telemetry hash does not match manifest")
        expected_draws = manifest.normalized_plan["artifact_filename"]
        draws = normalized["draws_artifact"]
        if not isinstance(draws, str) or draws != expected_draws:
            raise ValueError("draws_artifact does not match manifest")
        geometry, fixed = normalized["geometry"], normalized["fixed"]
        if not isinstance(geometry, dict) or not isinstance(fixed, dict):
            raise ValueError("geometry and fixed must be mappings")
        source = normalized["geometry_source"]
        scope = normalized["geometry_scope"]
        reason = normalized["geometry_unavailable_reason"]
        if source not in {"adapted", "pinned", "unavailable"}:
            raise ValueError("geometry_source must be adapted, pinned, or unavailable")
        if bool(geometry):
            if source == "unavailable" or scope not in {"shared", "per_chain"}:
                raise ValueError("available geometry must declare its source and scope")
            if reason is not None:
                raise ValueError("available geometry cannot have an unavailable reason")
            if any(value is None for value in geometry.values()):
                raise ValueError("available geometry cannot contain null values")
        elif (
            source != "unavailable"
            or scope is not None
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError("geometry availability reason is inconsistent")
        if set(geometry) - _GEOMETRY_FIELDS:
            raise ValueError("geometry has unsupported fields")
        expected_chains = manifest.executable_config["num_chains"]
        if (
            isinstance(expected_chains, bool)
            or not isinstance(expected_chains, int)
            or expected_chains <= 0
        ):
            raise ValueError("manifest num_chains must be a positive integer")
        if scope == "per_chain":
            for name, value in geometry.items():
                if (
                    isinstance(value, dict)
                    and value.get("type") == "low_rank_inverse_mass_matrix"
                ):
                    sigma = value.get("sigma")
                    if (
                        not isinstance(sigma, list)
                        or not sigma
                        or not isinstance(sigma[0], list)
                        or len(sigma) != expected_chains
                    ):
                        raise ValueError(
                            f"per-chain geometry field {name!r} does not match num_chains"
                        )
                elif not isinstance(value, list) or len(value) != expected_chains:
                    raise ValueError(
                        f"per-chain geometry field {name!r} does not match num_chains"
                    )

        for name, value in geometry.items():
            if (
                isinstance(value, dict)
                and value.get("type") == "low_rank_inverse_mass_matrix"
            ):
                if name != "inverse_mass_matrix":
                    raise ValueError(
                        "low-rank geometry is only valid as inverse_mass_matrix"
                    )
                sigma = value.get("sigma")
                if (
                    scope == "shared"
                    and isinstance(sigma, list)
                    and sigma
                    and isinstance(sigma[0], list)
                ):
                    raise ValueError("shared low-rank geometry cannot be batched")
                continue
            shape = _numeric_shape(value, path=f"geometry.{name}")
            if scope == "shared":
                if name in _SCALAR_GEOMETRY_FIELDS and shape:
                    raise ValueError(f"shared geometry field {name!r} must be scalar")
                if name == "inverse_mass_matrix" and len(shape) > 2:
                    raise ValueError(
                        "shared inverse_mass_matrix must be scalar, vector, or matrix"
                    )
            elif scope == "per_chain":
                if not shape or shape[0] != expected_chains:
                    raise ValueError(
                        f"per-chain geometry field {name!r} does not match num_chains"
                    )
                if name in _SCALAR_GEOMETRY_FIELDS and len(shape) != 1:
                    raise ValueError(
                        f"per-chain geometry field {name!r} must contain scalars"
                    )
                if name == "inverse_mass_matrix" and len(shape) > 3:
                    raise ValueError(
                        "per-chain inverse_mass_matrix entries must be scalar, "
                        "vector, or matrix"
                    )
            if name in _POSITIVE_GEOMETRY_FIELDS and any(
                leaf <= 0 for leaf in _numeric_leaves(value)
            ):
                raise ValueError(f"geometry field {name!r} must be positive")

        def _check_markers(value: Any) -> None:
            if isinstance(value, dict):
                if value.get("type") == "low_rank_inverse_mass_matrix":
                    _validate_low_rank(value, expected_chains=expected_chains)
                for child in value.values():
                    _check_markers(child)
            elif isinstance(value, list):
                for child in value:
                    _check_markers(child)

        _check_markers(geometry)
        if set(fixed) - {"num_integration_steps"}:
            raise ValueError("fixed has unsupported fields")
        if "num_integration_steps" in fixed:
            value = fixed["num_integration_steps"]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    "fixed num_integration_steps must be a positive integer"
                )
        timing = normalized["timing_seconds"]
        if not isinstance(timing, dict) or set(timing) != {
            "warmup",
            "sampling",
            "total",
        }:
            raise ValueError("timing_seconds keys must be warmup, sampling, total")
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v < 0
            for v in timing.values()
        ):
            raise ValueError("timing_seconds values must be finite and nonnegative")
        if timing["total"] < timing["warmup"] + timing["sampling"]:
            raise ValueError("total timing must include warmup and sampling")
        count, count_reason = (
            normalized["warmup_grad_evals"],
            normalized["warmup_grad_evals_reason"],
        )
        if not isinstance(count_reason, str) or not count_reason.strip():
            raise ValueError("warmup_grad_evals_reason must be non-empty")
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count < 0
        ):
            raise ValueError("warmup_grad_evals must be a nonnegative integer or null")
        resolved_step_policy = _validate_resolved_step_policy(
            normalized.get("resolved_step_policy")
        )
        return cls(
            normalized["schema"],
            normalized["plan_hash"],
            normalized["executable_config_hash"],
            draws,
            _freeze(geometry),
            source,
            scope,
            reason,
            _freeze(fixed),
            _freeze(timing),
            count,
            count_reason,
            _freeze(resolved_step_policy),
        )

    @classmethod
    def from_json(
        cls, text: str | bytes, manifest: ExecutionManifest
    ) -> ExecutionTelemetry:
        def reject_constant(value: str) -> Any:
            raise ValueError(f"non-finite JSON constant: {value}")

        return cls.from_dict(
            json.loads(
                text,
                object_pairs_hook=_reject_duplicates,
                parse_constant=reject_constant,
            ),
            manifest,
        )

    @classmethod
    def read_bytes(
        cls, payload: bytes, manifest: ExecutionManifest
    ) -> ExecutionTelemetry:
        return cls.from_json(payload, manifest)

    from_bytes = read_bytes

    @classmethod
    def read_path(
        cls, path: str | Path, manifest: ExecutionManifest
    ) -> ExecutionTelemetry:
        return cls.read_bytes(Path(path).read_bytes(), manifest)

    from_path = read_path

    def as_dict(self) -> dict[str, Any]:
        fields = _LEGACY_FIELDS if self.schema == LEGACY_TELEMETRY_SCHEMA else _FIELDS
        return json.loads(canonical_json({k: getattr(self, k) for k in fields}))

    def to_json(self) -> str:
        return canonical_json(self.as_dict())


__all__ = [
    "ExecutionTelemetry",
    "LEGACY_TELEMETRY_SCHEMA",
    "TELEMETRY_SCHEMA",
]
