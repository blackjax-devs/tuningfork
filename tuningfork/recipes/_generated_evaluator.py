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

"""Pure validation and post-launch evaluation for generated artifacts."""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import numpy as np
from blackjax.mcmc.metrics import LowRankInverseMassMatrix

from tuningfork.base_method import BASE_METHODS
from tuningfork.metrics.grad_counter import total_grad_evals
from tuningfork.recipes._execution_manifest import ExecutionManifest
from tuningfork.recipes._execution_telemetry import ExecutionTelemetry
from tuningfork.recipes._sample_stats import SAMPLE_STAT_PREFIX, validate_sample_stats


@dataclass(frozen=True)
class GeneratedRunData:
    """Validated arrays and provenance from one generated ``.npz`` artifact."""

    positions: Mapping[str, np.ndarray]
    chain_stats: Mapping[str, np.ndarray]
    infos: Any
    source: str
    num_chains: int
    num_samples: int
    base_method_name: str


@dataclass(frozen=True)
class GeometryResult:
    geometry: Mapping[str, Any] | None
    source: str
    reason: str | None = None


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    result.setflags(write=False)
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_freeze(v) for v in value))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _field_name(name: str) -> str:
    field = (
        name[len(SAMPLE_STAT_PREFIX) :] if name.startswith(SAMPLE_STAT_PREFIX) else name
    )
    if not field or not field.isidentifier() or field.startswith("__"):
        raise ValueError(f"invalid statistic name: {name!r}")
    return field


def _is_real_array(value: np.ndarray) -> bool:
    return np.issubdtype(value.dtype, np.bool_) or (
        np.issubdtype(value.dtype, np.number)
        and not np.issubdtype(value.dtype, np.complexfloating)
    )


def load_generated_artifact(
    path: str | Path, manifest: ExecutionManifest
) -> GeneratedRunData:
    """Load and strictly validate a generated draws artifact against ``manifest``."""
    if not isinstance(manifest, ExecutionManifest):
        raise TypeError("manifest must be an ExecutionManifest")
    source = str(path)
    expected = str(manifest.normalized_plan["artifact_filename"])
    if Path(source).name != Path(expected).name:
        raise ValueError("artifact filename does not match manifest")
    expected_chains = manifest.executable_config["num_chains"]
    expected_samples = manifest.executable_config["num_samples"]
    base_method_name = manifest.executable_config.get("base_method_name")
    if not isinstance(base_method_name, str) or base_method_name not in BASE_METHODS:
        raise ValueError("manifest base_method_name must name a registered method")
    if (
        not isinstance(expected_chains, int)
        or isinstance(expected_chains, bool)
        or expected_chains <= 0
    ):
        raise ValueError("manifest num_chains must be positive")
    if (
        not isinstance(expected_samples, int)
        or isinstance(expected_samples, bool)
        or expected_samples <= 0
    ):
        raise ValueError("manifest num_samples must be positive")
    positions: dict[str, np.ndarray] = {}
    stats: dict[str, np.ndarray] = {}
    with np.load(source, allow_pickle=False) as loaded:
        for name in loaded.files:
            value = np.asarray(loaded[name])
            if name.startswith(SAMPLE_STAT_PREFIX):
                field = _field_name(name)
                if field in stats:
                    raise ValueError(f"duplicate statistic name: {field!r}")
                if value.ndim < 2 or value.shape[:2] != (
                    expected_chains,
                    expected_samples,
                ):
                    raise ValueError(
                        f"statistic {name!r} must have shape (num_chains, num_samples)"
                    )
                if not _is_real_array(value):
                    raise ValueError(f"statistic {name!r} must be real numeric")
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"statistic {name!r} contains non-finite values")
                stats[field] = _readonly(value)
            else:
                if not name:
                    raise ValueError(f"invalid position name: {name!r}")
                if value.ndim < 2 or value.shape[:2] != (
                    expected_chains,
                    expected_samples,
                ):
                    raise ValueError(f"position {name!r} has incorrect leading shape")
                if not _is_real_array(value):
                    raise ValueError(f"position {name!r} must be real numeric")
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"position {name!r} contains non-finite values")
                positions[name] = _readonly(value)
    if not positions:
        raise ValueError("artifact contains no position arrays")
    validate_sample_stats(stats, base_method_name)
    fields = sorted(stats)
    info_type: Any = namedtuple(  # type: ignore[misc]
        "GeneratedInfo", fields or ["dummy"]
    )
    if fields:
        infos = info_type(*(jnp.asarray(stats[k]) for k in fields))
    else:
        infos = info_type(  # type: ignore[call-arg]
            jnp.zeros((expected_chains, expected_samples), dtype=jnp.int32)
        )
    return GeneratedRunData(
        MappingProxyType(positions),
        MappingProxyType(stats),
        infos,
        source,
        expected_chains,
        expected_samples,
        base_method_name,
    )


def sampling_grad_evals(run_data: GeneratedRunData) -> int:
    """Compute sampling gradient evaluations from validated chain statistics."""
    if not isinstance(run_data, GeneratedRunData):
        raise TypeError("run_data must be GeneratedRunData")
    entry = BASE_METHODS[run_data.base_method_name]
    return total_grad_evals(run_data.infos, entry.grad_count_per_step)


def _chain0(value: Any, *, select_chain: bool = True) -> Any:
    if isinstance(value, Mapping):
        if value.get("type") == "low_rank_inverse_mass_matrix":
            sigma = jnp.asarray(value["sigma"])
            U = jnp.asarray(value["U"])
            lam = jnp.asarray(value["lam"])
            if select_chain and sigma.ndim > 1:
                sigma, U, lam = sigma[0], U[0], lam[0]
            return LowRankInverseMassMatrix(
                sigma=sigma,
                U=U,
                lam=lam,
            )
        return {k: _chain0(v, select_chain=select_chain) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return value[0] if select_chain else value
    return value


def chain0_geometry(telemetry: ExecutionTelemetry) -> GeometryResult:
    """Select chain zero from telemetry without inventing unavailable geometry."""
    if not isinstance(telemetry, ExecutionTelemetry):
        raise TypeError("telemetry must be an ExecutionTelemetry")
    if telemetry.geometry_source == "unavailable":
        return GeometryResult(
            None, "unavailable", telemetry.geometry_unavailable_reason
        )
    selected = _chain0(
        telemetry.geometry,
        select_chain=telemetry.geometry_scope == "per_chain",
    )
    return GeometryResult(_freeze(selected), telemetry.geometry_source)


__all__ = [
    "GeneratedRunData",
    "GeometryResult",
    "load_generated_artifact",
    "sampling_grad_evals",
    "chain0_geometry",
]
