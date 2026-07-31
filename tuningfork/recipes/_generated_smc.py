"""Strict loading and pure evaluation of generated SMC particle artifacts.

The archive is deliberately small and namespaced: ``particle__<site>`` arrays,
``smc__weights``, ``smc__lambda`` and ``smc__ess`` history, and optional
``inner__<name>`` final parameters.  This module never executes a sampler or
writes a recipe.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from tuningfork.base_method import BASE_METHODS
from tuningfork.calibration.smc_gate import SMCGateVerdict, smc_gate
from tuningfork.model import MODELS
from tuningfork.recipes._execution_plan import _freeze
from tuningfork.recipes._ground_truth_reference import GroundTruthReference
from tuningfork.recipes._ground_truth_reference import align_ground_truth
from tuningfork.smc import SMC_METHODS


def _ro(value: Any) -> np.ndarray:
    result = np.asarray(value)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GeneratedSMCArtifact:
    particles: Mapping[str, np.ndarray]
    weights: np.ndarray
    lambda_history: np.ndarray
    ess_history: np.ndarray
    final_inner_params: Mapping[str, np.ndarray]
    source: str
    num_particles: int
    smc_method_name: str
    inner_method_name: str
    model_name: str

    @property
    def lambda_final(self) -> float:
        return float(self.lambda_history[-1])


@dataclass(frozen=True)
class GeneratedSMCEvaluation:
    gate: SMCGateVerdict
    headline_metric: float | None
    total_cost: int
    lambda_final: float
    history: Mapping[str, np.ndarray]
    ground_truth_identity: Mapping[str, Any]


def _config(config: Any) -> Mapping[str, Any]:
    value = getattr(config, "executable_config", None)
    if value is None:
        if isinstance(config, Mapping):
            value = config
        else:
            as_dict = getattr(config, "as_dict", None)
            if not callable(as_dict):
                raise TypeError(
                    "config must be an execution config, manifest, or mapping"
                )
            value = as_dict()
    if not isinstance(value, Mapping):
        raise TypeError("config must be an ExecutionManifest-like mapping")
    if value.get("execution_family") != "smc":
        raise ValueError("config execution_family must be 'smc'")
    return value


def _required_text(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"config {key} must be a non-empty string")
    return value


def _positive_int(
    config: Mapping[str, Any], key: str, default: int | None = None
) -> int:
    value = config.get(key, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"config {key} must be a positive integer")
    return int(value)


def load_generated_smc_artifact(path: str | Path, config: Any) -> GeneratedSMCArtifact:
    """Load and validate one generated SMC NPZ archive, returning immutable arrays."""
    cfg = _config(config)
    model = _required_text(cfg, "model_name")
    smc_name = _required_text(cfg, "smc_method_name")
    inner = _required_text(cfg, "inner_method_name")
    if model not in MODELS:
        raise ValueError(f"config model_name is not registered: {model!r}")
    if smc_name not in SMC_METHODS:
        raise ValueError(f"config smc_method_name is not registered: {smc_name!r}")
    if inner not in BASE_METHODS:
        raise ValueError(
            f"config inner_method_name is not a registered method: {inner!r}"
        )
    if inner not in SMC_METHODS[smc_name].compatible_inner_methods:
        raise ValueError(
            f"config inner method {inner!r} is incompatible with {smc_name!r}"
        )
    n = _positive_int(cfg, "num_particles")
    particles: dict[str, np.ndarray] = {}
    params: dict[str, np.ndarray] = {}
    source = str(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            names = set(archive.files)
            required = {"smc__weights", "smc__lambda", "smc__ess"}
            missing = required - names
            if missing:
                raise ValueError(
                    f"SMC artifact missing required arrays: {sorted(missing)!r}"
                )
            for name in archive.files:
                value = np.asarray(archive[name])
                if name.startswith("particle__"):
                    site = name.removeprefix("particle__")
                    if not site or site in particles:
                        raise ValueError(
                            f"invalid or duplicate particle site: {name!r}"
                        )
                    if value.ndim < 1 or value.shape[0] != n:
                        raise ValueError(
                            f"particle {name!r} must have leading shape ({n},)"
                        )
                    if not np.issubdtype(value.dtype, np.number) or np.issubdtype(
                        value.dtype, np.complexfloating
                    ):
                        raise ValueError(f"particle {name!r} must be real numeric")
                    if not np.all(np.isfinite(value)):
                        raise ValueError(
                            f"particle {name!r} contains non-finite values"
                        )
                    particles[site] = _ro(value)
                elif name.startswith("inner__"):
                    key = name.removeprefix("inner__")
                    if not key or key in params:
                        raise ValueError(
                            f"invalid or duplicate inner parameter: {name!r}"
                        )
                    if (
                        not np.issubdtype(value.dtype, np.number)
                        or np.issubdtype(value.dtype, np.complexfloating)
                        or not np.all(np.isfinite(value))
                    ):
                        raise ValueError(
                            f"inner parameter {name!r} must be finite real numeric"
                        )
                    params[key] = _ro(value)
                elif name not in required:
                    raise ValueError(f"unsupported SMC artifact array: {name!r}")
            if not particles:
                raise ValueError("SMC artifact contains no particle sites")
            weights = np.asarray(archive["smc__weights"])
            lambdas = np.asarray(archive["smc__lambda"])
            ess = np.asarray(archive["smc__ess"])
    except (OSError, ValueError):
        raise
    except Exception as exc:
        raise ValueError(f"could not read SMC artifact {source}: {exc}") from exc
    for label, array in (("weights", weights), ("lambda", lambdas), ("ess", ess)):
        if (
            not np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.complexfloating)
            or not np.all(np.isfinite(array))
        ):
            raise ValueError(f"SMC {label} history must be finite real numeric")
    if weights.ndim != 1 or weights.shape != (n,):
        raise ValueError(f"SMC weights must have shape ({n},)")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-6):
        raise ValueError("SMC weights must be nonnegative and normalized")
    if (
        lambdas.ndim != 1
        or ess.ndim != 1
        or lambdas.shape != ess.shape
        or not lambdas.size
    ):
        raise ValueError(
            "SMC lambda and ESS histories must be non-empty, matching 1-D arrays"
        )
    if np.any(np.diff(lambdas) < 0):
        raise ValueError("SMC lambda history must be monotone nondecreasing")
    lambda_tolerance = (
        8 * np.finfo(lambdas.dtype).eps
        if np.issubdtype(lambdas.dtype, np.floating)
        else 0.0
    )
    if np.any(lambdas < -lambda_tolerance) or np.any(
        lambdas > 1.0 + lambda_tolerance
    ):
        raise ValueError("SMC lambda history must stay within [0, 1]")
    if np.any(ess <= 0):
        raise ValueError("SMC ESS history must be strictly positive")
    if np.any(ess > n * (1.0 + 1e-6)):
        raise ValueError("SMC ESS history cannot exceed the particle count")
    return GeneratedSMCArtifact(
        MappingProxyType(particles),
        _ro(weights),
        _ro(lambdas),
        _ro(ess),
        MappingProxyType(params),
        source,
        n,
        smc_name,
        inner,
        model,
    )


def evaluate_generated_smc(
    artifact: GeneratedSMCArtifact,
    config: Any,
    reference: GroundTruthReference,
) -> GeneratedSMCEvaluation:
    """Evaluate a validated artifact using ``smc_gate`` and recorded cost history."""
    if isinstance(artifact, (str, Path)):
        artifact = load_generated_smc_artifact(artifact, config)
    if not isinstance(artifact, GeneratedSMCArtifact):
        raise TypeError("artifact must be GeneratedSMCArtifact or an NPZ path")
    cfg = _config(config)
    configured_smc = _required_text(cfg, "smc_method_name")
    if (
        artifact.model_name != cfg.get("model_name")
        or artifact.smc_method_name != configured_smc
        or artifact.inner_method_name != cfg.get("inner_method_name")
    ):
        raise ValueError("artifact identity does not match execution config")
    if not isinstance(reference, GroundTruthReference):
        raise TypeError("reference must be GroundTruthReference")
    if reference.model_name != artifact.model_name:
        raise ValueError("ground-truth model does not match generated SMC artifact")
    aligned = align_ground_truth(reference, artifact.particles)
    summary = {
        name: {"mean": stats["mean"], "std": stats["std"]}
        for name, stats in aligned.items()
    }
    gate = smc_gate(
        dict(artifact.particles),
        artifact.weights,
        summary,
        model_name=artifact.model_name,
        num_particles=artifact.num_particles,
        lambda_final=artifact.lambda_final,
    )
    steps = int(artifact.lambda_history.size)
    smc_params = cfg.get("smc_params")
    if not isinstance(smc_params, Mapping):
        raise ValueError("config smc_params must be a mapping")
    mcmc = _positive_int(smc_params, "num_mcmc_steps", 10)
    if artifact.inner_method_name == "rwm":
        cost_per_inner_step = 1
    elif artifact.inner_method_name == "hmc":
        cost_per_inner_step = _positive_int(
            smc_params, "num_integration_steps", 10
        )
    else:
        raise ValueError(
            "exact SMC cost accounting is not implemented for inner method "
            f"{artifact.inner_method_name!r}"
        )
    cost = artifact.num_particles * steps * mcmc * cost_per_inner_step
    headline = (
        None if gate.particle_ess is None or cost <= 0 else gate.particle_ess / cost
    )
    return GeneratedSMCEvaluation(
        gate,
        headline,
        cost,
        artifact.lambda_final,
        MappingProxyType(
            {"lambda": artifact.lambda_history, "ess": artifact.ess_history}
        ),
        _freeze(copy.deepcopy(reference.identity)),
    )


load_generated_smc = load_generated_smc_artifact

__all__ = [
    "GeneratedSMCArtifact",
    "GeneratedSMCEvaluation",
    "load_generated_smc_artifact",
    "load_generated_smc",
    "evaluate_generated_smc",
]
