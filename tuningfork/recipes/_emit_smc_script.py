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
"""Emit a standalone, manifest-bound SMC sampling program."""

from __future__ import annotations

from collections.abc import Mapping
from string import Template
from typing import TYPE_CHECKING, Any

from tuningfork._version import __version__
from tuningfork.recipes._execution_manifest import ExecutionManifest
from tuningfork.recipes._smc_execution_plan import resolve_smc_execution_plan
from tuningfork.recipes._smc_execution_telemetry import SMC_TELEMETRY_SCHEMA

if TYPE_CHECKING:
    from tuningfork.recipes._base_smc import SMCRecipe

_SUPPORTED_ROUTES = frozenset(
    {
        ("adaptive_tempered_smc", "rwm"),
        ("inner_kernel_tuning", "hmc"),
    }
)
_ALLOWED_SMC_PARAMS = {
    ("adaptive_tempered_smc", "rwm"): frozenset({"target_ess", "num_mcmc_steps"}),
    ("inner_kernel_tuning", "hmc"): frozenset(
        {"target_ess", "num_mcmc_steps", "num_integration_steps"}
    ),
}
_ALLOWED_INNER_PARAMS = {
    ("adaptive_tempered_smc", "rwm"): frozenset({"sigma"}),
    ("inner_kernel_tuning", "hmc"): frozenset({"step_size", "inverse_mass_matrix"}),
}


def _validate_codegen_route(config: Mapping[str, Any]) -> tuple[str, str]:
    route = (str(config["smc_method_name"]), str(config["inner_method_name"]))
    if route not in _SUPPORTED_ROUTES:
        raise NotImplementedError(
            "SMC code generation does not implement route "
            f"{route[0]!r} with {route[1]!r}; add the missing generated call shape"
        )
    smc_params = config["smc_params"]
    inner_params = config["inner_params_init"]
    if not isinstance(smc_params, Mapping) or not isinstance(inner_params, Mapping):
        raise TypeError("resolved SMC and inner parameters must be mappings")
    unknown_smc = set(smc_params) - _ALLOWED_SMC_PARAMS[route]
    unknown_inner = set(inner_params) - _ALLOWED_INNER_PARAMS[route]
    if unknown_smc:
        raise ValueError(
            f"SMC code generation would ignore parameters: {sorted(unknown_smc)!r}"
        )
    if unknown_inner:
        raise ValueError(
            f"SMC code generation would ignore inner parameters: "
            f"{sorted(unknown_inner)!r}"
        )
    strategy = config["parameter_update_strategy"]
    if route[0] == "adaptive_tempered_smc" and strategy != "none":
        raise ValueError(
            "adaptive_tempered_smc has no parameter-update layer; "
            "parameter_update_strategy must be 'none'"
        )
    return route


def emit_smc_script(recipe: SMCRecipe) -> str:
    """Render the exact supported SMC route selected by ``recipe``."""
    plan = resolve_smc_execution_plan(recipe)
    route = _validate_codegen_route(plan.config.as_dict())
    manifest = ExecutionManifest.from_plan(plan, generator_version=__version__)
    lifecycle = (
        _RWM_ADAPTIVE_LIFECYCLE
        if route == ("adaptive_tempered_smc", "rwm")
        else _HMC_TUNING_LIFECYCLE
    )
    lifecycle = lifecycle.replace(
        "$parameter_update",
        _parameter_update_snippet(
            plan.config.parameter_update_strategy,
            plan.config.parameter_update_strategy_kwargs,
        ),
    )
    x64_line = (
        'jax.config.update("jax_enable_x64", True)' if plan.config.requires_x64 else ""
    )
    return _PROGRAM.substitute(
        manifest_literal=repr(manifest.to_json()),
        lifecycle=lifecycle,
        telemetry_schema=repr(SMC_TELEMETRY_SCHEMA),
        x64_line=x64_line,
    )


_RWM_ADAPTIVE_LIFECYCLE = r"""
from functools import partial
from blackjax.base import SamplingAlgorithm
import blackjax.mcmc.random_walk as _random_walk

_sigma = float(_cfg["inner_params_init"]["sigma"])

def _proposal(rng_key, position):
    flat, unravel = ravel_pytree(position)
    noise = jax.random.normal(rng_key, flat.shape) * _sigma
    return unravel(flat + noise)

_raw_rwm = _random_walk.build_rmh()
_inner = SamplingAlgorithm(
    init=_random_walk.init,
    step=partial(_raw_rwm, transition_generator=_proposal),
)
_algorithm = blackjax.adaptive_tempered_smc(
    logprior_fn=_logprior,
    loglikelihood_fn=_loglikelihood,
    mcmc_step_fn=_inner.step,
    mcmc_init_fn=_inner.init,
    mcmc_parameters={},
    resampling_fn=blackjax.smc.resampling.systematic,
    target_ess=float(_cfg["smc_params"]["target_ess"]),
    num_mcmc_steps=int(_cfg["smc_params"]["num_mcmc_steps"]),
)
_state = _algorithm.init(jax.tree.map(jnp.asarray, _initial_particles))
for _ in range(_max_steps):
    _run_key, _step_key = jax.random.split(_run_key)
    _state, _step_info = _algorithm.step(_step_key, _state)
    _record_history(_state)
    if float(np.asarray(_state.tempering_param)) >= 1.0:
        break
_sampler_state = _state
_final_inner_params = {}
"""


def _parameter_update_snippet(strategy: str, kwargs: Mapping[str, Any]) -> str:
    """Select standalone update code for a validated strategy."""
    target = repr(float(kwargs.get("target_acceptance", 0.65)))
    snippets = {
        "none": """
def _update_none(rng_key, smc_state, smc_info):
    return dict(_inner_params)

_update = _update_none
""",
        "step_size_from_acceptance_rate": f"""
from blackjax.smc.tuning.from_kernel_info import update_scale_from_acceptance_rate
_target_acceptance = {target}
def _update_step(rng_key, smc_state, smc_info):
    result = dict(_inner_params)
    result["step_size"] = update_scale_from_acceptance_rate(
        _inner_params["step_size"], smc_info.update_info.acceptance_rate,
        target_acceptance_rate=_target_acceptance,
    )
    return result
_update = _update_step
""",
        "imm_from_particles": """
from blackjax.smc.tuning.from_particles import particles_as_rows
def _update_imm(rng_key, smc_state, smc_info):
    result = dict(_inner_params)
    variance = jnp.maximum(
        jnp.var(particles_as_rows(smc_state.particles), axis=0), 1e-6
    )
    shape = jnp.asarray(_inner_params["inverse_mass_matrix"]).shape
    result["inverse_mass_matrix"] = (
        variance if len(shape) == 1 else jnp.broadcast_to(variance, shape)
    )
    return result
_update = _update_imm
""",
        "step_size_and_imm_from_particles": f"""
from blackjax.smc.tuning.from_kernel_info import update_scale_from_acceptance_rate
from blackjax.smc.tuning.from_particles import particles_as_rows
_target_acceptance = {target}
def _update_combined(rng_key, smc_state, smc_info):
    result = dict(_inner_params)
    result["step_size"] = update_scale_from_acceptance_rate(
        _inner_params["step_size"], smc_info.update_info.acceptance_rate,
        target_acceptance_rate=_target_acceptance,
    )
    variance = jnp.maximum(
        jnp.var(particles_as_rows(smc_state.particles), axis=0), 1e-6
    )
    shape = jnp.asarray(_inner_params["inverse_mass_matrix"]).shape
    result["inverse_mass_matrix"] = (
        variance if len(shape) == 1 else jnp.broadcast_to(variance, shape)
    )
    return result
_update = _update_combined
""",
    }
    return snippets[strategy]


_HMC_TUNING_LIFECYCLE = r"""
from functools import partial
from blackjax.base import SamplingAlgorithm

_model_dim = int(sum(np.asarray(x).size for x in jax.tree.leaves(_init_position)))
_step_size = jnp.asarray(_cfg["inner_params_init"]["step_size"])
if _step_size.ndim == 0:
    _step_size = jnp.full((_num_particles,), _step_size)
_inverse_mass_matrix = jnp.asarray(
    _cfg["inner_params_init"]["inverse_mass_matrix"]
)
if _inverse_mass_matrix.ndim == 1:
    _inverse_mass_matrix = jnp.tile(
        _inverse_mass_matrix[None, :], (_num_particles, 1)
    )
if _step_size.shape != (_num_particles,):
    raise ValueError("HMC step_size must be scalar or one value per particle")
if _inverse_mass_matrix.shape != (_num_particles, _model_dim):
    raise ValueError(
        "HMC inverse_mass_matrix must be one diagonal vector or one per particle"
    )
_inner_params = {
    "step_size": _step_size,
    "inverse_mass_matrix": _inverse_mass_matrix,
}
_raw_hmc = blackjax.hmc.build_kernel()
_inner = SamplingAlgorithm(
    init=blackjax.hmc.init,
    step=partial(
        _raw_hmc,
        num_integration_steps=int(
            _cfg["smc_params"]["num_integration_steps"]
        ),
    ),
)
$parameter_update
_algorithm = blackjax.smc.inner_kernel_tuning.as_top_level_api(
    smc_algorithm=blackjax.adaptive_tempered_smc,
    logprior_fn=_logprior,
    loglikelihood_fn=_loglikelihood,
    mcmc_step_fn=_inner.step,
    mcmc_init_fn=_inner.init,
    resampling_fn=blackjax.smc.resampling.systematic,
    mcmc_parameter_update_fn=_update,
    initial_parameter_value=_inner_params,
    num_mcmc_steps=int(_cfg["smc_params"]["num_mcmc_steps"]),
    target_ess=float(_cfg["smc_params"]["target_ess"]),
)
_state = _algorithm.init(jax.tree.map(jnp.asarray, _initial_particles))
for _ in range(_max_steps):
    _run_key, _step_key = jax.random.split(_run_key)
    _state, _step_info = _algorithm.step(_step_key, _state)
    _record_history(_state.sampler_state)
    if float(np.asarray(_state.sampler_state.tempering_param)) >= 1.0:
        break
_sampler_state = _state.sampler_state
_final_inner_params = _state.parameter_override
"""


_PROGRAM = Template(
    r'''#!/usr/bin/env python3
"""Standalone SMC program generated from a versioned execution manifest."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
import time

import blackjax
import jax
$x64_line
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_prior_sample_fn, build_smc_logfns

EXECUTION_MANIFEST_JSON = $manifest_literal
_manifest = json.loads(EXECUTION_MANIFEST_JSON)
_cfg = _manifest["executable_config"]
_artifact_name = _manifest["normalized_plan"]["artifact_filename"]
_telemetry_name = _manifest["normalized_plan"]["telemetry_artifact_filename"]
_num_particles = int(_cfg["num_particles"])
_max_steps = int(_cfg["max_steps"])
_total_t0 = time.perf_counter()
_initialization_t0 = time.perf_counter()

_model = MODELS[_cfg["model_name"]]
_master_key = jax.random.key(int(_cfg["seed"]))
_logfns_key, _particles_key, _run_key = jax.random.split(_master_key, 3)
_init_position, _logprior, _loglikelihood, _postprocess = build_smc_logfns(
    _logfns_key, _model
)
_prior_sample_fn = build_prior_sample_fn(_model)
_initial_particles = _prior_sample_fn(_particles_key, _num_particles)
_initialization_seconds = time.perf_counter() - _initialization_t0

_lambda_history = []
_ess_history = []

def _record_history(sampler_state):
    weights = np.asarray(sampler_state.weights, dtype=float)
    _lambda_history.append(float(np.asarray(sampler_state.tempering_param)))
    _ess_history.append(float(1.0 / np.sum(weights ** 2)))

_sampling_t0 = time.perf_counter()
$lifecycle
jax.block_until_ready(_sampler_state)
_sampling_seconds = time.perf_counter() - _sampling_t0

if not isinstance(_sampler_state.particles, Mapping):
    raise TypeError("generated SMC particles must be a site mapping")
_output = {
    f"particle__{name}": np.asarray(value)
    for name, value in _sampler_state.particles.items()
}
_output["smc__weights"] = np.asarray(_sampler_state.weights)
_output["smc__lambda"] = np.asarray(_lambda_history)
_output["smc__ess"] = np.asarray(_ess_history)
for _name, _value in _final_inner_params.items():
    _output[f"inner__{_name}"] = np.asarray(_value)
np.savez(Path(_artifact_name), **_output)

_total_seconds = time.perf_counter() - _total_t0
_telemetry = {
    "schema": $telemetry_schema,
    "plan_hash": _manifest["plan_hash"],
    "executable_config_hash": _manifest["executable_config_hash"],
    "draws_artifact": _artifact_name,
    "num_particles": _num_particles,
    "num_smc_steps": len(_lambda_history),
    "lambda_final": (
        None if not _lambda_history else _lambda_history[-1]
    ),
    "timing_seconds": {
        "initialization": _initialization_seconds,
        "sampling": _sampling_seconds,
        "total": _total_seconds,
    },
}
Path(_telemetry_name).write_text(
    json.dumps(_telemetry, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
print(
    "TUNINGFORK_TIMINGS "
    + json.dumps(
        {
            "warmup_seconds": _initialization_seconds,
            "sampling_seconds": _sampling_seconds,
            "total_seconds": _total_seconds,
        },
        sort_keys=True,
        allow_nan=False,
    )
)
print("DONE")
'''
)


emit_smc_source = emit_smc_script

__all__ = ["SMC_TELEMETRY_SCHEMA", "emit_smc_script", "emit_smc_source"]
