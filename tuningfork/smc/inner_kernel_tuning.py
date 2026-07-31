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
"""inner_kernel_tuning descriptor for generated SMC execution plans."""

from __future__ import annotations

from tuningfork.base_method._base import HyperparamSpace
from tuningfork.smc._base import SMCMethod

__all__ = ["ENTRY"]


# Inner methods compatible with tempering (MH-based; excludes microcanonical).
_COMPATIBLE_INNER = (
    "rwm",
    "irmh",
    "mala",
    "barker",
    "hmc",
    "nuts",
    "ghmc",
    "dynamic_hmc",
)


ENTRY = SMCMethod(
    name="inner_kernel_tuning",
    family="smc",
    compatible_inner_methods=_COMPATIBLE_INNER,
    default_inner_method="rwm",
    num_particles_default=1000,
    default_hp_space=(
        HyperparamSpace("target_ess", "uniform", low=0.3, high=0.95),
        HyperparamSpace("num_mcmc_steps", "int", low=1, high=50),
    ),
    step_kwargs_schema=(),  # standard step(key, state) signature
    notes=(
        "Inner Kernel Tuning SMC (meta-SMC that adapts inner-kernel parameters). "
        "Wraps blackjax.smc.inner_kernel_tuning.as_top_level_api. At each SMC "
        "step, applies mcmc_parameter_update_fn(rng_key, smc_state, smc_info) "
        "to compute new per-step inner-kernel parameters for the NEXT mutation. "
        "State type: StateWithParameterOverride with _fields = "
        "('sampler_state', 'parameter_override'). Particles live at "
        "state.sampler_state.particles (not state.particles directly). "
        "Generated source applies the parameter update function to the current "
        "particle state before each mutation. "
        "Inner kernel must be MH-based — MCLMC family excluded (microcanonical "
        "invariance violated by tempering). Default inner: rwm. "
        "num_particles=1000 default. Declared SMC parameters: num_mcmc_steps (int [1, 50]). "
        "Resampling default: systematic."
    ),
)
