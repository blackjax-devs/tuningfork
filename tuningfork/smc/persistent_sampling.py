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
"""persistent_sampling descriptor for generated SMC execution plans."""

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
    name="persistent_sampling_smc",
    family="smc",
    compatible_inner_methods=_COMPATIBLE_INNER,
    default_inner_method="rwm",
    num_particles_default=1000,
    default_hp_space=(
        HyperparamSpace("n_schedule", "int", low=5, high=50),
        HyperparamSpace("num_mcmc_steps", "int", low=1, high=50),
    ),
    step_kwargs_schema=("lmbda",),  # step_fn(key, state, lmbda)
    notes=(
        "Persistent Sampling SMC (Karamanis et al. 2025). "
        "Keeps track of all particles from all previous iterations, building a "
        "growing ensemble for more stable posterior and marginal-likelihood "
        "estimation at the cost of higher memory usage. "
        "CRITICAL step-fn divergence: step_fn(rng_key, state, lmbda) requires "
        "an extra 'lmbda' argument (the tempering parameter) at each call; "
        "see step_kwargs_schema. Caller is responsible for the tempering schedule. "
        "CRITICAL memory constraint: 'n_schedule' must be supplied at construction "
        "to pre-allocate arrays of shape (n_schedule + 1, num_particles). "
        "CRITICAL tempering constraint: schedule must START at 0.0 (enforced "
        "upstream); if the supplied schedule also starts at 0.0, the first step "
        "is applied twice. "
        "Inner kernel must be MH-based — MCLMC family excluded (microcanonical "
        "invariance violated by tempering). Default inner: rwm (statistician "
        "verdict: RWM/IRMH first). num_particles=1000 default. "
        "Declared SMC parameters: n_schedule (int [5, 50]) and num_mcmc_steps "
        "(int [1, 50]). Resampling default: systematic. "
        "Generated source binds inner-kernel parameters before constructing the "
        "BlackJAX kernel."
    ),
)
