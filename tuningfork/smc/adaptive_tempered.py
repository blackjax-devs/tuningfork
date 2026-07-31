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
"""adaptive_tempered descriptor for generated SMC execution plans."""

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
    name="adaptive_tempered_smc",
    family="smc",
    compatible_inner_methods=_COMPATIBLE_INNER,
    default_inner_method="rwm",  # statistician verdict: RWM/IRMH first
    num_particles_default=1000,
    default_hp_space=(
        HyperparamSpace("target_ess", "uniform", low=0.3, high=0.95),
        HyperparamSpace("num_mcmc_steps", "int", low=1, high=50),
    ),
    step_kwargs_schema=(),  # standard step(key, state) signature
    notes=(
        "Adaptive Tempered SMC (Del Moral et al. 2006, adapted via target ESS). "
        "Temperature schedule chosen at each step by dichotomy root-solving on "
        "ESS(loglikelihood_fn(particles)) hitting target_ess * num_particles. "
        "Inner kernel must be MH-based — MCLMC family excluded (microcanonical "
        "invariance violated by tempering). Default inner: rwm (statistician "
        "verdict: RWM/IRMH first). num_particles=1000 default. Declared SMC parameters: "
        "target_ess (uniform [0.3, 0.95]) and num_mcmc_steps (int [1, 50]). "
        "Resampling default: systematic. step_fn standard (key, state); no extras. "
        "Inner-kernel parameters are represented declaratively; generated source "
        "binds non-array parameters before constructing the BlackJAX kernel."
    ),
)
