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
"""partial_posteriors descriptor for generated SMC execution plans."""

from __future__ import annotations

from tuningfork.base_method._base import HyperparamSpace
from tuningfork.smc._base import SMCMethod

__all__ = ["ENTRY"]


# Inner methods compatible with data-tempering (MH-based; excludes microcanonical).
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
    name="partial_posteriors_smc",
    family="smc",
    compatible_inner_methods=_COMPATIBLE_INNER,
    default_inner_method="rwm",
    num_particles_default=1000,
    default_hp_space=(HyperparamSpace("num_mcmc_steps", "int", low=1, high=50),),
    step_kwargs_schema=("data_mask",),  # step_fn(key, state, data_mask)
    notes=(
        "Partial Posteriors SMC (data-tempering SMC). "
        "Rather than annealing a temperature, this SMC path progressively includes "
        "more data points by stepping through increasingly complete data masks. "
        "See Section 2.2 of https://arxiv.org/pdf/2007.11936. "
        "API divergence: this route uses a partial-logposterior mapping rather "
        "than a standard prior/likelihood pair. "
        "CRITICAL step-fn divergence: step_fn(rng_key, state, data_mask) requires "
        "an extra 'data_mask' argument at each call; see step_kwargs_schema. "
        "CRITICAL init-fn divergence: init_fn(particles, num_observations) — "
        "num_observations is the total data count for the initial all-zeros mask. "
        "Inner kernel must be MH-based — MCLMC family excluded (microcanonical "
        "invariance violated by tempering). Default inner: rwm. "
        "num_particles=1000 default. Declared SMC parameters: num_mcmc_steps (int [1, 50]). "
        "Resampling default: systematic. "
        "CRITICAL inner-kernel contract: mcmc_parameters dict must contain ONLY JAX "
        "arrays — callable params (e.g. random_step for RWM) must be bound via "
        "functools.partial at build time."
    ),
)
