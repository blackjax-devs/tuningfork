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
"""Descriptor for adjusted Microcanonical Langevin Monte Carlo.

Recipes resolve ``(step_size, L)``; code generation translates these values to
the upstream ``integration_steps_params`` convention. The default isokinetic
integrator evaluates two gradients per integration step, and upstream init
does not require an RNG key. Dedicated adaptation uses
``blackjax.adjusted_mclmc_find_L_and_step_size`` with target 0.9.
"""

import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]

ENTRY = BaseMethod(
    name="adjusted_mclmc",
    family="mcmc",
    grad_count_per_step=lambda info: jnp.asarray(2 * info.num_integration_steps),
    grad_count_convention="2 × info.num_integration_steps",
    default_hp_space=(
        HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
        HyperparamSpace("L", "loguniform", low=0.1, high=100.0),
    ),
    needs_mass_matrix=True,
    target_acceptance_rate=0.9,
    notes=(
        "Metropolis-adjusted MCLMC (adjusted_mclmc). "
        "Codegen translates (step_size, L) to integration_steps_params=(N,) "
        "where N = max(1, round(L / step_size)) for upstream BlackJAX. "
        "grad_count_per_step = 2 * info.num_integration_steps. "
        "init: blackjax.adjusted_mclmc.init(position, logdensity_fn) — no rng_key. "
        "Dedicated adaptation: blackjax.adjusted_mclmc_find_L_and_step_size with target=0.9. "
        "needs_mass_matrix=True: IMM from diagonal_preconditioning."
    ),
)
