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
"""Descriptor for dynamic adjusted MCLMC.

Upstream ``blackjax.adjusted_mclmc_dynamic`` samples a random integration-step
count. Recipes resolve ``(step_size, L)`` and code generation emits the
trajectory-length function and average ``max(1, L / step_size)`` parameter.
The default integrator evaluates two gradients per realized step; upstream
init requires an RNG key. Adaptation uses the static adjusted-MCLMC routine.
"""

import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]

ENTRY = BaseMethod(
    name="adjusted_mclmc_dynamic",
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
        "Dynamic Metropolis-adjusted MCLMC. Codegen translates (step_size, L) "
        "to integration_steps_params=(avg,), avg=max(1, L / step_size); "
        "upstream integration_steps_fn samples uniformly around avg. "
        "grad_count_per_step = 2 * info.num_integration_steps (realized count). "
        "init requires rng_key for random_generator_arg. "
        "Adaptation: adjusted_mclmc_find_L_and_step_size (static kernel), target=0.9. "
        "needs_mass_matrix=True: IMM from diagonal_preconditioning."
    ),
)
