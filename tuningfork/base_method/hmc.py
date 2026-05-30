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
"""HMC algorithm entry for the tuningfork algorithm registry.

Fixed-L Hamiltonian Monte Carlo.  Both ``step_size`` and
``num_integration_steps`` are BO-tunable; ``inverse_mass_matrix`` comes from
window adaptation warmup.

Grad cost per step: ``info.num_integration_steps`` (1 gradient per leapfrog
step).  Optimal target acceptance rate ≈ 0.65 (Beskos et al. 2013).
"""

import blackjax
import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]

ENTRY = BaseMethod(
    name="hmc",
    family="mcmc",
    factory=blackjax.hmc,  # called as factory(logdensity_fn, **trial_params)
    grad_count_per_step=lambda info: jnp.asarray(info.num_integration_steps),
    grad_count_convention="info.num_integration_steps",
    default_hp_space=(
        HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
        HyperparamSpace("num_integration_steps", "int", low=1, high=128),
    ),
    needs_mass_matrix=True,
    target_acceptance_rate=0.65,
    notes=(
        "Beskos et al. optimal accept ≈ 0.65 for fixed-L HMC; both step_size "
        "and num_integration_steps are BO-tunable. inverse_mass_matrix comes "
        "from warmup adaptation, not BO."
    ),
)
