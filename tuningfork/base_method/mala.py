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
"""MALA algorithm entry for the tuningfork algorithm registry.

Metropolis-Adjusted Langevin Algorithm. Only ``step_size`` is recipe-resolved;
no mass matrix is needed (MALA uses a fixed isotropic metric).

Grad cost per step: constant 1.  The MALAState caches ``logdensity_grad``
from the accepted proposal; a single ``value_and_grad`` call is made per
step to evaluate the candidate.  Optimal target acceptance rate ≈ 0.574
(Roberts & Rosenthal 1998).
"""

import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]

ENTRY = BaseMethod(
    name="mala",
    family="mcmc",
    grad_count_per_step=lambda info: jnp.asarray(1),
    grad_count_convention="1",
    default_hp_space=(HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),),
    needs_mass_matrix=False,
    target_acceptance_rate=0.574,
    # MALA does not use inverse_mass_matrix but window_adaptation adapts one and
    # Generated plans may thread an adapted mass matrix for uniformity.
    notes=(
        "Roberts & Rosenthal '98 optimal accept ≈ 0.574 in high-D. "
        "Constant 1 grad/step (cached MALAState.logdensity_grad reused "
        "for MH ratio; single value_and_grad at proposal). "
        "No mass matrix required — isotropic Langevin dynamics."
    ),
)
