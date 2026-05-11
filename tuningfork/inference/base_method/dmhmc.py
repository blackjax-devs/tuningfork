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
"""Dynamic Multinomial HMC (dmhmc) algorithm entry for the bjx-bench algorithm
registry.

Dynamic HMC with **multinomial trajectory proposal**.  Like ``dynamic_hmc``,
the number of leapfrog steps is sampled from a length distribution at each step
rather than fixed.  The trajectory-length randomization is controlled by
``integration_steps_fn`` and ``next_random_arg_fn``, which CHEES adaptation
sets internally — they are not directly BO-tunable.  The only BO-tunable scalar
hyperparameter is ``step_size``; the ``inverse_mass_matrix`` is warmup-derived
from CHEES.

The multinomial proposal replaces the standard slice-sampling trajectory
selector with multinomial sampling over the trajectory (Betancourt 2017,
"A Conceptual Introduction to Hamiltonian Monte Carlo", §A.2).  This changes
*trajectory selection*, not the HP surface — the HP space is identical to
``dynamic_hmc``.

Each kernel call uses a *random* number of leapfrog steps drawn from the
adapted distribution; ``info.num_integration_steps`` exposes the realized count
so grad-accounting is exact.
"""

import blackjax
import jax.numpy as jnp

from tuningfork.inference.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]

ENTRY = BaseMethod(
    name="dmhmc",
    family="mcmc",
    factory=blackjax.dmhmc,  # called as factory(logdensity_fn, **trial_params)
    grad_count_per_step=lambda info: jnp.asarray(info.num_integration_steps),
    default_hp_space=(
        HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
        # inverse_mass_matrix is NOT BO-tunable — it comes from CHEES warmup.
        # integration_steps_fn / next_random_arg_fn are callables set by CHEES;
        # not representable as Optuna search space HPs.
    ),
    needs_mass_matrix=True,  # inverse_mass_matrix from CHEES warmup, not BO
    target_acceptance_rate=0.651,  # CHEES upstream default (slightly above HMC 0.65)
    notes=(
        "Dynamic Multinomial HMC (Betancourt 2017 §A.2 + Hoffman et al. 2022). "
        "Dynamic HMC with multinomial trajectory proposal; HP surface is identical "
        "to dynamic_hmc. Each step samples a random number of leapfrog steps from "
        "a length distribution adapted by CHEES. "
        "grad_count_per_step = info.num_integration_steps (realized count per step). "
        "Only BO-tunable HP: step_size (loguniform). "
        "inverse_mass_matrix from CHEES warmup; "
        "integration_steps_fn / next_random_arg_fn set internally by CHEES. "
        "target_acceptance_rate=0.651 matches CHEES upstream default."
    ),
)
