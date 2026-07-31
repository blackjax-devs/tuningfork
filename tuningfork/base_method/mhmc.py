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
"""Multinomial HMC (mhmc) algorithm entry for the tuningfork algorithm registry.

Fixed-L Hamiltonian Monte Carlo with **multinomial trajectory proposal**.  Both
``step_size`` and ``num_integration_steps`` are recipe-resolved;
``inverse_mass_matrix`` comes from window adaptation warmup.

The multinomial proposal replaces the standard slice-sampling trajectory
selector with multinomial sampling over the trajectory (Betancourt 2017,
"A Conceptual Introduction to Hamiltonian Monte Carlo", §A.2).  This changes
*trajectory selection*, not the HP surface — the HP space is identical to HMC.

Grad cost per step: ``info.num_integration_steps`` (1 gradient per leapfrog
step).  Optimal target acceptance rate ≈ 0.65 (Beskos et al. 2013; same as
standard HMC because the trajectory budget is identical).

**Alias note**: ``blackjax.multinomial_hmc is blackjax.mhmc`` is ``True``
(alias for backward-compat; confirmed 2026-05-10).  We expose only ``mhmc``
in the registry and document the alias here.
"""

import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]

ENTRY = BaseMethod(
    name="mhmc",
    family="mcmc",
    grad_count_per_step=lambda info: jnp.asarray(info.num_integration_steps),
    grad_count_convention="info.num_integration_steps",
    default_hp_space=(
        HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
        HyperparamSpace("num_integration_steps", "int", low=1, high=128),
    ),
    needs_mass_matrix=True,
    target_acceptance_rate=0.65,
    notes=(
        "Multinomial HMC (Betancourt 2017 §A.2). Replaces HMC's slice-sampling "
        "trajectory selector with multinomial sampling; HP surface is identical to HMC. "
        "Beskos et al. optimal accept ≈ 0.65 for fixed-L HMC; both step_size "
        "and num_integration_steps are recipe-resolved. inverse_mass_matrix comes "
        "from warmup adaptation. "
        "Alias: blackjax.multinomial_hmc is blackjax.mhmc (confirmed 2026-05-10). "
        "multinomial_hmc_proposal is at blackjax.mcmc.hmc.multinomial_hmc_proposal "
        "(NOT blackjax.multinomial_hmc_proposal — that's the SamplingAPI alias)."
    ),
)
