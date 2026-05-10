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
"""Dynamic HMC (also known as ``dhmc``) algorithm entry for the bjx-bench
algorithm registry.

Dynamic HMC (Hoffman et al. 2022, "Tuning-Free Generalized Hamiltonian Monte
Carlo") generalizes standard HMC by sampling the **number of leapfrog steps**
from a length distribution at each step rather than fixing it.  The
trajectory-length randomization is controlled by ``integration_steps_fn`` and
``next_random_arg_fn``, which CHEES adaptation sets internally — they are not
directly BO-tunable.  The only BO-tunable scalar hyperparameter is
``step_size``; the ``inverse_mass_matrix`` is warmup-derived from CHEES.

Each kernel call uses a *random* number of leapfrog steps drawn from the
adapted distribution; ``info.num_integration_steps`` exposes the realized count
so grad-accounting is exact.

**Alias note**: ``blackjax.dhmc is blackjax.dynamic_hmc`` is ``True`` (confirmed
2026-05-09).  We expose only ``dynamic_hmc`` in the registry and document the
alias here.

References
----------
- Hoffman, M. D., Radul, A., & Sountsov, P. (2022). An adaptive-MCMC scheme
  for setting trajectory lengths in Hamiltonian Monte Carlo. In *AISTATS 2022*.
  (ChEES-HMC / dynamic HMC paper.)
"""

import blackjax
import jax.numpy as jnp

from bjx_bench.inference.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]

ENTRY = BaseMethod(
    name="dynamic_hmc",
    family="mcmc",
    factory=blackjax.dynamic_hmc,  # blackjax.dhmc is blackjax.dynamic_hmc (alias)
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
        "Dynamic HMC (Hoffman et al. 2022). Each step samples a random number "
        "of leapfrog steps from a length distribution adapted by CHEES. "
        "grad_count_per_step = info.num_integration_steps (realized count per step). "
        "Only BO-tunable HP: step_size (loguniform). "
        "inverse_mass_matrix from CHEES warmup; "
        "integration_steps_fn / next_random_arg_fn set internally by CHEES. "
        "Alias: blackjax.dhmc is blackjax.dynamic_hmc (confirmed 2026-05-09). "
        "target_acceptance_rate=0.651 matches CHEES upstream default."
    ),
)
