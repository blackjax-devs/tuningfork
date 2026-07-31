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
"""Descriptor for IRMH recipe emission.

The upstream method is ``blackjax.irmh``
(``blackjax.mcmc.random_walk.irmh_as_top_level_api``).

The proposal is a full ``Callable`` supplied by generated emission — typically
fitted from a VI / Pathfinder / Laplace approximation.  Unlike RWM, the
proposal ``q(y)`` does not depend on the current state ``x``.

``extra_required_kwargs=("proposal_distribution",)``: generated emission does
not currently support this method. Enabling it requires typed recipe inputs
for the proposal (and optionally its log-density) plus a corresponding sampler
emitter.

Hyperparameter-free: no declared scalar HPs (``default_hp_space=()``).
The proposal is a full callable; it cannot be reduced to a declared scalar.
Gradient-free: ``grad_count_per_step=0``.
``target_acceptance_rate=None``: depends entirely on proposal-vs-target
overlap; no universal optimal rate.

References
----------
- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., &
  Teller, E. (1953). Equation of state calculations by fast computing
  machines. *The Journal of Chemical Physics*, 21(6), 1087–1092.
- Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*
  (2nd ed.). Springer. §7.3 Independent Metropolis-Hastings.
"""

import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod

__all__ = ["ENTRY"]


ENTRY = BaseMethod(
    name="irmh",
    family="mcmc",
    grad_count_per_step=lambda info: jnp.asarray(0),  # gradient-free MH
    grad_count_convention="0 (gradient-free)",
    default_hp_space=(),  # truly HP-free; proposal is a full callable
    needs_mass_matrix=False,
    target_acceptance_rate=None,  # depends entirely on proposal-vs-target overlap
    extra_required_kwargs=("proposal_distribution",),
    notes=(
        "Independent Metropolis-Hastings (Wang et al. 2022 reference; standard textbook "
        "MH variant where the proposal q(y) is independent of current state x). The proposal "
        "is a Callable (rng_key -> position) supplied as a typed recipe input, typically fitted from "
        "a VI / Pathfinder / Laplace approximation. For non-symmetric proposals, also supply "
        "proposal_logdensity_fn. Gradient-free (grad_count_per_step=0). RWInfo carries "
        "acceptance_rate, is_accepted, proposal. extra_required_kwargs=('proposal_distribution',); "
        "generated emission currently reports this method as unsupported. Enabling it "
        "requires typed recipe inputs for the proposal and a corresponding sampler emitter. "
        "Also used as an SMC inner kernel — the standalone "
        "non-SMC entry point; the SMC integration lives in the smc module."
    ),
)
