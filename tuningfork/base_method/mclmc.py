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
"""MCLMC (Microcanonical Langevin Monte Carlo) algorithm wrapper.

Note: each MCLMC step performs one integrator step. The default integrator
(isokinetic_mclachlan) costs 2 gradient evaluations per integrator step
(palindromic [b1,a1,b2,a1,b1]
scheme → 2 position updates) → constant 2 grads per kernel step.
MCLMCInfo does NOT carry num_integration_steps; the trajectory length L (in
time units) controls momentum-resample cadence (L / step_size time units).

Init note: blackjax.mclmc.init requires an rng_key to generate the initial
unit-vector momentum. Call kernel.init(position, rng_key) rather than the
rng_key-free form used by HMC/MALA/Barker.

Requires pytree_size(position) >= 2 (enforced by blackjax upstream).

Adaptation: BlackJAX provides blackjax.mclmc_find_L_and_step_size as a
dedicated warmup routine. BO tuning will dispatch to it based on
BaseMethod.name. This module only declares the entry.
"""

import blackjax
import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]

ENTRY = BaseMethod(
    name="mclmc",
    family="mcmc",
    factory=blackjax.mclmc,  # signature: (logdensity_fn, L, step_size, ...)
    grad_count_per_step=lambda info: jnp.asarray(2),
    grad_count_convention="2",
    default_hp_space=(
        HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
        HyperparamSpace("L", "loguniform", low=0.1, high=100.0),
    ),
    needs_mass_matrix=False,
    target_acceptance_rate=None,  # rejection-free; not applicable
    notes=(
        "Constant 2 grads/step (default isokinetic_mclachlan integrator). "
        "MCLMCInfo._fields = ('logdensity', 'kinetic_change', 'energy_change', 'nonans'); "
        "no num_integration_steps field. "
        "inverse_mass_matrix=1.0 default is scalar (global preconditioner). "
        "init requires rng_key: kernel.init(position, rng_key). "
        "Dedicated adaptation: blackjax.mclmc_find_L_and_step_size — not window_adaptation. "
        "pytree_size(position) >= 2 required by upstream."
    ),
)
