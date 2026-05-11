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
"""NUTS algorithm entry for the bjx-bench algorithm registry.

NUTS (No-U-Turn Sampler) is the default reference sampler in BlackJAX.
The only BO-tunable hyperparameter is ``step_size``; the
``inverse_mass_matrix`` comes from window adaptation warmup and is not
searched by Bayesian optimisation.

Grad cost per step: ``info.num_integration_steps`` (1 gradient per
leapfrog step, same accounting as HMC).
"""

import blackjax
import jax.numpy as jnp

from tuningfork.inference.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]

ENTRY = BaseMethod(
    name="nuts",
    family="mcmc",
    factory=blackjax.nuts,  # called as factory(logdensity_fn, **trial_params)
    grad_count_per_step=lambda info: jnp.asarray(info.num_integration_steps),
    default_hp_space=(HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),),
    needs_mass_matrix=True,
    target_acceptance_rate=0.80,
    notes=(
        "Stan-default target acceptance 0.80; inverse_mass_matrix supplied "
        "by warmup adaptation, not BO. step_size is the only BO-tunable HP."
    ),
)
