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
"""Descriptor for random-walk Metropolis (RWM).

Recipes resolve isotropic Gaussian proposal scale ``sigma``; code generation
emits the upstream ``blackjax.rmh`` proposal generator for arbitrary pytrees.
RWM evaluates only log density (zero gradient cost), with optimal acceptance
near 0.234 (Gelman, Roberts & Gilks 1996).
"""

import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]

ENTRY = BaseMethod(
    name="rwm",
    family="mcmc",
    grad_count_per_step=lambda info: jnp.asarray(0),
    grad_count_convention="0 (gradient-free)",
    default_hp_space=(HyperparamSpace("sigma", "loguniform", low=1e-3, high=10.0),),
    needs_mass_matrix=False,
    target_acceptance_rate=0.234,
    notes=(
        "Isotropic Gaussian proposal; sigma is the proposal scale. "
        "Codegen builds proposal_generator via ravel_pytree for arbitrary JAX pytrees. "
        "grad_count=0: RWM evaluates logdensity only. "
        "Optimal accept ≈ 0.234 (Gelman, Roberts & Gilks 1996)."
    ),
)
