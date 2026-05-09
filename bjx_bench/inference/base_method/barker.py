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
"""Barker algorithm entry for the bjx-bench algorithm registry.

Barker proposal MCMC (Livingstone & Zanella 2022).  Uses a gradient-based
proposal with a sigmoid accept/reject step.  ``step_size`` is BO-tunable;
``inverse_mass_matrix`` is supplied by warmup adaptation (like HMC/NUTS).

Grad cost per step: constant 1.  The Barker kernel evaluates the gradient
once per proposed step (the gradient is used to bias the proposal direction
via the Barker function).  Optimal target acceptance rate ≈ 0.40
(Livingstone & Zanella 2022, Theorem 2.2).

Note on ``inverse_mass_matrix``: ``blackjax.barker`` accepts
``inverse_mass_matrix=None`` (identity metric) or an explicit array.  In
Tier-B, the BO trial passes only ``step_size``; the mass matrix is either
left as ``None`` (identity) or supplied by the warmup adaptation.
``needs_mass_matrix=True`` signals the Tier-B runner to thread one through.
"""

from __future__ import annotations

import blackjax
import jax.numpy as jnp

from bjx_bench.inference.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]

ENTRY = BaseMethod(
    name="barker",
    family="mcmc",
    factory=blackjax.barker,  # called as factory(logdensity_fn, step_size=..., inverse_mass_matrix=...)
    grad_count_per_step=lambda info: jnp.asarray(1),
    default_hp_space=(HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),),
    needs_mass_matrix=True,
    target_acceptance_rate=0.40,
    notes=(
        "Livingstone & Zanella '22 optimal accept ≈ 0.40. "
        "Constant 1 grad/step. inverse_mass_matrix from warmup adaptation "
        "(defaults to identity/None if no warmup). BO tunes step_size only; "
        "IMM is passed separately by the Tier-B runner."
    ),
)
