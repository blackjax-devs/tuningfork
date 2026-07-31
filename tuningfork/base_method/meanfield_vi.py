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
"""Descriptor for mean-field VI recipe emission.

This module contains metadata consumed by code generation.  The generated
sampler runs the variational optimisation and then draws from the fitted
mean-field distribution.

The declared hyperparameter is ``num_optimization_steps``; the optimizer is a
fixed recipe-time choice.

Grad cost approximation: ``grad_count_per_step = lambda info: 1``.  Each VI
optimisation step requires one gradient of the log-density (via the ELBO
gradient); after the loop, each ``step`` call draws one sample.  The
approximation is intentionally simple — see ``notes`` for the full accounting.

Generated emission uses Adam and a recipe-specified optimisation budget.
"""

import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]


ENTRY = BaseMethod(
    name="meanfield_vi",
    family="vi",
    grad_count_per_step=lambda info: jnp.asarray(1),
    grad_count_convention="1",
    default_hp_space=(
        HyperparamSpace("num_optimization_steps", "int", low=1_000, high=50_000),
    ),
    needs_mass_matrix=False,
    target_acceptance_rate=None,  # VI is not a MH sampler
    notes=(
        "Mean-field variational inference (MFVI) descriptor for generated emission. "
        "Generated code runs num_optimization_steps Adam steps and emits one "
        "sample from the fitted mean-field Gaussian (N(mu, diag(exp(rho)))). "
        "The declared trial hyperparameter is num_optimization_steps; the "
        "optimizer is a fixed recipe-time choice. "
        "grad_count_per_step=1 is an approximation: during the optimisation "
        "phase, each step consumes one ELBO gradient (via reparameterisation); "
        "at sample time no gradient is evaluated. The approximation slightly "
        "over-counts for the sampling phase but is correct for the dominant "
        "optimisation cost. "
        "Preferred variant for d > 30 (fullrank_vi is recommended only for d <= 30)."
    ),
)
