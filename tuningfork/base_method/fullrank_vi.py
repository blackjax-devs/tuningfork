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
"""Descriptor for full-rank VI recipe emission.

This module contains metadata consumed by code generation.  The generated
sampler runs the variational optimisation and then draws from the fitted
full-rank distribution.

The declared hyperparameter is ``num_optimization_steps``; the optimizer is a
fixed recipe-time choice.

Grad cost approximation: ``grad_count_per_step = lambda info: 1``.  Each VI
optimisation step requires one gradient of the log-density (via the ELBO
gradient); after the loop, each ``step`` call draws one sample.  The
approximation is intentionally simple — see ``notes`` for the full accounting.

Generated emission uses Adam and a recipe-specified optimisation budget.

**Applicability**: full-rank VI is recommended only for ``d <= 30``.  For
higher-dimensional problems, use ``meanfield_vi`` instead.  The cholesky
parameterisation has ``O(d^2)`` parameters which become expensive at large
dimension.
"""

import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]


ENTRY = BaseMethod(
    name="fullrank_vi",
    family="vi",
    grad_count_per_step=lambda info: jnp.asarray(1),
    grad_count_convention="1",
    default_hp_space=(
        HyperparamSpace("num_optimization_steps", "int", low=2_000, high=100_000),
    ),
    needs_mass_matrix=False,
    target_acceptance_rate=None,  # VI is not a MH sampler
    notes=(
        "Full-rank variational inference (FRVI) descriptor for generated emission. "
        "Generated code runs num_optimization_steps Adam steps and emits one "
        "sample from the fitted full-rank Gaussian (N(mu, L @ L.T) where L "
        "is recovered from the flattened chol_params via _unflatten_cholesky). "
        "The declared trial hyperparameter is num_optimization_steps; the "
        "optimizer is a fixed recipe-time choice. "
        "grad_count_per_step=1 is an approximation: during optimisation each "
        "step consumes one ELBO gradient; at sample time no gradient is needed. "
        "Recommended ONLY for d <= 30: the cholesky parameterisation has "
        "O(d^2) parameters which become expensive and slow to converge at "
        "high dimension. Use meanfield_vi for d > 30."
    ),
)
