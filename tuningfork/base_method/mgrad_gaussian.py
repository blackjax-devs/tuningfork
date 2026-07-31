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
"""Descriptor for marginal-gradient Gaussian recipe emission.

The upstream ``blackjax.mgrad_gaussian`` method (Titsias 2018) is a marginal
sampler for latent-Gaussian models of the form::

    q(x) ∝ exp(f(x)) * N(x; mean, cov)

Uses a first-order approximation to the log-likelihood and samples via a
Gaussian proposal derived from the marginal.  ``logdensity_fn`` is the full
posterior log-density (Titsias 2018 marginal form internally subtracts the
Gaussian prior contribution).

The sole declared scalar hyperparameter is ``step_size`` (delta in Titsias 2018).
Upstream guidance: calibrate ``step_size`` so that acceptance rate ≈ 50%.

Grad cost per step: 1 (``jax.value_and_grad(logdensity_fn)`` once per step).

``extra_required_kwargs=("prior_cov", "prior_mean")``: generated emission
does not currently support this method. Enabling it requires typed recipe
inputs for the Gaussian prior and a corresponding sampler emitter.

References
----------
- Titsias, M. K. (2018). Auxiliary gradient-based sampling algorithms.
  *Journal of the Royal Statistical Society: Series B (Statistical
  Methodology)*, 80(4), 749–767.
"""

import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]


ENTRY = BaseMethod(
    name="mgrad_gaussian",
    family="mcmc",
    grad_count_per_step=lambda info: jnp.asarray(1),  # one value_and_grad per step
    grad_count_convention="1 (one value_and_grad per step)",
    default_hp_space=(HyperparamSpace("step_size", "loguniform", low=1e-3, high=10.0),),
    needs_mass_matrix=False,
    target_acceptance_rate=0.5,  # upstream docstring guidance
    extra_required_kwargs=("prior_cov", "prior_mean"),
    # Gradient-based, but no adapted mass matrix from HMC warmup; step_size is
    # recipe-resolved and shared across chains.
    notes=(
        "Titsias 2018 marginal sampler for latent-Gaussian models q(x) ∝ exp(f(x)) * "
        "N(x; mean, cov). Uses a first-order approximation to the log-likelihood; sole "
        "declared scalar is delta (step_size). MarginalInfo carries (acceptance_rate, is_accepted, "
        "proposal). 1 grad/step (jax.value_and_grad inside the kernel). Upstream guidance: "
        "target accept ≈ 0.5. extra_required_kwargs=('prior_cov', 'prior_mean'); generated "
        "emission currently reports this method as unsupported. Enabling it requires typed "
        "recipe inputs for the Gaussian prior and a corresponding sampler emitter. Internally precomputes "
        "the SVD of prior_cov on every factory call — for repeated trials with the "
        "same cov, cache via cov_svd kwarg if profiling shows it matters."
    ),
)
