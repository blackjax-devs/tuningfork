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
"""Marginal-gradient Gaussian (mgrad_gaussian) algorithm entry for the tuningfork
algorithm registry.

Wraps ``blackjax.mgrad_gaussian`` (Titsias 2018, marginal sampler for
latent-Gaussian models) for models of the form::

    q(x) ∝ exp(f(x)) * N(x; mean, cov)

Uses a first-order approximation to the log-likelihood and samples via a
Gaussian proposal derived from the marginal.  ``logdensity_fn`` is the full
posterior log-density (Titsias 2018 marginal form internally subtracts the
Gaussian prior contribution).

The sole BO-tunable hyperparameter is ``step_size`` (delta in Titsias 2018).
Upstream guidance: calibrate ``step_size`` so that acceptance rate ≈ 50%.

Grad cost per step: 1 (``jax.value_and_grad(logdensity_fn)`` once per step).

``extra_required_kwargs=("prior_cov", "prior_mean")``: the ``no_warmup`` runner raises
``NotImplementedError`` for this method.  a specialised path is required.

References
----------
- Titsias, M. K. (2018). Auxiliary gradient-based sampling algorithms.
  *Journal of the Royal Statistical Society: Series B (Statistical
  Methodology)*, 80(4), 749–767.
"""

import blackjax
import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY", "_factory"]


def _factory(logdensity_fn, *, prior_cov, prior_mean, step_size=1.0, **kwargs):
    """Build a ``blackjax.mgrad_gaussian`` kernel.

    Parameters
    ----------
    logdensity_fn
        Full posterior log-density callable.  Titsias 2018 marginal form
        internally subtracts the Gaussian prior contribution; the caller
        must provide the joint log-posterior (including the prior term).
    prior_cov
        Prior covariance matrix of shape ``(d, d)``.  Passed as
        ``covariance`` to the upstream API; the SVD is computed internally
        on each factory call.
    prior_mean
        Prior mean vector of shape ``(d,)``.
    step_size
        Delta in Titsias 2018.  Upstream guidance: target acceptance rate
        ≈ 0.5.  Default 1.0.
    **kwargs
        Accepted for interface uniformity; ignored.

    Returns
    -------
    SamplingAlgorithm
        A BlackJAX kernel object with ``.init`` and ``.step`` methods.
    """
    # blackjax.mgrad_gaussian wraps marginal_latent_gaussian.as_top_level_api.
    # logdensity_fn is the full posterior log-density (Titsias 2018 marginal
    # form internally subtracts the Gaussian prior contribution).
    return blackjax.mgrad_gaussian(
        logdensity_fn, covariance=prior_cov, mean=prior_mean, step_size=step_size
    )


ENTRY = BaseMethod(
    name="mgrad_gaussian",
    family="mcmc",
    factory=_factory,
    grad_count_per_step=lambda info: jnp.asarray(1),  # one value_and_grad per step
    default_hp_space=(HyperparamSpace("step_size", "loguniform", low=1e-3, high=10.0),),
    needs_mass_matrix=False,
    target_acceptance_rate=0.5,  # upstream docstring guidance
    extra_required_kwargs=("prior_cov", "prior_mean"),
    notes=(
        "Titsias 2018 marginal sampler for latent-Gaussian models q(x) ∝ exp(f(x)) * "
        "N(x; mean, cov). Uses a first-order approximation to the log-likelihood; sole "
        "tunable is delta (step_size). MarginalInfo carries (acceptance_rate, is_accepted, "
        "proposal). 1 grad/step (jax.value_and_grad inside the kernel). Upstream guidance: "
        "target accept ≈ 0.5. extra_required_kwargs=('prior_cov', 'prior_mean'); no_warmup raises "
        "NotImplementedError; a specialised path is required. Internally precomputes "
        "the SVD of prior_cov on every factory call — for repeated trials with the "
        "same cov, cache via cov_svd kwarg if profiling shows it matters."
    ),
)
