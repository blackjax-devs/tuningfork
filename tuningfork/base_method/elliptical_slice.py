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
"""Elliptical Slice Sampler algorithm entry for the tuningfork algorithm registry.

Wraps ``blackjax.elliptical_slice`` (Murray, Adams & MacKay 2010) for
latent-Gaussian models of the form::

    p(f | y) ∝ N(f; mean, cov) * likelihood(y | f)

The Gaussian prior is encoded in ``prior_cov`` + ``prior_mean`` at factory
time; the caller supplies the **likelihood-only** function as ``logdensity_fn``
(NOT the joint log-posterior).

IMPORTANT: the upstream BlackJAX parameter is named ``loglikelihood_fn`` but
our registry wrapper names its first argument ``logdensity_fn`` for interface
uniformity.  Callers must supply a likelihood-only function here.

Hyperparameter-free: no BO-tunable parameters (``default_hp_space=()``).
Gradient-free: ``grad_count_per_step=0``.
No MH step: ``target_acceptance_rate=None`` (slice sampler always accepts).

``extra_required_kwargs=("prior_cov", "prior_mean")``: the ``no_warmup`` runner raises
``NotImplementedError`` for this method; a specialised wiring path is required.

References
----------
- Murray, I., Adams, R., & MacKay, D. J. C. (2010). Elliptical slice sampling.
  In *Proceedings of the 13th International Conference on Artificial
  Intelligence and Statistics (AISTATS 2010)*, JMLR W&CP 9.
"""

import blackjax
import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace  # noqa: F401

__all__ = ["ENTRY", "_factory"]


def _factory(logdensity_fn, *, prior_cov, prior_mean, **kwargs):
    """Build a ``blackjax.elliptical_slice`` kernel.

    Parameters
    ----------
    logdensity_fn
        **Likelihood-only** log-density callable ``f -> log p(y | f)``.
        The Gaussian prior is encoded via ``prior_mean`` and ``prior_cov``
        and must NOT be included here.  Note: upstream BlackJAX names this
        parameter ``loglikelihood_fn`` — our wrapper renames it for registry
        uniformity, but the semantic is the same.
    prior_cov
        Prior covariance matrix of shape ``(d, d)``.
    prior_mean
        Prior mean vector of shape ``(d,)``.
    **kwargs
        Accepted for interface uniformity; ignored.

    Returns
    -------
    SamplingAlgorithm
        A BlackJAX kernel object with ``.init`` and ``.step`` methods.
    """
    # Pass-through wrapper.  blackjax.elliptical_slice's first arg is
    # named loglikelihood_fn upstream; conceptually it is the LIKELIHOOD
    # ONLY (the Gaussian prior is encoded in mean + cov).  Callers must
    # supply a likelihood-only function as logdensity_fn here.
    return blackjax.elliptical_slice(logdensity_fn, mean=prior_mean, cov=prior_cov)


ENTRY = BaseMethod(
    name="elliptical_slice",
    family="mcmc",
    factory=_factory,
    grad_count_per_step=lambda info: jnp.asarray(0),  # gradient-free
    default_hp_space=(),  # truly HP-free
    needs_mass_matrix=False,
    target_acceptance_rate=None,  # slice sampler; no MH step
    extra_required_kwargs=("prior_cov", "prior_mean"),
    notes=(
        "Murray, Adams & MacKay 2010 elliptical slice sampler for latent-Gaussian "
        "models p(f|y) ∝ N(f; mean, cov) * likelihood(y|f). The 'logdensity_fn' "
        "argument MUST be the likelihood ONLY (not the joint log posterior); the "
        "Gaussian prior is encoded via prior_cov + prior_mean. Hyperparameter-free. "
        "Gradient-free (grad_count_per_step=0). EllipSliceInfo carries (momentum, "
        "theta, subiter); no acceptance_rate field — slice sampling always accepts "
        "after a finite number of bracket-shrink subiters. extra_required_kwargs=('prior_cov', 'prior_mean'); "
        "no_warmup raises NotImplementedError; a specialised wiring path is required."
    ),
)
