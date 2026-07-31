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
"""Descriptor for Elliptical Slice recipe emission.

The upstream ``blackjax.elliptical_slice`` method (Murray, Adams & MacKay
2010) targets latent-Gaussian models of the form::

    p(f | y) ∝ N(f; mean, cov) * likelihood(y | f)

The Gaussian prior is encoded in ``prior_cov`` + ``prior_mean``. A future
generated route must supply the **likelihood-only** function, not the joint
log-posterior.

The upstream BlackJAX parameter is named ``loglikelihood_fn``.

Hyperparameter-free: no declared scalar parameters (``default_hp_space=()``).
Gradient-free: ``grad_count_per_step=0``.
No MH step: ``target_acceptance_rate=None`` (slice sampler always accepts).

``extra_required_kwargs=("prior_cov", "prior_mean")``: generated emission
does not currently support this method. Enabling it requires typed recipe
inputs for the Gaussian prior and a corresponding sampler emitter.

References
----------
- Murray, I., Adams, R., & MacKay, D. J. C. (2010). Elliptical slice sampling.
  In *Proceedings of the 13th International Conference on Artificial
  Intelligence and Statistics (AISTATS 2010)*, JMLR W&CP 9.
"""

import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod

__all__ = ["ENTRY"]


ENTRY = BaseMethod(
    name="elliptical_slice",
    family="mcmc",
    grad_count_per_step=lambda info: jnp.asarray(0),  # gradient-free
    grad_count_convention="0 (gradient-free)",
    default_hp_space=(),  # truly HP-free
    needs_mass_matrix=False,
    target_acceptance_rate=None,  # slice sampler; no MH step
    extra_required_kwargs=("prior_cov", "prior_mean"),
    # Gradient-free: no adapted step_size or mass matrix from warmup.
    notes=(
        "Murray, Adams & MacKay 2010 elliptical slice sampler for latent-Gaussian "
        "models p(f|y) ∝ N(f; mean, cov) * likelihood(y|f). The 'logdensity_fn' "
        "argument MUST be the likelihood ONLY (not the joint log posterior); the "
        "Gaussian prior is encoded via prior_cov + prior_mean. Hyperparameter-free. "
        "Gradient-free (grad_count_per_step=0). EllipSliceInfo carries (momentum, "
        "theta, subiter); no acceptance_rate field — slice sampling always accepts "
        "after a finite number of bracket-shrink subiters. extra_required_kwargs=('prior_cov', 'prior_mean'); "
        "generated emission currently reports this method as unsupported. Enabling it "
        "requires typed recipe inputs for the Gaussian prior and a corresponding sampler emitter."
    ),
)
