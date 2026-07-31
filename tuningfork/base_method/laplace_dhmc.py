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
"""Descriptor for Laplace-marginal dynamic HMC recipe emission.

Generated emission targets ``blackjax.laplace_dhmc``
(``blackjax/mcmc/laplace_dynamic_hmc.py``). It combines the Laplace
marginalisation of :mod:`~blackjax.mcmc.laplace_hmc` with the quasi-random
integration-step schedule of :mod:`~blackjax.mcmc.dynamic_hmc`.

Algorithm summary
-----------------
Identical to ``laplace_hmc`` except the number of leapfrog steps is drawn
quasi-randomly each transition (CHEES-style) rather than fixed.  The state
carries an extra ``random_generator_arg`` field (Halton index or PRNG key)
advanced each step by ``next_random_arg_fn``.

For the full algorithm description, see:
``sampling-book/book/algorithms/laplace_hmc_demo.md``

Grad cost approximation
-----------------------
Because the step count varies per transition, ``info.num_integration_steps``
is used as the per-step count.  The inner L-BFGS cost multiplier is ~5
(same approximation as ``laplace_hmc``)::

    grad_count_per_step = _laplace_grad_count

``extra_required_kwargs=("log_joint_fn", "theta_init")``: generated emission
splits the configured model into ``phi`` and ``theta`` and supplies a typed
joint log-density plus latent initializer.  It raises ``ValueError`` when no
phi/theta split is registered for the model.

Use case
--------
Same as ``laplace_hmc`` (hierarchical funnel-geometry models) but avoids
periodic-orbit sensitivity by randomising trajectory length.  Preferred over
``laplace_hmc`` when the fixed-L orbit has high autocorrelation.

References
----------
- ``blackjax/mcmc/laplace_dynamic_hmc.py`` — upstream implementation.
- ``sampling-book/book/algorithms/laplace_hmc_demo.md`` — full algorithm description.
"""

from tuningfork.base_method._base import BaseMethod, HyperparamSpace
from tuningfork.base_method._laplace_common import _laplace_grad_count

__all__ = ["ENTRY"]


ENTRY = BaseMethod(
    name="laplace_dhmc",
    family="mcmc",
    # Grad cost: (num_integration_steps + 1) × lbfgs_iter_num — measured proxy.
    # Replaces the hardcoded ×5 heuristic (blackjax PR #928).
    grad_count_per_step=_laplace_grad_count,
    grad_count_convention="(info.num_integration_steps + 1) × info.lbfgs_iter_num (lower bound; line-search ≈ 1)",
    default_hp_space=(HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),),
    needs_mass_matrix=True,
    target_acceptance_rate=0.8,
    extra_required_kwargs=("log_joint_fn", "theta_init"),
    # The generated Laplace preamble supplies the phi/theta model split and
    # initializes the sampler-specific state.
    notes=(
        "Dynamic HMC on the Laplace-approximated marginal log-density. Combines "
        "Laplace marginalisation (latent theta integrated out via L-BFGS at each "
        "leapfrog step) with a quasi-random number of leapfrog steps per transition "
        "(CHEES-style, avoids periodic-orbit sensitivity). "
        "State carries theta_star (MAP latent) and random_generator_arg (Halton index). "
        "Only step_size is recipe-resolved; trajectory length is adapted internally. "
        "Grad cost approximation: num_integration_steps * 5 (coarse; varies per "
        "transition due to random step count). "
        "extra_required_kwargs=('log_joint_fn', 'theta_init'); generated emission "
        "requires a registered model phi/theta split and fails explicitly when "
        "that configuration is absent. "
        "See sampling-book/book/algorithms/laplace_hmc_demo.md for full algorithm description."
    ),
)
