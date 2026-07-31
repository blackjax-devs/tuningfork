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
"""Laplace-marginal dynamic HMC (binomial proposal) entry for the tuningfork registry.

Wraps ``blackjax.laplace_dhmc`` (``blackjax/mcmc/laplace_dynamic_hmc.py``) for
the tuningfork algorithm registry.  Combines the Laplace marginalisation of
:mod:`~blackjax.mcmc.laplace_hmc` with the quasi-random integration-step
schedule of :mod:`~blackjax.mcmc.dynamic_hmc`.

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

``extra_required_kwargs=("log_joint_fn", "theta_init")``: the standard
``no_warmup`` runner raises ``NotImplementedError`` for this method.

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

from typing import Any

import blackjax

from tuningfork.base_method._base import BaseMethod, HyperparamSpace
from tuningfork.base_method._laplace_common import _laplace_grad_count

__all__ = ["ENTRY", "_factory"]


def _factory(
    logdensity_fn: Any,  # NOT USED — Laplace family uses log_joint_fn instead
    *,
    log_joint_fn: Any,
    theta_init: Any,
    step_size: float,
    inverse_mass_matrix: Any,
    **kwargs: Any,  # forwarded to **optimizer_kwargs (e.g. maxiter, gtol, ftol)
) -> Any:
    """Build a ``blackjax.laplace_dhmc`` kernel.

    Parameters
    ----------
    logdensity_fn
        NOT USED.  Present for interface uniformity with the standard factory
        signature ``(logdensity_fn, **hp_params) -> SamplingAlgorithm``.
        The Laplace-marginal family uses ``log_joint_fn`` as the actual
        primary callable.
    log_joint_fn
        Full log joint ``log p(theta, phi, y)`` as a callable
        ``(theta, phi) -> float``.  Both arguments may be arbitrary PyTrees.
        Must be at least C³ in ``theta`` for the Laplace approximation to be
        valid.
    theta_init
        Initial guess for the latent variables ``theta``.  Fixes the PyTree
        structure for all subsequent L-BFGS solves.
    step_size
        Leapfrog step size.
    inverse_mass_matrix
        Inverse mass matrix (1-D array for diagonal, scalar for isotropic).
        Supplied by warmup adaptation; not a declared scalar parameter.
    **kwargs
        Forwarded to ``blackjax.mcmc.laplace_dynamic_hmc.as_top_level_api`` as
        ``**optimizer_kwargs`` (e.g. ``maxiter=100``, ``gtol=1e-6``).

    Returns
    -------
    SamplingAlgorithm
        A BlackJAX kernel object with ``.init`` and ``.step`` methods.
        ``.init(phi_init, rng_key)`` runs a cold-start L-BFGS and seeds the
        random step-count generator; returns a ``LaplaceDynamicHMCState``.
    """
    return blackjax.laplace_dhmc(
        log_joint_fn,
        theta_init,
        step_size,
        inverse_mass_matrix,
        **kwargs,
    )


ENTRY = BaseMethod(
    name="laplace_dhmc",
    family="mcmc",
    factory=_factory,
    # Grad cost: (num_integration_steps + 1) × lbfgs_iter_num — measured proxy.
    # Replaces the hardcoded ×5 heuristic (blackjax PR #928).
    grad_count_per_step=_laplace_grad_count,
    grad_count_convention="(info.num_integration_steps + 1) × info.lbfgs_iter_num (lower bound; line-search ≈ 1)",
    default_hp_space=(HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),),
    needs_mass_matrix=True,
    target_acceptance_rate=0.8,
    extra_required_kwargs=("log_joint_fn", "theta_init"),
    # T2.3 descriptors: standard HMC family per-chain params.
    per_chain_param_keys=("step_size", "inverse_mass_matrix"),
    reinit_state=True,  # laplace_dhmc needs LaplaceDynamicHMCState (theta_star + rng_arg);
    # HMCState from window_adaptation is incompatible → per-chain kernel.init() required.
    extra_kwarg_builder=None,  # Laplace component construction is model-specific;
    # handled via _build_laplace_components runner helper, not a portable descriptor.
    notes=(
        "Dynamic HMC on the Laplace-approximated marginal log-density. Combines "
        "Laplace marginalisation (latent theta integrated out via L-BFGS at each "
        "leapfrog step) with a quasi-random number of leapfrog steps per transition "
        "(CHEES-style, avoids periodic-orbit sensitivity). "
        "State carries theta_star (MAP latent) and random_generator_arg (Halton index). "
        "Only step_size is recipe-resolved; trajectory length is adapted internally. "
        "Grad cost approximation: num_integration_steps * 5 (coarse; varies per "
        "transition due to random step count). "
        "extra_required_kwargs=('log_joint_fn', 'theta_init'); no_warmup raises "
        "NotImplementedError; a specialised wiring path is required. "
        "See sampling-book/book/algorithms/laplace_hmc_demo.md for full algorithm description."
    ),
)
