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
"""Laplace-marginal dynamic multinomial HMC entry for the tuningfork registry.

Wraps ``blackjax.laplace_dmhmc`` (``blackjax/mcmc/laplace_dynamic_hmc.py``
with ``build_proposal=multinomial_hmc_proposal``) for the tuningfork algorithm
registry.

This is the combination of both dynamic trajectory length (quasi-random step
count, as in ``laplace_dhmc``) and multinomial trajectory proposal (as in
``laplace_mhmc``).  The "2×2" variant that eliminates both periodic-orbit
sensitivity and the endpoint-proposal acceptance bias.

Algorithm summary
-----------------
Same as ``laplace_dhmc`` (dynamic Laplace HMC) but with multinomial sampling
over the full quasi-random trajectory.  The state carries both ``theta_star``
(MAP latent warm-start) and ``random_generator_arg`` (Halton step-count seed).

For the full algorithm description, see:
``sampling-book/book/algorithms/laplace_hmc_demo.md``

Grad cost approximation
-----------------------
Same approximation as ``laplace_dhmc`` (step count varies per transition)::

    grad_count_per_step = _laplace_grad_count

``extra_required_kwargs=("log_joint_fn", "theta_init")``: the standard
``no_warmup`` runner raises ``NotImplementedError`` for this method.

References
----------
- ``blackjax/mcmc/laplace_dynamic_hmc.py`` — upstream implementation
  (``laplace_dmhmc`` is ``laplace_dynamic_hmc`` with
  ``build_proposal=blackjax.mcmc.hmc.multinomial_hmc_proposal``).
- ``sampling-book/book/algorithms/laplace_hmc_demo.md`` — full algorithm description.
- Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. §A.2.
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
    """Build a ``blackjax.laplace_dmhmc`` kernel.

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
        Must be at least C³ in ``theta``.
    theta_init
        Initial guess for the latent variables ``theta``.  Fixes the PyTree
        structure for all subsequent L-BFGS solves.
    step_size
        Leapfrog step size.
    inverse_mass_matrix
        Inverse mass matrix (1-D array for diagonal, scalar for isotropic).
        Supplied by warmup adaptation; not BO-tunable.
    **kwargs
        Forwarded to ``blackjax.mcmc.laplace_dynamic_hmc.as_top_level_api`` as
        ``**optimizer_kwargs`` (e.g. ``maxiter=100``, ``gtol=1e-6``).

    Returns
    -------
    SamplingAlgorithm
        A BlackJAX kernel object with ``.init`` and ``.step`` methods.
        ``.init(phi_init, rng_key)`` seeds the random step-count generator;
        returns a ``LaplaceDynamicHMCState``.  Uses multinomial trajectory
        sampling (``blackjax.laplace_dmhmc`` = ``laplace_dynamic_hmc`` with
        ``build_proposal=multinomial_hmc_proposal``).
    """
    return blackjax.laplace_dmhmc(
        log_joint_fn,
        theta_init,
        step_size,
        inverse_mass_matrix,
        **kwargs,
    )


ENTRY = BaseMethod(
    name="laplace_dmhmc",
    family="mcmc",
    factory=_factory,
    # Grad cost: (num_integration_steps + 1) × lbfgs_iter_num — measured proxy.
    # num_integration_steps varies per transition (quasi-random schedule).
    grad_count_per_step=_laplace_grad_count,
    grad_count_convention="(info.num_integration_steps + 1) × info.lbfgs_iter_num (lower bound; line-search ≈ 1)",
    default_hp_space=(HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),),
    needs_mass_matrix=True,
    target_acceptance_rate=0.8,
    extra_required_kwargs=("log_joint_fn", "theta_init"),
    notes=(
        "Dynamic multinomial HMC on the Laplace-approximated marginal log-density. "
        "Combines dynamic trajectory length (quasi-random step count, avoids periodic-orbit "
        "sensitivity) with multinomial trajectory sampling (Betancourt 2017 §A.2, avoids "
        "endpoint-proposal acceptance bias). The '2×2' variant in the Laplace-marginal family. "
        "blackjax.laplace_dmhmc = laplace_dynamic_hmc with "
        "build_proposal=multinomial_hmc_proposal (blackjax.mcmc.hmc.multinomial_hmc_proposal). "
        "State carries theta_star (MAP latent warm-start) and random_generator_arg (Halton index). "
        "Only step_size is BO-tunable; trajectory length is adapted internally. "
        "Grad cost approximation: num_integration_steps * 5 (coarse; varies per transition). "
        "extra_required_kwargs=('log_joint_fn', 'theta_init'); no_warmup raises "
        "NotImplementedError; a specialised wiring path is required. "
        "See sampling-book/book/algorithms/laplace_hmc_demo.md for full algorithm description."
    ),
)
