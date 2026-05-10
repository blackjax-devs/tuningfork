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
"""Laplace-marginal multinomial HMC (static trajectory) entry for the bjx-bench registry.

Wraps ``blackjax.laplace_mhmc`` (``blackjax/mcmc/laplace_hmc.py`` with
``build_proposal=multinomial_hmc_proposal``) for the bjx-bench algorithm registry.

This is identical to ``laplace_hmc`` except the M-H endpoint proposal is replaced
by multinomial sampling over the full trajectory (Betancourt 2017 §A.2).  This
changes *trajectory selection*, not the hyperparameter surface — the HP space is
identical to ``laplace_hmc``.

Algorithm summary
-----------------
Same as ``laplace_hmc`` (HMC on Laplace-approximated marginal) but proposes
from the full trajectory via multinomial sampling rather than the trajectory
endpoint.  This typically yields better ESS per gradient at the cost of no
rejection step.

For the full algorithm description, see:
``sampling-book/book/algorithms/laplace_hmc_demo.md``

Grad cost approximation
-----------------------
Same approximation as ``laplace_hmc``::

    grad_count_per_step = lambda info: jnp.asarray(info.num_integration_steps * 5)

``extra_required_kwargs=("log_joint_fn", "theta_init")``: the standard
``no_warmup`` runner raises ``NotImplementedError`` for this method.

References
----------
- ``blackjax/mcmc/laplace_hmc.py`` — upstream implementation (``laplace_mhmc``
  is ``laplace_hmc`` with ``build_proposal=blackjax.mcmc.hmc.multinomial_hmc_proposal``).
- ``sampling-book/book/algorithms/laplace_hmc_demo.md`` — full algorithm description.
- Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. §A.2.
"""

from typing import Any

import blackjax
import jax.numpy as jnp

from bjx_bench.inference.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY", "_factory"]


def _factory(
    logdensity_fn: Any,  # NOT USED — Laplace family uses log_joint_fn instead
    *,
    log_joint_fn: Any,
    theta_init: Any,
    step_size: float,
    inverse_mass_matrix: Any,
    num_integration_steps: int = 5,
    **kwargs: Any,  # forwarded to **optimizer_kwargs (e.g. maxiter, gtol, ftol)
) -> Any:
    """Build a ``blackjax.laplace_mhmc`` kernel.

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
        HMC leapfrog step size.
    inverse_mass_matrix
        Inverse mass matrix (1-D array for diagonal, scalar for isotropic).
        Supplied by warmup adaptation; not BO-tunable.
    num_integration_steps
        Number of leapfrog steps per HMC transition.  Default 5.
    **kwargs
        Forwarded to ``blackjax.mcmc.laplace_hmc.as_top_level_api`` as
        ``**optimizer_kwargs`` (e.g. ``maxiter=100``, ``gtol=1e-6``).

    Returns
    -------
    SamplingAlgorithm
        A BlackJAX kernel object with ``.init`` and ``.step`` methods.
        Uses multinomial trajectory sampling (``blackjax.laplace_mhmc`` is
        ``laplace_hmc`` with ``build_proposal=multinomial_hmc_proposal``).
    """
    return blackjax.laplace_mhmc(
        log_joint_fn,
        theta_init,
        step_size,
        inverse_mass_matrix,
        num_integration_steps,
        **kwargs,
    )


ENTRY = BaseMethod(
    name="laplace_mhmc",
    family="mcmc",
    factory=_factory,
    # Grad cost approximation: num_integration_steps * ~5 inner L-BFGS grads.
    grad_count_per_step=lambda info: jnp.asarray(info.num_integration_steps * 5),
    default_hp_space=(
        HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
        HyperparamSpace("num_integration_steps", "int", low=1, high=20),
    ),
    needs_mass_matrix=True,
    target_acceptance_rate=0.8,
    extra_required_kwargs=("log_joint_fn", "theta_init"),
    notes=(
        "Multinomial HMC on the Laplace-approximated marginal log-density. "
        "Identical to laplace_hmc but proposes from the full trajectory via multinomial "
        "sampling (Betancourt 2017 §A.2) rather than the endpoint. "
        "Typically yields better ESS per gradient at the cost of no rejection step. "
        "blackjax.laplace_mhmc = laplace_hmc with build_proposal=multinomial_hmc_proposal "
        "(blackjax.mcmc.hmc.multinomial_hmc_proposal — NOT the top-level alias). "
        "Grad cost approximation: num_integration_steps * 5 (coarse). "
        "extra_required_kwargs=('log_joint_fn', 'theta_init'); no_warmup raises "
        "NotImplementedError; a specialised wiring path is required. "
        "See sampling-book/book/algorithms/laplace_hmc_demo.md for full algorithm description."
    ),
)
