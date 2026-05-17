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
"""Laplace-marginal HMC (static trajectory) entry for the tuningfork registry.

Wraps ``blackjax.laplace_hmc`` (``blackjax/mcmc/laplace_hmc.py``) for the
tuningfork algorithm registry.  This is HMC on the Laplace-approximated marginal
log-density of a hierarchical model, integrating out latent variables ``theta``
via L-BFGS at each leapfrog step, with gradients w.r.t. ``phi`` computed via the
implicit function theorem.

Algorithm summary
-----------------
For a hierarchical model ``log p(theta, phi, y)``, this sampler:

1. At each leapfrog step, computes ``theta*(phi) = argmax_theta log_joint(theta, phi)``
   via L-BFGS warm-started from the previous ``theta_star``.
2. Evaluates the Laplace log-marginal:
   ``log p̂(phi) ≈ log p(theta*, phi) - 0.5 * log |H_theta(theta*, phi)|``
3. Computes gradients w.r.t. ``phi`` via the implicit function theorem
   (``jax.lax.custom_root`` — L-BFGS iterations are NOT unrolled).
4. Runs standard HMC (fixed-L, endpoint + M-H proposal) on ``phi`` alone.

The state carries ``theta_star`` (the MAP latent at current ``phi``) as a
warm-start hint for the next leapfrog step.

For the full algorithm description and derivation, see:
``sampling-book/book/algorithms/laplace_hmc_demo.md``

Grad cost approximation
-----------------------
Each leapfrog step calls L-BFGS internally (warm-started).  The inner
optimiser uses ~5 gradient evaluations per step under typical conditions,
so the per-transition cost is approximately
``num_integration_steps * (1 + 5) = 6 * num_integration_steps`` gradients.
This is a coarse approximation; actual cost depends on the landscape and
convergence tolerance.  The factory default is::

    grad_count_per_step = lambda info: jnp.asarray(info.num_integration_steps * 5)

``extra_required_kwargs=("log_joint_fn", "theta_init")``: the standard
``no_warmup`` runner raises ``NotImplementedError`` for this method because
the factory cannot be called with just ``logdensity_fn``; the caller must
supply a joint log-density ``log_joint_fn(theta, phi)`` and an initial
latent position ``theta_init``.

Use case
--------
Hierarchical models where direct sampling of ``(theta, phi)`` jointly suffers
from funnel geometry (e.g. Neal's funnel, 8-schools), and where the inner
conditional ``p(theta | phi)`` is log-concave (guaranteeing the Laplace
approximation quality required for the implicit-function-theorem gradient).

References
----------
- ``blackjax/mcmc/laplace_hmc.py`` — upstream implementation.
- ``sampling-book/book/algorithms/laplace_hmc_demo.md`` — full algorithm description.
"""

from typing import Any

import blackjax
import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

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
    """Build a ``blackjax.laplace_hmc`` kernel.

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
        ``.init(phi_init)`` runs a cold-start L-BFGS to find ``theta_star``
        and returns a ``LaplaceHMCState``.  ``.step(rng_key, state)`` runs
        one warm-started HMC transition.
    """
    return blackjax.laplace_hmc(
        log_joint_fn,
        theta_init,
        step_size,
        inverse_mass_matrix,
        num_integration_steps,
        **kwargs,
    )


ENTRY = BaseMethod(
    name="laplace_hmc",
    family="mcmc",
    factory=_factory,
    # Grad cost approximation: num_integration_steps * ~5 inner L-BFGS grads.
    # The factor 5 is a coarse default assuming warm-started L-BFGS converges
    # in ~5 iterations per leapfrog step; actual cost depends on the landscape.
    grad_count_per_step=lambda info: jnp.asarray(info.num_integration_steps * 5),
    default_hp_space=(
        HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
        HyperparamSpace("num_integration_steps", "int", low=1, high=20),
    ),
    needs_mass_matrix=True,
    target_acceptance_rate=0.8,
    extra_required_kwargs=("log_joint_fn", "theta_init"),
    notes=(
        "HMC on the Laplace-approximated marginal log-density of hierarchical models. "
        "Latent vars theta integrated out via L-BFGS at each leapfrog step (warm-started "
        "from the previous MCMC state). Gradients w.r.t. phi via the implicit function "
        "theorem (jax.lax.custom_root — L-BFGS iterations NOT unrolled). "
        "State carries theta_star (MAP latent at current phi) for warm-start efficiency. "
        "Grad cost approximation: num_integration_steps * 5 (coarse; actual cost depends "
        "on inner L-BFGS convergence speed). "
        "Use for hierarchical models with latent Gaussian theta and hyperparameters phi "
        "where direct sampling of (theta, phi) jointly suffers from funnel geometry. "
        "extra_required_kwargs=('log_joint_fn', 'theta_init'); no_warmup raises "
        "NotImplementedError; a specialised wiring path is required. "
        "See sampling-book/book/algorithms/laplace_hmc_demo.md for full algorithm description."
    ),
)
