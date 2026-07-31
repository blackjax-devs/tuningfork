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
"""Descriptor for Laplace-marginal HMC recipe emission.

Generated emission targets ``blackjax.laplace_hmc``
(``blackjax/mcmc/laplace_hmc.py``). This is HMC on the
Laplace-approximated marginal log-density of a hierarchical model, integrating
out latent variables ``theta`` via L-BFGS at each leapfrog step, with gradients
w.r.t. ``phi`` computed via the implicit function theorem.

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

``extra_required_kwargs=("log_joint_fn", "theta_init")``: generated emission
splits the configured model into hyperparameters ``phi`` and latent variables
``theta``, then supplies a typed joint log-density and latent initializer to
the BlackJAX factory.  Emission raises ``ValueError`` when the model has no
registered phi/theta split; it does not guess a configuration.

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

from tuningfork.base_method._base import BaseMethod, HyperparamSpace
from tuningfork.base_method._laplace_common import _laplace_grad_count

__all__ = ["ENTRY"]


ENTRY = BaseMethod(
    name="laplace_hmc",
    family="mcmc",
    # Grad cost: (num_integration_steps + 1) × lbfgs_iter_num — measured via the
    # post-accept L-BFGS iter count as a proxy for per-leapfrog inner iters.
    # Replaces the hardcoded ×5 heuristic with the measured value from info.
    # See blackjax.mcmc.laplace_marginal.laplace_lbfgs_grad_evals for the formula.
    grad_count_per_step=_laplace_grad_count,
    grad_count_convention="(info.num_integration_steps + 1) × info.lbfgs_iter_num (lower bound; line-search ≈ 1)",
    default_hp_space=(
        HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
        HyperparamSpace("num_integration_steps", "int", low=1, high=20),
    ),
    needs_mass_matrix=True,
    target_acceptance_rate=0.8,
    extra_required_kwargs=("log_joint_fn", "theta_init"),
    # The generated Laplace preamble supplies the phi/theta model split and
    # initializes the sampler-specific state.
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
        "extra_required_kwargs=('log_joint_fn', 'theta_init'); generated emission "
        "requires a registered model phi/theta split and fails explicitly when "
        "that configuration is absent. "
        "See sampling-book/book/algorithms/laplace_hmc_demo.md for full algorithm description."
    ),
)
