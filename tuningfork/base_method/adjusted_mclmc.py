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
"""Adjusted MCLMC (Metropolis-adjusted Microcanonical Langevin Monte Carlo).

Adjusted MCLMC adds a Metropolis-Hastings correction step on top of the
MCLMC trajectory, improving exactness of the stationary distribution at the
cost of a higher rejection rate.

The BO hyperparameter space is ``(step_size, L)`` — matching vanilla MCLMC —
but the upstream factory does not accept ``L`` directly.  Instead, the number
of integration steps is derived as ``N = max(1, round(L / step_size))``, and
``integration_steps_params=(N,)`` is passed to the upstream factory.  The
translation uses JAX-native ``jnp.maximum`` / ``jnp.round`` so that the factory
is safe to call inside ``jax.vmap`` (where ``step_size`` is a traced array).
Both BO tuning (concrete float args) and the recipe runner (traced vmap args)
are handled without a separate code path.

Grad cost: the default integrator (isokinetic_mclachlan, a palindromic
[b1,a1,b2,a1,b1] scheme) evaluates the gradient twice per integrator step
(at each leapfrog position update) → ``grad_count_per_step = 2 * N`` where
``N = info.num_integration_steps``.

Init: ``blackjax.adjusted_mclmc.init(position, logdensity_fn)`` — NO rng_key
required (unlike vanilla MCLMC whose momentum requires a random unit vector).

Adaptation: ``blackjax.adjusted_mclmc_find_L_and_step_size`` with
``target=0.9`` (canonical adjusted-MCLMC acceptance target from upstream
tests).
"""

import blackjax
import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]


def _factory(logdensity_fn, *, step_size, L, inverse_mass_matrix=1.0, **kwargs):
    """Build a BlackJAX adjusted_mclmc SamplingAlgorithm.

    Translates the ``(step_size, L)`` BO hyperparameter space used by
    the benchmark into the ``integration_steps_params=(N,)`` convention
    expected by upstream ``blackjax.adjusted_mclmc``.

    Parameters
    ----------
    logdensity_fn
        BlackJAX-compatible log-density function.
    step_size
        Leapfrog step size.
    L
        Target trajectory length in time units.  Converted to an integer
        number of integration steps via ``N = max(1, round(L / step_size))``.
        Uses ``jnp.maximum`` / ``jnp.round`` so the factory is safe to call
        inside ``jax.vmap`` when ``step_size`` is a traced array.
    inverse_mass_matrix
        Diagonal preconditioning matrix (scalar or 1-D array).
        Default ``1.0`` (identity preconditioning).
    **kwargs
        Ignored; present for interface uniformity with the runner.

    Returns
    -------
    blackjax.SamplingAlgorithm
        Object with ``.init`` and ``.step`` methods.
    """
    # Use jnp ops so the factory is safe inside jax.vmap (step_size may be a
    # traced array).  Works with concrete values from BO tuning too.
    n_steps = jnp.maximum(1, jnp.round(L / step_size)).astype(jnp.int32)
    return blackjax.adjusted_mclmc(
        logdensity_fn,
        step_size=step_size,
        integration_steps_params=(n_steps,),
        inverse_mass_matrix=inverse_mass_matrix,
    )


ENTRY = BaseMethod(
    name="adjusted_mclmc",
    family="mcmc",
    factory=_factory,
    grad_count_per_step=lambda info: jnp.asarray(2 * info.num_integration_steps),
    grad_count_convention="2 × info.num_integration_steps",
    default_hp_space=(
        HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
        HyperparamSpace("L", "loguniform", low=0.1, high=100.0),
    ),
    needs_mass_matrix=True,
    target_acceptance_rate=0.9,
    # T2.3 descriptors: MCLMC family — L is also per-chain from adjusted_mclmc_tuning warmup.
    per_chain_param_keys=("step_size", "inverse_mass_matrix", "L"),
    reinit_state=False,  # HMCState from adjusted_mclmc_tuning is directly usable.
    extra_kwarg_builder=None,  # No extra kwargs beyond logdensity_fn + HP-space.
    notes=(
        "Metropolis-adjusted MCLMC (adjusted_mclmc). "
        "Factory translates (step_size, L) -> integration_steps_params=(N,) "
        "where N = jnp.maximum(1, jnp.round(L / step_size)) (jax-vmap safe). "
        "grad_count_per_step = 2 * info.num_integration_steps "
        "(isokinetic_mclachlan default integrator: 2 grads/step). "
        "init: blackjax.adjusted_mclmc.init(position, logdensity_fn) — no rng_key. "
        "Dedicated adaptation: blackjax.adjusted_mclmc_find_L_and_step_size with target=0.9. "
        "needs_mass_matrix=True: IMM from adjusted_mclmc_find_L_and_step_size diagonal_preconditioning. "
        "target_acceptance_rate=0.9: canonical adjusted-MCLMC acceptance target."
    ),
)
