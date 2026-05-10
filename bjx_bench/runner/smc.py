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
"""SMC runner helpers: particle initialisation + drive-to-convergence loop.

These utilities form the bridge between the SMC method registry
(``bjx_bench/inference/smc/``) and the recipe layer (Phase 6).  They are
deliberately pure functions with no global state and no side effects, and
are fully JIT-compatible.

Public API
----------
``init_particles_from_prior``
    Draw initial particles from an explicit prior sampler callable.
``run_smc``
    Drive an SMC step kernel until tempering reaches ``lambda_target`` or
    ``max_steps`` is exhausted.

TemperedSMCState uses field name ``tempering_param`` (NOT ``lmbda``).
The while_loop termination check and history recording use
``state.tempering_param``.  Detection of adaptive_tempered_smc uses
``hasattr(state, 'tempering_param')`` (present on TemperedSMCState;
absent on PartialPosteriorsSMCState and any inner_kernel_tuning wrapped state).
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

__all__ = ["init_particles_from_prior", "run_smc"]


def init_particles_from_prior(
    rng_key: jax.Array,
    *,
    prior_sample_fn: Callable[[jax.Array, int], jax.Array],
    num_particles: int,
) -> jax.Array:
    """Draw ``num_particles`` initial particles from the prior.

    We do NOT sample from a bare ``logprior_fn`` (rejection sampling in
    arbitrary dimension is intractable).  The caller supplies an explicit
    ``prior_sample_fn(key, n) -> array of shape (n, ...)``.

    For NumPyro models, the typical caller wires this as::

        prior_sample_fn = lambda key, n: prior_dist.sample(key, (n,))

    The test suite and recipe builders construct ``prior_sample_fn``
    explicitly from the posterior's prior distribution.

    Parameters
    ----------
    rng_key
        JAX random key used for particle sampling.
    prior_sample_fn
        Callable with signature ``(key: jax.Array, n: int) -> jax.Array``
        where the returned array has shape ``(n, ...)``.  Must be
        JAX-traceable so that this function can be JIT-compiled.
    num_particles
        Number of particles to draw.

    Returns
    -------
    jax.Array
        Array of shape ``(num_particles, ...)`` containing the initial
        particle positions drawn from the prior.

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> key = jax.random.key(0)
    >>> particles = init_particles_from_prior(
    ...     key,
    ...     prior_sample_fn=lambda k, n: jax.random.normal(k, (n, 5)),
    ...     num_particles=100,
    ... )
    >>> particles.shape
    (100, 5)
    """
    return prior_sample_fn(rng_key, num_particles)


def run_smc(
    rng_key: jax.Array,
    *,
    smc_init_state: Any,
    smc_step_fn: Callable,
    max_steps: int = 100,
    lambda_target: float = 1.0,
) -> tuple[Any, dict]:
    """Drive an SMC step kernel until convergence or step budget exhausted.

    For ``adaptive_tempered_smc`` — detected by ``hasattr(state,
    'tempering_param')`` — uses ``jax.lax.while_loop`` to run until
    ``state.tempering_param >= lambda_target`` or ``max_steps`` is
    exhausted.  History records ``tempering_param`` and
    ``log_likelihood_increment`` per step.

    For ``partial_posteriors_smc`` and ``inner_kernel_tuning`` — no
    tempering field — uses ``jax.lax.scan`` for a fixed ``max_steps``
    run.  History arrays are empty (shape ``(0,)``).  The caller must
    pick a sensible ``max_steps`` and inspect the final state directly.

    Both branches are JIT-compatible.

    Parameters
    ----------
    rng_key
        JAX random key.
    smc_init_state
        Initial SMC state, as returned by ``smc_alg.init(...)``.
    smc_step_fn
        Step function with signature ``(rng_key, state) -> (new_state, info)``.
        For ``adaptive_tempered_smc`` this is the standard 2-arg form;
        for ``partial_posteriors_smc`` the caller must pre-bind
        ``data_mask`` via ``functools.partial`` before passing here.
    max_steps
        Maximum number of SMC steps.  Default 100.
    lambda_target
        Tempering target for adaptive_tempered_smc.  Loop exits when
        ``state.tempering_param >= lambda_target``.  Ignored for
        non-tempering variants.  Default 1.0.

    Returns
    -------
    tuple[Any, dict]
        ``(final_state, history_dict)`` where ``history_dict`` has keys:

        - ``"lmbda"`` — shape ``(num_steps_run,)`` array of
          ``tempering_param`` values per step (adaptive_tempered only;
          empty array for other variants).
        - ``"log_likelihood_increment"`` — shape ``(num_steps_run,)``
          array of ``info.log_likelihood_increment`` per step
          (adaptive_tempered only; empty array for other variants).

    Notes
    -----
    ``jax.lax.while_loop`` requires all loop-carried values to have fixed
    dtypes and shapes.  For the while_loop branch we pre-allocate history
    buffers of size ``max_steps`` and track a step counter, then slice
    the filled prefix in the return value.

    Examples
    --------
    >>> import jax
    >>> key = jax.random.key(0)
    >>> # ... build smc_alg, state = smc_alg.init(particles) ...
    >>> final_state, history = run_smc(
    ...     key,
    ...     smc_init_state=state,
    ...     smc_step_fn=smc_alg.step,
    ...     max_steps=50,
    ...     lambda_target=1.0,
    ... )
    >>> history["lmbda"].shape
    (num_steps_run,)
    """
    is_adaptive_tempered = hasattr(smc_init_state, "tempering_param")

    if is_adaptive_tempered:
        return _run_smc_while_loop(
            rng_key,
            smc_init_state=smc_init_state,
            smc_step_fn=smc_step_fn,
            max_steps=max_steps,
            lambda_target=lambda_target,
        )
    else:
        return _run_smc_scan(
            rng_key,
            smc_init_state=smc_init_state,
            smc_step_fn=smc_step_fn,
            max_steps=max_steps,
        )


# ---------------------------------------------------------------------------
# Internal implementations
# ---------------------------------------------------------------------------


def _run_smc_while_loop(
    rng_key: jax.Array,
    *,
    smc_init_state: Any,
    smc_step_fn: Callable,
    max_steps: int,
    lambda_target: float,
) -> tuple[Any, dict]:
    """while_loop implementation for adaptive_tempered_smc.

    Loop-carried state: (step, rng_key, smc_state, lmbda_buf, loglik_buf)
    where ``lmbda_buf`` and ``loglik_buf`` are pre-allocated float32 buffers
    of length ``max_steps``.
    """
    lmbda_buf = jnp.zeros(max_steps, dtype=jnp.float32)
    loglik_buf = jnp.zeros(max_steps, dtype=jnp.float32)

    init_carry = (
        jnp.int32(0),
        rng_key,
        smc_init_state,
        lmbda_buf,
        loglik_buf,
    )

    def _cond(carry):
        step, _key, state, _lb, _llb = carry
        not_done_lmbda = state.tempering_param < jnp.float32(lambda_target)
        not_done_steps = step < jnp.int32(max_steps)
        return not_done_lmbda & not_done_steps

    def _body(carry):
        step, key, state, lb, llb = carry
        key, subkey = jax.random.split(key)
        new_state, info = smc_step_fn(subkey, state)
        lb = lb.at[step].set(jnp.float32(new_state.tempering_param))
        llb = llb.at[step].set(jnp.float32(info.log_likelihood_increment))
        return (step + jnp.int32(1), key, new_state, lb, llb)

    final_step, _, final_state, lmbda_buf, loglik_buf = jax.lax.while_loop(
        _cond, _body, init_carry
    )
    # Slice the filled prefix (dynamic slice on static-shape buffer)
    lmbda_hist = lmbda_buf[:final_step]
    loglik_hist = loglik_buf[:final_step]

    return final_state, {"lmbda": lmbda_hist, "log_likelihood_increment": loglik_hist}


def _run_smc_scan(
    rng_key: jax.Array,
    *,
    smc_init_state: Any,
    smc_step_fn: Callable,
    max_steps: int,
) -> tuple[Any, dict]:
    """scan implementation for non-tempering SMC variants (fixed step count).

    Returns empty history arrays (shape ``(0,)``).
    """

    def _step(carry, _):
        key, state = carry
        key, subkey = jax.random.split(key)
        new_state, _info = smc_step_fn(subkey, state)
        return (key, new_state), None

    (_, final_state), _ = jax.lax.scan(
        _step, (rng_key, smc_init_state), None, length=max_steps
    )

    empty = jnp.zeros(0, dtype=jnp.float32)
    return final_state, {"lmbda": empty, "log_likelihood_increment": empty}
