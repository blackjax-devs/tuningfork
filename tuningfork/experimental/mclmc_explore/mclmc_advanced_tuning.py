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
"""Coordinate-whitened (dense preconditioned) MCLMC wrapper and LRD kernel factory."""

from collections.abc import Callable
from typing import Any

import blackjax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def make_lrd_kernel(lrd_imm: Any) -> Callable:
    """Wrap the upstream blackjax mclmc kernel with a statically-bound LRD mass matrix.

    Now that ``blackjax.mcmc.integrators.isokinetic_mclachlan`` dispatches natively on
    ``LowRankInverseMassMatrix`` (landed in blackjax PR #936), this thin closure is the
    only tuningfork-side wiring needed: it binds the LRD mass matrix so that
    ``mclmc_find_L_and_step_size(..., diagonal_preconditioning=False)`` always sees the
    correct geometry during warmup regardless of the placeholder IMM the warmup passes.

    Parameters
    ----------
    lrd_imm
        A ``blackjax.mcmc.metrics.LowRankInverseMassMatrix`` instance.

    Returns
    -------
    kernel
        A callable with the same signature as ``blackjax.mclmc.build_kernel()``'s output
        but with ``lrd_imm`` baked in.
    """
    base_kernel = blackjax.mclmc.build_kernel()

    def kernel(rng_key, state, logdensity_fn, inverse_mass_matrix, L, step_size):
        # Always route through lrd_imm; the warmup passes a placeholder diagonal
        # when diagonal_preconditioning=False, which we deliberately override here.
        return base_kernel(rng_key, state, logdensity_fn, lrd_imm, L, step_size)

    return kernel


def whiten_logdensity_fn(
    logdensity_fn: Callable[[Any], jax.Array],
    init_position: Any,
    dense_imm: jax.Array,
) -> tuple[
    Callable[[jax.Array], jax.Array],
    Callable[[jax.Array], Any],
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Transform logdensity_fn to a whitened space where position y has identity covariance.

    Parameters
    ----------
    logdensity_fn
        The original logdensity function mapping PyTree position x to scalar.
    init_position
        A sample PyTree position x used to establish the structure and shape.
    dense_imm
        A dense inverse mass matrix (covariance matrix of position, Σ), of shape (d, d).

    Returns
    -------
    whitened_logdensity
        A function mapping whitened 1D array y of shape (d,) to scalar log-density.
    unravel_fn
        A function mapping a flat 1D array of shape (d,) back to the original PyTree.
    L
        The lower Cholesky factor of dense_imm, shape (d, d).
    L_inv
        The inverse of L, shape (d, d).
    x_ref
        The reference centering point (zeros of the same flat shape as x).
    """
    flat_init, unravel_fn = ravel_pytree(init_position)
    d = flat_init.shape[0]

    if dense_imm.shape != (d, d):
        raise ValueError(
            f"dense_imm shape mismatch: expected {(d, d)}, got {dense_imm.shape}"
        )

    # Compute Cholesky decomposition of dense inverse mass matrix (Σ)
    L = jnp.linalg.cholesky(dense_imm)
    # Compute L_inv for coordinate projection
    L_inv = jnp.linalg.solve(L, jnp.eye(d))

    # Center at 0 in flat space
    x_ref = jnp.zeros_like(flat_init)

    def whitened_logdensity(y: jax.Array) -> jax.Array:
        # Transform whitened coordinate y back to original flat space: x = L y + x_ref
        flat_x = L @ y + x_ref
        # Unflatten flat_x to the original PyTree
        x = unravel_fn(flat_x)
        return logdensity_fn(x)

    return whitened_logdensity, unravel_fn, L, L_inv, x_ref


def run_dense_mclmc(
    logdensity_fn: Callable[[Any], jax.Array],
    init_position: Any,
    dense_imm: jax.Array,
    rng_key: jax.Array,
    *,
    n_warmup: int = 1000,
    n_samples: int = 1000,
    num_chains: int = 4,
) -> tuple[Any, Any]:
    """Run coordinate-whitened MCLMC using a dense inverse mass matrix.

    This scales unadjusted MCLMC to handle rotational ill-conditioning by running
    standard isotropic MCLMC (diagonal mass matrix = ones) on the whitened coordinate
    space and mapping the samples back.

    Parameters
    ----------
    logdensity_fn
        The original logdensity function.
    init_position
        A sample PyTree position.
    dense_imm
        The dense inverse mass matrix (covariance matrix of position), shape (d, d).
    rng_key
        Master JAX random key.
    n_warmup
        MCLMC tuning/warmup steps (default 1000).
    n_samples
        Post-warmup sampling draws (default 1000).
    num_chains
        Number of parallel chains (default 4).

    Returns
    -------
    x_samples
        Unflattened PyTree of samples in original space, with leading dims (num_chains, n_samples).
    sampling_infos
        MCLMC sampling information batched over chains.
    """
    # 1. Coordinate whitening transformation
    whitened_logdensity, unravel_fn, L, L_inv, x_ref = whiten_logdensity_fn(
        logdensity_fn, init_position, dense_imm
    )

    flat_init, _ = ravel_pytree(init_position)
    # Project the initial position into the whitened space
    y0 = L_inv @ (flat_init - x_ref)

    # 2. Parallel Warmup/Tuning in whitened space
    warmup_key, sampling_key = jax.random.split(rng_key)
    warmup_keys = jax.random.split(warmup_key, num_chains)

    @jax.vmap
    def run_warmup_one(k: jax.Array, y_start: jax.Array) -> tuple[Any, Any]:
        init_k, tune_k = jax.random.split(k)
        # Standard MCLMC state init in whitened space
        state = blackjax.mcmc.mclmc.init(y_start, whitened_logdensity, init_k)
        mclmc_kernel = blackjax.mclmc.build_kernel()
        # Tune L and step_size with diagonal preconditioning in whitened space
        adapted_state, adaptation_state, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel,
            num_steps=n_warmup,
            state=state,
            rng_key=tune_k,
            logdensity_fn=whitened_logdensity,
            diagonal_preconditioning=True,
        )
        return adapted_state, adaptation_state

    # Replicate initial position across chains
    y0_replicated = jnp.tile(y0, (num_chains, 1))
    adapted_states, adaptation_states = run_warmup_one(warmup_keys, y0_replicated)

    # 3. Parallel Sampling in whitened space
    sampling_keys = jax.random.split(sampling_key, num_chains)

    @jax.vmap
    def run_sampling_one(
        k: jax.Array, state: Any, params: Any
    ) -> tuple[jax.Array, Any]:
        kernel = blackjax.mclmc(
            whitened_logdensity,
            step_size=params.step_size,
            L=params.L,
            inverse_mass_matrix=params.inverse_mass_matrix,
        )

        def body_fn(
            rng_key: jax.Array, state: Any
        ) -> tuple[Any, tuple[jax.Array, Any]]:
            state, info = kernel.step(rng_key, state)
            return state, (state.position, info)

        _, (y_positions, infos) = jax.lax.scan(
            lambda s, key: body_fn(key, s),
            state,
            jax.random.split(k, n_samples),
        )
        return y_positions, infos

    y_samples, sampling_infos = run_sampling_one(
        sampling_keys, adapted_states, adaptation_states
    )

    # 4. Transform samples back to original PyTree space
    @jax.vmap  # over chains
    @jax.vmap  # over samples
    def transform_back(y: jax.Array) -> Any:
        flat_x = L @ y + x_ref
        return unravel_fn(flat_x)

    x_samples = transform_back(y_samples)

    return x_samples, sampling_infos
