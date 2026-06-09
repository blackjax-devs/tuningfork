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
"""Low-Rank + Diagonal (LRD) MCLMC research utilities — permanent production home.

These utilities implement the full LRD preconditioning pipeline validated during
the MCLMC-LRD integration experiment (blackjax PR #936 / tuningfork PR #176):

    NUTS pilot run → SVD extraction → LRD kernel → mclmc_tuning warmup → sampling

The integrator ladder (dense Cholesky oracle → external LRD → adaptive LRD →
internal LRD) is documented in ``tuningfork/catalog/mclmc-routing-taxonomy.md``.

Public API
----------
``make_lrd_kernel``
    Canonical entry point.  Re-exported from ``tuningfork.base_method.mclmc``.
``decompose_covariance_low_rank``
    Decompose a known dense covariance into LRD format (oracle variant).
``extract_lrd_from_samples``
    Extract LRD parameters from a set of posterior samples (adaptive variant).
``run_pilot_nuts``
    Cheap 1-chain NUTS pilot to collect geometry samples for LRD extraction.
``run_low_rank_mclmc``
    External LRD coordinate-whitened MCLMC (modifies logdensity_fn).
``run_adaptive_low_rank_mclmc``
    Adaptive external LRD MCLMC with mean centering (modifies logdensity_fn).
``run_internal_lrd_mclmc``
    Native internal LRD MCLMC — does NOT modify logdensity_fn.  Production path.
``whiten_logdensity_fn``
    Dense Cholesky coordinate whitening helper (oracle; used in test_dense_mclmc).
``run_dense_mclmc``
    Dense preconditioned MCLMC (oracle baseline, O(d²) — for comparison only).
"""

from collections.abc import Callable
from typing import Any

import blackjax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tuningfork.base_method.mclmc import make_lrd_kernel  # re-export

__all__ = [
    "make_lrd_kernel",
    "decompose_covariance_low_rank",
    "extract_lrd_from_samples",
    "run_pilot_nuts",
    "run_low_rank_mclmc",
    "run_adaptive_low_rank_mclmc",
    "run_internal_lrd_mclmc",
    "whiten_logdensity_fn",
    "run_dense_mclmc",
]


# ──────────────────────────────────────────────────────────────────────────────
# LRD decomposition helpers
# ──────────────────────────────────────────────────────────────────────────────


def decompose_covariance_low_rank(cov: jax.Array, k: int):
    """Decompose dense covariance Σ into Low-Rank + Diagonal format.

    Returns (sigma, U, lam) such that
        Σ ≈ diag(σ)(I + U(Λ - I)Uᵀ)diag(σ)

    Parameters
    ----------
    cov
        Dense covariance matrix, shape (d, d).
    k
        Approximation rank.

    Returns
    -------
    sigma : jax.Array, shape (d,)
        Diagonal scaling vector (square-root of diagonal of cov).
    U : jax.Array, shape (d, k)
        Eigenvectors matrix with orthonormal columns.
    lam : jax.Array, shape (k,)
        Eigenvalues of the correlation matrix (top k by |lam - 1|).
    """
    sigma = jnp.sqrt(jnp.diagonal(cov))
    inv_sigma = 1.0 / sigma
    C = cov * inv_sigma[:, None] * inv_sigma[None, :]
    eigenvals, eigenvectors = jnp.linalg.eigh(C)
    sort_idx = jnp.argsort(jnp.abs(eigenvals - 1.0))[::-1]
    top_idx = sort_idx[:k]
    lam = eigenvals[top_idx]
    U = eigenvectors[:, top_idx]
    return sigma, U, lam


def extract_lrd_from_samples(positions, k: int):
    """Extract LRD preconditioning parameters from posterior samples.

    Parameters
    ----------
    positions
        PyTree of samples, first axis is the sample dimension.
    k
        Rank of the LRD approximation.

    Returns
    -------
    mean : jax.Array, shape (d,)
    sigma : jax.Array, shape (d,)
    U : jax.Array, shape (d, k)
    lam : jax.Array, shape (k,)
    """
    first_position = jax.tree.map(lambda x: x[0], positions)
    _, unravel_fn = ravel_pytree(first_position)
    flat_positions = jax.vmap(lambda p: ravel_pytree(p)[0])(positions)

    mean = jnp.mean(flat_positions, axis=0)
    sigma = jnp.std(flat_positions, axis=0)
    sigma = jnp.where(sigma == 0.0, 1.0, sigma)

    flat_positions_std = (flat_positions - mean[None, :]) / sigma[None, :]
    _, S, Vt = jnp.linalg.svd(flat_positions_std, full_matrices=False)
    V = Vt.T

    N = flat_positions_std.shape[0]
    lam = (S**2) / N
    sort_idx = jnp.argsort(jnp.abs(lam - 1.0))[::-1]
    top_idx = sort_idx[:k]

    lam_k = lam[top_idx]
    U_k = V[:, top_idx]
    return mean, sigma, U_k, lam_k


# ──────────────────────────────────────────────────────────────────────────────
# Pilot NUTS helper
# ──────────────────────────────────────────────────────────────────────────────


def run_pilot_nuts(
    logdensity_fn, init_position, rng_key, n_warmup=1000, n_samples=1000
):
    """Run a single diagonal-NUTS pilot chain to collect geometry samples.

    Used as the first stage of the adaptive LRD pipeline:
    pilot samples → ``extract_lrd_from_samples`` → LRD kernel.
    """
    warmup_key, sampling_key = jax.random.split(rng_key)
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
    (state, params), _ = warmup.run(warmup_key, init_position, n_warmup)

    step_size = (
        params["step_size"]
        if isinstance(params, dict)
        else getattr(params, "step_size")
    )
    inverse_mass_matrix = (
        params["inverse_mass_matrix"]
        if isinstance(params, dict)
        else getattr(params, "inverse_mass_matrix")
    )
    kernel = blackjax.nuts(
        logdensity_fn,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    def body_fn(state, key):
        state, info = kernel.step(key, state)
        return state, state.position

    _, positions = jax.lax.scan(
        body_fn, state, jax.random.split(sampling_key, n_samples)
    )
    return positions


# ──────────────────────────────────────────────────────────────────────────────
# External LRD MCLMC (coordinate-whitening variants — modifies logdensity_fn)
# ──────────────────────────────────────────────────────────────────────────────


def _low_rank_whitening_fn(logdensity_fn, init_position, sigma, U, lam):
    """Build whitened logdensity + transform functions for external LRD MCLMC."""
    flat_init, unravel_fn = ravel_pytree(init_position)
    sqrt_lam = jnp.sqrt(lam)
    inv_sqrt_lam = 1.0 / sqrt_lam

    def L_fn(y):
        return sigma * (y + U @ ((sqrt_lam - 1.0) * (U.T @ y)))

    def L_inv_fn(x):
        x_scaled = x / sigma
        return x_scaled + U @ ((inv_sqrt_lam - 1.0) * (U.T @ x_scaled))

    x_ref = jnp.zeros_like(flat_init)

    def whitened_logdensity(y):
        flat_x = L_fn(y) + x_ref
        x = unravel_fn(flat_x)
        return logdensity_fn(x)

    return whitened_logdensity, unravel_fn, L_fn, L_inv_fn, x_ref


def run_low_rank_mclmc(
    logdensity_fn,
    init_position,
    sigma,
    U,
    lam,
    rng_key,
    *,
    n_warmup: int = 1000,
    n_samples: int = 1000,
    num_chains: int = 4,
):
    """Run MCLMC with O(dk) external low-rank coordinate whitening.

    This is the external-whitening variant: it wraps ``logdensity_fn`` in a
    change-of-variables.  Avoid for hierarchical models (breaks prior-centering).
    Use ``run_internal_lrd_mclmc`` for production.
    """
    whitened_logdensity, unravel_fn, L_fn, L_inv_fn, x_ref = _low_rank_whitening_fn(
        logdensity_fn, init_position, sigma, U, lam
    )
    flat_init, _ = ravel_pytree(init_position)
    y0 = L_inv_fn(flat_init - x_ref)

    warmup_key, sampling_key = jax.random.split(rng_key)
    warmup_keys = jax.random.split(warmup_key, num_chains)

    @jax.vmap
    def run_warmup_one(k, y_start):
        init_k, tune_k = jax.random.split(k)
        state = blackjax.mcmc.mclmc.init(y_start, whitened_logdensity, init_k)
        mclmc_kernel = blackjax.mclmc.build_kernel()
        adapted_state, adaptation_state, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel,
            num_steps=n_warmup,
            state=state,
            rng_key=tune_k,
            logdensity_fn=whitened_logdensity,
            diagonal_preconditioning=True,
        )
        return adapted_state, adaptation_state

    y0_replicated = jnp.tile(y0, (num_chains, 1))
    adapted_states, adaptation_states = run_warmup_one(warmup_keys, y0_replicated)

    sampling_keys = jax.random.split(sampling_key, num_chains)

    @jax.vmap
    def run_sampling_one(k, state, params):
        kernel = blackjax.mclmc(
            whitened_logdensity,
            step_size=params.step_size,
            L=params.L,
            inverse_mass_matrix=params.inverse_mass_matrix,
        )

        def body_fn(state, rng_key):
            state, info = kernel.step(rng_key, state)
            return state, (state.position, info)

        _, (y_positions, infos) = jax.lax.scan(
            body_fn, state, jax.random.split(k, n_samples)
        )
        return y_positions, infos

    y_samples, sampling_infos = run_sampling_one(
        sampling_keys, adapted_states, adaptation_states
    )

    @jax.vmap
    @jax.vmap
    def transform_back(y):
        flat_x = L_fn(y) + x_ref
        return unravel_fn(flat_x)

    x_samples = transform_back(y_samples)
    return x_samples, sampling_infos


def run_adaptive_low_rank_mclmc(
    logdensity_fn,
    init_position,
    mean,
    sigma,
    U,
    lam,
    rng_key,
    *,
    n_warmup: int = 1000,
    n_samples: int = 1000,
    num_chains: int = 4,
):
    """Run MCLMC with adaptive O(dk) external LRD whitening (mean-centred).

    Includes mean-centering: ``x = L(y) + mean``.
    Avoid for hierarchical models.  Use ``run_internal_lrd_mclmc`` for production.
    """
    flat_init, unravel_fn = ravel_pytree(init_position)
    sqrt_lam = jnp.sqrt(lam)
    inv_sqrt_lam = 1.0 / sqrt_lam

    def L_fn(y):
        return sigma * (y + U @ ((sqrt_lam - 1.0) * (U.T @ y)))

    def L_inv_fn(x):
        x_centered = x - mean
        x_scaled = x_centered / sigma
        return x_scaled + U @ ((inv_sqrt_lam - 1.0) * (U.T @ x_scaled))

    def whitened_logdensity(y):
        flat_x = L_fn(y) + mean
        x = unravel_fn(flat_x)
        return logdensity_fn(x)

    y0 = L_inv_fn(flat_init)

    warmup_key, sampling_key = jax.random.split(rng_key)
    warmup_keys = jax.random.split(warmup_key, num_chains)

    @jax.vmap
    def run_warmup_one(k, y_start):
        init_k, tune_k = jax.random.split(k)
        state = blackjax.mcmc.mclmc.init(y_start, whitened_logdensity, init_k)
        mclmc_kernel = blackjax.mclmc.build_kernel()
        adapted_state, adaptation_state, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel,
            num_steps=n_warmup,
            state=state,
            rng_key=tune_k,
            logdensity_fn=whitened_logdensity,
            diagonal_preconditioning=True,
        )
        return adapted_state, adaptation_state

    y0_replicated = jnp.tile(y0, (num_chains, 1))
    adapted_states, adaptation_states = run_warmup_one(warmup_keys, y0_replicated)

    sampling_keys = jax.random.split(sampling_key, num_chains)

    @jax.vmap
    def run_sampling_one(k, state, params):
        kernel = blackjax.mclmc(
            whitened_logdensity,
            step_size=params.step_size,
            L=params.L,
            inverse_mass_matrix=params.inverse_mass_matrix,
        )

        def body_fn(state, rng_key):
            state, info = kernel.step(rng_key, state)
            return state, (state.position, info)

        _, (y_positions, infos) = jax.lax.scan(
            body_fn, state, jax.random.split(k, n_samples)
        )
        return y_positions, infos

    y_samples, sampling_infos = run_sampling_one(
        sampling_keys, adapted_states, adaptation_states
    )

    @jax.vmap
    @jax.vmap
    def transform_back(y):
        flat_x = L_fn(y) + mean
        return unravel_fn(flat_x)

    x_samples = transform_back(y_samples)
    return x_samples, sampling_infos


# ──────────────────────────────────────────────────────────────────────────────
# Internal LRD MCLMC  (does NOT modify logdensity_fn — production path)
# ──────────────────────────────────────────────────────────────────────────────


def run_internal_lrd_mclmc(
    logdensity_fn: Callable[[Any], jax.Array],
    init_position: Any,
    lrd_imm: Any,
    rng_key: jax.Array,
    *,
    n_warmup: int = 1000,
    n_samples: int = 1000,
    num_chains: int = 4,
):
    """Run MCLMC using the upstream LRD-dispatching isokinetic_mclachlan integrator.

    This is the **production path** for LRD-preconditioned MCLMC.  Unlike the
    external-whitening variants (``run_low_rank_mclmc``, ``run_adaptive_low_rank_mclmc``),
    this function does NOT wrap or translate ``logdensity_fn`` — it operates natively
    in the original coordinate frame, preserving prior-centering for hierarchical models.

    The LRD inverse mass matrix is statically bound via ``make_lrd_kernel``, which
    overrides the placeholder IMM that ``mclmc_find_L_and_step_size`` uses when
    ``diagonal_preconditioning=False``.

    Parameters
    ----------
    logdensity_fn
        Log-density function, PyTree position → scalar.
    init_position
        Starting PyTree position.
    lrd_imm
        A ``blackjax.mcmc.metrics.LowRankInverseMassMatrix`` instance.
    rng_key
        Master JAX random key.
    n_warmup
        Number of mclmc_tuning warmup steps (adapts L and step_size).
    n_samples
        Post-warmup sampling draws per chain.
    num_chains
        Number of parallel chains.

    Returns
    -------
    samples
        PyTree of samples with leading dims ``(num_chains, n_samples)``.
    sampling_infos
        MCLMC info batched over chains.
    """
    warmup_key, sampling_key = jax.random.split(rng_key)
    warmup_keys = jax.random.split(warmup_key, num_chains)
    sampling_keys = jax.random.split(sampling_key, num_chains)

    @jax.vmap
    def _warmup_one_chain(k):
        init_k, tune_k = jax.random.split(k)
        state = blackjax.mcmc.mclmc.init(init_position, logdensity_fn, init_k)
        kernel = make_lrd_kernel(lrd_imm)
        adapted_state, adaptation_state, _ = blackjax.mclmc_find_L_and_step_size(
            kernel,
            num_steps=n_warmup,
            state=state,
            rng_key=tune_k,
            logdensity_fn=logdensity_fn,
            diagonal_preconditioning=False,
        )
        return adapted_state, adaptation_state

    adapted_states, adaptation_states = _warmup_one_chain(warmup_keys)

    @jax.vmap
    def run_sampling_one(k, state, params):
        kernel = make_lrd_kernel(lrd_imm)

        def body_fn(state, rng_key):
            state, info = kernel(
                rng_key,
                state,
                logdensity_fn,
                inverse_mass_matrix=lrd_imm,
                L=params.L,
                step_size=params.step_size,
            )
            return state, (state.position, info)

        _, (positions, infos) = jax.lax.scan(
            body_fn, state, jax.random.split(k, n_samples)
        )
        return positions, infos

    samples, sampling_infos = run_sampling_one(
        sampling_keys, adapted_states, adaptation_states
    )
    return samples, sampling_infos


# ──────────────────────────────────────────────────────────────────────────────
# Dense Cholesky coordinate whitening (oracle / comparison only, O(d²))
# ──────────────────────────────────────────────────────────────────────────────


def whiten_logdensity_fn(
    logdensity_fn: Callable[[Any], jax.Array],
    init_position: Any,
    dense_imm: jax.Array,
):
    """Transform logdensity_fn to a whitened space with identity covariance.

    Uses the full dense Cholesky factor — O(d²) cost.  For comparison / oracle
    use only; use LRD variants for production.

    Returns
    -------
    whitened_logdensity, unravel_fn, L, L_inv, x_ref
    """
    flat_init, unravel_fn = ravel_pytree(init_position)
    d = flat_init.shape[0]
    if dense_imm.shape != (d, d):
        raise ValueError(
            f"dense_imm shape mismatch: expected {(d, d)}, got {dense_imm.shape}"
        )
    L = jnp.linalg.cholesky(dense_imm)
    L_inv = jnp.linalg.solve(L, jnp.eye(d))
    x_ref = jnp.zeros_like(flat_init)

    def whitened_logdensity(y: jax.Array) -> jax.Array:
        flat_x = L @ y + x_ref
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
):
    """Run coordinate-whitened MCLMC using a dense inverse mass matrix (oracle).

    O(d²) cost — for oracle comparison only.  Use ``run_internal_lrd_mclmc``
    for production.

    Returns
    -------
    x_samples
        Samples in the original PyTree space, shape ``(num_chains, n_samples, ...)``.
    sampling_infos
        MCLMC info batched over chains.
    """
    whitened_logdensity, unravel_fn, L, L_inv, x_ref = whiten_logdensity_fn(
        logdensity_fn, init_position, dense_imm
    )
    flat_init, _ = ravel_pytree(init_position)
    y0 = L_inv @ (flat_init - x_ref)

    warmup_key, sampling_key = jax.random.split(rng_key)
    warmup_keys = jax.random.split(warmup_key, num_chains)

    @jax.vmap
    def run_warmup_one(k, y_start):
        init_k, tune_k = jax.random.split(k)
        state = blackjax.mcmc.mclmc.init(y_start, whitened_logdensity, init_k)
        mclmc_kernel = blackjax.mclmc.build_kernel()
        adapted_state, adaptation_state, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel,
            num_steps=n_warmup,
            state=state,
            rng_key=tune_k,
            logdensity_fn=whitened_logdensity,
            diagonal_preconditioning=True,
        )
        return adapted_state, adaptation_state

    y0_replicated = jnp.tile(y0, (num_chains, 1))
    adapted_states, adaptation_states = run_warmup_one(warmup_keys, y0_replicated)

    sampling_keys = jax.random.split(sampling_key, num_chains)

    @jax.vmap
    def run_sampling_one(k, state, params):
        kernel = blackjax.mclmc(
            whitened_logdensity,
            step_size=params.step_size,
            L=params.L,
            inverse_mass_matrix=params.inverse_mass_matrix,
        )

        def body_fn(state, rng_key):
            state, info = kernel.step(rng_key, state)
            return state, (state.position, info)

        _, (y_positions, infos) = jax.lax.scan(
            body_fn, state, jax.random.split(k, n_samples)
        )
        return y_positions, infos

    y_samples, sampling_infos = run_sampling_one(
        sampling_keys, adapted_states, adaptation_states
    )

    @jax.vmap
    @jax.vmap
    def transform_back(y):
        flat_x = L @ y + x_ref
        return unravel_fn(flat_x)

    x_samples = transform_back(y_samples)
    return x_samples, sampling_infos
