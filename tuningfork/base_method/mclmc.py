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
"""MCLMC (Microcanonical Langevin Monte Carlo) algorithm wrapper.

Note: each MCLMC step performs one integrator step. The default integrator
(isokinetic_mclachlan) costs 2 gradient evaluations per integrator step
(palindromic [b1,a1,b2,a1,b1]
scheme → 2 position updates) → constant 2 grads per kernel step.
MCLMCInfo does NOT carry num_integration_steps; the trajectory length L (in
time units) controls momentum-resample cadence (L / step_size time units).

Init note: blackjax.mclmc.init requires an rng_key to generate the initial
unit-vector momentum. Call kernel.init(position, rng_key) rather than the
rng_key-free form used by HMC/MALA/Barker.

Requires pytree_size(position) >= 2 (enforced by blackjax upstream).

Adaptation: BlackJAX provides blackjax.mclmc_find_L_and_step_size as a
dedicated warmup routine. Generated recipe planning dispatches to it based on
BaseMethod.name. This module only declares the entry.

LRD Preconditioning
-------------------
``make_lrd_kernel`` is the permanent production entry point for running MCLMC
with a Low-Rank + Diagonal (LRD) inverse mass matrix.  It wraps the upstream
blackjax kernel (whose isokinetic_mclachlan integrator dispatches natively on
``LowRankInverseMassMatrix`` since blackjax PR #936) and statically binds the
LRD mass matrix so that ``blackjax.mclmc_find_L_and_step_size``
(which uses ``diagonal_preconditioning=False``) always receives the correct
geometry during warmup.

Full LRD pipeline helpers (``decompose_covariance_low_rank``,
``extract_lrd_from_samples``, ``run_pilot_nuts``, ``run_internal_lrd_mclmc``)
are also exported from this module.
"""

from collections.abc import Callable
from typing import Any

import blackjax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = [
    "ENTRY",
    "make_lrd_kernel",
    "decompose_covariance_low_rank",
    "extract_lrd_from_samples",
    "run_pilot_nuts",
    "run_internal_lrd_mclmc",
]


# ──────────────────────────────────────────────────────────────────────────────
# LRD kernel factory
# ──────────────────────────────────────────────────────────────────────────────


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

    Example
    -------
    >>> from blackjax.mcmc.metrics import LowRankInverseMassMatrix
    >>> from tuningfork.base_method.mclmc import make_lrd_kernel
    >>> lrd_imm = LowRankInverseMassMatrix(sigma=sigma, U=U, lam=lam)
    >>> kernel = make_lrd_kernel(lrd_imm)
    >>> adapted_state, params, _ = blackjax.mclmc_find_L_and_step_size(
    ...     kernel, num_steps=1000, state=state, rng_key=key,
    ...     logdensity_fn=logdensity_fn, diagonal_preconditioning=False)
    """
    base_kernel = blackjax.mclmc.build_kernel()

    def kernel(rng_key, state, logdensity_fn, inverse_mass_matrix, L, step_size):
        # Always route through lrd_imm; the warmup passes a placeholder diagonal
        # when diagonal_preconditioning=False, which we deliberately override here.
        return base_kernel(rng_key, state, logdensity_fn, lrd_imm, L, step_size)

    return kernel


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
    external-whitening variants, this function does NOT wrap or translate
    ``logdensity_fn`` — it operates natively in the original coordinate frame,
    preserving prior-centering for hierarchical models.

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
# BaseMethod entry
# ──────────────────────────────────────────────────────────────────────────────

ENTRY = BaseMethod(
    name="mclmc",
    family="mcmc",
    factory=blackjax.mclmc,  # signature: (logdensity_fn, L, step_size, ...)
    grad_count_per_step=lambda info: jnp.asarray(2),
    grad_count_convention="2",
    default_hp_space=(
        HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
        HyperparamSpace("L", "loguniform", low=0.1, high=100.0),
    ),
    needs_mass_matrix=False,
    target_acceptance_rate=None,  # rejection-free; not applicable
    # T2.3 descriptors: MCLMC family — L is also per-chain from mclmc_tuning warmup.
    per_chain_param_keys=("step_size", "inverse_mass_matrix", "L"),
    reinit_state=False,  # MCLMCState from mclmc_tuning is directly usable.
    extra_kwarg_builder=None,  # No extra kwargs beyond logdensity_fn + HP-space.
    notes=(
        "Constant 2 grads/step (default isokinetic_mclachlan integrator). "
        "MCLMCInfo._fields = ('logdensity', 'kinetic_change', 'energy_change', 'nonans'); "
        "no num_integration_steps field. "
        "inverse_mass_matrix=1.0 default is scalar (global preconditioner). "
        "init requires rng_key: kernel.init(position, rng_key). "
        "Dedicated adaptation: blackjax.mclmc_find_L_and_step_size — not window_adaptation. "
        "pytree_size(position) >= 2 required by upstream."
    ),
)
