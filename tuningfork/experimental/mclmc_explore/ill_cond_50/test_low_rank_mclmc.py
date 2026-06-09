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
"""Low-rank + diagonal coordinate-whitened MCLMC test on ill_cond_50."""

import blackjax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tuningfork.calibration.statistician_gate import auto_gate
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model.ill_cond_50 import COV


def decompose_covariance_low_rank(cov: jax.Array, k: int):
    """Decompose dense covariance matrix Σ into Low-Rank + Diagonal format:

        Σ ≈ diag(σ) (I + U(Λ - I)Uᵀ) diag(σ)

    Parameters
    ----------
    cov
        The dense covariance matrix, shape (d, d).
    k
        The approximation rank.

    Returns
    -------
    sigma
        Diagonal scaling vector, shape (d,).
    U
        Eigenvectors matrix with orthonormal columns, shape (d, k).
    lam
        Eigenvalues vector of the correlation matrix, shape (k,).
    """
    # σ_i = sqrt(Σ_ii)
    sigma = jnp.sqrt(jnp.diagonal(cov))

    # Correlation matrix: C = diag(1/σ) Σ diag(1/σ)
    inv_sigma = 1.0 / sigma
    C = cov * inv_sigma[:, None] * inv_sigma[None, :]

    # Symmetric Eigendecomposition of C: C = Q V Qᵀ
    eigenvals, eigenvectors = jnp.linalg.eigh(C)

    # Sort eigenvalues in descending order of deviation from 1.0 (preconditioning impact)
    # C - I has eigenvalues (eigenvals - 1.0). Sort by absolute magnitude of (eigenvals - 1).
    sort_idx = jnp.argsort(jnp.abs(eigenvals - 1.0))[::-1]

    top_idx = sort_idx[:k]
    lam = eigenvals[top_idx]
    U = eigenvectors[:, top_idx]

    return sigma, U, lam


def low_rank_whitening_fn(
    logdensity_fn,
    init_position,
    sigma: jax.Array,
    U: jax.Array,
    lam: jax.Array,
):
    """Transform logdensity_fn to a low-rank whitened coordinate space in O(dk) complexity.

    Parameters
    ----------
    logdensity_fn
        Original logdensity PyTree position -> scalar.
    init_position
        Sample PyTree position used for shape/structure.
    sigma
        Diagonal scaling vector, shape (d,).
    U
        Orthonormal low-rank eigenvectors, shape (d, k).
    lam
        Positive eigenvalues of the correlation matrix, shape (k,).
    """
    flat_init, unravel_fn = ravel_pytree(init_position)

    sqrt_lam = jnp.sqrt(lam)
    inv_sqrt_lam = 1.0 / sqrt_lam

    # L(y) = sigma * (y + U @ ((sqrt_lam - 1) * (Uᵀ @ y)))
    def L_fn(y):
        term = U.T @ y
        scaled_term = (sqrt_lam - 1.0) * term
        return sigma * (y + U @ scaled_term)

    # L_inv(x) = (x/sigma) + U @ ((inv_sqrt_lam - 1) * (Uᵀ @ (x/sigma)))
    def L_inv_fn(x):
        x_scaled = x / sigma
        term = U.T @ x_scaled
        scaled_term = (inv_sqrt_lam - 1.0) * term
        return x_scaled + U @ scaled_term

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
    """Run MCLMC with O(dk) low-rank coordinate whitening."""
    # 1. Low-rank whitening
    whitened_logdensity, unravel_fn, L_fn, L_inv_fn, x_ref = low_rank_whitening_fn(
        logdensity_fn, init_position, sigma, U, lam
    )

    flat_init, _ = ravel_pytree(init_position)
    y0 = L_inv_fn(flat_init - x_ref)

    # 2. Parallel Warmup/Tuning in whitened space
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

    # 3. Parallel Sampling in whitened space
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
            body_fn,
            state,
            jax.random.split(k, n_samples),
        )
        return y_positions, infos

    y_samples, sampling_infos = run_sampling_one(
        sampling_keys, adapted_states, adaptation_states
    )

    # 4. Transform samples back to original space
    @jax.vmap  # chains
    @jax.vmap  # samples
    def transform_back(y):
        flat_x = L_fn(y) + x_ref
        return unravel_fn(flat_x)

    x_samples = transform_back(y_samples)

    return x_samples, sampling_infos


def main():
    jax.config.update("jax_platform_name", "cpu")

    print("Loading ill_cond_50 model...")
    entry = MODELS["ill_cond_50"]

    master_key = jax.random.key(12345)
    init_key, run_key = jax.random.split(master_key)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)

    # We will test rank k=10, 20, 30, 40 to see the effect of rank k.
    for k in (10, 20, 30, 40):
        print(f"\n--- Decomposing COV with Rank k={k} ---")
        sigma, U, lam = decompose_covariance_low_rank(COV, k)

        print(f"Running Low-Rank Coordinate-Whitened MCLMC (k={k})...")
        lr_samples, _ = run_low_rank_mclmc(
            logdensity_fn,
            init_position,
            sigma,
            U,
            lam,
            run_key,
            n_warmup=1000,
            n_samples=1000,
        )
        gate_result = auto_gate(lr_samples)

        print(f"Low-Rank (k={k}) MCLMC Max R-hat: {gate_result.rhat_max:.4f}")
        print(f"Low-Rank (k={k}) MCLMC Min ESS: {gate_result.min_bulk_ess:.1f}")
        print(f"Low-Rank (k={k}) MCLMC Verdict: {gate_result.verdict}")

        # Verify it successfully passed or got a review verdict for higher ranks
        if k >= 30:
            assert (
                gate_result.rhat_max < 1.05
            ), f"Low-Rank k={k} failed to achieve low R-hat: {gate_result.rhat_max:.4f}"
            assert (
                gate_result.min_bulk_ess >= 100.0
            ), f"Low-Rank k={k} has extremely low ESS: {gate_result.min_bulk_ess:.1f}"
            assert gate_result.verdict in (
                "PASS",
                "REVIEW",
            ), f"Low-Rank k={k} verdict is FAIL: {gate_result.verdict}"

    print(
        "\nSUCCESS: Low-rank coordinate-whitened MCLMC successfully bypassed ill-conditioning with O(dk) cost!"
    )


if __name__ == "__main__":
    main()
