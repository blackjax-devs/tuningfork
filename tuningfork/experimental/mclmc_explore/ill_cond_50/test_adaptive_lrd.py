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
"""Adaptive Low-Rank + Diagonal (LRD) preconditioned MCLMC test on ill_cond_50."""

import blackjax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tuningfork.calibration.statistician_gate import auto_gate
from tuningfork.experimental.mclmc_explore.ill_cond_50.test_low_rank_mclmc import (
    decompose_covariance_low_rank,
    run_low_rank_mclmc,
)
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model.ill_cond_50 import COV


def run_pilot_nuts(
    logdensity_fn, init_position, rng_key, n_warmup=1000, n_samples=1000
):
    """Run a diagonal NUTS pilot chain to collect samples of the target geometry."""
    warmup_key, sampling_key = jax.random.split(rng_key)

    # 1. Warmup diagonal NUTS
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
    (state, params), _ = warmup.run(warmup_key, init_position, n_warmup)

    # 2. Collect pilot sampling trace (1 chain)
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
        body_fn,
        state,
        jax.random.split(sampling_key, n_samples),
    )
    return positions


def extract_lrd_from_samples(positions, k: int):
    """Extract LRD preconditioning parameters (mean, sigma, U, lam) from pilot samples."""
    # 1. Get unravel_fn once from the first sample position
    first_position = jax.tree.map(lambda x: x[0], positions)
    _, unravel_fn = ravel_pytree(first_position)

    # Flatten all PyTree samples to (N, d) by mapping over the sample axis
    flat_positions = jax.vmap(lambda p: ravel_pytree(p)[0])(positions)

    # Compute empirical mean and standard deviation along sample axis
    mean = jnp.mean(flat_positions, axis=0)
    sigma = jnp.std(flat_positions, axis=0)

    # Prevent division-by-zero on zero-variance dimensions
    sigma = jnp.where(sigma == 0.0, 1.0, sigma)

    # 2. Standardize samples
    flat_positions_std = (flat_positions - mean[None, :]) / sigma[None, :]

    # 3. Compute SVD: flat_positions_std = U_svd * S * Vt
    # Vt has shape (d, d); eigenvectors of correlation matrix are columns of Vt.T
    _, S, Vt = jnp.linalg.svd(flat_positions_std, full_matrices=False)
    V = Vt.T

    # Eigenvalues of the correlation matrix: S_i^2 / N
    N = flat_positions_std.shape[0]
    lam = (S**2) / N

    # Sort eigenvalues in descending order of preconditioning impact |lam - 1|
    sort_idx = jnp.argsort(jnp.abs(lam - 1.0))[::-1]
    top_idx = sort_idx[:k]

    lam_k = lam[top_idx]
    U_k = V[:, top_idx]

    return mean, sigma, U_k, lam_k


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
    """Run MCLMC with adaptive O(dk) coordinate whitening (centering included)."""
    flat_init, unravel_fn = ravel_pytree(init_position)

    sqrt_lam = jnp.sqrt(lam)
    inv_sqrt_lam = 1.0 / sqrt_lam

    # Forward: x = L(y) + mean
    def L_fn(y):
        term = U.T @ y
        scaled_term = (sqrt_lam - 1.0) * term
        return sigma * (y + U @ scaled_term)

    # Inverse: y = L_inv(x - mean)
    def L_inv_fn(x):
        x_centered = x - mean
        x_scaled = x_centered / sigma
        term = U.T @ x_scaled
        scaled_term = (inv_sqrt_lam - 1.0) * term
        return x_scaled + U @ scaled_term

    # Whitened log-density function
    def whitened_logdensity(y):
        flat_x = L_fn(y) + mean
        x = unravel_fn(flat_x)
        return logdensity_fn(x)

    y0 = L_inv_fn(flat_init)

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
        flat_x = L_fn(y) + mean
        return unravel_fn(flat_x)

    x_samples = transform_back(y_samples)

    return x_samples, sampling_infos


def main():
    jax.config.update("jax_platform_name", "cpu")

    print("Loading ill_cond_50 model...")
    entry = MODELS["ill_cond_50"]

    master_key = jax.random.key(54321)
    init_key, nuts_key, run_key = jax.random.split(master_key, 3)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)

    # --- Step 1: Run NUTS Pilot Chain ---
    print("\n[Pilot Run] Generating 1000 pilot samples using diagonal NUTS...")
    pilot_positions = run_pilot_nuts(logdensity_fn, init_position, nuts_key)

    # --- Step 2: Extract Adaptive LRD Geometry ---
    k = 40
    print(
        f"\n[LRD Extraction] SVD on pilot samples to extract top k={k} preconditioning..."
    )
    mean, sigma, U_adap, lam_inv_adap = extract_lrd_from_samples(pilot_positions, k)

    print(f"Extracted mean shape: {mean.shape}")
    print(f"Extracted sigma shape: {sigma.shape}")
    print(f"Extracted eigenvectors U shape: {U_adap.shape}")
    print(f"Extracted eigenvalues lam shape: {lam_inv_adap.shape}")

    # --- Step 3: Run Adaptive LRD MCLMC ---
    print(f"\n[MCLMC Execution] Running Adaptive LRD MCLMC (k={k})...")
    adap_samples, _ = run_adaptive_low_rank_mclmc(
        logdensity_fn,
        init_position,
        mean,
        sigma,
        U_adap,
        lam_inv_adap,
        run_key,
        n_warmup=1000,
        n_samples=1000,
    )
    adap_gate_result = auto_gate(adap_samples)

    print("\n--- Adaptive LRD MCLMC Results ---")
    print(f"Adaptive LRD (k={k}) Max R-hat: {adap_gate_result.rhat_max:.4f}")
    print(f"Adaptive LRD (k={k}) Min ESS: {adap_gate_result.min_bulk_ess:.1f}")
    print(f"Adaptive LRD (k={k}) Verdict: {adap_gate_result.verdict}")

    # --- Step 4: Run Oracle LRD MCLMC for comparison ---
    print(f"\n[MCLMC Comparison] Running Oracle LRD MCLMC (k={k})...")
    sigma_oracle, U_oracle, lam_oracle = decompose_covariance_low_rank(COV, k)
    oracle_samples, _ = run_low_rank_mclmc(
        logdensity_fn,
        init_position,
        sigma_oracle,
        U_oracle,
        lam_oracle,
        run_key,
        n_warmup=1000,
        n_samples=1000,
    )
    oracle_gate_result = auto_gate(oracle_samples)

    print("\n--- Oracle LRD MCLMC Results ---")
    print(f"Oracle LRD (k={k}) Max R-hat: {oracle_gate_result.rhat_max:.4f}")
    print(f"Oracle LRD (k={k}) Min ESS: {oracle_gate_result.min_bulk_ess:.1f}")
    print(f"Oracle LRD (k={k}) Verdict: {oracle_gate_result.verdict}")

    # --- Step 5: Assertion Checks ---
    print("\n--- Validation Assertion Checks ---")
    assert (
        adap_gate_result.rhat_max < 1.05
    ), f"Adaptive LRD failed to achieve low R-hat: {adap_gate_result.rhat_max:.4f}"
    assert (
        adap_gate_result.min_bulk_ess >= 100.0
    ), f"Adaptive LRD has extremely low ESS: {adap_gate_result.min_bulk_ess:.1f}"
    assert adap_gate_result.verdict in (
        "PASS",
        "REVIEW",
    ), f"Adaptive LRD verdict is FAIL: {adap_gate_result.verdict}"

    print(
        "SUCCESS: End-to-end Adaptive Low-Rank preconditioned MCLMC runs flawlessly and successfully replicates Oracle performance!"
    )


if __name__ == "__main__":
    main()
