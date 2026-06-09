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
"""Test dense preconditioned (coordinate-whitened) MCLMC vs diagonal MCLMC on ill_cond_50.

Integrator ladder step 1: dense Cholesky oracle (O(d²) — baseline only).
See catalog/mclmc-routing-taxonomy.md for the full ladder context.

Run: python -m tests.mclmc_lrd.test_dense_mclmc
"""

import blackjax
import jax
import jax.numpy as jnp

from tuningfork.base_method.mclmc_lrd_utils import run_dense_mclmc
from tuningfork.calibration.statistician_gate import auto_gate
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model.ill_cond_50 import COV


def run_diagonal_mclmc(
    logdensity_fn,
    init_position,
    rng_key,
    n_warmup=1000,
    n_samples=1000,
    num_chains=4,
):
    """Run standard diagonal preconditioned MCLMC for comparison."""
    warmup_key, sampling_key = jax.random.split(rng_key)
    warmup_keys = jax.random.split(warmup_key, num_chains)
    sampling_keys = jax.random.split(sampling_key, num_chains)

    @jax.vmap
    def run_warmup_one(k, x_start):
        init_k, tune_k = jax.random.split(k)
        state = blackjax.mcmc.mclmc.init(x_start, logdensity_fn, init_k)
        mclmc_kernel = blackjax.mclmc.build_kernel()
        adapted_state, adaptation_state, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel,
            num_steps=n_warmup,
            state=state,
            rng_key=tune_k,
            logdensity_fn=logdensity_fn,
            diagonal_preconditioning=True,
        )
        return adapted_state, adaptation_state

    init_positions = jax.tree.map(
        lambda x: jnp.tile(x, (num_chains, *([1] * x.ndim))), init_position
    )
    adapted_states, adaptation_states = run_warmup_one(warmup_keys, init_positions)

    @jax.vmap
    def run_sampling_one(k, state, params):
        kernel = blackjax.mclmc(
            logdensity_fn,
            step_size=params.step_size,
            L=params.L,
            inverse_mass_matrix=params.inverse_mass_matrix,
        )

        def body_fn(rng_key, state):
            state, info = kernel.step(rng_key, state)
            return state, (state.position, info)

        _, (positions, infos) = jax.lax.scan(
            lambda s, key: body_fn(key, s),
            state,
            jax.random.split(k, n_samples),
        )
        return positions, infos

    samples, sampling_infos = run_sampling_one(
        sampling_keys, adapted_states, adaptation_states
    )
    return samples, sampling_infos


def main():
    jax.config.update("jax_platform_name", "cpu")

    print("Loading ill_cond_50 model and true covariance...")
    entry = MODELS["ill_cond_50"]

    master_key = jax.random.key(12345)
    init_key, run_key = jax.random.split(master_key)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)

    print("\n--- Running Standard Diagonal MCLMC ---")
    diag_samples, _ = run_diagonal_mclmc(
        logdensity_fn, init_position, run_key, n_warmup=1000, n_samples=1000
    )
    diag_gate_result = auto_gate(diag_samples)
    print(f"Diagonal MCLMC Max R-hat: {diag_gate_result.rhat_max:.4f}")
    print(f"Diagonal MCLMC Min ESS: {diag_gate_result.min_bulk_ess:.1f}")
    print(f"Diagonal MCLMC Verdict: {diag_gate_result.verdict}")

    print("\n--- Running Coordinate-Whitened (Dense) MCLMC ---")
    dense_samples, _ = run_dense_mclmc(
        logdensity_fn, init_position, COV, run_key, n_warmup=1000, n_samples=1000
    )
    dense_gate_result = auto_gate(dense_samples)
    print(f"Dense preconditioned MCLMC Max R-hat: {dense_gate_result.rhat_max:.4f}")
    print(f"Dense preconditioned MCLMC Min ESS: {dense_gate_result.min_bulk_ess:.1f}")
    print(f"Dense preconditioned MCLMC Verdict: {dense_gate_result.verdict}")

    print("\n--- Validation Assertion Checks ---")
    assert (
        dense_gate_result.rhat_max < 1.05
    ), f"Dense MCLMC failed to achieve low R-hat: {dense_gate_result.rhat_max:.4f}"
    assert (
        dense_gate_result.min_bulk_ess >= 100.0
    ), f"Dense MCLMC has extremely low ESS: {dense_gate_result.min_bulk_ess:.1f}"
    assert dense_gate_result.verdict in (
        "PASS",
        "REVIEW",
    ), f"Dense MCLMC verdict is FAIL: {dense_gate_result.verdict}"
    assert (
        diag_gate_result.rhat_max > dense_gate_result.rhat_max
    ), "Dense preconditioning did not improve R-hat over diagonal!"
    print(
        "SUCCESS: Coordinate-whitened (dense) preconditioned MCLMC successfully bypassed rotational ill-conditioning!"
    )


if __name__ == "__main__":
    main()
