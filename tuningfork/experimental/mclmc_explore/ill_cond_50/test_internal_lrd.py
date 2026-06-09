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
"""Test the upstream LRD-dispatching isokinetic_mclachlan integrator on ill_cond_50.

The custom lrd_integrator.py has been retired.  We now use the standard
blackjax.mclmc kernel, which dispatches to the LRD path in
isokinetic_mclachlan when passed a LowRankInverseMassMatrix (blackjax PR #936).
"""

import blackjax
import jax
import jax.numpy as jnp
from blackjax.mcmc.metrics import LowRankInverseMassMatrix

from tuningfork.calibration.statistician_gate import auto_gate
from tuningfork.experimental.mclmc_explore import make_lrd_kernel
from tuningfork.experimental.mclmc_explore.ill_cond_50.test_low_rank_mclmc import (
    decompose_covariance_low_rank,
)
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model.ill_cond_50 import COV


def run_internal_lrd_mclmc(
    logdensity_fn,
    init_position,
    lrd_imm,
    rng_key,
    *,
    n_warmup: int = 1000,
    n_samples: int = 1000,
    num_chains: int = 4,
):
    """Run MCLMC using the upstream LRD-dispatching isokinetic_mclachlan integrator."""
    warmup_key, sampling_key = jax.random.split(rng_key)
    warmup_keys = jax.random.split(warmup_key, num_chains)
    sampling_keys = jax.random.split(sampling_key, num_chains)

    # 1. Warmup / Adaptation
    @jax.vmap
    def run_warmup_one(k, x_start):
        init_k, tune_k = jax.random.split(k)
        state = blackjax.mcmc.mclmc.init(x_start, logdensity_fn, init_k)

        # make_lrd_kernel binds lrd_imm so warmup's diagonal_preconditioning=False
        # path always calls isokinetic_mclachlan with the correct LRD geometry.
        kernel = make_lrd_kernel(lrd_imm)

        # Adapt only step_size and L; LRD mass matrix is held constant.
        adapted_state, adaptation_state, _ = blackjax.mclmc_find_L_and_step_size(
            kernel,
            num_steps=n_warmup,
            state=state,
            rng_key=tune_k,
            logdensity_fn=logdensity_fn,
            diagonal_preconditioning=False,
        )
        return adapted_state, adaptation_state

    # Replicate init_position across chains
    init_positions = jax.tree.map(
        lambda x: jnp.tile(x, (num_chains, *([1] * x.ndim))), init_position
    )
    adapted_states, adaptation_states = run_warmup_one(warmup_keys, init_positions)

    # 2. Parallel Sampling with the upstream LRD kernel
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
            body_fn,
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

    print("Loading ill_cond_50 model...")
    entry = MODELS["ill_cond_50"]

    master_key = jax.random.key(98765)
    init_key, run_key = jax.random.split(master_key)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)

    # --- Step 1: Decompose COV and Build LowRankInverseMassMatrix ---
    k = 40
    print(f"\n--- Decomposing COV with Rank k={k} ---")
    sigma, U, lam = decompose_covariance_low_rank(COV, k)

    # Construct standard LowRankInverseMassMatrix object
    lrd_imm = LowRankInverseMassMatrix(sigma=sigma, U=U, lam=lam)
    print(
        f"LowRankInverseMassMatrix constructed: sigma={lrd_imm.sigma.shape}, U={lrd_imm.U.shape}, lam={lrd_imm.lam.shape}"
    )

    # --- Step 2: Run MCLMC with Native Internal LRD Kernel ---
    print("\nRunning custom MCLMC kernel with native LRD support...")
    samples, _ = run_internal_lrd_mclmc(
        logdensity_fn, init_position, lrd_imm, run_key, n_warmup=1000, n_samples=1000
    )

    # --- Step 3: Evaluate Convergence ---
    gate_result = auto_gate(samples)
    print(f"\nInternal LRD MCLMC Max R-hat: {gate_result.rhat_max:.4f}")
    print(f"Internal LRD MCLMC Min ESS: {gate_result.min_bulk_ess:.1f}")
    print(f"Internal LRD MCLMC Verdict: {gate_result.verdict}")

    # Assertions: Should cleanly pass convergence checks
    assert (
        gate_result.rhat_max < 1.05
    ), f"Internal LRD MCLMC failed to achieve low R-hat: {gate_result.rhat_max:.4f}"
    assert (
        gate_result.min_bulk_ess >= 100.0
    ), f"Internal LRD MCLMC has extremely low ESS: {gate_result.min_bulk_ess:.1f}"
    assert gate_result.verdict in (
        "PASS",
        "REVIEW",
    ), f"Internal LRD MCLMC verdict is FAIL: {gate_result.verdict}"

    print(
        "\nSUCCESS: Custom internal LRD preconditioned MCLMC kernel runs flawlessly and solves ill_cond_50!"
    )


if __name__ == "__main__":
    main()
