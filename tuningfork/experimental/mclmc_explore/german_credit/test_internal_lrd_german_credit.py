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
"""Test upstream LRD-dispatching isokinetic_mclachlan on german_credit (d=26).

The custom lrd_integrator.py has been retired.  We now use the standard
blackjax.mclmc kernel, which dispatches to the LRD path in
isokinetic_mclachlan when passed a LowRankInverseMassMatrix (blackjax PR #936).
"""

import time

import blackjax
import jax
import jax.numpy as jnp
from blackjax.mcmc.metrics import LowRankInverseMassMatrix

from tuningfork.calibration.statistician_gate import auto_gate
from tuningfork.experimental.mclmc_explore import make_lrd_kernel
from tuningfork.experimental.mclmc_explore.ill_cond_50.test_adaptive_lrd import (
    extract_lrd_from_samples,
    run_pilot_nuts,
)
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn


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
    """Run MCLMC using the upstream LRD-dispatching isokinetic_mclachlan on german_credit."""
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
    # Force CPU backend
    jax.config.update("jax_platform_name", "cpu")

    print("Loading german_credit model...")
    entry = MODELS["german_credit"]

    master_key = jax.random.key(20260608)
    init_key, nuts_key, run_key = jax.random.split(master_key, 3)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)

    # --- Step 1: Run NUTS Pilot Chain ---
    print(
        "\n[Pilot Run] Generating 1000 pilot samples using diagonal NUTS on german_credit (26-D)..."
    )
    t0 = time.perf_counter()
    pilot_positions = run_pilot_nuts(
        logdensity_fn, init_position, nuts_key, n_warmup=1000, n_samples=1000
    )
    t_pilot = time.perf_counter() - t0
    print(f"Pilot run completed in {t_pilot:.1f}s.")

    # --- Step 2: Extract Adaptive LRD Geometry ---
    k = 26
    print(
        f"\n[LRD Extraction] SVD on pilot samples to extract top k={k} preconditioning..."
    )
    mean, sigma, U_adap, lam_inv_adap = extract_lrd_from_samples(pilot_positions, k)

    # Construct standard LowRankInverseMassMatrix object
    lrd_imm = LowRankInverseMassMatrix(sigma=sigma, U=U_adap, lam=lam_inv_adap)
    print(
        f"LowRankInverseMassMatrix constructed: sigma={lrd_imm.sigma.shape}, U={lrd_imm.U.shape}, lam={lrd_imm.lam.shape}"
    )

    # --- Step 3: Run Internal LRD MCLMC ---
    print(
        f"\n[MCLMC Execution] Running Custom Internal LRD MCLMC (k={k}) on german_credit (26-D)..."
    )
    t0 = time.perf_counter()
    adap_samples, _ = run_internal_lrd_mclmc(
        logdensity_fn, init_position, lrd_imm, run_key, n_warmup=1000, n_samples=1000
    )
    t_mclmc = time.perf_counter() - t0
    print(f"Internal LRD MCLMC completed in {t_mclmc:.1f}s.")

    # Evaluate metrics
    gate_result = auto_gate(adap_samples)

    print("\n--- Internal LRD MCLMC results on german_credit ---")
    print(f"Max R-hat: {gate_result.rhat_max:.4f}")
    print(f"Min Bulk ESS: {gate_result.min_bulk_ess:.1f}")
    print(f"Verdict: {gate_result.verdict}")


if __name__ == "__main__":
    main()
