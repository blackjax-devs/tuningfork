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
"""Test LRD-dispatching isokinetic_mclachlan with adjusted_mclmc_dynamic on horseshoe (204-D).

Verdict: REVIEW — R-hat=1.0193, ESS=270.7 (equivalent to diagonal baseline).
Finding: Cauchy sparsity funnel (local curvature) dominates; LRD provides no
benefit over diagonal. Routing: adjusted_mclmc_dynamic is correct; MCLMC LRD
adds no value here.

Step-size scaling (0.55): unadjusted mclmc_tuning adapts a large step_size
(MH-free); scaling by 0.55 targets ~94% acceptance in the adjusted sampler.

Run: python -m tests.mclmc_lrd.test_internal_lrd_horseshoe
"""

import time

import blackjax
import jax
import jax.numpy as jnp
from blackjax.mcmc.integrators import isokinetic_mclachlan
from blackjax.mcmc.metrics import LowRankInverseMassMatrix

from tuningfork.base_method.mclmc import make_lrd_kernel
from tuningfork.base_method.mclmc_lrd_utils import (
    extract_lrd_from_samples,
    run_pilot_nuts,
)
from tuningfork.calibration.statistician_gate import auto_gate
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn


def run_internal_lrd_mclmc_dynamic(
    logdensity_fn,
    init_position,
    lrd_imm,
    rng_key,
    *,
    n_warmup: int = 1000,
    n_samples: int = 1000,
    num_chains: int = 4,
):
    """Run adjusted_mclmc_dynamic with upstream LRD isokinetic_mclachlan on horseshoe."""
    warmup_key, sampling_key = jax.random.split(rng_key)
    warmup_keys = jax.random.split(warmup_key, num_chains)
    sampling_keys = jax.random.split(sampling_key, num_chains)

    @jax.vmap
    def run_warmup_one(k, x_start):
        init_k, tune_k = jax.random.split(k)
        state = blackjax.mcmc.mclmc.init(x_start, logdensity_fn, init_k)
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

    init_positions = jax.tree.map(
        lambda x: jnp.tile(x, (num_chains, *([1] * x.ndim))), init_position
    )
    adapted_states, adaptation_states = run_warmup_one(warmup_keys, init_positions)

    print(f"Adapted step_size: {adaptation_states.step_size}")
    print(f"Adapted L: {adaptation_states.L}")

    @jax.vmap
    def run_sampling_one(k, state_pos, params):
        init_k, sample_k = jax.random.split(k)
        state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
            state_pos, logdensity_fn, init_k
        )
        steps_fn = (
            blackjax.mcmc.adjusted_mclmc_dynamic.make_random_trajectory_length_fn(True)
        )
        lrd_dynamic_kernel = blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
            integration_steps_fn=steps_fn,
            integrator=isokinetic_mclachlan,
        )
        step_size_adj = params.step_size * 0.55
        avg = jnp.maximum(1.0, params.L / step_size_adj)

        def body_fn(state, rng_key):
            state, info = lrd_dynamic_kernel(
                rng_key,
                state,
                logdensity_fn,
                step_size=step_size_adj,
                inverse_mass_matrix=lrd_imm,
                integration_steps_params=(avg,),
            )
            return state, (state.position, info)

        _, (positions, infos) = jax.lax.scan(
            body_fn, state, jax.random.split(sample_k, n_samples)
        )
        return positions, infos

    x_starts = adapted_states.position
    samples, sampling_infos = run_sampling_one(
        sampling_keys, x_starts, adaptation_states
    )
    return samples, sampling_infos


def main():
    jax.config.update("jax_platform_name", "cpu")

    print("Loading horseshoe model...")
    entry = MODELS["horseshoe"]

    master_key = jax.random.key(20260608)
    init_key, nuts_key, run_key = jax.random.split(master_key, 3)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)

    print(
        "\n[Pilot Run] Generating 1000 pilot samples using diagonal NUTS on horseshoe (204-D)..."
    )
    t0 = time.perf_counter()
    pilot_positions = run_pilot_nuts(
        logdensity_fn, init_position, nuts_key, n_warmup=1000, n_samples=1000
    )
    t_pilot = time.perf_counter() - t0
    print(f"Pilot run completed in {t_pilot:.1f}s.")

    k = 50
    print(
        f"\n[LRD Extraction] SVD on pilot samples to extract top k={k} preconditioning..."
    )
    mean, sigma, U_adap, lam_inv_adap = extract_lrd_from_samples(pilot_positions, k)

    lrd_imm = LowRankInverseMassMatrix(sigma=sigma, U=U_adap, lam=lam_inv_adap)
    print(
        f"LowRankInverseMassMatrix constructed: sigma={lrd_imm.sigma.shape}, U={lrd_imm.U.shape}, lam={lrd_imm.lam.shape}"
    )

    print(
        f"\n[MCLMC Execution] Running Internal LRD adjusted_mclmc_dynamic (k={k}) on horseshoe (204-D)..."
    )
    t0 = time.perf_counter()
    adap_samples, infos = run_internal_lrd_mclmc_dynamic(
        logdensity_fn, init_position, lrd_imm, run_key, n_warmup=10000, n_samples=1000
    )
    t_mclmc = time.perf_counter() - t0
    print(f"Internal LRD adjusted_mclmc_dynamic completed in {t_mclmc:.1f}s.")

    gate_result = auto_gate(adap_samples)

    print("\n--- Internal LRD adjusted_mclmc_dynamic results on horseshoe ---")
    print(f"Max R-hat: {gate_result.rhat_max:.4f}")
    print(f"Min Bulk ESS: {gate_result.min_bulk_ess:.1f}")
    print(f"Verdict: {gate_result.verdict}")

    mean_p_accept = jnp.mean(infos.acceptance_rate)
    print(f"Mean Acceptance Probability: {mean_p_accept:.4f}")


if __name__ == "__main__":
    main()
