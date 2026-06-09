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
"""Cheap Discovery Phase test using blackjax.multipathfinder + LRD MCLMC."""

import time

import blackjax
import jax
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from jax.flatten_util import ravel_pytree

from tuningfork.calibration.statistician_gate import auto_gate
from tuningfork.experimental.mclmc_explore.german_credit.test_internal_lrd_german_credit import (
    run_internal_lrd_mclmc,
)
from tuningfork.experimental.mclmc_explore.ill_cond_50.test_adaptive_lrd import (
    extract_lrd_from_samples,
)
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn


def run_multipathfinder_discovery(
    logdensity_fn, init_position, rng_key, n_paths=16, n_samples=1000
):
    """Run multipathfinder and draw samples from the PSIS-weighted approximation."""
    init_key, run_key, sample_key = jax.random.split(rng_key, 3)

    # Flatten the logdensity for multipathfinder
    flat_init, unravel_fn = ravel_pytree(init_position)
    ndim = flat_init.shape[0]

    def flat_logdensity(y):
        return logdensity_fn(unravel_fn(y))

    # Generate initial positions for optimization paths
    # Adding small random perturbations around the initial position to encourage path exploration
    initial_positions = (
        flat_init[None, :] + jax.random.normal(init_key, (n_paths, ndim)) * 0.1
    )

    print(f"Initializing multipathfinder with {n_paths} paths in {ndim}-D...")
    algo = blackjax.multipathfinder(flat_logdensity)
    state, info = algo.init(run_key, initial_positions, num_samples=100)

    print("Drawing samples from multipathfinder PSIS-weighted approximation...")
    flat_samples = algo.sample(sample_key, state, n_samples)

    # Unflatten back to PyTree positions
    unravel_vmap = jax.vmap(unravel_fn)
    positions = unravel_vmap(flat_samples)

    return positions, info


def evaluate_target(model_name: str, k: int):
    print("\n==================================================")
    print(f"Evaluating {model_name} (SVD top k={k})...")
    print("==================================================")

    entry = MODELS[model_name]
    master_key = jax.random.key(13579)
    init_key, discovery_key, run_key = jax.random.split(master_key, 3)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)

    # --- Step 1: Cheap Discovery Phase via Multipathfinder ---
    t0 = time.perf_counter()
    pilot_positions, pf_info = run_multipathfinder_discovery(
        logdensity_fn, init_position, discovery_key, n_paths=16, n_samples=1000
    )
    t_discovery = time.perf_counter() - t0
    print(f"Cheap Discovery Phase completed in {t_discovery:.2f}s.")

    # --- Step 2: SVD LRD Extraction ---
    print(
        f"\n[LRD Extraction] SVD on Pathfinder samples to extract top k={k} preconditioning..."
    )
    mean, sigma, U_adap, lam_inv_adap = extract_lrd_from_samples(pilot_positions, k)

    # Print eigenvalues to check quality of approximation
    print(f"Extracted lam_inv_adap eigenvalues (top 10): {lam_inv_adap[:10]}")
    print(f"Extracted sigma stds (top 10): {sigma[:10]}")

    # Construct LowRankInverseMassMatrix
    lrd_imm = LowRankInverseMassMatrix(sigma=sigma, U=U_adap, lam=lam_inv_adap)
    print(
        f"LowRankInverseMassMatrix constructed: sigma={lrd_imm.sigma.shape}, U={lrd_imm.U.shape}, lam={lrd_imm.lam.shape}"
    )

    # --- Step 3: Run Adaptive Internal LRD MCLMC ---
    print(
        f"\n[MCLMC Execution] Running Custom Internal LRD MCLMC (k={k}) on {model_name}..."
    )
    t0 = time.perf_counter()
    samples, _ = run_internal_lrd_mclmc(
        logdensity_fn, init_position, lrd_imm, run_key, n_warmup=1000, n_samples=1000
    )
    t_mclmc = time.perf_counter() - t0
    print(f"Internal LRD MCLMC completed in {t_mclmc:.2f}s.")

    # --- Step 4: Evaluate Convergence ---
    gate_result = auto_gate(samples)
    print(f"\n--- {model_name} Pathfinder-LRD results ---")
    print(f"Max R-hat: {gate_result.rhat_max:.4f}")
    print(f"Min Bulk ESS: {gate_result.min_bulk_ess:.1f}")
    print(f"Verdict: {gate_result.verdict}")


def main():
    jax.config.update("jax_platform_name", "cpu")

    evaluate_target("german_credit", k=26)
    evaluate_target("ill_cond_50", k=40)


if __name__ == "__main__":
    main()
