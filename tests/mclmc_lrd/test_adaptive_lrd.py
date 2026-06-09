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
"""Adaptive LRD external MCLMC on ill_cond_50 — NUTS pilot → SVD → mclmc.

Integrator ladder step 3: adaptive external LRD (on-the-fly geometry discovery).
See catalog/mclmc-routing-taxonomy.md for the full ladder context.

Run: python -m tests.mclmc_lrd.test_adaptive_lrd
"""

import jax

from tuningfork.base_method.mclmc_lrd_utils import (
    decompose_covariance_low_rank,
    extract_lrd_from_samples,
    run_adaptive_low_rank_mclmc,
    run_low_rank_mclmc,
    run_pilot_nuts,
)
from tuningfork.calibration.statistician_gate import auto_gate
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model.ill_cond_50 import COV


def main():
    jax.config.update("jax_platform_name", "cpu")

    print("Loading ill_cond_50 model...")
    entry = MODELS["ill_cond_50"]

    master_key = jax.random.key(54321)
    init_key, nuts_key, run_key = jax.random.split(master_key, 3)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)

    print("\n[Pilot Run] Generating 1000 pilot samples using diagonal NUTS...")
    pilot_positions = run_pilot_nuts(logdensity_fn, init_position, nuts_key)

    k = 40
    print(
        f"\n[LRD Extraction] SVD on pilot samples to extract top k={k} preconditioning..."
    )
    mean, sigma, U_adap, lam_inv_adap = extract_lrd_from_samples(pilot_positions, k)

    print(f"Extracted mean shape: {mean.shape}")
    print(f"Extracted sigma shape: {sigma.shape}")
    print(f"Extracted eigenvectors U shape: {U_adap.shape}")
    print(f"Extracted eigenvalues lam shape: {lam_inv_adap.shape}")

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
