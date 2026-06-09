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
"""Low-rank + diagonal external coordinate-whitened MCLMC on ill_cond_50.

Integrator ladder step 2: external LRD (O(dk) — rank progression k=10..40).
See catalog/mclmc-routing-taxonomy.md for the full ladder context.

Run: python -m tests.mclmc_lrd.test_low_rank_mclmc
"""

import jax

from tuningfork.base_method.mclmc_lrd_utils import (
    decompose_covariance_low_rank,
    run_low_rank_mclmc,
)
from tuningfork.calibration.statistician_gate import auto_gate
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model.ill_cond_50 import COV


def main():
    jax.config.update("jax_platform_name", "cpu")

    print("Loading ill_cond_50 model...")
    entry = MODELS["ill_cond_50"]

    master_key = jax.random.key(12345)
    init_key, run_key = jax.random.split(master_key)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)

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
