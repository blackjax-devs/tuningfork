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

Integrator ladder step 4: internal LRD MCLMC (native ESH dispatch, no logdensity
wrapping).  This is the PRODUCTION path.  Certified PASS: R-hat=1.0030, ESS=2079.5
(@statistician multi-seed hardening, 2026-06-09).

Run: python -m tests.mclmc_lrd.test_internal_lrd
"""

import jax
from blackjax.mcmc.metrics import LowRankInverseMassMatrix

from tuningfork.base_method.mclmc_lrd_utils import (
    decompose_covariance_low_rank,
    run_internal_lrd_mclmc,
)
from tuningfork.calibration.statistician_gate import auto_gate
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model.ill_cond_50 import COV


def main():
    jax.config.update("jax_platform_name", "cpu")

    print("Loading ill_cond_50 model...")
    entry = MODELS["ill_cond_50"]

    master_key = jax.random.key(98765)
    init_key, run_key = jax.random.split(master_key)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)

    k = 40
    print(f"\n--- Decomposing COV with Rank k={k} ---")
    sigma, U, lam = decompose_covariance_low_rank(COV, k)

    lrd_imm = LowRankInverseMassMatrix(sigma=sigma, U=U, lam=lam)
    print(
        f"LowRankInverseMassMatrix constructed: sigma={lrd_imm.sigma.shape}, U={lrd_imm.U.shape}, lam={lrd_imm.lam.shape}"
    )

    print("\nRunning MCLMC with native LRD dispatch via isokinetic_mclachlan...")
    samples, _ = run_internal_lrd_mclmc(
        logdensity_fn, init_position, lrd_imm, run_key, n_warmup=1000, n_samples=1000
    )

    gate_result = auto_gate(samples)
    print(f"\nInternal LRD MCLMC Max R-hat: {gate_result.rhat_max:.4f}")
    print(f"Internal LRD MCLMC Min ESS: {gate_result.min_bulk_ess:.1f}")
    print(f"Internal LRD MCLMC Verdict: {gate_result.verdict}")

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
        "\nSUCCESS: Internal LRD preconditioned MCLMC kernel runs flawlessly and solves ill_cond_50!"
    )


if __name__ == "__main__":
    main()
