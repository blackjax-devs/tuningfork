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
"""Stress-test upstream LRD-dispatching isokinetic_mclachlan on stoch_vol (d=503).

Two results recorded (see catalog/stoch_vol/lessons.md for full analysis):
  - External whitening: R-hat=2.0536, FAIL — breaks prior-centering.
  - Internal LRD: R-hat=1.0498, ESS=156.1, REVIEW — preserves coordinate structure
    but non-linear funnel curvature limits mixing.

Prior-centering preservation principle: the internal integrator rescales/rotates
momentum only; it does NOT translate position coordinates.  This is why it
outperforms external whitening on hierarchical models.

Run: python -m tests.mclmc_lrd.test_internal_lrd_stoch_vol
"""

import time

import jax
from blackjax.mcmc.metrics import LowRankInverseMassMatrix

from tuningfork.base_method.mclmc_lrd_utils import (
    extract_lrd_from_samples,
    run_internal_lrd_mclmc,
    run_pilot_nuts,
)
from tuningfork.calibration.statistician_gate import auto_gate
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn


def main():
    jax.config.update("jax_platform_name", "cpu")

    print("Loading stoch_vol model...")
    entry = MODELS["stoch_vol"]

    master_key = jax.random.key(1234567)
    init_key, nuts_key, run_key = jax.random.split(master_key, 3)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)

    print(
        "\n[Pilot Run] Generating 1000 pilot samples using diagonal NUTS on stoch_vol (503-D)..."
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
        f"\n[MCLMC Execution] Running Internal LRD MCLMC (k={k}) on stoch_vol (503-D)..."
    )
    t0 = time.perf_counter()
    adap_samples, _ = run_internal_lrd_mclmc(
        logdensity_fn, init_position, lrd_imm, run_key, n_warmup=1000, n_samples=1000
    )
    t_mclmc = time.perf_counter() - t0
    print(f"Internal LRD MCLMC completed in {t_mclmc:.1f}s.")

    gate_result = auto_gate(adap_samples)

    print("\n--- Internal LRD MCLMC results on stoch_vol ---")
    print(f"Max R-hat: {gate_result.rhat_max:.4f}")
    print(f"Min Bulk ESS: {gate_result.min_bulk_ess:.1f}")
    print(f"Verdict: {gate_result.verdict}")

    print("\n--- Scientific Finding ---")
    print(
        "stoch_vol has non-linear varying curvature (AR(1) funnel geometry). "
        "Internal LRD preserves prior-centering (unlike external whitening) but "
        "cannot flatten non-linear curvature. REVIEW is the expected ceiling."
    )


if __name__ == "__main__":
    main()
