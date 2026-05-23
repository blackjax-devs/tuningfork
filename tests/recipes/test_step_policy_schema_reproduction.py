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
"""Schema wiring reproduction check — V0 step_policy=None baseline.

Confirms that wiring step_policy=None (V0, library default) through the
_recipe_runner pipeline reproduces the same FAIL verdict as the committed
FAILED recipes for:

  - ill_cond_50 × window_adaptation_diag_imm × dynamic_hmc
  - lotka_volterra × window_adaptation_diag_imm × dynamic_hmc

These cells FAILed in earlier work because the library default
integration_steps_fn (uniform L in [1, 10)) is too short for:
  - ill_cond_50 (κ≈1000, needs L ~ sqrt(κ) ≈ 32+)
  - lotka_volterra (stiff ODE; NIS_med=87 far exceeds L_max=10)

Purpose: confirm the step_policy schema wiring does NOT change V0 behaviour
before introducing non-V0 variants.

Exact numeric match is NOT required (PRNG / JAX version drift is expected).
The test confirms:
  - verdict is NOT PASS (auto-gate rejects, as expected)
  - rhat > 1.01 (chain not converged, as expected)
  - ESS < 400   (insufficient samples, as expected)

Marked ``@pytest.mark.slow`` — runs JAX warmup + sampling (~2–5 min/cell).
"""

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    "model_name,target_acceptance",
    [
        ("ill_cond_50", 0.8),
        # lotka_volterra at ta=0.99 per §7 of d-hmc-integration-steps-fn-matrix.md
        # (heuristic refinement #5: stiff ODE at high ta is the paradox case)
        ("lotka_volterra", 0.99),
    ],
)
def test_v0_step_policy_reproduces_failed_verdict(
    model_name: str,
    target_acceptance: float,
    tmp_path,
) -> None:
    """V0 step_policy=None on a known-FAIL cell still produces a FAIL verdict.

    This is the schema-wiring sanity check: confirms that wiring step_policy=None
    through the recipe runner pipeline does not accidentally change the sampler
    behaviour (it should reproduce the same FAIL outcome as earlier work).

    The test does NOT assert exact rhat/ESS values — those may drift slightly
    across JAX/BlackJAX versions.  It asserts only that:
      1. The verdict is FAIL or REVIEW (not PASS).
      2. rhat > 1.01 (chain not converged).
      3. ESS < 400 (insufficient effective samples).
    """
    from tuningfork.recipes._recipe_runner import (
        RECIPE_N_SAMPLES,
        RECIPE_N_WARMUP,
        RECIPE_NUM_CHAINS,
        RECIPE_SEED,
        emit_low_recipe_for_cell,
    )

    result = emit_low_recipe_for_cell(
        model_name=model_name,
        warmup_name="window_adaptation_diag_imm",
        sampler_name="dynamic_hmc",
        n_warmup=RECIPE_N_WARMUP,
        n_samples=RECIPE_N_SAMPLES,
        num_chains=RECIPE_NUM_CHAINS,
        seed=RECIPE_SEED,
        catalog_root=tmp_path,
        outcomes_file=tmp_path / "outcomes.md",
        verbose=True,
        target_acceptance=target_acceptance,
        step_policy=None,  # V0: library default (uniform L in [1, 10))
    )

    # 1. Must NOT pass the auto-gate (FAIL is expected)
    assert result.verdict != "PASS", (
        f"{model_name} × dynamic_hmc with V0 step_policy unexpectedly PASSED "
        f"(rhat={result.gate_rhat_max}, ESS={result.gate_min_ess}). "
        f"Something changed upstream — investigate."
    )

    # 2. rhat must be out-of-spec (chain not converged)
    assert result.gate_rhat_max is not None
    assert result.gate_rhat_max > 1.01, (
        f"{model_name}: rhat={result.gate_rhat_max:.4f} is unexpectedly near 1.0 "
        f"for a known-FAIL cell with V0 step_policy."
    )

    # 3. ESS must be sub-gate (insufficient effective samples)
    assert result.gate_min_ess is not None
    assert result.gate_min_ess < 400, (
        f"{model_name}: min_ess={result.gate_min_ess:.1f} unexpectedly ≥ 400 "
        f"for a known-FAIL cell with V0 step_policy."
    )
