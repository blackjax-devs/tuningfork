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
"""Phase B slow integration test — MEDIUM-with-policy emit path.

Tests that ``emit_low_recipe_for_cell`` with ``policy_tag`` + ``Effort.MEDIUM``
produces a MEDIUM recipe with the correct filename, effort, and step_policy.

Uses ``mvn_10 × window_adaptation_diag_imm × dynamic_hmc`` with a V7-style
empirical spec as the integration cell (fast-converging model; avoids the
known-FAIL ill_cond_50 cell which would make this test slow AND non-deterministic).

Verified properties:
- Recipe file is saved at ``medium__dynamic_hmc__window_adaptation_diag_imm__policy_test.json``
- ``recipe.effort == Effort.MEDIUM``
- ``recipe.step_policy["kind"] == "empirical"``
- ``recipe.step_policy["values"]`` and ``recipe.step_policy["weights"]`` match the input
- The PASS gate fires on mvn_10 (trivial geometry; any reasonable step_policy works)
"""

import pytest

pytestmark = pytest.mark.slow


def test_medium_with_policy_tag_mvn10(tmp_path):
    """MEDIUM recipe with policy_tag is saved at the correct tagged filename."""
    import numpy as np

    from tuningfork.recipes._base import Effort
    from tuningfork.recipes._recipe_runner import emit_low_recipe_for_cell

    # Deterministic empirical spec: bimodal at L=3 and L=7 (well within mvn_10 range)
    policy_spec = {
        "kind": "empirical",
        "values": [3, 5, 7],
        "weights": [0.25, 0.50, 0.25],
    }
    policy_tag = "policy_test"

    result = emit_low_recipe_for_cell(
        "mvn_10",
        "window_adaptation_diag_imm",
        "dynamic_hmc",
        n_warmup=500,
        n_samples=500,
        num_chains=2,
        seed=20260517,
        catalog_root=tmp_path,
        outcomes_file=tmp_path / "outcomes.md",
        verbose=False,
        step_policy=policy_spec,
        policy_tag=policy_tag,
        effort=Effort.MEDIUM,
    )

    # mvn_10 with any sane step_policy should PASS
    assert result.verdict == "PASS", (
        f"Expected PASS for mvn_10×dynamic_hmc with empirical policy; "
        f"got {result.verdict} (rhat={result.gate_rhat_max}, ess={result.gate_min_ess})"
    )

    # Recipe file should exist at the tagged MEDIUM path
    expected_path = (
        tmp_path
        / "mvn_10"
        / "recipes"
        / f"medium__dynamic_hmc__window_adaptation_diag_imm__{policy_tag}.json"
    )
    assert (
        expected_path.exists()
    ), f"Expected recipe at {expected_path}; got result.recipe_path={result.recipe_path}"

    # Load and verify the recipe
    from tuningfork.recipes._base import Recipe

    recipe = Recipe.load(expected_path)
    assert recipe.effort == Effort.MEDIUM
    assert recipe.step_policy is not None
    assert recipe.step_policy["kind"] == "empirical"
    assert recipe.step_policy["values"] == policy_spec["values"]
    assert np.allclose(recipe.step_policy["weights"], policy_spec["weights"], atol=1e-6)


def test_medium_with_policy_tag_none_preserves_low(tmp_path):
    """policy_tag=None (default) preserves LOW effort and canonical filename."""
    from tuningfork.recipes._recipe_runner import emit_low_recipe_for_cell

    result = emit_low_recipe_for_cell(
        "mvn_10",
        "window_adaptation_diag_imm",
        "dynamic_hmc",
        n_warmup=500,
        n_samples=500,
        num_chains=2,
        seed=20260517,
        catalog_root=tmp_path,
        outcomes_file=tmp_path / "outcomes.md",
        verbose=False,
        # policy_tag=None (default), effort=Effort.LOW (default)
    )

    assert (
        result.verdict == "PASS"
    ), f"Expected PASS for mvn_10×dynamic_hmc V0; got {result.verdict}"

    # Recipe file should be at the canonical LOW path (no tag)
    expected_path = (
        tmp_path
        / "mvn_10"
        / "recipes"
        / "low__dynamic_hmc__window_adaptation_diag_imm.json"
    )
    assert expected_path.exists(), f"Expected canonical LOW recipe at {expected_path}"

    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe.load(expected_path)
    assert recipe.effort == Effort.LOW
