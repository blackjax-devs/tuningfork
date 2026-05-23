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
"""Schema extension slow integration test — warmup_inner_kernel emit path.

Tests that ``emit_low_recipe_for_cell`` with ``warmup_inner_kernel="nuts"``
and ``sampler_name="hmc"`` produces a LOW recipe with:

- Correct filename including ``__inner_nuts`` tag
- ``recipe.warmup_inner_kernel == "nuts"``
- ``recipe.warmups`` list populated
- ``recipe.base_method_params`` includes ``num_integration_steps`` (injected
  from NUTS warmup NIS median via ``transform_warmup_state``)
- Auto-gate PASS on mvn_10 (well-behaved 10-D Gaussian)

Cell: mvn_10 × window_adaptation_diag_imm × hmc + inner_nuts
Budget: canonical 4 chains × 1000 warmup × 1000 samples (~30-60 s on CPU).

This is the "inner-kernel substitution rescue path" — NUTS warmup
adapts (step_size, IMM) for an HMC sampler, with the median NUTS NIS injected
as ``num_integration_steps`` for HMC. The emission should create the file:
  ``low__hmc__window_adaptation_diag_imm__inner_nuts.json``
"""

import pytest

pytestmark = pytest.mark.slow


def test_inner_nuts_hmc_emit_mvn10(tmp_path):
    """LOW recipe for hmc + inner_nuts warmup passes auto-gate on mvn_10.

    Verifies the schema-extension pipeline:
    1. emit_low_recipe_for_cell(warmup_inner_kernel="nuts", sampler_name="hmc")
    2. NUTS drives window_adaptation; NIS captured in warmup_info
    3. transform_warmup_state injects median(NIS) as num_integration_steps
    4. HMC sampler runs with NUTS-adapted (step_size, IMM, NIS-median L)
    5. Auto-gate PASS on mvn_10
    6. Recipe saved at low__hmc__window_adaptation_diag_imm__inner_nuts.json
    7. Recipe.load round-trips correctly with warmups list + warmup_inner_kernel
    """
    from tuningfork.recipes._base import Effort, Recipe
    from tuningfork.recipes._recipe_runner import emit_low_recipe_for_cell

    result = emit_low_recipe_for_cell(
        "mvn_10",
        "window_adaptation_diag_imm",
        "hmc",
        # Canonical schema-extension budget: 4 chains × 1000 warmup × 1000 samples.
        # mvn_10 is well-behaved; any reasonable (step_size, L) passes the gate.
        n_warmup=1000,
        n_samples=1000,
        num_chains=4,
        seed=20260517,
        catalog_root=tmp_path,
        outcomes_file=tmp_path / "outcomes.md",
        verbose=False,
        warmup_inner_kernel="nuts",  # Schema extension: explicit NUTS warmup for HMC
    )

    # --- Gate must PASS ---
    assert result.verdict == "PASS", (
        f"Expected PASS for mvn_10 × hmc + inner_nuts; "
        f"got {result.verdict} "
        f"(rhat_max={result.gate_rhat_max}, min_ess={result.gate_min_ess}, "
        f"n_div={result.gate_n_div})"
    )

    # --- Filename must include __inner_nuts tag (§3.5) ---
    expected_path = (
        tmp_path
        / "mvn_10"
        / "recipes"
        / "low__hmc__window_adaptation_diag_imm__inner_nuts.json"
    )
    assert expected_path.exists(), (
        f"Expected recipe at {expected_path}; "
        f"result.recipe_path={result.recipe_path}"
    )

    # --- Load and verify recipe fields ---
    recipe = Recipe.load(expected_path)

    assert recipe.effort == Effort.LOW
    assert recipe.base_method_name == "hmc"
    assert recipe.warmup_name == "window_adaptation_diag_imm"

    # Schema extension: warmup_inner_kernel persisted correctly.
    assert (
        recipe.warmup_inner_kernel == "nuts"
    ), f"Expected warmup_inner_kernel='nuts', got {recipe.warmup_inner_kernel!r}"

    # Schema extension: warmups list populated (not just flat fields).
    assert (
        recipe.warmups
    ), "recipe.warmups must be non-empty after schema-extension save/load"
    assert recipe.warmups[0]["name"] == "window_adaptation_diag_imm", (
        f"Expected warmups[0].name='window_adaptation_diag_imm', "
        f"got {recipe.warmups[0]['name']!r}"
    )

    # Schema extension: transform_warmup_state must have injected num_integration_steps
    # (nuts → hmc row in the resolution table: NIS median injection).
    assert "num_integration_steps" in recipe.base_method_params, (
        "Schema-extension nuts→hmc transform must inject num_integration_steps into "
        "base_method_params from NUTS warmup NIS median; field is missing. "
        f"Actual base_method_params keys: {list(recipe.base_method_params.keys())}"
    )
    nis_value = recipe.base_method_params["num_integration_steps"]
    assert (
        isinstance(nis_value, int) and nis_value >= 1
    ), f"num_integration_steps must be a positive int, got {nis_value!r}"

    # Schema extension: the recipe must NOT write legacy flat warmup fields to JSON.
    import json

    raw = json.loads(expected_path.read_text())
    assert "warmups" in raw, "New-format recipe JSON must contain 'warmups' key"
    assert "warmup_name" not in raw, (
        "Schema extension §2.4: Recipe.save must NOT write legacy 'warmup_name' field; "
        f"found it in {expected_path}"
    )
    assert "warmup_params" not in raw, (
        "Schema extension §2.4: Recipe.save must NOT write legacy 'warmup_params' field; "
        f"found it in {expected_path}"
    )
