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
"""Unit tests for warmup_num_chains schema field and dispatch logic.

Pure-logic and schema validation tests — no JAX traces, no chain runs.
All tests are @pytest.mark.fast.
"""

import json

import pytest

from tuningfork.recipes._base import Effort, Recipe, validate_warmup_num_chains

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_recipe(**overrides):
    """Return a minimal Recipe with optional field overrides."""
    defaults = dict(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1, "inverse_mass_matrix": [1.0] * 10},
        warmup_params={"n_warmup": 100, "num_chains": 4},
        warmups=[
            {
                "name": "window_adaptation_diag_imm",
                "params": {"n_warmup": 100, "num_chains": 4},
            }
        ],
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"trials": 0, "wall_seconds_estimate": 0.0},
        difficulty=None,
        instructions="",
        notes="",
        tuning_seed=42,
        tuningfork_version="0.0.0.dev0",
        blackjax_version="0.0.0",
        jax_version="0.0.0",
        timestamp_utc="2026-01-01T00:00:00Z",
    )
    defaults.update(overrides)
    return Recipe(**defaults)


# ---------------------------------------------------------------------------
# validate_warmup_num_chains
# ---------------------------------------------------------------------------


def test_validate_none_always_passes():
    """None is always valid (backward-compat default)."""
    validate_warmup_num_chains(None, n_phases=1)
    validate_warmup_num_chains(None, n_phases=3)


def test_validate_single_phase_valid():
    """[1], [4], [8] all valid for n_phases=1."""
    validate_warmup_num_chains([1], n_phases=1)
    validate_warmup_num_chains([4], n_phases=1)
    # W > S is also valid (more warmup chains than sampling chains).
    validate_warmup_num_chains([8], n_phases=1)


def test_validate_multi_phase_valid():
    """Matching length with all W>=1 is valid."""
    validate_warmup_num_chains([1, 1], n_phases=2)
    validate_warmup_num_chains([4, 4], n_phases=2)
    validate_warmup_num_chains([1, 4], n_phases=2)
    validate_warmup_num_chains([8, 2], n_phases=2)


def test_validate_w_greater_than_s_valid():
    """W > num_chains is explicitly valid (reduce+broadcast handles it)."""
    validate_warmup_num_chains([8], n_phases=1)
    validate_warmup_num_chains([16, 16], n_phases=2)


def test_validate_w_zero_raises():
    """W=0 raises ValueError (must be >= 1)."""
    with pytest.raises(ValueError, match="warmup_num_chains\\[0\\] must be >= 1"):
        validate_warmup_num_chains([0], n_phases=1)


def test_validate_w_negative_raises():
    """Negative W raises ValueError."""
    with pytest.raises(ValueError, match="warmup_num_chains\\[0\\] must be >= 1"):
        validate_warmup_num_chains([-1], n_phases=1)


def test_validate_length_mismatch_raises():
    """Length != n_phases raises ValueError."""
    with pytest.raises(ValueError, match="lengths must match"):
        validate_warmup_num_chains([1, 1], n_phases=1)
    with pytest.raises(ValueError, match="lengths must match"):
        validate_warmup_num_chains([1], n_phases=2)


def test_validate_not_list_raises():
    """Non-list (e.g. int) raises ValueError."""
    with pytest.raises(ValueError, match="must be a list\\[int\\]"):
        validate_warmup_num_chains(4, n_phases=1)  # type: ignore[arg-type]


def test_validate_non_int_element_raises():
    """Float element raises ValueError."""
    with pytest.raises(ValueError, match="warmup_num_chains\\[0\\] must be an int"):
        validate_warmup_num_chains([1.0], n_phases=1)  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Recipe field defaults and round-trip
# ---------------------------------------------------------------------------


def test_recipe_warmup_num_chains_default_none():
    """Recipe defaults warmup_num_chains to None."""
    recipe = _make_minimal_recipe()
    assert recipe.warmup_num_chains is None


def test_recipe_warmup_num_chains_set():
    """Recipe accepts warmup_num_chains=[1]."""
    recipe = _make_minimal_recipe(warmup_num_chains=[1])
    assert recipe.warmup_num_chains == [1]


def test_recipe_warmup_num_chains_w_gt_s():
    """Recipe accepts W > num_chains (W=8 with num_chains=4 is valid)."""
    recipe = _make_minimal_recipe(warmup_num_chains=[8])
    assert recipe.warmup_num_chains == [8]


def test_recipe_save_load_roundtrip_null(tmp_path):
    """warmup_num_chains=None round-trips through JSON."""
    recipe = _make_minimal_recipe(warmup_num_chains=None)
    saved_path = recipe.save(tmp_path)
    loaded = Recipe.load(saved_path)
    assert loaded.warmup_num_chains is None


def test_recipe_save_load_roundtrip_set(tmp_path):
    """warmup_num_chains=[1] round-trips through JSON."""
    recipe = _make_minimal_recipe(warmup_num_chains=[1])
    saved_path = recipe.save(tmp_path)
    loaded = Recipe.load(saved_path)
    assert loaded.warmup_num_chains == [1]


def test_recipe_save_load_roundtrip_w_gt_s(tmp_path):
    """warmup_num_chains=[8] (W > num_chains=4) round-trips through JSON."""
    recipe = _make_minimal_recipe(warmup_num_chains=[8])
    saved_path = recipe.save(tmp_path)
    loaded = Recipe.load(saved_path)
    assert loaded.warmup_num_chains == [8]


def test_recipe_load_legacy_json_no_field(tmp_path):
    """Legacy recipe JSON without warmup_num_chains loads with None default."""
    # Construct a minimal recipe, save it, then strip warmup_num_chains from JSON.
    recipe = _make_minimal_recipe(warmup_num_chains=[1])
    saved_path = recipe.save(tmp_path)
    # Remove the field from the on-disk JSON
    d = json.loads(saved_path.read_text())
    d.pop("warmup_num_chains", None)
    saved_path.write_text(json.dumps(d))
    # Load should succeed with default None
    loaded = Recipe.load(saved_path)
    assert loaded.warmup_num_chains is None


def test_recipe_load_multiphase_valid(tmp_path):
    """Multi-phase recipe with warmup_num_chains=[1, 1] round-trips."""
    recipe = _make_minimal_recipe(
        warmups=[
            {
                "name": "window_adaptation_diag_imm",
                "params": {"n_warmup": 50, "num_chains": 4},
            },
            {
                "name": "window_adaptation_dense_imm",
                "params": {"n_warmup": 50, "num_chains": 4},
            },
        ],
        warmup_num_chains=[1, 1],
    )
    saved_path = recipe.save(tmp_path)
    loaded = Recipe.load(saved_path)
    assert loaded.warmup_num_chains == [1, 1]


def test_recipe_load_validation_length_mismatch_raises(tmp_path):
    """Loading a recipe JSON with wrong-length warmup_num_chains raises ValueError."""
    recipe = _make_minimal_recipe(warmup_num_chains=[1])
    saved_path = recipe.save(tmp_path)
    # Corrupt the JSON: set warmup_num_chains to 2-element list but warmups has 1 phase
    d = json.loads(saved_path.read_text())
    d["warmup_num_chains"] = [1, 1]  # wrong length
    saved_path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="lengths must match"):
        Recipe.load(saved_path)


def test_recipe_load_validation_w_zero_raises(tmp_path):
    """Loading a recipe JSON with W=0 raises ValueError."""
    recipe = _make_minimal_recipe(warmup_num_chains=[1])
    saved_path = recipe.save(tmp_path)
    d = json.loads(saved_path.read_text())
    d["warmup_num_chains"] = [0]
    saved_path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="must be >= 1"):
        Recipe.load(saved_path)


# ---------------------------------------------------------------------------
# emit_script template selection
# ---------------------------------------------------------------------------


def test_emit_script_w1_forces_singlechain_template():
    """warmup_num_chains=[1] selects single-chain template regardless of progress_bar."""
    from tuningfork.recipes._emit_script import emit_script

    recipe = _make_minimal_recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="window_adaptation_diag_imm",
        warmup_num_chains=None,  # recipe default
    )
    # With progress_bar=False and no warmup_num_chains override, selects multichain.
    script_multi = emit_script(recipe, progress_bar=False)
    assert "_warmup_is_perchain = True" in script_multi  # multichain template marker

    # With warmup_num_chains=[1], forces single-chain template even with pb=False.
    script_single = emit_script(recipe, progress_bar=False, warmup_num_chains=[1])
    # Single-chain template sets _warmup_is_perchain = False.
    assert "_warmup_is_perchain = False" in script_single
    assert "_warmup_is_perchain = True" not in script_single


def test_emit_script_w_eq_s_same_as_none():
    """warmup_num_chains=[4] with num_chains=4 behaves the same as None."""
    from tuningfork.recipes._emit_script import emit_script

    recipe = _make_minimal_recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="window_adaptation_diag_imm",
    )
    script_none = emit_script(recipe, progress_bar=False, num_chains=4)
    script_w4 = emit_script(
        recipe, progress_bar=False, num_chains=4, warmup_num_chains=[4]
    )
    assert script_none == script_w4


def test_emit_script_recipe_warmup_num_chains_honored():
    """Recipe-stamped warmup_num_chains=[1] is used when no override passed."""
    from tuningfork.recipes._emit_script import emit_script

    recipe = _make_minimal_recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="window_adaptation_diag_imm",
        warmup_num_chains=[1],  # recipe-stamped
    )
    script = emit_script(recipe, progress_bar=False)
    # Should use single-chain template (_warmup_is_perchain = False).
    assert "_warmup_is_perchain = False" in script
    assert "_warmup_is_perchain = True" not in script


def test_emit_script_override_wins_over_recipe():
    """Call-time warmup_num_chains override wins over recipe-stamped value."""
    from tuningfork.recipes._emit_script import emit_script

    recipe = _make_minimal_recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="window_adaptation_diag_imm",
        warmup_num_chains=[1],  # recipe-stamped = 1 (single-chain)
    )
    # Override to W=4 (== num_chains default) → should select multichain template.
    script = emit_script(
        recipe, progress_bar=False, num_chains=4, warmup_num_chains=[4]
    )
    assert "_warmup_is_perchain = True" in script
