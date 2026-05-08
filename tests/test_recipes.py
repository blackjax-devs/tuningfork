"""Tests for bjx_bench.inference.recipes (Recipe dataclass + Effort enum).

Phase 2.5 commit 3: covers the LOW-effort path only. MEDIUM and HIGH
constructors are stubs (NotImplementedError).

Tests
-----
1. Effort enum — values are lowercase strings.
2. Recipe — constructs from kwargs; frozen (no mutation).
3. Recipe.from_default_config — NUTS + mvn_10 produces the expected LOW recipe.
4. Recipe.from_default_config — HMC + mvn_10 produces the expected LOW recipe.
5. Recipe.save / load — round-trip preserves all fields.
6. Recipe.from_warmup_only — raises NotImplementedError mentioning follow-up spawn.
7. Recipe.from_tuning_result — raises NotImplementedError similarly.
8. Starter JSONs — all 6 files exist, load cleanly, have effort=LOW, warmup_name="no_warmup",
   and base_method_name in BASE_METHODS.
9. render_instructions — returns non-empty prose for LOW recipes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from bjx_bench.calibration.tier_b import default_params_for
from bjx_bench.inference.base_method import BASE_METHODS
from bjx_bench.inference.recipes import Effort, Recipe
from bjx_bench.inference.recipes._instructions import render_instructions
from bjx_bench.model import MODELS

# Path to the committed starter recipes
_STARTER_ROOT = (
    Path(__file__).parent.parent / "bjx_bench" / "inference" / "recipes" / "starter"
)


# ---------------------------------------------------------------------------
# Test 1: Effort enum values
# ---------------------------------------------------------------------------


def test_effort_enum_values() -> None:
    """Effort enum members have lowercase string values."""
    assert Effort.LOW.value == "low"
    assert Effort.MEDIUM.value == "medium"
    assert Effort.HIGH.value == "high"
    # As a str subclass, the string representation is the value itself
    assert str(Effort.LOW) == "Effort.LOW"  # str(Enum) gives "ClassName.MEMBER"
    assert Effort.LOW == "low"  # str-Enum compares equal to its value


# ---------------------------------------------------------------------------
# Test 2: Recipe construction and frozen invariant
# ---------------------------------------------------------------------------


def test_recipe_construct_and_frozen() -> None:
    """Recipe constructs from kwargs and is immutable (frozen dataclass)."""
    recipe = Recipe(
        model_name="test_model",
        base_method_name="nuts",
        warmup_name="no_warmup",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1},
        warmup_params={},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"trials": 0, "wall_seconds_estimate": 0.0},
        difficulty=None,
        instructions="test instructions",
        notes="",
        tuning_seed=0,
        bjx_bench_version="0.0.0.dev0",
        blackjax_version="1.0.0",
        jax_version="0.4.0",
        timestamp_utc="2026-01-01T00:00:00Z",
    )
    assert recipe.model_name == "test_model"
    assert recipe.effort == Effort.LOW
    # Frozen: assignment must raise FrozenInstanceError
    with pytest.raises(Exception):
        recipe.model_name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test 3: from_default_config — NUTS + mvn_10
# ---------------------------------------------------------------------------


def test_from_default_config_nuts_mvn10() -> None:
    """from_default_config for NUTS + mvn_10 produces the expected LOW recipe."""
    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    recipe = Recipe.from_default_config(posterior, base_method)

    assert recipe.effort == Effort.LOW
    assert recipe.model_name == "mvn_10"
    assert recipe.base_method_name == "nuts"
    assert recipe.warmup_name == "no_warmup"
    assert recipe.warmup_params == {}
    assert recipe.headline_metric is None
    assert recipe.sample_quality is None
    assert recipe.difficulty is None
    assert recipe.calibration_budget == {"trials": 0, "wall_seconds_estimate": 0.0}
    assert recipe.tuning_seed == 0

    # base_method_params must match default_params_for(nuts)
    expected_params = default_params_for(base_method)
    assert recipe.base_method_params == expected_params

    # NUTS default: step_size = sqrt(1e-3 * 1.0) ≈ 0.03162
    assert math.isclose(recipe.base_method_params["step_size"], math.sqrt(1e-3 * 1.0))


# ---------------------------------------------------------------------------
# Test 4: from_default_config — HMC + mvn_10
# ---------------------------------------------------------------------------


def test_from_default_config_hmc_mvn10() -> None:
    """from_default_config for HMC + mvn_10 produces the expected LOW recipe."""
    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["hmc"]
    recipe = Recipe.from_default_config(posterior, base_method)

    assert recipe.effort == Effort.LOW
    assert recipe.base_method_name == "hmc"
    assert recipe.warmup_name == "no_warmup"

    expected_params = default_params_for(base_method)
    assert recipe.base_method_params == expected_params

    # HMC defaults:
    # step_size = sqrt(1e-3 * 1.0) ≈ 0.03162
    # num_integration_steps = (1 + 128) // 2 = 64
    assert math.isclose(recipe.base_method_params["step_size"], math.sqrt(1e-3 * 1.0))
    assert recipe.base_method_params["num_integration_steps"] == 64


# ---------------------------------------------------------------------------
# Test 5: save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Recipe.save(tmp_path) → Recipe.load(path) round-trips all fields."""
    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    recipe = Recipe.from_default_config(posterior, base_method)

    saved_path = recipe.save(tmp_path)
    assert saved_path.exists()

    loaded = Recipe.load(saved_path)
    # All fields must match
    assert loaded.model_name == recipe.model_name
    assert loaded.base_method_name == recipe.base_method_name
    assert loaded.warmup_name == recipe.warmup_name
    assert loaded.effort == recipe.effort
    assert loaded.base_method_params == recipe.base_method_params
    assert loaded.warmup_params == recipe.warmup_params
    assert loaded.headline_metric == recipe.headline_metric
    assert loaded.sample_quality == recipe.sample_quality
    assert loaded.calibration_budget == recipe.calibration_budget
    assert loaded.difficulty == recipe.difficulty
    assert loaded.instructions == recipe.instructions
    assert loaded.notes == recipe.notes
    assert loaded.tuning_seed == recipe.tuning_seed
    assert loaded.bjx_bench_version == recipe.bjx_bench_version
    assert loaded.blackjax_version == recipe.blackjax_version
    assert loaded.jax_version == recipe.jax_version
    assert loaded.timestamp_utc == recipe.timestamp_utc

    # Effort enum preserved correctly
    assert isinstance(loaded.effort, Effort)
    assert loaded.effort == Effort.LOW

    # Filename convention: <effort>__<method>__<warmup>.json
    assert saved_path.name == "low__nuts__no_warmup.json"
    assert saved_path.parent.name == "mvn_10"


def test_save_json_effort_is_string(tmp_path: Path) -> None:
    """The JSON file stores effort as a plain string (not 'Effort.LOW')."""
    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["hmc"]
    recipe = Recipe.from_default_config(posterior, base_method)
    saved_path = recipe.save(tmp_path)

    raw = json.loads(saved_path.read_text())
    assert raw["effort"] == "low"  # not "Effort.LOW" or similar
    assert isinstance(raw["effort"], str)


# ---------------------------------------------------------------------------
# Test 6: from_warmup_only stub
# ---------------------------------------------------------------------------


def test_from_warmup_only_raises() -> None:
    """from_warmup_only raises NotImplementedError mentioning follow-up spawn."""
    with pytest.raises(NotImplementedError, match="follow-up spawn"):
        Recipe.from_warmup_only()


# ---------------------------------------------------------------------------
# Test 7: from_tuning_result stub
# ---------------------------------------------------------------------------


def test_from_tuning_result_raises() -> None:
    """from_tuning_result raises NotImplementedError mentioning follow-up spawn."""
    with pytest.raises(NotImplementedError, match="follow-up spawn"):
        Recipe.from_tuning_result()


# ---------------------------------------------------------------------------
# Test 8: 6 starter JSONs exist and load cleanly
# ---------------------------------------------------------------------------

_EXPECTED_COMBOS = [
    ("mvn_10", "hmc"),
    ("mvn_10", "nuts"),
    ("neals_funnel", "hmc"),
    ("neals_funnel", "nuts"),
    ("eight_schools_ncp", "hmc"),
    ("eight_schools_ncp", "nuts"),
]


@pytest.mark.parametrize("model_name,method_name", _EXPECTED_COMBOS)
def test_starter_recipe_exists_and_loads(model_name: str, method_name: str) -> None:
    """Each of the 6 starter LOW recipes exists on disk and loads cleanly."""
    path = _STARTER_ROOT / model_name / f"low__{method_name}__no_warmup.json"
    assert path.exists(), f"Missing starter recipe: {path}"

    recipe = Recipe.load(path)

    assert recipe.effort == Effort.LOW
    assert recipe.effort == "low"  # str-Enum equality with value
    assert recipe.warmup_name == "no_warmup"
    assert recipe.model_name == model_name
    assert recipe.base_method_name == method_name
    assert method_name in BASE_METHODS
    assert recipe.headline_metric is None
    assert recipe.warmup_params == {}
    assert recipe.calibration_budget == {"trials": 0, "wall_seconds_estimate": 0.0}
    assert isinstance(recipe.instructions, str)
    assert len(recipe.instructions) > 0


# ---------------------------------------------------------------------------
# Test 9: render_instructions returns non-empty prose
# ---------------------------------------------------------------------------


def test_render_instructions_low() -> None:
    """render_instructions returns non-empty prose for a LOW recipe."""
    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    recipe = Recipe.from_default_config(posterior, base_method)
    prose = render_instructions(recipe)
    assert isinstance(prose, str)
    assert len(prose) > 20
    # LOW template should mention the algorithm name
    assert "nuts" in prose
    # LOW template does NOT try to format headline_metric as a float
    assert "zero-calibration" in prose


def test_render_instructions_medium_stub() -> None:
    """render_instructions with a stub MEDIUM recipe returns non-empty text."""
    # Build a Recipe manually with MEDIUM effort (simulating a future from_warmup_only)
    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="stan_window",
        effort=Effort.MEDIUM,
        base_method_params={"step_size": 0.1, "inverse_mass_matrix": [1.0] * 10},
        warmup_params={"n_warmup": 1000, "target_acceptance_rate": 0.8},
        headline_metric=0.0512,
        sample_quality=None,
        calibration_budget={"trials": 0, "wall_seconds_estimate": 45.0},
        difficulty=None,
        instructions="",
        tuning_seed=0,
        bjx_bench_version="0.0.0.dev0",
        blackjax_version="1.0.0",
        jax_version="0.4.0",
        timestamp_utc="2026-01-01T00:00:00Z",
    )
    prose = render_instructions(recipe)
    assert isinstance(prose, str)
    assert len(prose) > 10
    assert "medium" in prose.lower() or "Medium" in prose


def test_render_instructions_high_stub() -> None:
    """render_instructions with a stub HIGH recipe returns non-empty text."""
    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="stan_window",
        effort=Effort.HIGH,
        base_method_params={"step_size": 0.08, "num_integration_steps": 32},
        warmup_params={"n_warmup": 1000},
        headline_metric=0.0731,
        sample_quality=None,
        calibration_budget={"trials": 50, "wall_seconds_estimate": 1800.0},
        difficulty=None,
        instructions="",
        tuning_seed=42,
        bjx_bench_version="0.0.0.dev0",
        blackjax_version="1.0.0",
        jax_version="0.4.0",
        timestamp_utc="2026-01-01T00:00:00Z",
    )
    prose = render_instructions(recipe)
    assert isinstance(prose, str)
    assert len(prose) > 10
    assert "high" in prose.lower() or "High" in prose
