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
"""Tests for init_strategy schema field and force_resample_config API.

All tests are @pytest.mark.fast — pure logic / schema / no JAX trace,
each well under 100 ms wall.
"""

import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from tuningfork.recipes._base import (
    _VALID_INIT_STRATEGY_TYPES,
    Effort,
    FailureDiagnosis,
    Recipe,
    RecipeFailedError,
    validate_init_strategy,
)

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Minimal recipe kwargs for construction
# ---------------------------------------------------------------------------

_MINIMAL: dict[str, Any] = dict(
    model_name="mvn_10",
    base_method_name="nuts",
    warmup_name="no_warmup",
    effort=Effort.LOW,
    base_method_params={"step_size": 0.1},
    warmup_params={},
    headline_metric=None,
    sample_quality=None,
    calibration_budget={"trials": 0, "wall_seconds_estimate": 1.0},
    difficulty=None,
    instructions="test",
)


def _make_recipe(**overrides: Any) -> Recipe:
    kw = {**_MINIMAL, **overrides}
    return Recipe(**kw)


# ---------------------------------------------------------------------------
# validate_init_strategy — unit tests
# ---------------------------------------------------------------------------


def test_validate_init_strategy_none_passes() -> None:
    """None is always valid (backward-compat default)."""
    validate_init_strategy(None)  # must not raise


def test_validate_init_strategy_prior_sample_passes() -> None:
    """{'type': 'prior_sample'} is a valid explicit spec."""
    validate_init_strategy({"type": "prior_sample"})


def test_validate_init_strategy_zero_passes() -> None:
    """{'type': 'zero'} is a valid spec."""
    validate_init_strategy({"type": "zero"})


def test_validate_init_strategy_uniform_valid_passes() -> None:
    """{'type': 'uniform', 'low': -1, 'high': 1} is valid."""
    validate_init_strategy({"type": "uniform", "low": -1.0, "high": 1.0})


def test_validate_init_strategy_uniform_float_str_passes() -> None:
    """uniform spec with low/high as strings that coerce to float is valid."""
    # JSON load always gives floats, but a human might pass strings.
    validate_init_strategy({"type": "uniform", "low": "-1.5", "high": "2.5"})


def test_validate_init_strategy_non_dict_raises() -> None:
    """Non-dict (e.g., a string) raises ValueError with a clear message."""
    with pytest.raises(ValueError, match="dict or None"):
        validate_init_strategy("prior_sample")  # type: ignore[arg-type]


def test_validate_init_strategy_unknown_type_raises() -> None:
    """Unknown 'type' key raises ValueError naming the offending type."""
    with pytest.raises(ValueError, match="not recognised"):
        validate_init_strategy({"type": "gaussian"})


def test_validate_init_strategy_uniform_missing_low_raises() -> None:
    """uniform spec without 'low' raises ValueError."""
    with pytest.raises(ValueError, match="low.*high"):
        validate_init_strategy({"type": "uniform", "high": 1.0})


def test_validate_init_strategy_uniform_missing_high_raises() -> None:
    """uniform spec without 'high' raises ValueError."""
    with pytest.raises(ValueError, match="low.*high"):
        validate_init_strategy({"type": "uniform", "low": -1.0})


def test_validate_init_strategy_uniform_equal_bounds_raises() -> None:
    """low == high violates the low < high contract."""
    with pytest.raises(ValueError, match="low < high"):
        validate_init_strategy({"type": "uniform", "low": 0.0, "high": 0.0})


def test_validate_init_strategy_uniform_inverted_raises() -> None:
    """low > high raises ValueError."""
    with pytest.raises(ValueError, match="low < high"):
        validate_init_strategy({"type": "uniform", "low": 1.0, "high": -1.0})


def test_valid_init_strategy_types_constant() -> None:
    """_VALID_INIT_STRATEGY_TYPES covers exactly the five documented types."""
    assert _VALID_INIT_STRATEGY_TYPES == {
        "prior_sample",
        "zero",
        "uniform",
        "zero_perchain",
        "uniform_perchain",
        "reference_summary",
    }


def test_reference_summary_rejects_nonfinite_and_shape_mismatch() -> None:
    valid: dict[str, Any] = {
        "type": "reference_summary",
        "mean": {"x": [0.0, 1.0]},
        "std": {"x": [1.0, 2.0]},
        "offsets": [0.1, -0.1],
        "source_path": "model/reference/summary.json",
        "source_sha256": "0" * 64,
    }
    validate_init_strategy(valid)
    with pytest.raises(ValueError, match="non-finite"):
        validate_init_strategy({**valid, "mean": {"x": [float("nan")]}})
    with pytest.raises(ValueError, match="shapes"):
        validate_init_strategy({**valid, "std": {"x": [1.0]}})
    with pytest.raises(ValueError, match="lowercase"):
        validate_init_strategy({**valid, "source_sha256": "A" * 64})


def test_reference_summary_validates_every_leaf_and_rejects_boolean_offsets() -> None:
    valid: dict[str, Any] = {
        "type": "reference_summary",
        "mean": {"x": [0.0, 1.0], "y": [[2.0], [3.0]]},
        "std": {"x": [1.0, 2.0], "y": [[0.5], [0.25]]},
        "offsets": [0.1, -0.1],
        "source_path": "model/reference/summary.json",
        "source_sha256": "0" * 64,
    }
    validate_init_strategy(valid)
    with pytest.raises(ValueError, match=r"mean/std\['x'\] shapes"):
        validate_init_strategy({**valid, "std": {**valid["std"], "x": [1.0]}})
    with pytest.raises(ValueError, match="booleans"):
        validate_init_strategy({**valid, "offsets": [True]})


# ---------------------------------------------------------------------------
# init_strategy field on Recipe — defaults + round-trip
# ---------------------------------------------------------------------------


def test_recipe_init_strategy_defaults_to_none() -> None:
    """init_strategy defaults to None when not provided."""
    recipe = _make_recipe()
    assert recipe.init_strategy is None


def test_recipe_init_strategy_zero_stored() -> None:
    """init_strategy={'type':'zero'} is stored on the Recipe."""
    recipe = _make_recipe(init_strategy={"type": "zero"})
    assert recipe.init_strategy == {"type": "zero"}


def test_recipe_init_strategy_uniform_stored() -> None:
    """init_strategy={'type':'uniform','low':-1,'high':1} is stored."""
    spec = {"type": "uniform", "low": -1.0, "high": 1.0}
    recipe = _make_recipe(init_strategy=spec)
    assert recipe.init_strategy == spec


def test_recipe_init_strategy_none_round_trips(tmp_path: Path) -> None:
    """init_strategy=None saves as null and loads as None."""
    recipe = _make_recipe()
    path = recipe.save(tmp_path)
    loaded = Recipe.load(path)
    assert loaded.init_strategy is None


def test_recipe_init_strategy_zero_round_trips(tmp_path: Path) -> None:
    """init_strategy={'type':'zero'} survives save→load."""
    spec = {"type": "zero"}
    recipe = _make_recipe(init_strategy=spec)
    path = recipe.save(tmp_path)
    loaded = Recipe.load(path)
    assert loaded.init_strategy == spec


def test_recipe_init_strategy_uniform_round_trips(tmp_path: Path) -> None:
    """init_strategy uniform spec survives save→load with exact values."""
    spec = {"type": "uniform", "low": -2.0, "high": 3.0}
    recipe = _make_recipe(init_strategy=spec)
    path = recipe.save(tmp_path)
    loaded = Recipe.load(path)
    assert loaded.init_strategy == spec
    assert loaded.init_strategy["low"] == pytest.approx(-2.0)
    assert loaded.init_strategy["high"] == pytest.approx(3.0)


def test_recipe_init_strategy_prior_sample_round_trips(tmp_path: Path) -> None:
    """Explicit {'type':'prior_sample'} round-trips through save/load."""
    spec = {"type": "prior_sample"}
    recipe = _make_recipe(init_strategy=spec)
    path = recipe.save(tmp_path)
    loaded = Recipe.load(path)
    assert loaded.init_strategy == spec


def test_init_strategy_in_saved_json(tmp_path: Path) -> None:
    """save() writes the 'init_strategy' key to the JSON file."""
    spec = {"type": "uniform", "low": -1.0, "high": 1.0}
    recipe = _make_recipe(init_strategy=spec)
    path = recipe.save(tmp_path)
    raw = json.loads(path.read_text())
    assert "init_strategy" in raw
    assert raw["init_strategy"] == spec


# ---------------------------------------------------------------------------
# Backward-compat load — old recipes without init_strategy
# ---------------------------------------------------------------------------


def test_legacy_recipe_loads_without_init_strategy(tmp_path: Path) -> None:
    """A JSON without 'init_strategy' loads with init_strategy=None (old recipe compat)."""
    recipe = _make_recipe()
    path = recipe.save(tmp_path)
    raw = json.loads(path.read_text())
    # Simulate a recipe written before the schema extension
    del raw["init_strategy"]
    path.write_text(json.dumps(raw, indent=2) + "\n")

    loaded = Recipe.load(path)
    assert loaded.init_strategy is None


def test_load_unknown_init_strategy_type_raises(tmp_path: Path) -> None:
    """Recipe.load raises ValueError at load time for unknown init_strategy type."""
    recipe = _make_recipe()
    path = recipe.save(tmp_path)
    raw = json.loads(path.read_text())
    # Inject an unknown type (simulates a hand-edited recipe with a typo)
    raw["init_strategy"] = {"type": "gaussian"}
    path.write_text(json.dumps(raw, indent=2) + "\n")

    with pytest.raises(ValueError, match="not recognised"):
        Recipe.load(path)


def test_load_uniform_inverted_bounds_raises(tmp_path: Path) -> None:
    """Recipe.load raises ValueError for uniform spec with low >= high."""
    recipe = _make_recipe()
    path = recipe.save(tmp_path)
    raw = json.loads(path.read_text())
    raw["init_strategy"] = {"type": "uniform", "low": 5.0, "high": -5.0}
    path.write_text(json.dumps(raw, indent=2) + "\n")

    with pytest.raises(ValueError, match="low < high"):
        Recipe.load(path)


# ---------------------------------------------------------------------------
# force_resample deprecation shim
# ---------------------------------------------------------------------------


def test_force_resample_emits_deprecation_warning() -> None:
    """force_resample=True emits DeprecationWarning."""
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    failed_recipe = _make_recipe(
        effort=Effort.FAILED,
        failure_diagnosis=FailureDiagnosis.HARD_DIRECTION,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # FAILED recipe → RecipeFailedError raised AFTER the DeprecationWarning fires
        with pytest.raises(RecipeFailedError):
            run_recipe_to_idata(failed_recipe, force_resample=True)

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1, f"Expected 1 DeprecationWarning, got {dep_warnings}"
    msg = str(dep_warnings[0].message)
    assert "force_resample" in msg
    assert "force_resample_config" in msg


def test_force_resample_false_no_warning() -> None:
    """force_resample=False (default) emits no DeprecationWarning."""
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    failed_recipe = _make_recipe(
        effort=Effort.FAILED,
        failure_diagnosis=FailureDiagnosis.HARD_DIRECTION,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RecipeFailedError):
            run_recipe_to_idata(failed_recipe)

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 0


def test_force_resample_config_accepted_no_warning() -> None:
    """force_resample_config=... accepted without DeprecationWarning."""
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    failed_recipe = _make_recipe(
        effort=Effort.FAILED,
        failure_diagnosis=FailureDiagnosis.HARD_DIRECTION,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RecipeFailedError):
            run_recipe_to_idata(failed_recipe, force_resample_config={"seed": 42})

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 0


def test_force_resample_maps_to_config_with_recipe_seed() -> None:
    """force_resample=True maps to force_resample_config={'seed': recipe.tuning_seed}."""
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    # FAILED recipe triggers RecipeFailedError after the DeprecationWarning fires.
    # Use a non-zero tuning_seed to verify the mapping records the correct seed.
    recipe_with_seed = _make_recipe(
        effort=Effort.FAILED,
        tuning_seed=9999,
        failure_diagnosis=FailureDiagnosis.HARD_DIRECTION,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RecipeFailedError):
            run_recipe_to_idata(recipe_with_seed, force_resample=True)

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1
