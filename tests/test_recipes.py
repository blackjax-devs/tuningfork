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
"""Tests for bjx_bench.inference.recipes (Recipe dataclass + Effort enum).

Phase 3, P3.2: extends P2.5 commit 3 tests with MEDIUM and HIGH constructors.

Tests
-----
1.  Effort enum — values are lowercase strings.                 [fast]
2.  Recipe — constructs from kwargs; frozen (no mutation).      [fast]
3.  Recipe.from_default_config — NUTS + mvn_10.                 [fast]
4.  Recipe.from_default_config — HMC + mvn_10.                  [fast]
5.  Recipe.save / load — LOW recipe round-trip.                 [fast]
6.  JSON effort field stored as plain string.                   [fast]
7.  Starter JSONs — all 6 exist and load cleanly.               [fast]
8.  render_instructions — non-empty prose for LOW.              [fast]
9.  render_instructions — non-empty prose for stub MEDIUM.      [fast]
10. render_instructions — non-empty prose for stub HIGH.        [fast]
11. from_warmup_only (stan_window + NUTS) — MEDIUM recipe.
12. from_warmup_only (mclmc_tuning + MCLMC) — metadata threaded.
13. from_warmup_only — incompatible pair raises ValueError.
14. from_tuning_result (NUTS) — HIGH recipe from tune_algorithm.
15. from_tuning_result — save/load round-trip.
16. render_instructions — MEDIUM/HIGH prose contains expected tokens.
"""

import json
import math
from pathlib import Path

import jax
import pytest

from bjx_bench.calibration.tier_b import default_params_for
from bjx_bench.inference.base_method import BASE_METHODS
from bjx_bench.inference.recipes import Effort, Recipe
from bjx_bench.inference.recipes._instructions import render_instructions
from bjx_bench.inference.warmup import WARMUPS
from bjx_bench.model import MODELS

# Path to the committed starter recipes
_STARTER_ROOT = (
    Path(__file__).parent.parent / "bjx_bench" / "inference" / "recipes" / "starter"
)


# ---------------------------------------------------------------------------
# Test 1: Effort enum values
# ---------------------------------------------------------------------------


@pytest.mark.fast
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


@pytest.mark.fast
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


@pytest.mark.fast
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

    # NUTS default: step_size = 1e-3 * (1/1e-3)**0.7 (70th-pctile on log-scale, P4.0 tweak)
    assert math.isclose(
        recipe.base_method_params["step_size"], 1e-3 * (1.0 / 1e-3) ** 0.7
    )


# ---------------------------------------------------------------------------
# Test 4: from_default_config — HMC + mvn_10
# ---------------------------------------------------------------------------


@pytest.mark.fast
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
    # step_size = 1e-3 * (1/1e-3)**0.7 (70th-pctile on log-scale, P4.0 tweak)
    # num_integration_steps = (1 + 128) // 2 = 64
    assert math.isclose(
        recipe.base_method_params["step_size"], 1e-3 * (1.0 / 1e-3) ** 0.7
    )
    assert recipe.base_method_params["num_integration_steps"] == 64


# ---------------------------------------------------------------------------
# Test 5: save / load round-trip
# ---------------------------------------------------------------------------


@pytest.mark.fast
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


@pytest.mark.fast
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
# Test 7 (was 8): 6 starter JSONs exist and load cleanly
# ---------------------------------------------------------------------------

_EXPECTED_COMBOS = [
    ("mvn_10", "hmc"),
    ("mvn_10", "nuts"),
    ("neals_funnel", "hmc"),
    ("neals_funnel", "nuts"),
    ("eight_schools_ncp", "hmc"),
    ("eight_schools_ncp", "nuts"),
]


@pytest.mark.fast
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
# Test 8 (was 9): render_instructions returns non-empty prose
# ---------------------------------------------------------------------------


@pytest.mark.fast
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


@pytest.mark.fast
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


@pytest.mark.fast
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


# ---------------------------------------------------------------------------
# Tests 11–16: P3.2 — MEDIUM and HIGH constructors (require actual warmup)
# ---------------------------------------------------------------------------


def test_from_warmup_only_stan_window_nuts() -> None:
    """from_warmup_only with stan_window + NUTS returns a MEDIUM recipe.

    Verifies:
    - effort = MEDIUM
    - warmup_name = "stan_window"
    - base_method_params contains both step_size (from defaults) and
      inverse_mass_matrix (from warmup adaptation)
    - calibration_budget["n_warmup"] == 200
    - calibration_budget["wall_seconds_estimate"] > 0
    """
    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    warmup = WARMUPS["stan_window"]

    recipe = Recipe.from_warmup_only(
        posterior,
        base_method,
        warmup,
        n_warmup=200,
        rng_key=jax.random.key(0),
    )

    assert recipe.effort == Effort.MEDIUM
    assert recipe.warmup_name == "stan_window"
    assert recipe.model_name == "mvn_10"
    assert recipe.base_method_name == "nuts"

    # base_method_params must include both the default step_size (loguniform
    # 70th-pctile ≈ 0.126, P4.0 tweak) AND the warmup-adapted inverse_mass_matrix.
    assert "step_size" in recipe.base_method_params
    assert "inverse_mass_matrix" in recipe.base_method_params

    # IMM must be a list (coerced from jax.Array by _to_jsonable).
    imm = recipe.base_method_params["inverse_mass_matrix"]
    assert isinstance(imm, list), f"inverse_mass_matrix should be list, got {type(imm)}"
    assert len(imm) == 10  # mvn_10 is 10-D

    # calibration_budget fields
    assert recipe.calibration_budget["n_warmup"] == 200
    assert recipe.calibration_budget["wall_seconds_estimate"] > 0
    assert recipe.calibration_budget["trials"] == 0

    # warmup_params records the input config
    assert recipe.warmup_params["n_warmup"] == 200

    # headline_metric is None for MEDIUM (no post-warmup samples)
    assert recipe.headline_metric is None

    # instructions must be non-empty prose
    assert isinstance(recipe.instructions, str)
    assert len(recipe.instructions) > 10


def test_from_warmup_only_mclmc_tuning_metadata() -> None:
    """from_warmup_only with mclmc_tuning threads _total_tuning_steps into calibration_budget.

    P3.1 threads the ``_total_tuning_steps`` metadata key from mclmc_tuning into
    adapted_params with an underscore prefix.  from_warmup_only must capture it
    in calibration_budget and strip it from base_method_params.
    """
    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["mclmc"]
    warmup = WARMUPS["mclmc_tuning"]

    recipe = Recipe.from_warmup_only(
        posterior,
        base_method,
        warmup,
        n_warmup=200,
        rng_key=jax.random.key(1),
    )

    assert recipe.effort == Effort.MEDIUM
    assert recipe.warmup_name == "mclmc_tuning"

    # _total_tuning_steps must appear in calibration_budget (threaded from metadata).
    assert "_total_tuning_steps" in recipe.calibration_budget
    assert isinstance(recipe.calibration_budget["_total_tuning_steps"], int)

    # _total_tuning_steps must NOT appear in base_method_params (stripped).
    assert "_total_tuning_steps" not in recipe.base_method_params

    # MCLMC adapted params (L, step_size) must be in base_method_params.
    assert "step_size" in recipe.base_method_params
    assert "L" in recipe.base_method_params


def test_from_warmup_only_incompatible_raises() -> None:
    """from_warmup_only with an incompatible (warmup, base_method) pair raises ValueError."""
    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    # mclmc_tuning is only compatible with mclmc, not nuts.
    warmup = WARMUPS["mclmc_tuning"]

    with pytest.raises(ValueError, match="not compatible"):
        Recipe.from_warmup_only(
            posterior,
            base_method,
            warmup,
            n_warmup=100,
            rng_key=jax.random.key(0),
        )


def test_from_tuning_result_nuts() -> None:
    """from_tuning_result produces a HIGH recipe from tune_algorithm output.

    Verifies:
    - effort = HIGH
    - headline_metric > 0 (best_score was finite; mvn_10 + nuts is well-behaved)
    - difficulty dict contains expected keys
    - n_trials_completed matches n_trials arg
    """
    from bjx_bench.calibration.tier_b import tune_algorithm

    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    warmup = WARMUPS["stan_window"]

    tuning_result = tune_algorithm(
        posterior,
        base_method,
        n_trials=3,
        n_seeds=1,
        n_chains=1,
        n_samples=200,
        n_warmup=200,
        rng_key=jax.random.key(0),
    )

    recipe = Recipe.from_tuning_result(
        tuning_result,
        posterior=posterior,
        base_method=base_method,
        warmup=warmup,
    )

    assert recipe.effort == Effort.HIGH
    assert recipe.model_name == "mvn_10"
    assert recipe.base_method_name == "nuts"
    assert recipe.warmup_name == "stan_window"

    # headline_metric should be a finite float (mvn_10 doesn't diverge)
    assert isinstance(recipe.headline_metric, float)
    assert math.isfinite(recipe.headline_metric)

    # difficulty dict from TuningDifficulty.asdict()
    assert recipe.difficulty is not None
    assert isinstance(recipe.difficulty, dict)
    for key in (
        "default_score",
        "best_score",
        "threshold_score",
        "default_works",
        "n_trials_to_threshold",
        "n_trials_to_best",
    ):
        assert key in recipe.difficulty, f"Missing difficulty key: {key}"

    # calibration_budget
    assert recipe.calibration_budget["trials"] == 3
    assert recipe.calibration_budget["n_seeds"] == 1

    # instructions non-empty
    assert len(recipe.instructions) > 10


def test_from_tuning_result_save_load_roundtrip(tmp_path: Path) -> None:
    """HIGH recipe round-trips through Recipe.save / Recipe.load.

    Verifies in particular that:
    - inverse_mass_matrix (list[float]) in base_method_params round-trips.
    - difficulty dict (nested Python primitives) round-trips without JSON errors.
    """
    from bjx_bench.calibration.tier_b import tune_algorithm

    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    warmup = WARMUPS["stan_window"]

    tuning_result = tune_algorithm(
        posterior,
        base_method,
        n_trials=2,
        n_seeds=1,
        n_chains=1,
        n_samples=100,
        n_warmup=100,
        rng_key=jax.random.key(42),
    )

    recipe = Recipe.from_tuning_result(
        tuning_result,
        posterior=posterior,
        base_method=base_method,
        warmup=warmup,
    )

    saved_path = recipe.save(tmp_path)
    loaded = Recipe.load(saved_path)

    # Core identity fields
    assert loaded.effort == Effort.HIGH
    assert loaded.model_name == recipe.model_name
    assert loaded.base_method_name == recipe.base_method_name
    assert loaded.warmup_name == recipe.warmup_name

    # headline_metric round-trip (float precision)
    assert loaded.headline_metric == recipe.headline_metric

    # difficulty round-trip (nested dict, not a dataclass after load)
    assert loaded.difficulty == recipe.difficulty
    assert isinstance(loaded.difficulty, dict)

    # inverse_mass_matrix round-trip: list[float] after load
    if "inverse_mass_matrix" in recipe.base_method_params:
        orig_imm = recipe.base_method_params["inverse_mass_matrix"]
        loaded_imm = loaded.base_method_params["inverse_mass_matrix"]
        assert isinstance(loaded_imm, list)
        assert loaded_imm == orig_imm  # exact list equality (both are Python floats)

    # Filename convention
    assert saved_path.name == "high__nuts__stan_window.json"


def test_render_instructions_medium_and_high_real() -> None:
    """render_instructions on real MEDIUM and HIGH recipes returns meaningful prose."""
    from bjx_bench.calibration.tier_b import tune_algorithm

    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    warmup_sw = WARMUPS["stan_window"]

    # --- MEDIUM ---
    medium = Recipe.from_warmup_only(
        posterior,
        base_method,
        warmup_sw,
        n_warmup=100,
        rng_key=jax.random.key(7),
    )
    prose_m = render_instructions(medium)
    assert isinstance(prose_m, str)
    assert len(prose_m) > 20
    assert "stan_window" in prose_m

    # --- HIGH ---
    tuning_result = tune_algorithm(
        posterior,
        base_method,
        n_trials=2,
        n_seeds=1,
        n_chains=1,
        n_samples=100,
        n_warmup=100,
        rng_key=jax.random.key(8),
    )
    high = Recipe.from_tuning_result(
        tuning_result,
        posterior=posterior,
        base_method=base_method,
        warmup=warmup_sw,
    )
    prose_h = render_instructions(high)
    assert isinstance(prose_h, str)
    assert len(prose_h) > 20
    # HIGH template shows the number of trials
    assert str(tuning_result.n_trials_completed) in prose_h


# ---------------------------------------------------------------------------
# Tests 9-10: P3.3 wider LOW coverage + MEDIUM smoke
# ---------------------------------------------------------------------------

_LOW_OTHER_ALGOS = [
    (model, method)
    for model in ("mvn_10", "neals_funnel", "eight_schools_ncp")
    for method in ("mala", "barker", "rwm", "mclmc")
]


@pytest.mark.fast
@pytest.mark.parametrize("model_name,method_name", _LOW_OTHER_ALGOS)
def test_low_recipe_exists_for_other_algos(model_name: str, method_name: str) -> None:
    """P3.3: every (starter_model, base_method) pair has a LOW recipe on disk."""
    path = _STARTER_ROOT / model_name / f"low__{method_name}__no_warmup.json"
    assert path.exists(), f"Missing LOW recipe for {model_name} + {method_name}"
    recipe = Recipe.load(path)
    assert recipe.effort == Effort.LOW
    assert recipe.warmup_name == "no_warmup"
    assert recipe.model_name == model_name
    assert recipe.base_method_name == method_name


_MEDIUM_COMBOS = [
    (model, method)
    for model in ("mvn_10", "neals_funnel", "eight_schools_ncp")
    for method in ("hmc", "nuts")
]


@pytest.mark.parametrize("model_name,method_name", _MEDIUM_COMBOS)
def test_medium_recipe_exists_and_has_warmup_data(
    model_name: str, method_name: str
) -> None:
    """P3.3: each (starter_model, {hmc,nuts}) has a MEDIUM recipe via stan_window
    with non-empty warmup-adapted params and positive wall-clock."""
    path = _STARTER_ROOT / model_name / f"medium__{method_name}__stan_window.json"
    assert path.exists(), f"Missing MEDIUM recipe for {model_name} + {method_name}"
    recipe = Recipe.load(path)
    assert recipe.effort == Effort.MEDIUM
    assert recipe.warmup_name == "stan_window"
    assert recipe.calibration_budget["n_warmup"] == 1000
    assert recipe.calibration_budget["wall_seconds_estimate"] > 0
    # The warmup-adapted base_method_params must contain step_size AND
    # inverse_mass_matrix (a non-trivial adaptation, not just defaults).
    assert "step_size" in recipe.base_method_params
    assert "inverse_mass_matrix" in recipe.base_method_params
    imm = recipe.base_method_params["inverse_mass_matrix"]
    assert isinstance(imm, list)  # JSON deserialization gives list
    assert len(imm) > 0
    assert recipe.headline_metric is None  # MEDIUM doesn't measure post-warmup ESS


# ---------------------------------------------------------------------------
# Tests P3.4: 6 HIGH recipes exist and have correct schema
# ---------------------------------------------------------------------------

_HIGH_COMBOS = [
    (model, method)
    for model in ("mvn_10", "neals_funnel", "eight_schools_ncp")
    for method in ("hmc", "nuts")
]

_EXPECTED_DIFFICULTY_KEYS = (
    "default_score",
    "best_score",
    "threshold_score",
    "default_works",
    "n_trials_to_threshold",
    "n_trials_to_best",
    "wall_seconds_to_threshold",
    "wall_seconds_to_best",
)


@pytest.mark.fast
@pytest.mark.parametrize("model_name,method_name", _HIGH_COMBOS)
def test_high_recipe_exists_and_has_bo_data(model_name: str, method_name: str) -> None:
    """P3.4: each (starter_model, {hmc,nuts}) has a HIGH recipe via Tier-B BO
    at n_trials=20 with a valid headline_metric and TuningDifficulty profile."""
    path = _STARTER_ROOT / model_name / f"high__{method_name}__stan_window.json"
    assert (
        path.exists()
    ), f"Missing HIGH recipe for {model_name} + {method_name}: {path}"

    recipe = Recipe.load(path)

    # Identity and effort checks
    assert recipe.effort == Effort.HIGH
    assert recipe.warmup_name == "stan_window"
    assert recipe.model_name == model_name
    assert recipe.base_method_name == method_name

    # headline_metric must be a real, positive float (all starter models
    # are well-conditioned; divergence here would be a real failure).
    assert recipe.headline_metric is not None
    assert isinstance(recipe.headline_metric, float)
    assert math.isfinite(recipe.headline_metric)
    assert recipe.headline_metric > 0, (
        f"headline_metric={recipe.headline_metric} for {model_name}+{method_name}; "
        "expected > 0 for these well-conditioned starter models."
    )

    # calibration_budget shape
    assert recipe.calibration_budget["trials"] == 20
    assert "n_seeds" in recipe.calibration_budget

    # difficulty is a dict with all expected keys
    assert recipe.difficulty is not None
    assert isinstance(recipe.difficulty, dict)
    for key in _EXPECTED_DIFFICULTY_KEYS:
        assert key in recipe.difficulty, (
            f"Missing difficulty key {key!r} in HIGH recipe "
            f"{model_name}+{method_name}"
        )

    # Spot-check difficulty numeric types
    assert isinstance(recipe.difficulty["default_score"], float)
    assert isinstance(recipe.difficulty["best_score"], float)
    assert isinstance(recipe.difficulty["default_works"], bool)
    assert isinstance(recipe.difficulty["n_trials_to_threshold"], int)

    # Instructions non-empty prose
    assert isinstance(recipe.instructions, str)
    assert len(recipe.instructions) > 10


# ---------------------------------------------------------------------------
# Test 17 (P5.0): _generate_starter CLI flag filtering
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_emit_low_recipes_sampler_filter(tmp_path: Path, monkeypatch) -> None:
    """emit_low_recipes(sampler='nuts') emits NUTS recipes only.

    P5.0 (Q5.A) added per-cell flag filtering to ``_generate_starter.py``;
    this test locks the ``sampler`` filter behavior at the function-level
    so future refactors can't silently regress it.

    Uses monkeypatch to redirect _STARTER_ROOT into tmp_path so we don't
    clobber committed recipes. Only tests LOW because it's deterministic
    and zero-cost (no MCMC).
    """
    from bjx_bench.inference.recipes import _generate_starter

    monkeypatch.setattr(_generate_starter, "_STARTER_ROOT", tmp_path)
    paths = _generate_starter.emit_low_recipes(model_names=["mvn_10"], sampler="nuts")
    assert len(paths) == 1, f"Expected 1 recipe, got {len(paths)}: {paths}"
    assert paths[0].name == "low__nuts__no_warmup.json"


@pytest.mark.fast
def test_emit_low_recipes_no_filter_emits_all_methods(
    tmp_path: Path, monkeypatch
) -> None:
    """emit_low_recipes() with no sampler filter emits all 6 algos for the model."""
    from bjx_bench.inference.recipes import _generate_starter

    monkeypatch.setattr(_generate_starter, "_STARTER_ROOT", tmp_path)
    paths = _generate_starter.emit_low_recipes(model_names=["mvn_10"])
    # 6 algorithms: hmc, nuts, mala, barker, rwm, mclmc
    assert len(paths) == 6
    method_names = sorted(p.name.split("__")[1] for p in paths)
    assert method_names == sorted(["hmc", "nuts", "mala", "barker", "rwm", "mclmc"])


@pytest.mark.fast
def test_main_rejects_unknown_model(monkeypatch) -> None:
    """`--only <unknown>` raises SystemExit."""
    import sys

    from bjx_bench.inference.recipes import _generate_starter

    monkeypatch.setattr(
        sys,
        "argv",
        ["_generate_starter", "--only", "no_such_model"],
    )
    with pytest.raises(SystemExit, match="not in STARTER_MODEL_NAMES"):
        _generate_starter.main()


@pytest.mark.fast
def test_main_help_smoke() -> None:
    """`--help` exits with status 0 and prints flag descriptions."""
    import sys

    from bjx_bench.inference.recipes import _generate_starter

    saved_argv = sys.argv
    try:
        sys.argv = ["_generate_starter", "--help"]
        with pytest.raises(SystemExit) as exc_info:
            _generate_starter.main()
        # argparse's --help calls sys.exit(0) on success.
        assert exc_info.value.code == 0
    finally:
        sys.argv = saved_argv
