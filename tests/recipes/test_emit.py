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
"""Tests for Recipe emission logic (from warmup/tuning results).

This file contains all tests that run actual MCMC chains or tuning algorithms.
These are marked @pytest.mark.slow individually (not at module level, per PR-4 rules).

Tests: from_warmup_only, from_tuning_result, render_instructions_medium_and_high_real.

History: test_medium_recipe_exists_and_has_warmup_data (parametrized over 6
(model × {hmc, nuts}) combos) was removed 2026-05-17 as a slow-CI fix —
the MEDIUM placeholder recipes it asserted-existence-of had been deleted in
PR #6 commit 3 (715a82c, "recipes: delete stale low/medium/high starter
recipes"), but the test surgery in that commit missed this slow-only test
because we don't run slow locally. Real MEDIUM recipes are produced by
Recipe Phase 1+ pipeline; their existence-on-disk is no longer a test gate.
"""

import math
from pathlib import Path

import jax
import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.model import MODELS
from tuningfork.recipes import Effort, Recipe
from tuningfork.recipes._instructions import render_instructions
from tuningfork.warmup import WARMUPS

# ---------------------------------------------------------------------------
# MEDIUM and HIGH constructors (require actual warmup)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_from_warmup_only_window_adaptation_diag_imm_nuts() -> None:
    """from_warmup_only with window_adaptation_diag_imm + NUTS returns a MEDIUM recipe.

    Verifies:
    - effort = MEDIUM
    - warmup_name = "window_adaptation_diag_imm"
    - base_method_params contains both step_size (from defaults) and
      inverse_mass_matrix (from warmup adaptation)
    - calibration_budget["n_warmup"] == 200
    - calibration_budget["wall_seconds_estimate"] > 0
    """
    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    warmup = WARMUPS["window_adaptation_diag_imm"]

    recipe = Recipe.from_warmup_only(
        posterior,
        base_method,
        warmup,
        n_warmup=200,
        rng_key=jax.random.key(0),
    )

    assert recipe.effort == Effort.MEDIUM
    assert recipe.warmup_name == "window_adaptation_diag_imm"
    assert recipe.model_name == "mvn_10"
    assert recipe.base_method_name == "nuts"

    # base_method_params must include both the default step_size (loguniform
    # 70th-pctile ≈ 0.126) AND the warmup-adapted inverse_mass_matrix.
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


@pytest.mark.slow
def test_from_warmup_only_mclmc_tuning_metadata() -> None:
    """from_warmup_only with mclmc_tuning threads _total_tuning_steps into calibration_budget.

    Threading the ``_total_tuning_steps`` metadata key from mclmc_tuning into
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_from_tuning_result_nuts() -> None:
    """from_tuning_result produces a HIGH recipe from tune_algorithm output.

    Verifies:
    - effort = HIGH
    - headline_metric > 0 (best_score was finite; mvn_10 + nuts is well-behaved)
    - difficulty dict contains expected keys
    - n_trials_completed matches n_trials arg
    """
    from tuningfork.calibration.tune import tune_algorithm

    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    warmup = WARMUPS["window_adaptation_diag_imm"]

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
    assert recipe.warmup_name == "window_adaptation_diag_imm"

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


@pytest.mark.slow
def test_from_tuning_result_save_load_roundtrip(tmp_path: Path) -> None:
    """HIGH recipe round-trips through Recipe.save / Recipe.load.

    Verifies in particular that:
    - inverse_mass_matrix (list[float]) in base_method_params round-trips.
    - difficulty dict (nested Python primitives) round-trips without JSON errors.
    """
    from tuningfork.calibration.tune import tune_algorithm

    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    warmup = WARMUPS["window_adaptation_diag_imm"]

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
    assert saved_path.name == "high__nuts__window_adaptation_diag_imm.json"


@pytest.mark.slow
def test_render_instructions_medium_and_high_real() -> None:
    """render_instructions on real MEDIUM and HIGH recipes returns meaningful prose."""
    from tuningfork.calibration.tune import tune_algorithm

    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    warmup_sw = WARMUPS["window_adaptation_diag_imm"]

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
    assert "window_adaptation_diag_imm" in prose_m

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


# test_medium_recipe_exists_and_has_warmup_data (parametrized over 6 combos)
# removed 2026-05-17 — see module docstring "History" section.
