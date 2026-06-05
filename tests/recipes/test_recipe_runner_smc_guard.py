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
"""Tests for the SMC guard in run_recipe_to_idata.

``run_recipe_to_idata`` is MCMC-only; it accesses ``recipe.effort``,
``recipe.base_method_name``, and ``recipe.warmup_name``, none of which
exist on ``SMCRecipe``.  The guard added in this PR detects this at the
top of the function (before any JAX computation) and raises a clear
``TypeError`` pointing the caller to ``run_smc()``.

Without the guard the function would raise an opaque ``AttributeError``
deep inside the warmup/sampler dispatch code — confusing and misleading.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.fast


def test_run_recipe_to_idata_raises_typeerror_for_smc_recipe() -> None:
    """Passing an SMCRecipe to run_recipe_to_idata raises TypeError with a clear message.

    This is the core regression guard: before this fix, an AttributeError
    was raised inside the dispatch logic (recipe.effort / recipe.base_method_name /
    recipe.warmup_name absent on SMCRecipe).  The guard raises early with a
    message pointing to run_smc().
    """
    from tuningfork.recipes._base_smc import SMCRecipe
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    smc_recipe = SMCRecipe(
        model_name="gmm_25",
        smc_method_name="adaptive_tempered_smc",
        inner_method_name="rwm",
        num_particles=100,
        max_steps=50,
    )

    with pytest.raises(TypeError, match="run_smc"):
        run_recipe_to_idata(smc_recipe)  # type: ignore[arg-type]


def test_run_recipe_to_idata_smc_error_mentions_smc_recipe_typename() -> None:
    """The TypeError message includes the actual type name for debugging."""
    from tuningfork.recipes._base_smc import SMCRecipe
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    smc_recipe = SMCRecipe(
        model_name="neals_funnel",
        smc_method_name="inner_kernel_tuning",
        inner_method_name="hmc",
        num_particles=200,
        max_steps=50,
    )

    with pytest.raises(TypeError, match="SMCRecipe"):
        run_recipe_to_idata(smc_recipe)  # type: ignore[arg-type]


def test_run_recipe_to_idata_accepts_mcmc_recipe_schema() -> None:
    """run_recipe_to_idata does NOT raise TypeError for a valid MCMC Recipe.

    Ensures the guard only fires for objects without the 'effort' attribute
    (i.e. SMCRecipe), not for regular Recipe objects.  The test uses a
    GROUNDTRUTH Recipe which early-exits via load_idata — no JAX computation.
    """
    from unittest.mock import MagicMock, patch

    from tuningfork.recipes._base import Effort, Recipe
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.GROUNDTRUTH,
        base_method_params={"step_size": 0.5},
        warmup_params={"n_warmup": 1000},
        warmups=[{"name": "window_adaptation_diag_imm", "params": {"n_warmup": 1000}}],
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"trials": 0, "wall_seconds_estimate": 0.0},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )

    # For GROUNDTRUTH effort, run_recipe_to_idata delegates to load_idata;
    # mock that out so no actual I/O or JAX happens.
    fake_idata = MagicMock()
    with patch(
        "tuningfork.catalog.render.load_idata", return_value=fake_idata
    ) as mock_load:
        result = run_recipe_to_idata(recipe)

    # No TypeError raised → guard passed correctly
    mock_load.assert_called_once()
    assert result is fake_idata
