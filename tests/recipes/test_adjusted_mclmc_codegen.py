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

"""Generated-program coverage for adjusted-MCLMC variants."""

from pathlib import Path

import pytest

from tuningfork.recipes._base import Recipe
from tuningfork.recipes._emit_script import emit_script

_CATALOG = Path(__file__).parents[2] / "tuningfork" / "catalog" / "mvn_10" / "recipes"


@pytest.mark.e2e
def test_adjusted_mclmc_tuning_emits_and_runs(
    tmp_path: Path,
) -> None:
    from tuningfork.catalog import execute_recipe

    recipe = Recipe.load(_CATALOG / "low__adjusted_mclmc__adjusted_mclmc_tuning.json")
    result = execute_recipe(
        recipe,
        tmp_path,
        num_warmup=2,
        num_samples=2,
        num_chains=2,
        timeout=120,
    )
    assert result.artifact_path is not None


@pytest.mark.e2e
def test_adjusted_mclmc_dynamic_tuning_emits_and_runs(
    tmp_path: Path,
) -> None:
    from tuningfork.catalog import execute_recipe

    recipe = Recipe.load(
        _CATALOG / "low__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json"
    )
    result = execute_recipe(
        recipe,
        tmp_path,
        num_warmup=2,
        num_samples=2,
        num_chains=2,
        timeout=120,
    )
    assert result.artifact_path is not None


@pytest.mark.fast
def test_adjusted_mclmc_dynamic_keeps_continuous_average_trajectory_length() -> None:
    recipe = Recipe.load(
        _CATALOG / "low__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json"
    )
    source = emit_script(recipe, num_warmup=2, num_samples=2, num_chains=2)
    assert "_trajectory_parameter = jnp.maximum(1.0, _L / step_size)" in source
