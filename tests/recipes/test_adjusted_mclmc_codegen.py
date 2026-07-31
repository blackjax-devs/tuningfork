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
