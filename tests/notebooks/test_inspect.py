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
"""Fast tests for tuningfork.notebooks.inspect (load_recipe, summarize_recipe).

All tests are pure logic / schema — no JAX trace, no chain runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_recipe_json(tmp_path: Path) -> Path:
    """Write a minimal valid recipe JSON to a temp file and return the path."""
    recipe_data = {
        "model_name": "eight_schools_ncp",
        "base_method_name": "nuts",
        "warmup_name": "stan_window",
        "effort": "groundtruth",
        "base_method_params": {"step_size": 0.5, "inverse_mass_matrix": [1.0, 1.0]},
        "warmup_params": {"n_warmup": 1000},
        "headline_metric": None,
        "sample_quality": None,
        "calibration_budget": {"trials": 0, "wall_seconds_estimate": 42.0},
        "difficulty": None,
        "instructions": "Run NUTS.",
        "notes": "",
        "inverse_mass_matrix_path": None,
        "workflow": "",
        "gate_evidence": {
            "auto": {
                "rhat_max": 1.002,
                "min_bulk_ess": 500.0,
                "n_divergences": 0,
                "max_abs_mean_z": None,
                "verdict": "PASS",
                "margins": {},
            },
            "override": {
                "reason": "",
                "statistician_id": "",
                "decision": "",
            },
        },
        "tuning_seed": 42,
        "tuningfork_version": "0.1.0",
        "blackjax_version": "1.0.0",
        "jax_version": "0.4.0",
        "timestamp_utc": "2026-05-12T00:00:00Z",
    }
    recipe_file = tmp_path / "groundtruth__nuts__stan_window.json"
    recipe_file.write_text(json.dumps(recipe_data))
    return recipe_file


@pytest.fixture
def minimal_recipe_json_empty_gate(tmp_path: Path) -> Path:
    """Recipe JSON with empty gate_evidence (NOT_RUN verdict, no R̂ rows)."""
    recipe_data = {
        "model_name": "mvn_10",
        "base_method_name": "hmc",
        "warmup_name": "no_warmup",
        "effort": "low",
        "base_method_params": {"step_size": 0.1},
        "warmup_params": {},
        "headline_metric": None,
        "sample_quality": None,
        "calibration_budget": {"trials": 0, "wall_seconds_estimate": 0.0},
        "difficulty": None,
        "instructions": "",
        "notes": "",
        "inverse_mass_matrix_path": None,
        "workflow": "",
        "gate_evidence": {
            "auto": {
                "rhat_max": None,
                "min_bulk_ess": None,
                "n_divergences": None,
                "max_abs_mean_z": None,
                "verdict": "NOT_RUN",
                "margins": {},
            },
            "override": {
                "reason": "",
                "statistician_id": "",
                "decision": "",
            },
        },
        "tuning_seed": 0,
        "tuningfork_version": "0.1.0",
        "blackjax_version": "1.0.0",
        "jax_version": "0.4.0",
        "timestamp_utc": "2026-05-12T00:00:00Z",
    }
    recipe_file = tmp_path / "low__hmc__no_warmup.json"
    recipe_file.write_text(json.dumps(recipe_data))
    return recipe_file


# ---------------------------------------------------------------------------
# load_recipe tests
# ---------------------------------------------------------------------------


def test_load_recipe_absolute_path(minimal_recipe_json: Path) -> None:
    """load_recipe loads a recipe from an absolute path."""
    from tuningfork.notebooks import load_recipe

    recipe = load_recipe(minimal_recipe_json)
    assert recipe.model_name == "eight_schools_ncp"
    assert recipe.base_method_name == "nuts"
    assert recipe.warmup_name == "stan_window"


def test_load_recipe_relative_path_resolves_against_repo_root(
    minimal_recipe_json: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_recipe resolves a relative path against the tuningfork repo root."""
    import tuningfork as _tf
    from tuningfork.notebooks import load_recipe

    # The recipe is in a temp dir — make the repo root point to the parent
    # of the temp dir so relative resolution works.
    tmp_parent = minimal_recipe_json.parent.parent

    # Write a pyproject.toml sentinel in the temp parent so _repo_root detects it
    sentinel = tmp_parent / "pyproject.toml"
    sentinel.write_text("[tool.fake]")

    # Patch tuningfork.__file__ to point into the temp dir
    # so that _repo_root() returns tmp_parent
    fake_tf_path = tmp_parent / "tuningfork" / "__init__.py"
    fake_tf_path.parent.mkdir(parents=True, exist_ok=True)
    fake_tf_path.write_text("")

    monkeypatch.setattr(_tf, "__file__", str(fake_tf_path))

    # Build relative path: <temp-dir-name>/<filename>
    rel_path = minimal_recipe_json.relative_to(tmp_parent)
    recipe = load_recipe(rel_path)
    assert recipe.model_name == "eight_schools_ncp"


def test_load_recipe_missing_raises_file_not_found() -> None:
    """load_recipe raises FileNotFoundError with a helpful message on miss."""
    from tuningfork.notebooks import load_recipe

    with pytest.raises(FileNotFoundError, match="Recipe file not found"):
        load_recipe("/tmp/this_file_definitely_does_not_exist_abc123.json")


# ---------------------------------------------------------------------------
# summarize_recipe tests
# ---------------------------------------------------------------------------


def test_summarize_recipe_returns_dataframe(minimal_recipe_json: Path) -> None:
    """summarize_recipe returns a pd.DataFrame with expected shape."""
    from tuningfork.notebooks import load_recipe, summarize_recipe

    recipe = load_recipe(minimal_recipe_json)
    df = summarize_recipe(recipe)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Property", "Value"]
    assert len(df) >= 10  # at least 10 rows


def test_summarize_recipe_contains_expected_rows(minimal_recipe_json: Path) -> None:
    """summarize_recipe includes model, sampler, verdict, R̂_max, etc."""
    from tuningfork.notebooks import load_recipe, summarize_recipe

    recipe = load_recipe(minimal_recipe_json)
    df = summarize_recipe(recipe)
    props = set(df["Property"].tolist())

    for expected in (
        "model",
        "sampler",
        "effort",
        "stored gate verdict",
        "tuning_seed",
    ):
        assert expected in props, f"Expected property {expected!r} not found in summary"


def test_summarize_recipe_empty_gate_shows_na(
    minimal_recipe_json_empty_gate: Path,
) -> None:
    """When gate_evidence is NOT_RUN, R̂_max shows 'N/A'."""
    from tuningfork.notebooks import load_recipe, summarize_recipe

    recipe = load_recipe(minimal_recipe_json_empty_gate)
    df = summarize_recipe(recipe)

    rhat_row = df[df["Property"] == "R_hat_max"]
    assert len(rhat_row) == 1
    assert rhat_row.iloc[0]["Value"] == "N/A"

    verdict_row = df[df["Property"] == "stored gate verdict"]
    assert len(verdict_row) == 1
    assert verdict_row.iloc[0]["Value"] == "NOT_RUN"
