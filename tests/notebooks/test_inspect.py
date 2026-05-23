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
"""Fast tests for tuningfork.catalog.inspect (load_recipe, summarize_recipe).

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
        "warmup_name": "window_adaptation_diag_imm",
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
    recipe_file = tmp_path / "groundtruth__nuts__window_adaptation_diag_imm.json"
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
    from tuningfork.catalog.inspect import load_recipe

    recipe = load_recipe(minimal_recipe_json)
    assert recipe.model_name == "eight_schools_ncp"
    assert recipe.base_method_name == "nuts"
    assert recipe.warmup_name == "window_adaptation_diag_imm"


def test_load_recipe_relative_path_resolves_against_repo_root(
    minimal_recipe_json: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_recipe resolves a relative path against the tuningfork repo root."""
    import tuningfork as _tf
    from tuningfork.catalog.inspect import load_recipe

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
    from tuningfork.catalog.inspect import load_recipe

    with pytest.raises(FileNotFoundError, match="Recipe file not found"):
        load_recipe("/tmp/this_file_definitely_does_not_exist_abc123.json")


# ---------------------------------------------------------------------------
# summarize_recipe tests
# ---------------------------------------------------------------------------


def test_summarize_recipe_returns_dataframe(minimal_recipe_json: Path) -> None:
    """summarize_recipe returns a pd.DataFrame with expected shape."""
    from tuningfork.catalog.inspect import load_recipe, summarize_recipe

    recipe = load_recipe(minimal_recipe_json)
    df = summarize_recipe(recipe)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Property", "Value"]
    assert len(df) >= 10  # at least 10 rows


def test_summarize_recipe_contains_expected_rows(minimal_recipe_json: Path) -> None:
    """summarize_recipe includes model, sampler, verdict, R̂_max, etc."""
    from tuningfork.catalog.inspect import load_recipe, summarize_recipe

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
    from tuningfork.catalog.inspect import load_recipe, summarize_recipe

    recipe = load_recipe(minimal_recipe_json_empty_gate)
    df = summarize_recipe(recipe)

    rhat_row = df[df["Property"] == "R_hat_max"]
    assert len(rhat_row) == 1
    assert rhat_row.iloc[0]["Value"] == "N/A"

    verdict_row = df[df["Property"] == "stored gate verdict"]
    assert len(verdict_row) == 1
    assert verdict_row.iloc[0]["Value"] == "NOT_RUN"


# ---------------------------------------------------------------------------
# Sample-budget rows: num_chains / n_warmup / n_samples (emit-script-num-chains)
# ---------------------------------------------------------------------------

_CATALOG_ROOT = Path(__file__).resolve().parents[2] / "tuningfork" / "catalog"


def test_summarize_recipe_sample_budget_rows_low_recipe() -> None:
    """summarize_recipe includes num_chains / n_warmup / n_samples for a LOW recipe.

    Uses the committed eight_schools_ncp LOW nuts recipe which has:
      warmup_params.num_chains = 4, warmup_params.n_warmup = 1000
      calibration_budget.n_samples = 1000, calibration_budget.num_chains = 4
    """
    from tuningfork.catalog.inspect import load_recipe, summarize_recipe

    low_recipe_path = (
        _CATALOG_ROOT
        / "eight_schools_ncp"
        / "recipes"
        / "low__nuts__window_adaptation_diag_imm.json"
    )
    recipe = load_recipe(low_recipe_path)
    df = summarize_recipe(recipe)
    props = dict(zip(df["Property"].tolist(), df["Value"].tolist()))

    assert "num_chains" in props, "summarize_recipe must include 'num_chains' row"
    assert (
        props["num_chains"] == "4"
    ), f"Expected num_chains='4' for LOW recipe, got {props['num_chains']!r}"
    assert "n_warmup" in props, "summarize_recipe must include 'n_warmup' row"
    assert (
        props["n_warmup"] == "1000"
    ), f"Expected n_warmup='1000' for LOW recipe, got {props['n_warmup']!r}"
    assert "n_samples" in props, "summarize_recipe must include 'n_samples' row"
    assert (
        props["n_samples"] == "1000"
    ), f"Expected n_samples='1000' for LOW recipe, got {props['n_samples']!r}"


def test_summarize_recipe_sample_budget_rows_legacy_groundtruth(
    minimal_recipe_json: Path,
) -> None:
    """summarize_recipe shows 'N/A' for budget fields absent from legacy recipes.

    The minimal_recipe_json fixture has no num_chains in warmup_params or
    calibration_budget, and no n_samples anywhere — those fields didn't exist
    when the groundtruth protocol was defined.
    """
    from tuningfork.catalog.inspect import load_recipe, summarize_recipe

    recipe = load_recipe(minimal_recipe_json)
    df = summarize_recipe(recipe)
    props = dict(zip(df["Property"].tolist(), df["Value"].tolist()))

    # num_chains absent from both warmup_params and calibration_budget
    assert (
        props.get("num_chains") == "N/A"
    ), f"Expected num_chains='N/A' for legacy recipe, got {props.get('num_chains')!r}"
    # n_warmup present in warmup_params (n_warmup=1000 in the fixture)
    assert (
        props.get("n_warmup") == "1000"
    ), f"Expected n_warmup='1000', got {props.get('n_warmup')!r}"
    # n_samples absent
    assert (
        props.get("n_samples") == "N/A"
    ), f"Expected n_samples='N/A' for legacy recipe, got {props.get('n_samples')!r}"


# ---------------------------------------------------------------------------
# Phase B-2: warmup_inner_kernel surfacing in summarize_recipe
# ---------------------------------------------------------------------------


def test_summarize_recipe_warmup_inner_kernel_shown_when_set() -> None:
    """summarize_recipe includes warmup_inner_kernel row when explicitly set.

    Phase B-2: when a recipe has warmup_inner_kernel='nuts', the summary
    DataFrame must include a row ('warmup_inner_kernel', 'nuts') so the
    Statistician can immediately see the inner-kernel override without
    digging into recipe.warmup_inner_kernel directly.
    """
    from tuningfork.catalog.inspect import summarize_recipe
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1, "num_integration_steps": 10},
        warmup_params={"n_warmup": 200},
        warmups=[{"name": "window_adaptation_diag_imm", "params": {"n_warmup": 200}}],
        warmup_inner_kernel="nuts",  # Phase B-2 explicit override
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )

    df = summarize_recipe(recipe)
    props = dict(zip(df["Property"].tolist(), df["Value"].tolist()))

    assert "warmup_inner_kernel" in props, (
        "Phase B-2: summarize_recipe must include 'warmup_inner_kernel' row "
        "when recipe.warmup_inner_kernel is explicitly set."
    )
    assert (
        props["warmup_inner_kernel"] == "nuts"
    ), f"Expected warmup_inner_kernel='nuts', got {props['warmup_inner_kernel']!r}"


def test_summarize_recipe_warmup_inner_kernel_absent_when_none() -> None:
    """summarize_recipe omits warmup_inner_kernel row when the field is None.

    Phase B-2: legacy recipes (and new recipes without an explicit override)
    have warmup_inner_kernel=None. The summary must NOT include a
    warmup_inner_kernel row in that case — it would only add noise since
    the implicit substitute-family default is already well-known.
    """
    from tuningfork.catalog.inspect import summarize_recipe
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1, "num_integration_steps": 10},
        warmup_params={"n_warmup": 200},
        warmups=[{"name": "window_adaptation_diag_imm", "params": {"n_warmup": 200}}],
        warmup_inner_kernel=None,  # default — no override
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )

    df = summarize_recipe(recipe)
    props = set(df["Property"].tolist())

    assert "warmup_inner_kernel" not in props, (
        "Phase B-2: summarize_recipe must NOT include 'warmup_inner_kernel' row "
        "when recipe.warmup_inner_kernel is None (avoids noise for legacy recipes)."
    )
