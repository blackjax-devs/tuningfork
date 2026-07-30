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
"""Consumer tests for HEADLINE_ESTIMATOR_EXCLUDED_MODELS.

``catalog/_estimator_provenance.py`` already has invariant tests in
``test_emit.py`` that check the allowlist describes the catalog correctly
(no dead entries, no stale ones). Those never call either place that actually
CONSULTS the allowlist at read time: ``headline_estimator_of`` (surfaced to a
reader via ``summarize_recipe``) and ``reemit_sweep.reconstruct`` (which must
decline to re-verify an excluded model rather than comparing its legacy-
estimator headline against the current-estimator convention). A dict that is
only ever iterated by its own invariant tests is decorative; these tests
exercise the two call sites so an allowlist entry is provably load-bearing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_reemit_driver():
    spec = importlib.util.spec_from_file_location(
        "reemit_sweep", _REPO_ROOT / "tools" / "reemit_sweep.py"
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["reemit_sweep"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


driver = _load_reemit_driver()


# ---------------------------------------------------------------------------
# headline_estimator_of: pure-function behaviour
# ---------------------------------------------------------------------------


def test_excluded_model_reports_the_legacy_estimator_with_its_recorded_reason() -> None:
    """An excluded model's caveat is the reason recorded in the allowlist.

    Precedence matters: exclusion is checked before the recipe's own
    ``headline_basis``, so an excluded model is flagged even if some future
    re-emit accidentally stamps it with the current estimator.
    """
    from tuningfork.catalog._estimator_provenance import (
        HEADLINE_ESTIMATOR_EXCLUDED_MODELS,
        headline_estimator_of,
    )
    from tuningfork.metrics.headline import HEADLINE_ESS_ESTIMATOR, LEGACY_ESS_ESTIMATOR

    recipe = SimpleNamespace(
        model_name="gp_regression",
        headline_basis={"ess_estimator": HEADLINE_ESS_ESTIMATOR},
    )
    estimator, caveat = headline_estimator_of(recipe)
    assert estimator == LEGACY_ESS_ESTIMATOR
    assert caveat is not None
    assert caveat == HEADLINE_ESTIMATOR_EXCLUDED_MODELS["gp_regression"]


def test_non_excluded_model_on_the_headline_estimator_has_no_caveat() -> None:
    from tuningfork.catalog._estimator_provenance import headline_estimator_of
    from tuningfork.metrics.headline import HEADLINE_ESS_ESTIMATOR

    recipe = SimpleNamespace(
        model_name="banana",
        headline_basis={"ess_estimator": HEADLINE_ESS_ESTIMATOR},
    )
    estimator, caveat = headline_estimator_of(recipe)
    assert estimator == HEADLINE_ESS_ESTIMATOR
    assert caveat is None


def test_non_excluded_model_without_a_stamp_is_flagged_unrecorded_not_excluded() -> (
    None
):
    """A pre-stamp recipe and an excluded model are different failure modes.

    Collapsing them would make an unstamped legacy recipe look like a
    deliberate, reasoned exclusion instead of a gap to close by re-emitting.
    """
    from tuningfork.catalog._estimator_provenance import headline_estimator_of

    recipe = SimpleNamespace(model_name="banana", headline_basis=None)
    estimator, caveat = headline_estimator_of(recipe)
    assert estimator == "unrecorded"
    assert caveat is not None
    assert "gp_regression" not in caveat


# ---------------------------------------------------------------------------
# Consumer 1: tuningfork.catalog.inspect.summarize_recipe
# ---------------------------------------------------------------------------


def _write_recipe(tmp_path: Path, model_name: str, headline_basis) -> Path:
    recipe_data = {
        "model_name": model_name,
        "base_method_name": "hmc",
        "warmup_name": "window_adaptation_diag_imm",
        "effort": "low",
        "base_method_params": {"step_size": 0.5, "inverse_mass_matrix": [1.0]},
        "warmup_params": {"n_warmup": 1000},
        "headline_metric": 1.0,
        "headline_basis": headline_basis,
        "sample_quality": None,
        "calibration_budget": {"trials": 0, "wall_seconds_estimate": 1.0},
        "difficulty": None,
        "instructions": "Run HMC.",
        "notes": "",
        "inverse_mass_matrix_path": None,
        "workflow": "",
        "gate_evidence": {
            "auto": {
                "rhat_max": 1.001,
                "min_bulk_ess": 500.0,
                "n_divergences": 0,
                "max_abs_mean_z": None,
                "verdict": "PASS",
                "margins": {},
            },
            "override": {"reason": "", "statistician_id": "", "decision": ""},
        },
        "tuning_seed": 0,
        "tuningfork_version": "0.1.0",
        "blackjax_version": "1.0.0",
        "jax_version": "0.4.0",
        "timestamp_utc": "2026-05-12T00:00:00Z",
    }
    recipe_file = tmp_path / f"{model_name}.json"
    recipe_file.write_text(json.dumps(recipe_data))
    return recipe_file


def test_summarize_recipe_marks_an_excluded_models_headline_not_comparable(
    tmp_path: Path,
) -> None:
    from tuningfork.catalog.inspect import load_recipe, summarize_recipe
    from tuningfork.metrics.headline import HEADLINE_ESS_ESTIMATOR

    # Even a recipe stamped with the current-catalog estimator is overridden:
    # the model itself is off the migration, not just this one recipe.
    path = _write_recipe(
        tmp_path, "gp_regression", {"ess_estimator": HEADLINE_ESS_ESTIMATOR}
    )
    recipe = load_recipe(path)
    df = summarize_recipe(recipe)

    row = df.loc[df["Property"] == "headline_ESS_estimator", "Value"].iloc[0]
    assert "NOT COMPARABLE" in row
    assert "gp_regression" not in row  # the reason text, not the model name, carries it
    assert "63 hours" in row  # the recorded reason from the allowlist, verbatim


def test_summarize_recipe_shows_a_plain_estimator_for_a_non_excluded_model(
    tmp_path: Path,
) -> None:
    from tuningfork.catalog.inspect import load_recipe, summarize_recipe
    from tuningfork.metrics.headline import HEADLINE_ESS_ESTIMATOR

    path = _write_recipe(tmp_path, "banana", {"ess_estimator": HEADLINE_ESS_ESTIMATOR})
    recipe = load_recipe(path)
    df = summarize_recipe(recipe)

    row = df.loc[df["Property"] == "headline_ESS_estimator", "Value"].iloc[0]
    assert row == HEADLINE_ESS_ESTIMATOR
    assert "NOT COMPARABLE" not in row


# ---------------------------------------------------------------------------
# Consumer 2: tools/reemit_sweep.py's reconstruct()
# ---------------------------------------------------------------------------


def test_reconstruct_declines_an_excluded_model_before_any_other_check(
    tmp_path: Path,
) -> None:
    """The exclusion check runs first, so it fires even for an otherwise-valid cell.

    Directory name drives the check (``recipe_path.parent.parent.name``), not
    the recipe's own ``model_name`` field — pinning that keeps a future refactor
    from silently reading the wrong field.
    """
    model_dir = tmp_path / "gp_regression" / "recipes"
    model_dir.mkdir(parents=True)
    p = model_dir / "high__hmc__chees.json"
    p.write_text(
        json.dumps(
            {"effort": "high", "base_method_name": "hmc", "model_name": "banana"}
        )
    )
    result = driver.reconstruct(p)
    assert isinstance(result, driver.Skip)
    assert "excluded from the estimator migration" in result.reason


def test_reconstruct_does_not_decline_a_non_excluded_model_for_estimator_exclusion(
    tmp_path: Path,
) -> None:
    """The exclusion check is conditioned on allowlist membership, not a
    directory-name pattern — a model that merely sounds related is unaffected.
    """
    model_dir = tmp_path / "gp_regression_v2" / "recipes"
    model_dir.mkdir(parents=True)
    p = model_dir / "failed__elliptical_slice__no_warmup.json"
    p.write_text(
        json.dumps({"effort": "failed", "base_method_name": "elliptical_slice"})
    )
    result = driver.reconstruct(p)
    assert isinstance(result, driver.Skip)
    assert "excluded from the estimator migration" not in result.reason
    assert "failed recipe" in result.reason
