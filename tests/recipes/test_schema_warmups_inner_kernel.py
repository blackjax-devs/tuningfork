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
"""Schema extension unit tests: Recipe.save/load round-trips.

Fast unit tests — no JAX traces, no MCMC, no chain runs.

Test coverage:
  (a) Recipe.save round-trips through new schema (warmups list written, not flat fields).
  (b) Recipe.load accepts legacy warmup_name/warmup_params shape.
  (c) Backward-compat: loading an existing on-disk LOW recipe still works.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tuningfork.recipes._base import Effort, Recipe

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CATALOG_ROOT = Path(__file__).resolve().parents[2] / "tuningfork" / "catalog"

_KNOWN_LOW_RECIPE = (
    _CATALOG_ROOT / "mvn_10" / "recipes" / "low__nuts__window_adaptation_diag_imm.json"
)


def _make_minimal_recipe(**overrides: Any) -> Recipe:
    """Construct a minimal Recipe for testing."""
    kwargs: dict[str, Any] = dict(
        model_name="test_model",
        base_method_name="nuts",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.5, "inverse_mass_matrix": [1.0, 1.0]},
        warmup_params={"n_warmup": 500, "num_chains": 2, "target_acceptance": 0.8},
        warmups=[
            {
                "name": "window_adaptation_diag_imm",
                "params": {"n_warmup": 500, "num_chains": 2, "target_acceptance": 0.8},
            }
        ],
        headline_metric=0.42,
        sample_quality=None,
        calibration_budget={
            "trials": 0,
            "wall_seconds_estimate": 12.0,
            "n_samples": 1000,
        },
        difficulty=None,
        instructions="test instructions",
        notes="",
        gate_evidence={
            "auto": {
                "rhat_max": 1.001,
                "min_bulk_ess": 600.0,
                "n_divergences": 0,
                "max_abs_mean_z": None,
                "verdict": "PASS",
                "margins": {},
            },
            "override": {"reason": "", "statistician_id": "", "decision": ""},
        },
        tuning_seed=42,
        tuningfork_version="0.0.0.dev0",
        blackjax_version="0.9.0",
        jax_version="0.4.0",
        timestamp_utc="2026-05-23T00:00:00Z",
    )
    kwargs.update(overrides)
    return Recipe(**kwargs)


# ---------------------------------------------------------------------------
# (a) save/load round-trip: warmups written; legacy fields absent in JSON
# ---------------------------------------------------------------------------


def test_save_writes_warmups_not_legacy_fields(tmp_path: Path) -> None:
    """Recipe.save emits 'warmups' list; does NOT emit warmup_name/warmup_params."""
    recipe = _make_minimal_recipe()
    recipe.save(tmp_path)

    # Find the written file
    written = list(tmp_path.rglob("*.json"))
    assert len(written) == 1, f"expected 1 file, got {written}"
    d = json.loads(written[0].read_text())

    assert "warmups" in d, "warmups list must be present in new schema JSON"
    assert "warmup_name" not in d, "warmup_name must NOT be written in new schema"
    assert "warmup_params" not in d, "warmup_params must NOT be written in new schema"


def test_save_warmups_content(tmp_path: Path) -> None:
    """warmups list content matches the stage passed to Recipe.__init__."""
    recipe = _make_minimal_recipe()
    recipe.save(tmp_path)

    written = list(tmp_path.rglob("*.json"))
    d = json.loads(written[0].read_text())

    warmups = d["warmups"]
    assert len(warmups) == 1
    assert warmups[0]["name"] == "window_adaptation_diag_imm"
    assert warmups[0]["params"]["n_warmup"] == 500


def test_save_load_round_trip(tmp_path: Path) -> None:
    """Recipe.save followed by Recipe.load produces an equivalent recipe."""
    original = _make_minimal_recipe()
    path = original.save(tmp_path)

    loaded = Recipe.load(path)
    assert loaded.model_name == original.model_name
    assert loaded.base_method_name == original.base_method_name
    assert loaded.warmup_name == original.warmup_name
    assert loaded.warmup_params == original.warmup_params
    assert loaded.warmups == original.warmups
    assert loaded.warmup_inner_kernel == original.warmup_inner_kernel
    assert loaded.effort == original.effort
    assert loaded.headline_metric == pytest.approx(original.headline_metric)


def test_save_load_warmup_inner_kernel(tmp_path: Path) -> None:
    """warmup_inner_kernel is preserved across save/load."""
    recipe = _make_minimal_recipe(
        warmup_inner_kernel="nuts",
        base_method_name="hmc",
    )
    path = recipe.save(tmp_path)
    loaded = Recipe.load(path)
    assert loaded.warmup_inner_kernel == "nuts"


# ---------------------------------------------------------------------------
# (b) Legacy load: warmup_name/warmup_params flat fields → warmups constructed
# ---------------------------------------------------------------------------


def test_load_legacy_flat_fields(tmp_path: Path) -> None:
    """Recipe.load accepts legacy format with warmup_name/warmup_params flat fields."""
    legacy_dict = {
        "model_name": "test_model",
        "base_method_name": "nuts",
        "warmup_name": "window_adaptation_diag_imm",
        "effort": "low",
        "base_method_params": {"step_size": 0.3, "inverse_mass_matrix": [1.0]},
        "warmup_params": {"n_warmup": 750, "target_acceptance": 0.8},
        "headline_metric": 0.3,
        "sample_quality": None,
        "calibration_budget": {"trials": 0, "wall_seconds_estimate": 5.0},
        "difficulty": None,
        "instructions": "",
        "notes": "",
        "gate_evidence": {
            "auto": {
                "rhat_max": 1.002,
                "min_bulk_ess": 500.0,
                "n_divergences": 0,
                "max_abs_mean_z": None,
                "verdict": "PASS",
                "margins": {},
            },
            "override": {"reason": "", "statistician_id": "", "decision": ""},
        },
        "tuning_seed": 0,
        "tuningfork_version": "0.0.0.dev0",
        "blackjax_version": "",
        "jax_version": "",
        "timestamp_utc": "",
        "failure_diagnosis": None,
        "attempted_configurations": [],
        "step_policy": None,
        "inverse_mass_matrix_path": None,
        "workflow": "",
    }
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy_dict) + "\n")

    recipe = Recipe.load(legacy_path)

    # warmup_name / warmup_params must still be accessible
    assert recipe.warmup_name == "window_adaptation_diag_imm"
    assert recipe.warmup_params["n_warmup"] == 750

    # warmups must be constructed as single-element list
    assert len(recipe.warmups) == 1
    assert recipe.warmups[0]["name"] == "window_adaptation_diag_imm"
    assert recipe.warmups[0]["params"]["n_warmup"] == 750

    # warmup_inner_kernel must default to None
    assert recipe.warmup_inner_kernel is None


def test_load_existing_on_disk_recipe() -> None:
    """Loading an existing on-disk LOW recipe works without regen."""
    if not _KNOWN_LOW_RECIPE.exists():
        pytest.skip(f"on-disk recipe not found: {_KNOWN_LOW_RECIPE}")

    recipe = Recipe.load(_KNOWN_LOW_RECIPE)

    assert recipe.model_name == "mvn_10"
    assert recipe.base_method_name == "nuts"
    assert recipe.warmup_name == "window_adaptation_diag_imm"
    assert recipe.effort == Effort.LOW
    # warmups must be populated from legacy flat fields
    assert len(recipe.warmups) == 1
    assert recipe.warmups[0]["name"] == "window_adaptation_diag_imm"
    # warmup_inner_kernel defaults to None for legacy recipes
    assert recipe.warmup_inner_kernel is None
    # step_size is accessible
    assert recipe.base_method_params.get("step_size") is not None


# ---------------------------------------------------------------------------
# (c) Backward-compat: loading a legacy on-disk recipe produces correct step_size
# ---------------------------------------------------------------------------


def test_legacy_load_step_size_matches_original() -> None:
    """Legacy recipe load yields the same step_size as the original JSON value."""
    if not _KNOWN_LOW_RECIPE.exists():
        pytest.skip(f"on-disk recipe not found: {_KNOWN_LOW_RECIPE}")

    with open(_KNOWN_LOW_RECIPE) as f:
        raw = json.load(f)
    original_step_size = raw["base_method_params"]["step_size"]

    recipe = Recipe.load(_KNOWN_LOW_RECIPE)
    assert recipe.base_method_params["step_size"] == pytest.approx(original_step_size)
