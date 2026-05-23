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
"""Phase B-2 schema tests: Recipe.save/load round-trips + transform_warmup_state.

Fast unit tests — no JAX traces, no MCMC, no chain runs.

Test coverage:
  (a) Recipe.save round-trips through new schema (warmups list written, not flat fields).
  (b) Recipe.load accepts legacy warmup_name/warmup_params shape.
  (c) transform_warmup_state resolution table for nuts → {hmc, dynamic_hmc, mala}.
  (d) Backward-compat: loading an existing on-disk LOW recipe still works.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
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
# (c) transform_warmup_state resolution table
# ---------------------------------------------------------------------------


def _mock_warmup_info(nis_values: list[int]) -> MagicMock:
    """Build a mock warmup_info with num_integration_steps."""
    info = MagicMock()
    info.num_integration_steps = np.array(nis_values)
    return info


def _adapted_params(step_size: float = 0.1, imm: list[float] | None = None) -> dict:
    return {
        "step_size": np.array(step_size),
        "inverse_mass_matrix": np.array(imm or [1.0, 1.0]),
    }


class TestTransformWarmupState:
    """Unit tests for transform_warmup_state resolution table."""

    def test_nuts_to_nuts_identity(self) -> None:
        """nuts → nuts: identity transform (no NIS injection)."""
        from tuningfork.base_method._warmup_to_sampler_transform import (
            transform_warmup_state,
        )

        info = _mock_warmup_info([5, 7, 10, 15])
        result = transform_warmup_state("nuts", "nuts", _adapted_params(), info)

        assert "step_size" in result
        assert "inverse_mass_matrix" in result
        assert "num_integration_steps" not in result
        assert "step_policy" not in result

    def test_nuts_to_hmc_nis_median(self) -> None:
        """nuts → hmc: injects num_integration_steps = median(NIS)."""
        from tuningfork.base_method._warmup_to_sampler_transform import (
            transform_warmup_state,
        )

        nis = [10, 20, 30, 40]  # median = 25
        info = _mock_warmup_info(nis)
        result = transform_warmup_state("nuts", "hmc", _adapted_params(), info)

        assert "num_integration_steps" in result
        assert result["num_integration_steps"] == int(np.median(nis))

    def test_nuts_to_mhmc_nis_median(self) -> None:
        """nuts → mhmc: injects num_integration_steps = median(NIS)."""
        from tuningfork.base_method._warmup_to_sampler_transform import (
            transform_warmup_state,
        )

        nis = [8, 16, 32]  # median = 16
        info = _mock_warmup_info(nis)
        result = transform_warmup_state("nuts", "mhmc", _adapted_params(), info)

        assert result["num_integration_steps"] == 16

    def test_nuts_to_dynamic_hmc_empirical(self) -> None:
        """nuts → dynamic_hmc: injects empirical step_policy from NIS."""
        from tuningfork.base_method._warmup_to_sampler_transform import (
            transform_warmup_state,
        )

        nis = [5, 5, 10, 10, 10, 15]
        info = _mock_warmup_info(nis)
        result = transform_warmup_state("nuts", "dynamic_hmc", _adapted_params(), info)

        assert "step_policy" in result
        sp = result["step_policy"]
        assert sp["kind"] == "empirical"
        assert "values" in sp and "weights" in sp
        assert len(sp["values"]) == len(sp["weights"])
        # Sum of weights ≈ 1
        assert abs(sum(sp["weights"]) - 1.0) < 1e-6

    def test_nuts_to_dmhmc_empirical(self) -> None:
        """nuts → dmhmc: injects empirical step_policy from NIS."""
        from tuningfork.base_method._warmup_to_sampler_transform import (
            transform_warmup_state,
        )

        nis = [20, 20, 30, 40, 40]
        info = _mock_warmup_info(nis)
        result = transform_warmup_state("nuts", "dmhmc", _adapted_params(), info)

        assert "step_policy" in result
        sp = result["step_policy"]
        assert sp["kind"] == "empirical"

    def test_nuts_to_dynamic_hmc_with_policy_override(self) -> None:
        """When step_policy_override provided, use it instead of computing fresh."""
        from tuningfork.base_method._warmup_to_sampler_transform import (
            transform_warmup_state,
        )

        pinned_spec = {"kind": "uniform_int", "low": 50, "high": 200}
        nis = [5, 6, 7, 8]  # would produce different empirical spec
        info = _mock_warmup_info(nis)
        result = transform_warmup_state(
            "nuts",
            "dynamic_hmc",
            _adapted_params(),
            info,
            step_policy_override=pinned_spec,
        )

        assert result["step_policy"] == pinned_spec

    def test_hmc_to_hmc_identity(self) -> None:
        """hmc → hmc (self-warmup): identity transform."""
        from tuningfork.base_method._warmup_to_sampler_transform import (
            transform_warmup_state,
        )

        info = _mock_warmup_info([5])  # should be ignored
        result = transform_warmup_state("hmc", "hmc", _adapted_params(), info)

        assert "step_size" in result
        assert "inverse_mass_matrix" in result
        assert "num_integration_steps" not in result

    def test_mala_to_mala_identity(self) -> None:
        """mala → mala (self-warmup): identity transform."""
        from tuningfork.base_method._warmup_to_sampler_transform import (
            transform_warmup_state,
        )

        info = MagicMock()
        del info.num_integration_steps  # MALA has no NIS
        result = transform_warmup_state("mala", "mala", _adapted_params(), info)

        assert "step_size" in result
        assert "num_integration_steps" not in result

    def test_barker_to_barker_identity(self) -> None:
        """barker → barker (self-warmup): identity transform."""
        from tuningfork.base_method._warmup_to_sampler_transform import (
            transform_warmup_state,
        )

        info = MagicMock()
        result = transform_warmup_state("barker", "barker", _adapted_params(), info)

        assert "step_size" in result
        assert "num_integration_steps" not in result

    def test_implicit_kernel_none_for_substitute_family(self) -> None:
        """warmup_inner_kernel=None + dynamic_hmc → resolves to nuts → empirical."""
        from tuningfork.base_method._warmup_to_sampler_transform import (
            transform_warmup_state,
        )

        nis = [10, 20, 30]
        info = _mock_warmup_info(nis)
        result = transform_warmup_state(None, "dynamic_hmc", _adapted_params(), info)

        # Implicit resolution: dynamic_hmc → nuts substitute → empirical
        assert "step_policy" in result
        assert result["step_policy"]["kind"] == "empirical"

    def test_implicit_kernel_none_for_standard_method(self) -> None:
        """warmup_inner_kernel=None + nuts → resolves to nuts → identity."""
        from tuningfork.base_method._warmup_to_sampler_transform import (
            transform_warmup_state,
        )

        nis = [10, 20, 30]
        info = _mock_warmup_info(nis)
        result = transform_warmup_state(None, "nuts", _adapted_params(), info)

        assert "num_integration_steps" not in result
        assert "step_policy" not in result

    def test_multichain_nis_ravelled(self) -> None:
        """Multi-chain NIS shape is ravelled before median — one canonical median."""
        from tuningfork.base_method._warmup_to_sampler_transform import (
            transform_warmup_state,
        )

        # Simulate (4 chains, 100 warmup steps) NIS tensor
        nis_multichain = np.full((4, 100), fill_value=20)
        nis_multichain[2, :] = 40  # chain 2 has higher NIS

        info = MagicMock()
        info.num_integration_steps = nis_multichain

        result = transform_warmup_state("nuts", "hmc", _adapted_params(), info)

        # Ravelled median: 300*20 + 100*40 = 6000+4000=10000 values
        # 75% are 20, 25% are 40 → median = 20
        rav = nis_multichain.ravel()
        expected_median = int(np.median(rav))
        assert result["num_integration_steps"] == expected_median


# ---------------------------------------------------------------------------
# (d) Backward-compat: loading a legacy on-disk recipe produces correct step_size
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
