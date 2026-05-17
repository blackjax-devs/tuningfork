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
"""Tests for Recipe dataclass schema validation and logic (no emission).

This file contains all pure-logic tests: Recipe construction, field validation,
schema-conformance, serialization/deserialization, and rendering of instructions.
All tests are marked @pytest.mark.fast.
"""

import dataclasses
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.calibration.tune import default_params_for
from tuningfork.model import MODELS
from tuningfork.recipes import (
    AttemptedConfig,
    Effort,
    FailureDiagnosis,
    Recipe,
    RecipeFailedError,
)
from tuningfork.recipes._instructions import render_instructions

# Path to the committed catalog (post-R2 layout, 2026-05-17)
_CATALOG_ROOT = Path(__file__).resolve().parents[2] / "tuningfork" / "catalog"


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
        tuningfork_version="0.0.0.dev0",
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

    # NUTS default: step_size = 1e-3 * (1/1e-3)**0.7 (70th-pctile on log-scale)
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
    # step_size = 1e-3 * (1/1e-3)**0.7 (70th-pctile on log-scale)
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
    assert loaded.tuningfork_version == recipe.tuningfork_version
    assert loaded.blackjax_version == recipe.blackjax_version
    assert loaded.jax_version == recipe.jax_version
    assert loaded.timestamp_utc == recipe.timestamp_utc

    # Effort enum preserved correctly
    assert isinstance(loaded.effort, Effort)
    assert loaded.effort == Effort.LOW

    # Filename convention (post-R2, 2026-05-17): non-groundtruth recipes
    # live under <model>/recipes/<effort>__<method>__<warmup>.json.
    assert saved_path.name == "low__nuts__no_warmup.json"
    assert saved_path.parent.name == "recipes"
    assert saved_path.parent.parent.name == "mvn_10"


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
    # Under the canonical taxonomy, LOW = conventional pairing with library
    # defaults; the prose should label it as such.  When `from_default_config`
    # produces a not-yet-measured stub (headline_metric=None), the template
    # renders "not yet measured" rather than failing to format.
    assert "Low-effort recipe" in prose
    assert "conventional" in prose
    assert "not yet measured" in prose


@pytest.mark.fast
def test_render_instructions_medium_stub() -> None:
    """render_instructions with a stub MEDIUM recipe returns non-empty text."""
    # Build a Recipe manually with MEDIUM effort (simulating a future from_warmup_only)
    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.MEDIUM,
        base_method_params={"step_size": 0.1, "inverse_mass_matrix": [1.0] * 10},
        warmup_params={"n_warmup": 1000, "target_acceptance_rate": 0.8},
        headline_metric=0.0512,
        sample_quality=None,
        calibration_budget={"trials": 0, "wall_seconds_estimate": 45.0},
        difficulty=None,
        instructions="",
        tuning_seed=0,
        tuningfork_version="0.0.0.dev0",
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
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.HIGH,
        base_method_params={"step_size": 0.08, "num_integration_steps": 32},
        warmup_params={"n_warmup": 1000},
        headline_metric=0.0731,
        sample_quality=None,
        calibration_budget={"trials": 50, "wall_seconds_estimate": 1800.0},
        difficulty=None,
        instructions="",
        tuning_seed=42,
        tuningfork_version="0.0.0.dev0",
        blackjax_version="1.0.0",
        jax_version="0.4.0",
        timestamp_utc="2026-01-01T00:00:00Z",
    )
    prose = render_instructions(recipe)
    assert isinstance(prose, str)
    assert len(prose) > 10
    assert "high" in prose.lower() or "High" in prose


# ---------------------------------------------------------------------------
# Test 17: _generate_starter CLI flag filtering
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_emit_low_recipes_sampler_filter(tmp_path: Path, monkeypatch) -> None:
    """emit_low_recipes(sampler='nuts') emits NUTS recipes only.

    added per-cell flag filtering to ``_generate_starter.py``;
    this test locks the ``sampler`` filter behavior at the function-level
    so future refactors can't silently regress it.

    Uses monkeypatch to redirect _STARTER_ROOT into tmp_path so we don't
    clobber committed recipes. Only tests LOW because it's deterministic
    and zero-cost (no MCMC).
    """
    from tuningfork.recipes import _generate_starter

    monkeypatch.setattr(_generate_starter, "_CATALOG_ROOT", tmp_path)
    paths = _generate_starter.emit_low_recipes(model_names=["mvn_10"], sampler="nuts")
    assert len(paths) == 1, f"Expected 1 recipe, got {len(paths)}: {paths}"
    assert paths[0].name == "low__nuts__no_warmup.json"


@pytest.mark.fast
def test_emit_low_recipes_no_filter_emits_all_methods(
    tmp_path: Path, monkeypatch
) -> None:
    """emit_low_recipes() with no sampler filter emits all 6 algos for the model."""
    from tuningfork.recipes import _generate_starter

    monkeypatch.setattr(_generate_starter, "_CATALOG_ROOT", tmp_path)
    paths = _generate_starter.emit_low_recipes(model_names=["mvn_10"])
    # 6 algorithms: hmc, nuts, mala, barker, rwm, mclmc
    assert len(paths) == 6
    method_names = sorted(p.name.split("__")[1] for p in paths)
    assert method_names == sorted(["hmc", "nuts", "mala", "barker", "rwm", "mclmc"])


@pytest.mark.fast
def test_main_rejects_unknown_model(monkeypatch) -> None:
    """`--only <unknown>` raises SystemExit."""
    import sys

    from tuningfork.recipes import _generate_starter

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

    from tuningfork.recipes import _generate_starter

    saved_argv = sys.argv
    try:
        sys.argv = ["_generate_starter", "--help"]
        with pytest.raises(SystemExit) as exc_info:
            _generate_starter.main()
        # argparse's --help calls sys.exit(0) on success.
        assert exc_info.value.code == 0
    finally:
        sys.argv = saved_argv


# ---------------------------------------------------------------------------
# Tests for schema fields + IMM sidecar helpers
# ---------------------------------------------------------------------------

_RECIPE_KWARGS_MINIMAL: dict[str, Any] = dict(
    model_name="mvn_10",
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
)


@pytest.mark.fast
def test_recipe_has_phase5_fields() -> None:
    """new schema fields default to the expected values."""
    recipe = Recipe(**_RECIPE_KWARGS_MINIMAL)

    # inverse_mass_matrix_path defaults to None
    assert recipe.inverse_mass_matrix_path is None

    # workflow defaults to empty string
    assert recipe.workflow == ""

    # gate_evidence defaults to the prescribed nested dict shape
    ge = recipe.gate_evidence
    assert isinstance(ge, dict)
    assert "auto" in ge
    assert "override" in ge

    auto = ge["auto"]
    assert auto["rhat_max"] is None
    assert auto["min_bulk_ess"] is None
    assert auto["n_divergences"] is None
    assert auto["max_abs_mean_z"] is None
    assert auto["verdict"] == "NOT_RUN"
    assert auto["margins"] == {}

    override = ge["override"]
    assert override["reason"] == ""
    assert override["statistician_id"] == ""
    assert override["decision"] == ""

    # Verify default_factory produces independent dicts (no shared mutable state)
    recipe2 = Recipe(**_RECIPE_KWARGS_MINIMAL)
    assert recipe.gate_evidence is not recipe2.gate_evidence


@pytest.mark.fast
def test_recipe_phase5_fields_save_load_roundtrip(tmp_path: Path) -> None:
    """non-default schema field values round-trip through save/load."""
    custom_gate_evidence = {
        "auto": {
            "rhat_max": 1.005,
            "min_bulk_ess": 412.3,
            "n_divergences": 0,
            "max_abs_mean_z": 0.12,
            "verdict": "PASS",
            "margins": {"rhat_max": 0.005, "min_bulk_ess": 12.3},
        },
        "override": {
            "reason": "Looks fine",
            "statistician_id": "stat-007",
            "decision": "APPROVE",
        },
    }
    recipe = Recipe(
        **_RECIPE_KWARGS_MINIMAL,
        gate_evidence=custom_gate_evidence,
        workflow="ran NUTS, observed leapfrog mean=22",
        inverse_mass_matrix_path="test/path.imm.npz",
    )

    saved_path = recipe.save(tmp_path)
    loaded = Recipe.load(saved_path)

    assert loaded.inverse_mass_matrix_path == "test/path.imm.npz"
    assert loaded.workflow == "ran NUTS, observed leapfrog mean=22"
    assert loaded.gate_evidence == custom_gate_evidence
    assert loaded.gate_evidence["auto"]["verdict"] == "PASS"
    assert loaded.gate_evidence["override"]["decision"] == "APPROVE"


@pytest.mark.fast
def test_save_imm_sidecar_and_load_roundtrip(tmp_path: Path) -> None:
    """save_imm_sidecar writes .npz; load_imm_sidecar recovers the array."""
    recipe = Recipe(**_RECIPE_KWARGS_MINIMAL)
    original_imm = jnp.eye(10)

    rel_path = recipe.save_imm_sidecar(tmp_path, original_imm)

    # The file must exist at the expected location (post-R2 layout):
    # LOW IMM sidecar lives under <model>/recipes/<effort>__<...>.imm.npz.
    expected_file = tmp_path / "mvn_10" / "recipes" / "low__nuts__no_warmup.imm.npz"
    assert expected_file.exists(), f"Expected sidecar at {expected_file}"

    # rel_path is relative to tmp_path
    assert rel_path == str(Path("mvn_10") / "recipes" / "low__nuts__no_warmup.imm.npz")

    # Load via load_imm_sidecar (with inverse_mass_matrix_path set)
    recipe_with_path = Recipe(
        **_RECIPE_KWARGS_MINIMAL, inverse_mass_matrix_path=rel_path
    )
    loaded_imm = recipe_with_path.load_imm_sidecar(tmp_path)

    assert loaded_imm is not None
    assert jnp.allclose(loaded_imm, original_imm)


@pytest.mark.fast
def test_load_imm_sidecar_returns_none_when_path_unset(tmp_path: Path) -> None:
    """load_imm_sidecar returns None when inverse_mass_matrix_path is None."""
    recipe = Recipe(**_RECIPE_KWARGS_MINIMAL)
    assert recipe.inverse_mass_matrix_path is None

    result = recipe.load_imm_sidecar(tmp_path)
    assert result is None


# ---------------------------------------------------------------------------
# Tests for Effort.GROUNDTRUTH (Phase 0)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_effort_enum_has_five_values() -> None:
    """Effort enum has exactly 5 values: LOW, MEDIUM, HIGH, GROUNDTRUTH, FAILED."""
    assert len(Effort) == 5
    assert set(Effort) == {
        Effort.LOW,
        Effort.MEDIUM,
        Effort.HIGH,
        Effort.GROUNDTRUTH,
        Effort.FAILED,
    }
    assert Effort.GROUNDTRUTH.value == "groundtruth"
    assert Effort.GROUNDTRUTH == "groundtruth"
    assert Effort.FAILED.value == "failed"
    assert Effort.FAILED == "failed"


def _make_mock_cert(
    split_rhat_max: float = 1.005,
    min_chunk_bulk_ess: float = 450.0,
    num_divergences: int = 0,
    e_bfmi: float = 0.5,
) -> MagicMock:
    """Build a mock CertificationResult."""
    cert = MagicMock()
    cert.passed = True
    cert.split_rhat_max = split_rhat_max
    cert.min_chunk_bulk_ess = min_chunk_bulk_ess
    cert.num_divergences = num_divergences
    cert.e_bfmi = e_bfmi
    return cert


def _make_mock_adaptation(
    step_size: float = 0.05,
    imm_size: int = 10,
) -> MagicMock:
    """Build a mock AdaptationParams with a small diagonal IMM."""
    adapt = MagicMock()
    adapt.step_size = step_size
    adapt.inverse_mass_matrix = np.ones(imm_size)
    adapt.num_leapfrog_median = 8
    return adapt


@pytest.mark.fast
def test_from_groundtruth_run_returns_valid_recipe() -> None:
    """from_groundtruth_run returns a Recipe with GROUNDTRUTH effort and correct fields."""
    posterior = MODELS["mvn_10"]
    cert = _make_mock_cert()
    adaptation = _make_mock_adaptation(imm_size=10)

    recipe = Recipe.from_groundtruth_run(
        posterior,
        cert=cert,
        adaptation=adaptation,
        wall_seconds=42.5,
        tuning_seed=42,
        n_warmup=500,
        n_samples=2000,
        n_chunks=4,
        target_acceptance=0.80,
    )

    assert recipe.effort == Effort.GROUNDTRUTH
    assert recipe.model_name == "mvn_10"
    assert recipe.base_method_name == "nuts"
    assert recipe.warmup_name == "window_adaptation_diag_imm"
    assert recipe.headline_metric is None

    # gate_evidence
    auto = recipe.gate_evidence["auto"]
    assert auto["verdict"] == "PASS"
    assert auto["rhat_max"] == pytest.approx(cert.split_rhat_max)
    assert auto["min_bulk_ess"] == pytest.approx(cert.min_chunk_bulk_ess)
    assert auto["n_divergences"] == cert.num_divergences
    assert auto["max_abs_mean_z"] is None

    # calibration_budget
    budget = recipe.calibration_budget
    assert budget["trials"] == 0
    assert budget["wall_seconds_estimate"] == pytest.approx(42.5)
    assert budget["n_warmup"] == 500
    assert budget["n_samples"] == 2000

    # warmup_params
    assert recipe.warmup_params["n_chunks"] == 4
    assert recipe.warmup_params["target_acceptance"] == pytest.approx(0.80)

    # instructions non-empty
    assert isinstance(recipe.instructions, str)
    assert len(recipe.instructions) > 10


@pytest.mark.fast
def test_from_groundtruth_run_save_load_roundtrip(tmp_path: Path) -> None:
    """from_groundtruth_run recipe round-trips through save/load."""
    posterior = MODELS["mvn_10"]
    cert = _make_mock_cert()
    adaptation = _make_mock_adaptation(imm_size=10)

    recipe = Recipe.from_groundtruth_run(
        posterior,
        cert=cert,
        adaptation=adaptation,
        wall_seconds=10.0,
        tuning_seed=0,
        n_warmup=100,
        n_samples=500,
        n_chunks=4,
        target_acceptance=0.80,
    )
    saved_path = recipe.save(tmp_path)

    # Filename convention (post-R2): groundtruth recipes live at
    # <model>/groundtruth.json (no filename suffix; one path per model).
    assert saved_path.name == "groundtruth.json"
    assert saved_path.parent.name == "mvn_10"
    assert saved_path.exists()

    loaded = Recipe.load(saved_path)
    assert loaded.effort == Effort.GROUNDTRUTH
    assert loaded.model_name == "mvn_10"
    assert loaded.gate_evidence["auto"]["verdict"] == "PASS"
    assert loaded.calibration_budget["n_samples"] == 500


@pytest.mark.fast
def test_from_groundtruth_run_large_imm_uses_sentinel() -> None:
    """from_groundtruth_run with IMM.size > 50 stores 'sidecar' sentinel."""
    posterior = MODELS["mvn_10"]
    cert = _make_mock_cert()
    adaptation = _make_mock_adaptation(imm_size=51)  # > 50 threshold

    recipe = Recipe.from_groundtruth_run(
        posterior,
        cert=cert,
        adaptation=adaptation,
        wall_seconds=5.0,
        tuning_seed=0,
        n_warmup=100,
        n_samples=200,
        n_chunks=2,
        target_acceptance=0.80,
    )
    assert recipe.base_method_params["inverse_mass_matrix"] == "sidecar"


@pytest.mark.fast
def test_from_groundtruth_run_small_imm_inlined() -> None:
    """from_groundtruth_run with IMM.size <= 50 inlines the IMM as a list."""
    posterior = MODELS["mvn_10"]
    cert = _make_mock_cert()
    adaptation = _make_mock_adaptation(imm_size=10)  # <= 50 threshold

    recipe = Recipe.from_groundtruth_run(
        posterior,
        cert=cert,
        adaptation=adaptation,
        wall_seconds=5.0,
        tuning_seed=0,
        n_warmup=100,
        n_samples=200,
        n_chunks=2,
        target_acceptance=0.80,
    )
    imm = recipe.base_method_params["inverse_mass_matrix"]
    assert isinstance(imm, list)
    assert len(imm) == 10


@pytest.mark.fast
def test_from_groundtruth_run_imm_sidecar_pattern(tmp_path: Path) -> None:
    """Orchestrator pattern: sentinel → save_imm_sidecar → dataclasses.replace works."""
    posterior = MODELS["mvn_10"]
    cert = _make_mock_cert()
    adaptation = _make_mock_adaptation(imm_size=51)

    recipe = Recipe.from_groundtruth_run(
        posterior,
        cert=cert,
        adaptation=adaptation,
        wall_seconds=5.0,
        tuning_seed=0,
        n_warmup=100,
        n_samples=200,
        n_chunks=2,
        target_acceptance=0.80,
    )
    assert recipe.base_method_params["inverse_mass_matrix"] == "sidecar"

    # Save sidecar and replace path
    imm_array = np.ones(51)
    sidecar_rel = recipe.save_imm_sidecar(tmp_path, imm_array)
    recipe_with_sidecar = dataclasses.replace(
        recipe, inverse_mass_matrix_path=sidecar_rel
    )

    assert recipe_with_sidecar.inverse_mass_matrix_path is not None
    assert recipe_with_sidecar.inverse_mass_matrix_path.endswith(".imm.npz")
    assert "mvn_10" in recipe_with_sidecar.inverse_mass_matrix_path


@pytest.mark.fast
def test_load_cached_samples_returns_none_for_non_groundtruth() -> None:
    """load_cached_samples returns None for LOW/MEDIUM/HIGH recipes."""
    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    recipe = Recipe.from_default_config(posterior, base_method)
    assert recipe.effort == Effort.LOW
    assert recipe.load_cached_samples() is None


@pytest.mark.fast
def test_load_cached_samples_returns_none_on_cache_miss(tmp_path: Path) -> None:
    """load_cached_samples returns None when no cache exists (empty tmp_path)."""
    posterior = MODELS["mvn_10"]
    cert = _make_mock_cert()
    adaptation = _make_mock_adaptation()
    recipe = Recipe.from_groundtruth_run(
        posterior,
        cert=cert,
        adaptation=adaptation,
        wall_seconds=5.0,
        tuning_seed=0,
        n_warmup=100,
        n_samples=200,
        n_chunks=2,
        target_acceptance=0.80,
    )
    assert recipe.effort == Effort.GROUNDTRUTH
    # tmp_path is empty → cache miss
    result = recipe.load_cached_samples(cache_dir=tmp_path)
    assert result is None


@pytest.mark.fast
def test_load_cached_samples_returns_draws_on_hit(tmp_path: Path) -> None:
    """load_cached_samples returns draws dict when cache is populated."""
    import datetime

    import tuningfork

    posterior = MODELS["mvn_10"]
    cert = _make_mock_cert()
    adaptation = _make_mock_adaptation()

    recipe = Recipe.from_groundtruth_run(
        posterior,
        cert=cert,
        adaptation=adaptation,
        wall_seconds=5.0,
        tuning_seed=0,
        n_warmup=100,
        n_samples=50,
        n_chunks=2,
        target_acceptance=0.80,
    )

    # Populate cache manually using _io internals
    from tuningfork._cache_io import (
        _atomic_write_json,
        _atomic_write_npz,
        _draws_path,
        _get_code_sha,
        _metadata_path,
    )

    draws_data = {"x": np.random.randn(50, 10).astype(np.float32)}
    _atomic_write_npz(_draws_path("mvn_10", tmp_path), draws_data)

    cert_dict = {"passed": True}
    metadata = {
        "name": "mvn_10",
        "tuningfork_version": tuningfork.__version__,
        "code_sha": _get_code_sha(tmp_path),
        "generator": "analytic",
        "num_samples": 50,
        "seed": 0,
        "timestamp_utc": datetime.datetime.now(datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "certification": cert_dict,
    }
    _atomic_write_json(_metadata_path("mvn_10", tmp_path), metadata)

    result = recipe.load_cached_samples(cache_dir=tmp_path)
    assert result is not None
    assert "x" in result
    assert result["x"].shape == (50, 10)


@pytest.mark.fast
def test_render_instructions_groundtruth_non_empty() -> None:
    """render_instructions for a GROUNDTRUTH recipe returns non-empty prose."""
    posterior = MODELS["mvn_10"]
    cert = _make_mock_cert()
    adaptation = _make_mock_adaptation()

    recipe = Recipe.from_groundtruth_run(
        posterior,
        cert=cert,
        adaptation=adaptation,
        wall_seconds=5.0,
        tuning_seed=0,
        n_warmup=500,
        n_samples=2000,
        n_chunks=4,
        target_acceptance=0.80,
    )
    prose = render_instructions(recipe)
    assert isinstance(prose, str)
    assert len(prose) > 20
    assert "mvn_10" in prose
    assert "ground-truth" in prose.lower() or "Ground-truth" in prose


@pytest.mark.fast
def test_try_load_cached_draws_returns_none_on_miss(tmp_path: Path) -> None:
    """try_load_cached_draws returns None when no cache exists."""
    from tuningfork._cache_io import try_load_cached_draws

    posterior = MODELS["mvn_10"]
    result = try_load_cached_draws(posterior, cache_dir=tmp_path)
    assert result is None


@pytest.mark.fast
def test_try_load_cached_draws_returns_draws_on_hit(tmp_path: Path) -> None:
    """try_load_cached_draws returns draws dict when cache is populated."""
    import datetime

    import tuningfork
    from tuningfork._cache_io import (
        _atomic_write_json,
        _atomic_write_npz,
        _draws_path,
        _get_code_sha,
        _metadata_path,
        try_load_cached_draws,
    )

    posterior = MODELS["mvn_10"]

    draws_data = {"x": np.random.randn(100, 10).astype(np.float32)}
    _atomic_write_npz(_draws_path("mvn_10", tmp_path), draws_data)

    metadata = {
        "name": "mvn_10",
        "tuningfork_version": tuningfork.__version__,
        "code_sha": _get_code_sha(tmp_path),
        "generator": "analytic",
        "num_samples": 100,
        "seed": 0,
        "timestamp_utc": datetime.datetime.now(datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "certification": {"passed": True},
    }
    _atomic_write_json(_metadata_path("mvn_10", tmp_path), metadata)

    # n=None: load all
    result = try_load_cached_draws(posterior, cache_dir=tmp_path)
    assert result is not None
    assert "x" in result
    assert result["x"].shape == (100, 10)

    # n=50: load first 50
    result_sliced = try_load_cached_draws(posterior, n=50, cache_dir=tmp_path)
    assert result_sliced is not None
    assert result_sliced["x"].shape == (50, 10)


@pytest.mark.fast
def test_try_load_cached_draws_returns_none_when_n_too_large(tmp_path: Path) -> None:
    """try_load_cached_draws returns None when requested n > cached num_samples."""
    import datetime

    import tuningfork
    from tuningfork._cache_io import (
        _atomic_write_json,
        _atomic_write_npz,
        _draws_path,
        _get_code_sha,
        _metadata_path,
        try_load_cached_draws,
    )

    posterior = MODELS["mvn_10"]

    draws_data = {"x": np.random.randn(50, 10).astype(np.float32)}
    _atomic_write_npz(_draws_path("mvn_10", tmp_path), draws_data)

    metadata = {
        "name": "mvn_10",
        "tuningfork_version": tuningfork.__version__,
        "code_sha": _get_code_sha(tmp_path),
        "generator": "analytic",
        "num_samples": 50,  # only 50 cached
        "seed": 0,
        "timestamp_utc": datetime.datetime.now(datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "certification": {"passed": True},
    }
    _atomic_write_json(_metadata_path("mvn_10", tmp_path), metadata)

    # n=100 > 50 → cache miss
    result = try_load_cached_draws(posterior, n=100, cache_dir=tmp_path)
    assert result is None


# ---------------------------------------------------------------------------
# Tests for FAILED recipes (R1 restructure phase)
# ---------------------------------------------------------------------------


class TestFailedRecipe:
    """Coverage for the FAILED effort tier + forking-path log."""

    @pytest.mark.fast
    def test_effort_enum_has_five_values(self) -> None:
        """Effort now has LOW, MEDIUM, HIGH, GROUNDTRUTH, FAILED."""
        assert set(Effort) == {
            Effort.LOW,
            Effort.MEDIUM,
            Effort.HIGH,
            Effort.GROUNDTRUTH,
            Effort.FAILED,
        }
        assert Effort.FAILED == "failed"

    @pytest.mark.fast
    def test_failure_diagnosis_enum_has_five_values(self) -> None:
        """FailureDiagnosis covers the 5 canonical buckets."""
        assert set(FailureDiagnosis) == {
            FailureDiagnosis.OUT_OF_SCOPE,
            FailureDiagnosis.REQUIRES_ALT_SAMPLER,
            FailureDiagnosis.REQUIRES_MODEL_CHANGE,
            FailureDiagnosis.TRIVIAL_FIX_DEFERRED,
            FailureDiagnosis.HARD_DIRECTION,
        }

    @pytest.mark.fast
    def test_attempted_config_round_trips_via_asdict(self) -> None:
        """AttemptedConfig serializes and deserializes cleanly."""
        cfg = AttemptedConfig(
            base_method_params={"step_size": 0.01},
            warmup_params={"n_warmup": 1000},
            seed=42,
            gate_verdict={
                "verdict": "FAIL",
                "rhat_max": 1.05,
                "min_bulk_ess": 287.0,
                "n_divergences": 0,
            },
            wall_seconds=95.0,
            note="closest attempt",
        )
        roundtrip = AttemptedConfig(**dataclasses.asdict(cfg))
        assert roundtrip == cfg

    @pytest.mark.fast
    def test_recipe_failed_round_trip(self, tmp_path: Path) -> None:
        """A FAILED Recipe save → load round-trip preserves new fields."""
        recipe = Recipe(
            model_name="stoch_vol",
            base_method_name="mclmc",
            warmup_name="mclmc_tuning",
            effort=Effort.FAILED,
            failure_diagnosis=FailureDiagnosis.HARD_DIRECTION,
            attempted_configurations=[
                AttemptedConfig(
                    base_method_params={"step_size": 0.01, "L": 5.0},
                    warmup_params={"n_warmup": 1000},
                    seed=42,
                    gate_verdict={
                        "verdict": "FAIL",
                        "rhat_max": 1.18,
                        "min_bulk_ess": 42.1,
                        "n_divergences": 0,
                    },
                    wall_seconds=89.0,
                    note="default MCLMC tuning — ESS too low",
                ),
                AttemptedConfig(
                    base_method_params={"step_size": 0.001, "L": 5.0},
                    warmup_params={"n_warmup": 1000},
                    seed=42,
                    gate_verdict={
                        "verdict": "FAIL",
                        "rhat_max": 1.04,
                        "min_bulk_ess": 287.0,
                        "n_divergences": 0,
                    },
                    wall_seconds=95.0,
                    note="10× smaller step_size — closest attempt",
                ),
            ],
            workflow="MCLMC on 503-D stoch_vol — 2 forking paths attempted, neither cleared the gate.",
            base_method_params={"step_size": 0.001, "L": 5.0},
            warmup_params={"n_warmup": 1000},
            headline_metric=None,
            sample_quality=None,
            calibration_budget={"trials": 0, "wall_seconds_estimate": 0.0},
            difficulty=None,
            instructions="test instructions",
            tuning_seed=42,
            tuningfork_version="0.0.0.dev0",
            blackjax_version="1.0.0",
            jax_version="0.4.0",
            timestamp_utc="2026-01-01T00:00:00Z",
        )

        # save() creates a directory structure, returns the full path
        saved_path = recipe.save(tmp_path)

        loaded = Recipe.load(saved_path)
        assert loaded.effort == Effort.FAILED
        assert loaded.failure_diagnosis == FailureDiagnosis.HARD_DIRECTION
        assert len(loaded.attempted_configurations) == 2
        assert loaded.attempted_configurations[0].seed == 42
        assert (
            loaded.attempted_configurations[1].note
            == "10× smaller step_size — closest attempt"
        )

    @pytest.mark.fast
    def test_is_failed_method(self) -> None:
        """is_failed() returns True only for FAILED recipes."""
        recipe_failed = Recipe(
            model_name="mvn_10",
            base_method_name="mclmc",
            warmup_name="mclmc_tuning",
            effort=Effort.FAILED,
            failure_diagnosis=FailureDiagnosis.HARD_DIRECTION,
            base_method_params={"step_size": 0.01},
            warmup_params={"n_warmup": 1000},
            headline_metric=None,
            sample_quality=None,
            calibration_budget={"trials": 0, "wall_seconds_estimate": 0.0},
            difficulty=None,
            instructions="test instructions",
        )
        assert recipe_failed.is_failed()

        recipe_low = Recipe(
            model_name="mvn_10",
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
        )
        assert not recipe_low.is_failed()

    @pytest.mark.fast
    def test_recipe_failed_error_carries_recipe(self) -> None:
        """RecipeFailedError records the recipe + diagnosis in its message."""
        recipe = Recipe(
            model_name="mvn_10",
            base_method_name="mclmc",
            warmup_name="mclmc_tuning",
            effort=Effort.FAILED,
            failure_diagnosis=FailureDiagnosis.OUT_OF_SCOPE,
            base_method_params={"step_size": 0.01},
            warmup_params={"n_warmup": 1000},
            headline_metric=None,
            sample_quality=None,
            calibration_budget={"trials": 0, "wall_seconds_estimate": 0.0},
            difficulty=None,
            instructions="test instructions",
        )
        with pytest.raises(RecipeFailedError) as exc_info:
            raise RecipeFailedError(recipe)
        assert "FAILED" in str(exc_info.value)
        assert "out_of_scope" in str(exc_info.value)
        assert exc_info.value.recipe is recipe

    @pytest.mark.fast
    def test_prior_recipe_loads_with_default_new_fields(self, tmp_path: Path) -> None:
        """A pre-R1 recipe JSON (no failure_diagnosis, no attempted_configurations)
        loads cleanly with the new fields defaulting to None / []."""
        # Build a JSON dict matching the pre-R1 schema — i.e., a LOW recipe
        # omitting the two new fields. Save it as a temp file, then load.
        pre_r1_json = {
            "model_name": "mvn_10",
            "base_method_name": "nuts",
            "warmup_name": "no_warmup",
            "effort": "low",
            "base_method_params": {"step_size": 0.1},
            "warmup_params": {},
            "headline_metric": None,
            "sample_quality": None,
            "calibration_budget": {"trials": 0, "wall_seconds_estimate": 0.0},
            "difficulty": None,
            "instructions": "test instructions",
            "notes": "",
            "tuning_seed": 0,
            "tuningfork_version": "0.0.0.dev0",
            "blackjax_version": "1.0.0",
            "jax_version": "0.4.0",
            "timestamp_utc": "2026-01-01T00:00:00Z",
            # Missing: failure_diagnosis, attempted_configurations
        }

        recipe_path = tmp_path / "mvn_10" / "low__nuts__no_warmup.json"
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        recipe_path.write_text(json.dumps(pre_r1_json))

        # Load should succeed and fill in defaults
        loaded = Recipe.load(recipe_path)
        assert loaded.failure_diagnosis is None
        assert loaded.attempted_configurations == []

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "recipe_path,expected_diagnosis",
        [
            (
                "gmm_25/recipes/failed__nuts__window_adaptation_diag_imm.json",
                FailureDiagnosis.OUT_OF_SCOPE,
            ),
            (
                "mvn_10/recipes/failed__elliptical_slice__no_warmup.json",
                FailureDiagnosis.REQUIRES_MODEL_CHANGE,
            ),
            (
                "neals_funnel/recipes/failed__meanfield_vi__no_warmup.json",
                FailureDiagnosis.OUT_OF_SCOPE,
            ),
            (
                "stoch_vol/recipes/failed__hmc__no_warmup.json",
                FailureDiagnosis.HARD_DIRECTION,
            ),
            (
                "horseshoe/recipes/failed__rmhmc__window_adaptation_diag_imm.json",
                FailureDiagnosis.REQUIRES_MODEL_CHANGE,
            ),
            (
                "mvn_10/recipes/failed__laplace_hmc__no_warmup.json",
                FailureDiagnosis.REQUIRES_MODEL_CHANGE,
            ),
            (
                "radon/recipes/failed__nuts__fullrank_vi.json",
                FailureDiagnosis.REQUIRES_MODEL_CHANGE,
            ),
        ],
    )
    def test_committed_failed_recipe_loads_with_diagnosis(
        self,
        recipe_path: str,
        expected_diagnosis: FailureDiagnosis,
    ) -> None:
        """Each committed FAILED recipe loads cleanly and carries the expected diagnosis."""
        catalog_root = Path(__file__).resolve().parents[2] / "tuningfork" / "catalog"
        path = catalog_root / recipe_path
        assert path.exists(), f"Missing committed FAILED recipe: {path}"
        recipe = Recipe.load(path)
        assert recipe.is_failed()
        assert recipe.effort == Effort.FAILED
        assert recipe.failure_diagnosis == expected_diagnosis
        assert len(recipe.workflow) > 50, "workflow must be non-trivial prose"
        assert recipe.gate_evidence["auto"]["verdict"] == "FAIL"
