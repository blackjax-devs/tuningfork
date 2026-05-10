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

import json
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

from bjx_bench.calibration.tier_b import default_params_for
from bjx_bench.inference.base_method import BASE_METHODS
from bjx_bench.inference.recipes import Effort, Recipe
from bjx_bench.inference.recipes._instructions import render_instructions
from bjx_bench.model import MODELS

# Path to the committed starter recipes
_STARTER_ROOT = (
    Path(__file__).resolve().parents[2]
    / "bjx_bench"
    / "inference"
    / "recipes"
    / "starter"
)


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
        bjx_bench_version="0.0.0.dev0",
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

    # NUTS default: step_size = 1e-3 * (1/1e-3)**0.7 (70th-pctile on log-scale, P4.0 tweak)
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
    # step_size = 1e-3 * (1/1e-3)**0.7 (70th-pctile on log-scale, P4.0 tweak)
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
    assert loaded.bjx_bench_version == recipe.bjx_bench_version
    assert loaded.blackjax_version == recipe.blackjax_version
    assert loaded.jax_version == recipe.jax_version
    assert loaded.timestamp_utc == recipe.timestamp_utc

    # Effort enum preserved correctly
    assert isinstance(loaded.effort, Effort)
    assert loaded.effort == Effort.LOW

    # Filename convention: <effort>__<method>__<warmup>.json
    assert saved_path.name == "low__nuts__no_warmup.json"
    assert saved_path.parent.name == "mvn_10"


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
# Test 7 (was 8): 6 starter JSONs exist and load cleanly
# ---------------------------------------------------------------------------

_EXPECTED_COMBOS = [
    ("mvn_10", "hmc"),
    ("mvn_10", "nuts"),
    ("neals_funnel", "hmc"),
    ("neals_funnel", "nuts"),
    ("eight_schools_ncp", "hmc"),
    ("eight_schools_ncp", "nuts"),
]


@pytest.mark.fast
@pytest.mark.parametrize("model_name,method_name", _EXPECTED_COMBOS)
def test_starter_recipe_exists_and_loads(model_name: str, method_name: str) -> None:
    """Each of the 6 starter LOW recipes exists on disk and loads cleanly."""
    path = _STARTER_ROOT / model_name / f"low__{method_name}__no_warmup.json"
    assert path.exists(), f"Missing starter recipe: {path}"

    recipe = Recipe.load(path)

    assert recipe.effort == Effort.LOW
    assert recipe.effort == "low"  # str-Enum equality with value
    assert recipe.warmup_name == "no_warmup"
    assert recipe.model_name == model_name
    assert recipe.base_method_name == method_name
    assert method_name in BASE_METHODS
    assert recipe.headline_metric is None
    assert recipe.warmup_params == {}
    assert recipe.calibration_budget == {"trials": 0, "wall_seconds_estimate": 0.0}
    assert isinstance(recipe.instructions, str)
    assert len(recipe.instructions) > 0


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
    # Under canonical-C taxonomy, LOW = conventional pairing with library
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
        warmup_name="stan_window",
        effort=Effort.MEDIUM,
        base_method_params={"step_size": 0.1, "inverse_mass_matrix": [1.0] * 10},
        warmup_params={"n_warmup": 1000, "target_acceptance_rate": 0.8},
        headline_metric=0.0512,
        sample_quality=None,
        calibration_budget={"trials": 0, "wall_seconds_estimate": 45.0},
        difficulty=None,
        instructions="",
        tuning_seed=0,
        bjx_bench_version="0.0.0.dev0",
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
        warmup_name="stan_window",
        effort=Effort.HIGH,
        base_method_params={"step_size": 0.08, "num_integration_steps": 32},
        warmup_params={"n_warmup": 1000},
        headline_metric=0.0731,
        sample_quality=None,
        calibration_budget={"trials": 50, "wall_seconds_estimate": 1800.0},
        difficulty=None,
        instructions="",
        tuning_seed=42,
        bjx_bench_version="0.0.0.dev0",
        blackjax_version="1.0.0",
        jax_version="0.4.0",
        timestamp_utc="2026-01-01T00:00:00Z",
    )
    prose = render_instructions(recipe)
    assert isinstance(prose, str)
    assert len(prose) > 10
    assert "high" in prose.lower() or "High" in prose


# ---------------------------------------------------------------------------
# Tests 9-10: P3.3 wider LOW coverage + MEDIUM smoke
# ---------------------------------------------------------------------------

_LOW_OTHER_ALGOS = [
    (model, method)
    for model in ("mvn_10", "neals_funnel", "eight_schools_ncp")
    for method in ("mala", "barker", "rwm", "mclmc")
]


@pytest.mark.fast
@pytest.mark.parametrize("model_name,method_name", _LOW_OTHER_ALGOS)
def test_low_recipe_exists_for_other_algos(model_name: str, method_name: str) -> None:
    """P3.3: every (starter_model, base_method) pair has a LOW recipe on disk."""
    path = _STARTER_ROOT / model_name / f"low__{method_name}__no_warmup.json"
    assert path.exists(), f"Missing LOW recipe for {model_name} + {method_name}"
    recipe = Recipe.load(path)
    assert recipe.effort == Effort.LOW
    assert recipe.warmup_name == "no_warmup"
    assert recipe.model_name == model_name
    assert recipe.base_method_name == method_name


# ---------------------------------------------------------------------------
# Tests P3.4: 6 HIGH recipes exist and have correct schema
# ---------------------------------------------------------------------------

_HIGH_COMBOS = [
    (model, method)
    for model in ("mvn_10", "neals_funnel", "eight_schools_ncp")
    for method in ("hmc", "nuts")
]

_EXPECTED_DIFFICULTY_KEYS = (
    "default_score",
    "best_score",
    "threshold_score",
    "default_works",
    "n_trials_to_threshold",
    "n_trials_to_best",
    "wall_seconds_to_threshold",
    "wall_seconds_to_best",
)


@pytest.mark.fast
@pytest.mark.parametrize("model_name,method_name", _HIGH_COMBOS)
def test_high_recipe_exists_and_has_bo_data(model_name: str, method_name: str) -> None:
    """P3.4: each (starter_model, {hmc,nuts}) has a HIGH recipe via Tier-B BO
    at n_trials=20 with a valid headline_metric and TuningDifficulty profile."""
    path = _STARTER_ROOT / model_name / f"high__{method_name}__stan_window.json"
    assert (
        path.exists()
    ), f"Missing HIGH recipe for {model_name} + {method_name}: {path}"

    recipe = Recipe.load(path)

    # Identity and effort checks
    assert recipe.effort == Effort.HIGH
    assert recipe.warmup_name == "stan_window"
    assert recipe.model_name == model_name
    assert recipe.base_method_name == method_name

    # headline_metric must be a real, positive float (all starter models
    # are well-conditioned; divergence here would be a real failure).
    assert recipe.headline_metric is not None
    assert isinstance(recipe.headline_metric, float)
    assert math.isfinite(recipe.headline_metric)
    assert recipe.headline_metric > 0, (
        f"headline_metric={recipe.headline_metric} for {model_name}+{method_name}; "
        "expected > 0 for these well-conditioned starter models."
    )

    # calibration_budget shape
    assert recipe.calibration_budget["trials"] == 20
    assert "n_seeds" in recipe.calibration_budget

    # difficulty is a dict with all expected keys
    assert recipe.difficulty is not None
    assert isinstance(recipe.difficulty, dict)
    for key in _EXPECTED_DIFFICULTY_KEYS:
        assert key in recipe.difficulty, (
            f"Missing difficulty key {key!r} in HIGH recipe "
            f"{model_name}+{method_name}"
        )

    # Spot-check difficulty numeric types
    assert isinstance(recipe.difficulty["default_score"], float)
    assert isinstance(recipe.difficulty["best_score"], float)
    assert isinstance(recipe.difficulty["default_works"], bool)
    assert isinstance(recipe.difficulty["n_trials_to_threshold"], int)

    # Instructions non-empty prose
    assert isinstance(recipe.instructions, str)
    assert len(recipe.instructions) > 10


# ---------------------------------------------------------------------------
# Test 17 (P5.0): _generate_starter CLI flag filtering
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_emit_low_recipes_sampler_filter(tmp_path: Path, monkeypatch) -> None:
    """emit_low_recipes(sampler='nuts') emits NUTS recipes only.

    P5.0 (Q5.A) added per-cell flag filtering to ``_generate_starter.py``;
    this test locks the ``sampler`` filter behavior at the function-level
    so future refactors can't silently regress it.

    Uses monkeypatch to redirect _STARTER_ROOT into tmp_path so we don't
    clobber committed recipes. Only tests LOW because it's deterministic
    and zero-cost (no MCMC).
    """
    from bjx_bench.inference.recipes import _generate_starter

    monkeypatch.setattr(_generate_starter, "_STARTER_ROOT", tmp_path)
    paths = _generate_starter.emit_low_recipes(model_names=["mvn_10"], sampler="nuts")
    assert len(paths) == 1, f"Expected 1 recipe, got {len(paths)}: {paths}"
    assert paths[0].name == "low__nuts__no_warmup.json"


@pytest.mark.fast
def test_emit_low_recipes_no_filter_emits_all_methods(
    tmp_path: Path, monkeypatch
) -> None:
    """emit_low_recipes() with no sampler filter emits all 6 algos for the model."""
    from bjx_bench.inference.recipes import _generate_starter

    monkeypatch.setattr(_generate_starter, "_STARTER_ROOT", tmp_path)
    paths = _generate_starter.emit_low_recipes(model_names=["mvn_10"])
    # 6 algorithms: hmc, nuts, mala, barker, rwm, mclmc
    assert len(paths) == 6
    method_names = sorted(p.name.split("__")[1] for p in paths)
    assert method_names == sorted(["hmc", "nuts", "mala", "barker", "rwm", "mclmc"])


@pytest.mark.fast
def test_main_rejects_unknown_model(monkeypatch) -> None:
    """`--only <unknown>` raises SystemExit."""
    import sys

    from bjx_bench.inference.recipes import _generate_starter

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

    from bjx_bench.inference.recipes import _generate_starter

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
# Tests P5.0a: Phase 5 schema fields + IMM sidecar helpers
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
    """P5.0a: new Phase 5 fields default to the expected values."""
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
    """P5.0a: non-default Phase 5 field values round-trip through save/load."""
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
    """P5.0a: save_imm_sidecar writes .npz; load_imm_sidecar recovers the array."""
    recipe = Recipe(**_RECIPE_KWARGS_MINIMAL)
    original_imm = jnp.eye(10)

    rel_path = recipe.save_imm_sidecar(tmp_path, original_imm)

    # The file must exist at the expected location
    expected_file = tmp_path / "mvn_10" / "low__nuts__no_warmup.imm.npz"
    assert expected_file.exists(), f"Expected sidecar at {expected_file}"

    # rel_path is relative to tmp_path
    assert rel_path == str(Path("mvn_10") / "low__nuts__no_warmup.imm.npz")

    # Load via load_imm_sidecar (with inverse_mass_matrix_path set)
    recipe_with_path = Recipe(
        **_RECIPE_KWARGS_MINIMAL, inverse_mass_matrix_path=rel_path
    )
    loaded_imm = recipe_with_path.load_imm_sidecar(tmp_path)

    assert loaded_imm is not None
    assert jnp.allclose(loaded_imm, original_imm)


@pytest.mark.fast
def test_load_imm_sidecar_returns_none_when_path_unset(tmp_path: Path) -> None:
    """P5.0a: load_imm_sidecar returns None when inverse_mass_matrix_path is None."""
    recipe = Recipe(**_RECIPE_KWARGS_MINIMAL)
    assert recipe.inverse_mass_matrix_path is None

    result = recipe.load_imm_sidecar(tmp_path)
    assert result is None


_BACKWARD_COMPAT_RECIPES = [
    "mvn_10/low__nuts__no_warmup.json",
    "mvn_10/medium__nuts__stan_window.json",
]


@pytest.mark.fast
@pytest.mark.parametrize("rel_path", _BACKWARD_COMPAT_RECIPES)
def test_existing_starter_recipes_still_load(rel_path: str) -> None:
    """P5.0a: existing committed recipes load without error and have Phase 5 defaults."""
    path = _STARTER_ROOT / rel_path
    assert path.exists(), f"Missing recipe: {path}"

    recipe = Recipe.load(path)

    # New fields must default correctly on old recipes that lack these keys
    assert recipe.inverse_mass_matrix_path is None
    assert recipe.workflow == ""
    assert isinstance(recipe.gate_evidence, dict)
    assert recipe.gate_evidence["auto"]["verdict"] == "NOT_RUN"
    assert recipe.gate_evidence["override"]["decision"] == ""
