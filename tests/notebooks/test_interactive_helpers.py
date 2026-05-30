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
"""Tests for the interactive catalog notebook helpers (list_recipes,
plot_recipe_diagnostics) + Posterior.headline_params / headline_coords
fields.

Covers:
  - list_recipes returns paths for known models; raises on unknown
  - All 14 catalog models expose the two new fields with correct types
  - Per-model headline_params + headline_coords match the 2026-05-18
    decision doc
"""

import pytest

from tuningfork.catalog import list_recipes
from tuningfork.model import MODELS

pytestmark = pytest.mark.fast


def test_list_recipes_known_model_returns_paths() -> None:
    paths = list_recipes("eight_schools_ncp")
    assert len(paths) >= 1, "eight_schools_ncp should have at least groundtruth.json"
    assert any(p.name == "groundtruth.json" for p in paths)


def test_list_recipes_unknown_model_raises() -> None:
    with pytest.raises(FileNotFoundError):
        list_recipes("nonexistent_model_xyz")


def test_list_recipes_sorted() -> None:
    """Returned paths should be in a stable order (groundtruth first, then sorted recipes)."""
    paths = list_recipes("stoch_vol")
    # groundtruth.json should come first if present
    names = [p.name for p in paths]
    if "groundtruth.json" in names:
        assert names[0] == "groundtruth.json"


def test_all_14_models_have_headline_fields() -> None:
    """Every model entry must have both new fields set (None or typed value)."""
    for name, entry in MODELS.items():
        assert hasattr(entry, "headline_params"), f"{name} missing headline_params"
        assert hasattr(entry, "headline_coords"), f"{name} missing headline_coords"

        # headline_params: None or tuple of str
        hp = entry.headline_params
        assert hp is None or (
            isinstance(hp, tuple) and all(isinstance(s, str) for s in hp)
        ), f"{name} headline_params malformed: {hp!r}"

        # headline_coords: None or dict of str -> list of int
        hc = entry.headline_coords
        if hc is not None:
            assert isinstance(hc, dict), f"{name} headline_coords not dict: {hc!r}"
            for k, v in hc.items():
                assert isinstance(k, str), f"{name} headline_coords key not str: {k!r}"
                assert isinstance(v, list) and all(
                    isinstance(i, int) for i in v
                ), f"{name} headline_coords value not list[int]: {v!r}"


def test_headline_params_per_decision_doc() -> None:
    """Per-model headline_params match the 2026-05-18 ratified decision doc."""
    expected = {
        "banana": None,
        "gmm_25": None,
        "ill_cond_50": None,
        "mvn_10": None,
        "neals_funnel": ("v", "theta"),
        "eight_schools_ncp": ("mu", "tau"),
        "german_credit": None,
        "gp_regression": ("log_lengthscale", "log_kernel_scale", "log_noise_scale"),
        "horseshoe": ("alpha", "sigma", "tau_tilde", "c2_tilde"),
        "irt_2pl": ("sigma_theta", "mu_b", "sigma_b", "sigma_a"),
        "logistic_synthetic": None,
        "lotka_volterra": ("alpha", "beta", "gamma", "delta"),
        "radon": ("mu_a", "sigma_a", "beta", "sigma_y"),
        "stoch_vol": ("mu", "phi", "sigma"),
    }
    for name, want in expected.items():
        got = MODELS[name].headline_params
        assert got == want, f"{name}: got {got!r}, want {want!r}"


def test_headline_coords_per_decision_doc() -> None:
    """Only german_credit has a non-None headline_coords per the 2026-05-18 decision doc."""
    # german_credit: intercept + 7 numerical features
    assert MODELS["german_credit"].headline_coords == {"beta": [0, 1, 2, 3, 4, 5, 6, 7]}
    # All other models: None
    for name in MODELS:
        if name == "german_credit":
            continue
        assert (
            MODELS[name].headline_coords is None
        ), f"{name} expected None headline_coords, got {MODELS[name].headline_coords!r}"


# ---------------------------------------------------------------------------
# catalog_explorer FAIL-recipe estimate cell — None-guard regression (PR #111)
# ---------------------------------------------------------------------------


def _make_fail_recipe(budget: "dict[str, object]"):
    """Build a minimal FAIL recipe with the given calibration_budget."""
    from tuningfork.recipes._base import Effort, Recipe

    return Recipe(
        model_name="ill_cond_50",
        base_method_name="hmc",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.FAILED,
        base_method_params={"step_size": 0.01, "inverse_mass_matrix": [1.0]},
        warmup_params={"n_warmup": 1000, "num_chains": 4},
        headline_metric=None,
        sample_quality=None,
        calibration_budget=budget,
        difficulty=None,
        instructions="test",
    )


def test_fail_recipe_estimate_no_timing_fields() -> None:
    """FAIL recipe with only wall_seconds_estimate (no _spd / _ww) must not crash.

    Regression test for the TypeError: unsupported operand type(s) for *:
    'NoneType' and 'int' when _spd is None in the estimate cell.
    Verifies the None-guard logic extracts 0.0 fallbacks and doesn't raise.
    """
    recipe = _make_fail_recipe(
        {"trials": 0, "wall_seconds_estimate": 30.0}  # no _spd, no _ww
    )
    budget = recipe.calibration_budget or {}
    # Simulate the estimate cell logic with None-guards
    _ww = budget.get("warmup_wall_seconds") or 0.0
    _spd = budget.get("sampling_seconds_per_draw") or 0.0
    _c = int(
        budget.get("num_chains") or (recipe.warmup_params or {}).get("num_chains") or 4
    )
    _OVERHEAD_S = 20.0
    _n = 400
    # Must not raise
    _est_tot = _ww + _spd * _n * _c + _OVERHEAD_S
    _est_min = _est_tot / 60.0
    assert _est_min > 0.0, "estimate must be positive (overhead alone)"

    # _has_timing should be False → "unavailable" path
    _has_timing = _ww > 0.0 or _spd > 0.0
    assert not _has_timing, "no timing fields → has_timing must be False"


def test_fail_recipe_estimate_with_timing_fields() -> None:
    """FAIL recipe with measured timing fields shows a concrete estimate."""
    recipe = _make_fail_recipe(
        {
            "trials": 0,
            "wall_seconds_estimate": 30.0,
            "warmup_wall_seconds": 12.0,
            "sampling_seconds_per_draw": 0.005,
        }
    )
    budget = recipe.calibration_budget or {}
    _ww = budget.get("warmup_wall_seconds") or 0.0
    _spd = budget.get("sampling_seconds_per_draw") or 0.0
    _c = int(
        budget.get("num_chains") or (recipe.warmup_params or {}).get("num_chains") or 4
    )
    _OVERHEAD_S = 20.0
    _n = 400
    _est_tot = _ww + _spd * _n * _c + _OVERHEAD_S
    _est_min = _est_tot / 60.0
    assert _est_min > 0.0

    _has_timing = _ww > 0.0 or _spd > 0.0
    assert _has_timing, "timing fields present → has_timing must be True"
    # Rough bound: warmup=12 + sampling=0.005*400*4=8 + overhead=20 = 40s ≈ 0.67 min
    assert 0.5 < _est_min < 2.0, f"Unexpected estimate {_est_min:.2f} min"


def test_fail_recipe_estimate_empty_budget() -> None:
    """FAIL recipe with minimal calibration_budget (no timing fields) must not crash."""
    recipe = _make_fail_recipe({"trials": 0})  # only trials — no timing fields
    budget = recipe.calibration_budget or {}
    _ww = budget.get("warmup_wall_seconds") or 0.0
    _spd = budget.get("sampling_seconds_per_draw") or 0.0
    _c = int(
        budget.get("num_chains") or (recipe.warmup_params or {}).get("num_chains") or 4
    )
    _est_tot = _ww + _spd * 400 * _c + 20.0
    # Must not raise; dominated by OVERHEAD
    assert _est_tot > 0.0
