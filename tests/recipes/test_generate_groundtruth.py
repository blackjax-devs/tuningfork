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
"""Tests for the ground-truth orchestrator (_generate_groundtruth.py).

Fast tests: analytic-path smoke (no JAX trace).
Slow tests: NUTS-path with downscaled settings (actual JAX compilation + sampling).
"""

from pathlib import Path

import pytest

from tuningfork.inference.recipes._base import Effort
from tuningfork.inference.recipes._generate_groundtruth import (
    generate_groundtruth_recipe,
    sweep_all,
)
from tuningfork.model import MODELS


@pytest.mark.fast
def test_generate_groundtruth_analytic_returns_none_and_populates_cache(
    tmp_path: Path,
) -> None:
    """generate_groundtruth_recipe for an analytic model (mvn_10) returns None
    and populates draws/summaries/metadata cache files."""
    entry = MODELS["mvn_10"]

    result = generate_groundtruth_recipe(
        entry,
        seed=42,
        cache_dir=tmp_path,
        # n_samples doesn't drive a NUTS run for analytic models, but we use
        # a small value to keep the test fast (fewer i.i.d. draws to write)
        n_samples=500,
    )

    # Analytic path: no recipe emitted
    assert result is None

    # Cache files must be populated
    assert (tmp_path / "draws" / "mvn_10.npz").exists(), "draws npz missing"
    assert (tmp_path / "summaries" / "mvn_10.json").exists(), "summaries json missing"
    assert (tmp_path / "metadata" / "mvn_10.json").exists(), "metadata json missing"

    # No adaptation file for analytic models
    assert not (tmp_path / "adaptation" / "mvn_10.json").exists()


@pytest.mark.slow
def test_generate_groundtruth_nuts_returns_recipe_and_saves_json(
    tmp_path: Path,
) -> None:
    """generate_groundtruth_recipe for eight_schools_ncp (NUTS path) returns a
    GROUNDTRUTH Recipe with PASS verdict and saves the recipe JSON.

    Uses downscaled settings (n_samples=4000, n_warmup=1000, n_chunks=4) to
    keep wall time under ~60s on CPU. The WORKLOG watch TestCertifyNutsPassesGate
    documents that seed=42 passes at n=4000 with min_chunk_bulk_ess≈554.
    """
    entry = MODELS["eight_schools_ncp"]
    tmp_recipe = tmp_path / "recipes"
    tmp_cache = tmp_path / "cache"

    # n_warmup=500, n_samples=4000 at seed=42 passes the gate with min_ess≈554
    # (per WORKLOG TestCertifyNutsPassesGate watch + tests/reference/test_nuts.py).
    recipe = generate_groundtruth_recipe(
        entry,
        seed=42,
        n_samples=4000,
        n_warmup=500,
        n_chunks=4,
        target_acceptance=0.80,
        cache_dir=tmp_cache,
        recipe_root=tmp_recipe,
    )

    # Must return a Recipe (not None) for NUTS-path model
    assert recipe is not None
    assert recipe.effort == Effort.GROUNDTRUTH
    assert recipe.model_name == "eight_schools_ncp"
    assert recipe.base_method_name == "nuts"
    assert recipe.warmup_name == "stan_window"

    # Gate evidence must be PASS
    auto = recipe.gate_evidence["auto"]
    assert auto["verdict"] == "PASS", (
        f"Certification failed: rhat={auto['rhat_max']:.4f}, "
        f"min_ess={auto['min_bulk_ess']:.1f}, n_div={auto['n_divergences']}"
    )

    # Recipe JSON must be saved at the expected path
    recipe_path = (
        tmp_recipe / "eight_schools_ncp" / "groundtruth__nuts__stan_window.json"
    )
    assert recipe_path.exists(), f"Recipe JSON not found at {recipe_path}"

    # Recipe round-trips through load
    from tuningfork.inference.recipes._base import Recipe

    loaded = Recipe.load(recipe_path)
    assert loaded.effort == Effort.GROUNDTRUTH
    assert loaded.gate_evidence["auto"]["verdict"] == "PASS"


@pytest.mark.slow
def test_sweep_all_analytic_models_pass(tmp_path: Path) -> None:
    """sweep_all on two analytic models returns 2-entry summary with both passed=True."""
    # mvn_10 and neals_funnel are both analytic (no NUTS chain)
    results = sweep_all(
        models=["mvn_10", "neals_funnel"],
        seed=42,
        n_samples=500,
        cache_dir=tmp_path,
    )

    assert set(results.keys()) == {"mvn_10", "neals_funnel"}

    for name, summary in results.items():
        assert summary["passed"] is True, f"{name} failed: {summary}"
        assert summary["generator"] == "analytic"
        assert summary["wall_seconds"] >= 0.0
        assert summary["recipe_path"] is None  # no recipe for analytic
        assert summary["cert_diagnostics"] is None  # no cert for analytic
