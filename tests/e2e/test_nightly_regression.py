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
"""Regression guards for nightly XFAIL_CELLS: seeds that should pass.

Each test here confirms that a GOOD seed for an XFAIL_CELLS entry continues
to clear z < 4.0 in e2e mode.  If one of these starts failing, the XFAIL
cell has regressed beyond the known seed-sensitivity scope and needs triage.

Marker: @pytest.mark.slow (full warmup + sampling, ~200s for lotka).
Run via:
    make test-slow
    JAX_PLATFORM_NAME=cpu uv run pytest tests/e2e/test_nightly_regression.py -v
"""
from __future__ import annotations

import pytest

_RECIPE_ROOT = None  # resolved lazily below


def _catalog_root():
    """Return the tuningfork catalog root (lazy import to avoid heavy init at collection)."""
    from pathlib import Path

    return Path(__file__).parent.parent.parent / "tuningfork" / "catalog"


@pytest.mark.slow
def test_lotka_dense_imm_inner_nuts_seed_20260713_passes() -> None:
    """Seed 20260713 must clear z < 4.0 for the step-collapse XFAIL_CELLS entry.

    The e2e cell tier2/lotka_volterra/low__hmc__window_adaptation_dense_imm__inner_nuts
    is in XFAIL_CELLS because window_adaptation_dense_imm warmup step-collapses for some
    nightly seeds on lotka's stiff ODE (20260712 partial, 20260714 confirmed z=27.7).

    Seed 20260713 was verified clean (z < 4.0) during the Phase 1 diagnosis run.
    This test pins that: while some seeds fail, valid seeds exist and must stay passing.
    If this test fails, the recipe has regressed beyond seed-sensitivity.

    Expected runtime: ~200s (full warmup + 1000 sampling steps).
    Tracked: github.com/blackjax-devs/tuningfork/issues/232
    """
    import sys
    from pathlib import Path

    # benchmarks/ is not on sys.path by default for tests/; add it once
    _bench_dir = str(Path(__file__).parent.parent.parent / "benchmarks")
    if _bench_dir not in sys.path:
        sys.path.insert(0, _bench_dir)

    from benchmarks._benchmark_helpers import compute_max_abs_mean_z
    from tuningfork.catalog.inspect import load_recipe
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    catalog = _catalog_root()
    recipe_path = (
        catalog
        / "lotka_volterra"
        / "recipes"
        / "low__hmc__window_adaptation_dense_imm__inner_nuts.json"
    )
    if not recipe_path.exists():
        pytest.skip(f"Recipe not found: {recipe_path}")

    recipe = load_recipe(recipe_path)
    idata = run_recipe_to_idata(
        recipe,
        skip_warmup=False,
        n_samples=1000,
        force_resample_config={"seed": 20260713, "n_samples": 1000},
        _suppress_print=True,
    )
    z = compute_max_abs_mean_z(idata, "lotka_volterra")
    assert (
        z is not None
    ), "compute_max_abs_mean_z returned None — reference/summary.json missing?"
    assert z < 4.0, (
        f"Seed 20260713 should pass z < 4.0 (it is a clean seed), got z={z:.3f}. "
        "If this seed now fails, the step-collapse has worsened — see issue #232."
    )
