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
"""Shared helpers for the recipe benchmark suite.

Used by both test_fast_recipes.py and test_e2e_recipes.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_CATALOG_ROOT = Path(__file__).resolve().parents[1] / "tuningfork" / "catalog"
_N_SAMPLES = 1000  # matches recipe-cert n_samp=4000 (1000×4 chains) for z<2.0
_Z_THRESHOLD = 2.0  # PASS gate
_BENCHMARK_SEED = 20260531  # fixed seed for reproducible runs


def bench_id(cell: tuple[str, str, str, str]) -> str:
    """Stable pytest ID for a BENCH_CELLS entry."""
    tier, model, recipe_file, mode = cell
    stem = recipe_file.replace(".json", "")
    return f"{tier}-{model}-{stem}-{mode}"


def compute_max_abs_mean_z(idata: Any, model_name: str) -> float | None:
    """Compute max |z| vs GT reference using the recipe-cert auto_gate formula.

    z_i = |sample_mean_i − gt_mean_i| / max(SE_sample_i, SE_gt_i)
    where SE_sample_i = sample_std_i / sqrt(min_bulk_ESS).

    Returns None when reference/summary.json is unavailable (graceful skip).
    """
    summary_path = _CATALOG_ROOT / model_name / "reference" / "summary.json"
    if not summary_path.exists():
        return None
    if not hasattr(idata, "posterior"):
        return None

    gt_summaries = json.loads(summary_path.read_text())
    posterior = idata.posterior
    mc_samples: dict[str, Any] = {
        var: np.asarray(posterior[var].values) for var in posterior.data_vars
    }
    if not mc_samples:
        return None

    class _StubInfo:
        pass

    from tuningfork.calibration.statistician_gate import auto_gate
    from tuningfork.model import MODELS

    result = auto_gate(
        mc_samples,
        _StubInfo(),
        ground_truth_summaries=gt_summaries,
        posterior=MODELS.get(model_name),
        n_chunks=1,
    )
    return result.max_abs_mean_z


def run_benchmark_cell(
    benchmark: Any,
    model_name: str,
    recipe_file: str,
    mode: str,
) -> None:
    """Run a single benchmark cell: time it + assert GT-correctness.

    Shared implementation used by both test_fast_recipes and test_e2e_recipes.
    """
    from tuningfork.catalog.inspect import load_recipe
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe_path = _CATALOG_ROOT / model_name / "recipes" / recipe_file
    if not recipe_path.exists():
        pytest.skip(f"Recipe not found on disk: {recipe_path}")
    recipe = load_recipe(recipe_path)

    skip_warmup = mode == "calibrated"

    def run() -> Any:
        return run_recipe_to_idata(
            recipe,
            skip_warmup=skip_warmup,
            n_samples=_N_SAMPLES,
            force_resample_config=(
                None
                if skip_warmup
                else {"seed": _BENCHMARK_SEED, "n_samples": _N_SAMPLES}
            ),
            _suppress_print=True,
        )

    idata = benchmark(run)

    z = compute_max_abs_mean_z(idata, model_name)
    if z is not None:
        assert z < _Z_THRESHOLD, (
            f"GT-correctness FAILED for {model_name}/{recipe_file} ({mode}): "
            f"max_abs_mean_z={z:.3f} ≥ {_Z_THRESHOLD}"
        )
