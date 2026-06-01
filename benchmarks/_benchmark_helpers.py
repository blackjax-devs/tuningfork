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
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_CATALOG_ROOT = Path(__file__).resolve().parents[1] / "tuningfork" / "catalog"
_N_SAMPLES = 1000  # matches recipe-cert n_samp=4000 (1000×4 chains) for z<2.0
_Z_THRESHOLD = 4.0  # PASS gate: mock proved 2.0 false-positives hard geometries (z≤3.8)
_BENCHMARK_SEED = (
    20260531  # fixed seed for reproducible runs (legacy; overridden by nightly)
)


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

    # P0 bug fix: summary.json has {"mean": {"param": v}, "std": {"param": v}} but
    # auto_gate expects {"param": {"mean": v, "std": v}}.  Restructure before passing.
    gt_per_param = {
        param: {stat: gt_summaries[stat][param] for stat in ("mean", "std")}
        for param in gt_summaries.get("mean", {})
        if param in mc_samples
    }
    if not gt_per_param:
        return None

    class _StubInfo:
        pass

    from tuningfork.calibration.statistician_gate import auto_gate
    from tuningfork.model import MODELS

    result = auto_gate(
        mc_samples,
        _StubInfo(),
        ground_truth_summaries=gt_per_param,
        posterior=MODELS.get(model_name),
        n_chunks=1,
    )
    return result.max_abs_mean_z


# ---------------------------------------------------------------------------
# Nightly seed scheme (date-seeded 3-seed scheme)
# ---------------------------------------------------------------------------


def get_nightly_seeds(run_date: date | None = None) -> tuple[int, int, int]:
    """Return the 3 seeds for a nightly run: (date-1, date, date+1).

    Each seed is ``int(YYYYMMDD)``.  Night D and D+1 share seeds {D, D+1},
    giving 2 overlapping seeds for cross-date regression detection.
    """
    d = run_date or date.today()
    return (
        int((d - timedelta(days=1)).strftime("%Y%m%d")),
        int(d.strftime("%Y%m%d")),
        int((d + timedelta(days=1)).strftime("%Y%m%d")),
    )


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def extract_cell_metrics(
    idata: Any,
    model_name: str,
    warmup_wall_s: float = 0.0,
    sample_wall_s: float = 0.0,
) -> dict[str, Any]:
    """Extract all benchmark metrics from idata.

    Returns a dict suitable for storage in the benchmark-results branch.
    """
    import arviz as az

    # ESS: DataTree → DatasetView
    ess_dt = az.ess(idata.posterior, method="bulk")
    try:
        min_ess = float(min(float(ess_dt.ds[v].min()) for v in ess_dt.ds.data_vars))
    except Exception:  # noqa: BLE001
        min_ess = float("nan")

    # Divergences
    try:
        n_div = int(idata.sample_stats.ds["diverging"].values.sum())
    except Exception:  # noqa: BLE001
        n_div = -1

    # z-score correctness
    z = compute_max_abs_mean_z(idata, model_name)

    return {
        "n_divergences": n_div,
        "min_bulk_ess": min_ess,
        "max_abs_mean_z": z,
        "runtime_warmup_s": warmup_wall_s,
        "runtime_sample_s": sample_wall_s,
        "correctness_passed": (z < _Z_THRESHOLD) if z is not None else None,
    }


# ---------------------------------------------------------------------------
# JIT warmup pass (P1)
# ---------------------------------------------------------------------------


def run_jit_warmup(seed: int = _BENCHMARK_SEED) -> None:
    """Run one throwaway calibrated cell to warm the XLA JIT cache.

    The mock showed up to 63% ESS difference between cold-start (first cell) and
    warm runs.  This pass stabilises subsequent cells.  Silently skips if the
    logistic_synthetic/nuts recipe is absent (CI bootstraps cleanly).
    """
    from tuningfork.catalog.inspect import load_recipe
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    warmup_recipe_path = (
        _CATALOG_ROOT
        / "logistic_synthetic"
        / "recipes"
        / "low__nuts__window_adaptation_diag_imm.json"
    )
    if not warmup_recipe_path.exists():
        return  # absent on fresh checkouts — skip silently
    try:
        recipe = load_recipe(warmup_recipe_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_recipe_to_idata(
                recipe,
                skip_warmup=True,
                n_samples=100,
                force_resample_config={"seed": seed, "n_samples": 100},
                _suppress_print=True,
            )
    except Exception:  # noqa: BLE001
        pass  # best-effort; never break the benchmark run


def run_benchmark_cell(
    benchmark: Any,
    model_name: str,
    recipe_file: str,
    mode: str,
    seed: int = _BENCHMARK_SEED,
) -> dict[str, Any]:
    """Run a single benchmark cell: time it + assert GT-correctness.

    Returns a metrics dict (extract_cell_metrics output) for result persistence.
    Shared implementation used by both test_fast_recipes and test_e2e_recipes.

    Parameters
    ----------
    benchmark
        pytest-benchmark fixture.
    model_name, recipe_file, mode
        Cell identity.
    seed
        RNG seed for the sampling run.  Defaults to ``_BENCHMARK_SEED``; nightly
        CI passes the date-derived seed so results are cross-night comparable.
    """
    import time

    from tuningfork.catalog.inspect import load_recipe
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe_path = _CATALOG_ROOT / model_name / "recipes" / recipe_file
    if not recipe_path.exists():
        pytest.skip(f"Recipe not found on disk: {recipe_path}")
    recipe = load_recipe(recipe_path)

    skip_warmup = mode == "calibrated"
    t_warmup_start = time.perf_counter()

    def run() -> Any:
        return run_recipe_to_idata(
            recipe,
            skip_warmup=skip_warmup,
            n_samples=_N_SAMPLES,
            force_resample_config=(
                None if skip_warmup else {"seed": seed, "n_samples": _N_SAMPLES}
            ),
            _suppress_print=True,
        )

    idata = benchmark(run)
    t_total = time.perf_counter() - t_warmup_start

    z = compute_max_abs_mean_z(idata, model_name)
    if z is not None:
        assert z < _Z_THRESHOLD, (
            f"GT-correctness FAILED for {model_name}/{recipe_file} ({mode}): "
            f"max_abs_mean_z={z:.3f} ≥ {_Z_THRESHOLD}"
        )

    # Extract and return metrics for nightly result persistence
    return extract_cell_metrics(
        idata,
        model_name,
        warmup_wall_s=t_total,  # approximate (benchmark times the full run)
        sample_wall_s=0.0,
    )
