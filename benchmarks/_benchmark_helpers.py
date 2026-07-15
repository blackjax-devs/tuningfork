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
_Z_THRESHOLD = (
    4.0  # PASS gate: mock proved 2.0 false-positives hard geometries (z<=3.8)
)
_BENCHMARK_SEED = (
    20260531  # fixed seed for reproducible runs (legacy; overridden by nightly)
)

# Per-cell JAX/blackjax-drift detection tolerances (provisional; statistician to refine)
_JAX_DRIFT_ESS_FLOOR_FACTOR = 0.5  # warn if rerun ESS < 50% of committed recipe value
_JAX_DRIFT_Z_DELTA = 2.0  # warn if rerun z > committed_z + 2.0
_WITHIN_SEED_ESS_REL_TOL = 0.05  # warn if 2 warm same-seed runs differ >5% ESS
_WITHIN_SEED_Z_ABS_TOL = 0.2  # warn if 2 warm same-seed runs differ >0.2 in z


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
    #
    # Also include n_samples (top-level scalar in summary.json) per param so the
    # gate can compute se_gt = std / sqrt(n_samples) using the TRUE reference count
    # (e.g. 40000 for a single-chain GT run) rather than falling back to
    # n_chains * n_draws from the benchmark run (1 * 1000 = 1000), which would
    # make se_gt 6× too large and the correctness check too lenient.
    #
    # Note: models with summary_v2.json (multichain GT) still route through
    # reference/summary.json here (legacy SE).  A follow-up PR should upgrade
    # migrated models to use between_chain_se from summary_v2.json.
    n_samples_gt = gt_summaries.get("n_samples")
    gt_per_param = {
        param: {
            "mean": gt_summaries["mean"][param],
            "std": gt_summaries["std"][param],
            **({"n_samples": n_samples_gt} if n_samples_gt is not None else {}),
        }
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
                _no_tap=True,  # benchmark context: never touch tap regardless of env var
            )
    except Exception:  # noqa: BLE001
        pass  # best-effort; never break the benchmark run


# ---------------------------------------------------------------------------
# Per-cell compile-warmup and JAX-drift detection helpers
# ---------------------------------------------------------------------------


def _get_committed_metrics(
    recipe: Any,
) -> tuple[int | None, float | None, float | None]:
    """Extract (tuning_seed, committed_min_bulk_ess, committed_max_abs_mean_z).

    Used to seed the compile-warmup run and compare the rerun against the
    recipe's certified metrics for JAX/blackjax-drift detection.
    """
    tuning_seed: int | None = getattr(recipe, "tuning_seed", None)
    gate_evidence: dict[str, Any] = getattr(recipe, "gate_evidence", None) or {}
    gate_auto: dict[str, Any] = gate_evidence.get("auto", {}) or {}
    committed_ess: float | None = gate_auto.get("min_bulk_ess")
    committed_z: float | None = gate_auto.get("max_abs_mean_z")
    return tuning_seed, committed_ess, committed_z


def _check_jax_drift(
    metrics: dict[str, Any],
    committed_ess: float | None,
    committed_z: float | None,
    model_name: str,
    recipe_file: str,
) -> tuple[bool, list[str]]:
    """Return (drift_flag, drift_details) comparing rerun vs committed recipe metrics.

    Criterion (statistician-ratified 2026-06-01):
    - ESS < committed_ess × 50%          → flag
    - z   > committed_z  + 2.0 (abs)     → flag
    - n_divergences excluded (too noisy)

    Does NOT print or emit GHA annotations — caller (``run_nightly.py`` via
    ``emit_gha_annotations``) decides when to emit ``::warning::JAX_DRIFT``.
    Additive and non-blocking: never changes GREEN/REVIEW/REGRESSION verdict.
    """
    cell_id = f"{model_name}/{recipe_file}"
    ess = metrics.get("min_bulk_ess")
    z = metrics.get("max_abs_mean_z")

    raw_details: list[str] = []

    if (
        ess is not None
        and committed_ess is not None
        and committed_ess > 0
        and ess < committed_ess * _JAX_DRIFT_ESS_FLOOR_FACTOR
    ):
        raw_details.append(
            f"ESS={ess:.0f} vs committed={committed_ess:.0f}"
            f" (< {_JAX_DRIFT_ESS_FLOOR_FACTOR:.0%})"
        )

    if z is not None and committed_z is not None:
        if z > committed_z + _JAX_DRIFT_Z_DELTA:
            raw_details.append(
                f"z={z:.3f} vs committed={committed_z:.3f}"
                f" (delta > {_JAX_DRIFT_Z_DELTA})"
            )

    drift_flag = bool(raw_details)
    drift_details: list[str] = (
        [f"JAX_DRIFT {cell_id}: " + "; ".join(raw_details)] if drift_flag else []
    )
    return drift_flag, drift_details


def _check_within_seed_determinism(
    m1: dict[str, Any],
    m2: dict[str, Any],
    seed: int,
    model_name: str,
    recipe_file: str,
) -> None:
    """Log a warning when two warm same-seed runs disagree beyond tolerance.

    Warm + fixed-seed runs should be fully deterministic.  Disagreement signals
    non-determinism in JAX/XLA (e.g. parallelism, compiler changes).
    """
    cell_id = f"{model_name}/{recipe_file}"
    ess1 = m1.get("min_bulk_ess")
    ess2 = m2.get("min_bulk_ess")
    z1 = m1.get("max_abs_mean_z")
    z2 = m2.get("max_abs_mean_z")

    if ess1 is not None and ess2 is not None:
        ref = max(abs(ess1), abs(ess2), 1.0)
        if abs(ess1 - ess2) / ref > _WITHIN_SEED_ESS_REL_TOL:
            print(
                f"::warning::DETERMINISM_WARN {cell_id} seed={seed}: "
                f"ESS run1={ess1:.0f} run2={ess2:.0f}"
                f" (rel diff > {_WITHIN_SEED_ESS_REL_TOL:.0%})"
            )

    if z1 is not None and z2 is not None:
        if abs(z1 - z2) > _WITHIN_SEED_Z_ABS_TOL:
            print(
                f"::warning::DETERMINISM_WARN {cell_id} seed={seed}: "
                f"z run1={z1:.3f} run2={z2:.3f}"
                f" (abs diff > {_WITHIN_SEED_Z_ABS_TOL})"
            )


def _mean_metrics(m1: dict[str, Any], m2: dict[str, Any]) -> dict[str, Any]:
    """Return the element-wise mean of two metric dicts.

    Numeric values are averaged; non-numeric (None, bool, str) are taken from m1.
    Runtime fields (``runtime_*``) are summed rather than averaged since they
    represent total elapsed time across both runs.
    """
    result: dict[str, Any] = {}
    for k in m1:
        v1, v2 = m1.get(k), m2.get(k)
        # Guard: bool subclasses int in Python — treat bools as non-numeric
        is_numeric = (
            isinstance(v1, (int, float))
            and not isinstance(v1, bool)
            and isinstance(v2, (int, float))
            and not isinstance(v2, bool)
        )
        if is_numeric:
            n1: float = float(v1)  # type: ignore[arg-type]
            n2: float = float(v2)  # type: ignore[arg-type]
            if k.startswith("runtime_"):
                result[k] = n1 + n2  # total elapsed, not mean
            else:
                result[k] = (n1 + n2) / 2.0
        else:
            result[k] = v1
    return result


def run_benchmark_cell(
    benchmark: Any,
    model_name: str,
    recipe_file: str,
    mode: str,
    run_date: date | None = None,
) -> dict[int, dict[str, Any]]:
    """Run a single benchmark cell with compile-warmup + 3 date-seeds x 2 warm runs.

    Cell shape: **7 runs total** per cell.

    1. **Compile-warmup** (1 run, discarded from metrics):
       Runs the cell with the recipe's committed ``tuning_seed`` to compile the
       XLA executable so all date-seeds execute on a warm JIT cache.
       The warmup-run metrics are compared to the recipe's committed
       ``gate_evidence.auto`` values: a significant drop in ESS or spike in z
       emits a ``::warning::JAX_DRIFT`` annotation (JAX/blackjax numeric drift).

    2. **3 date-seeds x 2 warm runs** (6 runs, all warm):
       Each seed is run twice.  Per-seed metric = mean of the 2 runs.
       If the 2 runs of a seed disagree beyond tolerance a
       ``::warning::DETERMINISM_WARN`` annotation is emitted (warm + fixed-seed
       should be fully deterministic).

    Per-seed means are stored in ``benchmark.extra_info["per_seed_metrics"]``
    for downstream result persistence by ``run_nightly.py``.

    GT-correctness (z < 4.0) is asserted on the per-seed mean before returning.

    Parameters
    ----------
    benchmark
        pytest-benchmark fixture.
    model_name, recipe_file, mode
        Cell identity.
    run_date
        Date to derive seeds from (defaults to today).
    """
    import time

    from tuningfork.catalog.inspect import load_recipe
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe_path = _CATALOG_ROOT / model_name / "recipes" / recipe_file
    if not recipe_path.exists():
        pytest.skip(f"Recipe not found on disk: {recipe_path}")
    recipe = load_recipe(recipe_path)

    seeds = get_nightly_seeds(run_date)
    skip_warmup = mode == "calibrated"
    per_seed_metrics: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Step 1: compile-warmup with recipe's committed tuning_seed.
    # Outside the benchmark() call — not timed, result discarded.
    # Best-effort: never break the benchmark run on warmup failure.
    # JAX-drift flag + details stored in extra_info for run_nightly.py.
    # ------------------------------------------------------------------
    tuning_seed, committed_ess, committed_z = _get_committed_metrics(recipe)
    compile_seed = tuning_seed if tuning_seed is not None else seeds[0]
    jax_drift_flag: bool = False
    jax_drift_details: list[str] = []
    try:
        idata_warmup = run_recipe_to_idata(
            recipe,
            skip_warmup=skip_warmup,
            n_samples=_N_SAMPLES,
            force_resample_config={"seed": compile_seed, "n_samples": _N_SAMPLES},
            _suppress_print=True,
            _no_tap=True,  # compile-warmup is outside the timed block; never tap
        )
        warmup_metrics = extract_cell_metrics(idata_warmup, model_name)
        jax_drift_flag, jax_drift_details = _check_jax_drift(
            warmup_metrics, committed_ess, committed_z, model_name, recipe_file
        )
    except Exception:  # noqa: BLE001
        pass  # compile-warmup is best-effort; proceed to seed runs regardless

    # ------------------------------------------------------------------
    # Step 2: 3 date-seeds x 2 warm runs each.
    # Timed by benchmark() as a single block.
    # With --benchmark-min-rounds=1 --benchmark-max-time=0 runs exactly once.
    # ------------------------------------------------------------------
    def run_all_seeds() -> None:
        """3 seeds x 2 warm runs; per-seed metric = mean of the pair."""
        for s in seeds:
            runs: list[dict[str, Any]] = []
            for _ in range(2):
                t0 = time.perf_counter()
                idata = run_recipe_to_idata(
                    recipe,
                    skip_warmup=skip_warmup,
                    n_samples=_N_SAMPLES,
                    force_resample_config=(
                        None if skip_warmup else {"seed": s, "n_samples": _N_SAMPLES}
                    ),
                    _suppress_print=True,
                    _no_tap=True,  # timed body: structurally gates tap from all timing
                )
                t_run = time.perf_counter() - t0
                runs.append(
                    extract_cell_metrics(idata, model_name, warmup_wall_s=t_run)
                )
            _check_within_seed_determinism(runs[0], runs[1], s, model_name, recipe_file)
            per_seed_metrics[s] = _mean_metrics(runs[0], runs[1])

    benchmark(run_all_seeds)

    # Assert GT-correctness on per-seed means
    for s, metrics in per_seed_metrics.items():
        z = metrics.get("max_abs_mean_z")
        if z is not None:
            assert z < _Z_THRESHOLD, (
                f"GT-correctness FAILED for {model_name}/{recipe_file} ({mode}) "
                f"seed={s}: max_abs_mean_z={z:.3f} >= {_Z_THRESHOLD}"
            )

    # Store per-seed means and JAX-drift flag in extra_info for run_nightly.py
    benchmark.extra_info["per_seed_metrics"] = {
        str(s): m for s, m in per_seed_metrics.items()
    }
    benchmark.extra_info["jax_drift"] = {
        "flag": jax_drift_flag,
        "details": jax_drift_details,
    }

    return per_seed_metrics
