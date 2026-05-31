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
"""Phase 8 v1 benchmark suite — cross-sampler correctness + performance.

Runs via:
    make benchmark               # all tiers, e2e + calibrated, nightly
    make benchmark-pr            # Tier 1 calibrated only (fast, ~60s, per-PR)

Each benchmark:
  1. Runs a PASS recipe's sampler via ``run_recipe_to_idata`` (e2e or calibrated).
  2. Times the run with pytest-benchmark (wall time).
  3. Asserts GT-correctness post-timing: ``max_abs_mean_z < 2.0``.

This makes each benchmark a **correctness regression test**, not just timing — a
regression in the sampler's GT-agreement shows up as a failing benchmark.

Families covered (Phase 8 v1):
  nuts, hmc, dynamic_hmc, mhmc, dmhmc       — Tier 1 (standard candle) + Tier 2
  mclmc                                      — Tier 1+2, e2e only (skip_warmup raises)
  adjusted_mclmc, adjusted_mclmc_dynamic     — Tier 1, e2e + calibrated
  laplace_dhmc, laplace_dmhmc, laplace_hmc,
  laplace_mhmc                               — Tier 1, e2e only (phi-space mismatch)

Gaps (Phase 8B):  SMC, VI, elliptical_slice, rmhmc — 0 good recipes; need coverage first.

n_samples=500: minimum for reliable z-scores at threshold 2.0
  (SE ∝ 1/sqrt(ESS); at n=500, 4 chains → ESS≈200–1000 → SE manageable).

D5 budget cap (CONTRIBUTING.md): select < 180 s, exec ≤ 240 s per cell.

References:
  - worklog/threads/phase8-benchmark-scope.md (statistician scope doc)
  - worklog/decisions/2026-05-28-max-abs-mean-z-threshold.md (z-threshold)
  - CONTRIBUTING.md § Benchmark suite
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_CATALOG_ROOT = Path(__file__).resolve().parents[1] / "tuningfork" / "catalog"
_N_SAMPLES = 500  # TL-ratified: minimum for reliable z-scores at threshold=2.0
_Z_THRESHOLD = 2.0  # PASS gate (2026-05-28-max-abs-mean-z-threshold.md)
_BENCHMARK_SEED = 20260531  # fixed seed for reproducible benchmark runs

# ---------------------------------------------------------------------------
# BENCH_CELLS: (tier, model_name, recipe_filename, mode)
# ---------------------------------------------------------------------------
#
# mode: "e2e"        → run_recipe_to_idata (full warmup + sample)
#       "calibrated" → run_recipe_to_idata(skip_warmup=True) (sample only)
#
# Tier 1 = standard candle (always fast, cross-family comparison baseline)
# Tier 2 = interesting geometry (algorithm-specific stress tests)
#
# PASS-only, one IMM per family (TL ratification 2026-05-31).
# Recipes verified on-disk at scope-doc commit 1304e59.

_BENCH_CELLS: list[tuple[str, str, str, str]] = [
    # ── Tier 1: Standard candles (cross-family comparison baseline) ──────
    # (tier, model, recipe_file, mode)
    #
    # nuts
    (
        "tier1",
        "logistic_synthetic",
        "low__nuts__window_adaptation_diag_imm.json",
        "e2e",
    ),
    (
        "tier1",
        "logistic_synthetic",
        "low__nuts__window_adaptation_diag_imm.json",
        "calibrated",
    ),
    # hmc
    ("tier1", "logistic_synthetic", "low__hmc__window_adaptation_diag_imm.json", "e2e"),
    (
        "tier1",
        "logistic_synthetic",
        "low__hmc__window_adaptation_diag_imm.json",
        "calibrated",
    ),
    # dynamic_hmc
    (
        "tier1",
        "logistic_synthetic",
        "low__dynamic_hmc__window_adaptation_diag_imm.json",
        "e2e",
    ),
    (
        "tier1",
        "logistic_synthetic",
        "low__dynamic_hmc__window_adaptation_diag_imm.json",
        "calibrated",
    ),
    # mhmc
    (
        "tier1",
        "logistic_synthetic",
        "low__mhmc__window_adaptation_diag_imm.json",
        "e2e",
    ),
    (
        "tier1",
        "logistic_synthetic",
        "low__mhmc__window_adaptation_diag_imm.json",
        "calibrated",
    ),
    # dmhmc
    (
        "tier1",
        "logistic_synthetic",
        "low__dmhmc__window_adaptation_diag_imm.json",
        "e2e",
    ),
    (
        "tier1",
        "logistic_synthetic",
        "low__dmhmc__window_adaptation_diag_imm.json",
        "calibrated",
    ),
    # mclmc — e2e only (skip_warmup raises: momentum init not handled)
    ("tier1", "logistic_synthetic", "low__mclmc__mclmc_tuning.json", "e2e"),
    # adjusted_mclmc — both modes (PR #103 fixed skip_warmup rng_key threading)
    ("tier1", "mvn_10", "low__adjusted_mclmc__adjusted_mclmc_tuning.json", "e2e"),
    (
        "tier1",
        "mvn_10",
        "low__adjusted_mclmc__adjusted_mclmc_tuning.json",
        "calibrated",
    ),
    # adjusted_mclmc_dynamic — both modes
    (
        "tier1",
        "logistic_synthetic",
        "low__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json",
        "e2e",
    ),
    (
        "tier1",
        "logistic_synthetic",
        "low__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json",
        "calibrated",
    ),
    # laplace family — e2e only (phi-space GT-means mismatch for skip_warmup)
    (
        "tier1",
        "eight_schools_ncp",
        "low__laplace_dhmc__window_adaptation_diag_imm.json",
        "e2e",
    ),
    (
        "tier1",
        "eight_schools_ncp",
        "low__laplace_dmhmc__window_adaptation_diag_imm.json",
        "e2e",
    ),
    (
        "tier1",
        "eight_schools_ncp",
        "low__laplace_hmc__window_adaptation_diag_imm.json",
        "e2e",
    ),
    # laplace_mhmc: dense IMM is the best-headline variant per scope doc
    (
        "tier1",
        "eight_schools_ncp",
        "low__laplace_mhmc__window_adaptation_dense_imm.json",
        "e2e",
    ),
    # ── Tier 2: Interesting geometry (algorithm-specific stress tests) ────
    # nuts × eight_schools: hierarchical NCP funnel
    ("tier2", "eight_schools_ncp", "low__nuts__window_adaptation_diag_imm.json", "e2e"),
    (
        "tier2",
        "eight_schools_ncp",
        "low__nuts__window_adaptation_diag_imm.json",
        "calibrated",
    ),
    # nuts × stoch_vol: high-d AR(1) (~45s)
    ("tier2", "stoch_vol", "low__nuts__window_adaptation_diag_imm.json", "e2e"),
    ("tier2", "stoch_vol", "low__nuts__window_adaptation_diag_imm.json", "calibrated"),
    # dynamic_hmc × banana: curved geometry with step policy
    (
        "tier2",
        "banana",
        "medium__dynamic_hmc__window_adaptation_diag_imm__policy_v1-medium.json",
        "e2e",
    ),
    (
        "tier2",
        "banana",
        "medium__dynamic_hmc__window_adaptation_diag_imm__policy_v1-medium.json",
        "calibrated",
    ),
    # hmc × lotka_volterra: stiff ODE (~42s)
    (
        "tier2",
        "lotka_volterra",
        "low__hmc__window_adaptation_dense_imm__inner_nuts.json",
        "e2e",
    ),
    (
        "tier2",
        "lotka_volterra",
        "low__hmc__window_adaptation_dense_imm__inner_nuts.json",
        "calibrated",
    ),
    # dmhmc × horseshoe: extreme geometry (~81s, ≤5min budget)
    ("tier2", "horseshoe", "low__dmhmc__window_adaptation_dense_imm.json", "e2e"),
    (
        "tier2",
        "horseshoe",
        "low__dmhmc__window_adaptation_dense_imm.json",
        "calibrated",
    ),
    # dmhmc × stoch_vol: high-d (~20s)
    ("tier2", "stoch_vol", "low__dmhmc__window_adaptation_dense_imm.json", "e2e"),
    (
        "tier2",
        "stoch_vol",
        "low__dmhmc__window_adaptation_dense_imm.json",
        "calibrated",
    ),
    # mclmc × eight_schools: MCLMC vs NUTS on NCP (~18s), e2e only
    ("tier2", "eight_schools_ncp", "low__mclmc__mclmc_tuning.json", "e2e"),
]

# ---------------------------------------------------------------------------
# GT-correctness helper
# ---------------------------------------------------------------------------


def _compute_max_abs_mean_z(idata: Any, model_name: str) -> float | None:
    """Compute max |z| for all posterior parameters vs the GT reference.

    Reuses the recipe-cert standard (2026-05-28-max-abs-mean-z-threshold.md):
    z = (sample_mean - gt_mean) / sqrt(SE_sample^2 + SE_gt^2)

    Returns None when reference/summary.json is unavailable (graceful skip).
    """
    summary_path = _CATALOG_ROOT / model_name / "reference" / "summary.json"
    if not summary_path.exists():
        return None

    gt = json.loads(summary_path.read_text())
    gt_means: dict[str, float] = {k: float(v) for k, v in gt.get("mean", {}).items()}
    gt_stds: dict[str, float] = {k: float(v) for k, v in gt.get("std", {}).items()}
    n_gt = int(gt.get("num_samples", 40000))

    if not hasattr(idata, "posterior"):
        return None

    posterior = idata.posterior
    z_scores: list[float] = []
    for var in posterior.data_vars:
        if var not in gt_means or var not in gt_stds:
            continue
        samples = np.asarray(posterior[var].values)
        # Flatten chain × draw → single array; compute mean
        flat = (
            samples.reshape(-1, *samples.shape[2:])
            if samples.ndim > 2
            else samples.ravel()
        )
        n_samp = len(flat)
        if n_samp == 0:
            continue
        sample_mean = float(np.mean(flat))
        gt_mean = gt_means[var]
        gt_std = gt_stds[var]
        if gt_std <= 0:
            continue
        se_sample = gt_std / np.sqrt(n_samp)
        se_gt = gt_std / np.sqrt(n_gt)
        se_combined = float(np.sqrt(se_sample**2 + se_gt**2))
        if se_combined <= 0:
            continue
        z = abs(sample_mean - gt_mean) / se_combined
        z_scores.append(z)

    return float(max(z_scores)) if z_scores else None


# ---------------------------------------------------------------------------
# Parametrized benchmark
# ---------------------------------------------------------------------------


def _bench_id(cell: tuple[str, str, str, str]) -> str:
    tier, model, recipe_file, mode = cell
    stem = recipe_file.replace(".json", "")
    return f"{tier}-{model}-{stem}-{mode}"


@pytest.mark.benchmark(group="recipes")
@pytest.mark.parametrize(
    "tier,model_name,recipe_file,mode",
    _BENCH_CELLS,
    ids=[_bench_id(c) for c in _BENCH_CELLS],
)
def test_recipe_perf(
    benchmark: Any,
    tier: str,
    model_name: str,
    recipe_file: str,
    mode: str,
) -> None:
    """Benchmark a recipe's sampler and assert GT-correctness.

    Timing is measured by pytest-benchmark (wall time, warmup round).
    GT-correctness (max_abs_mean_z < 2.0) is asserted AFTER the timed run
    so the assertion overhead doesn't pollute the benchmark measurement.

    Parameters
    ----------
    tier
        "tier1" (standard candle, runs in PR CI calibrated mode) or
        "tier2" (stress test, nightly only).
    model_name
        Model name key in MODELS registry.
    recipe_file
        Recipe JSON filename under ``catalog/<model>/recipes/``.
    mode
        ``"e2e"`` (full warmup+sample) or ``"calibrated"`` (skip_warmup=True).
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

    # Timed run (pytest-benchmark handles warmup + multiple rounds internally)
    idata = benchmark(run)

    # --- GT-correctness check (asserted outside the timed block) ---
    z = _compute_max_abs_mean_z(idata, model_name)
    if z is not None:
        assert z < _Z_THRESHOLD, (
            f"GT-correctness FAILED for {model_name}/{recipe_file} ({mode}): "
            f"max_abs_mean_z={z:.3f} ≥ {_Z_THRESHOLD} "
            f"(recipe may have regressed in sampling correctness)"
        )
