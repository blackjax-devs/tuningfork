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
"""Benchmark cell registry — single source of truth for recipe benchmark suite.

Each entry is a 4-tuple: (tier, model_name, recipe_filename, mode)
  tier  : "tier1" (standard candle) or "tier2" (interesting geometry)
  model : model name key in MODELS registry
  recipe: JSON filename under catalog/<model>/recipes/
  mode  : "e2e" (full warmup+sample) or "calibrated" (skip_warmup=True)

Wall-time routing (from round-4 CI run 26707364194, one timed run per cell):
  FAST cells (≤60s, in test_fast_recipes.py):  31 cells, ~8 min total nightly
  SLOW cells (>60s, in test_e2e_recipes.py):    2 cells, lotka_volterra e2e
                                                  (204s) + horseshoe e2e (75s)

The split keeps the fast suite quick enough for local smoke + targeted CI
triggers, while the slow e2e cells run nightly only.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# FAST cells (≤60s each in CI) — test_fast_recipes.py
# ---------------------------------------------------------------------------

FAST_CELLS: list[tuple[str, str, str, str]] = [
    # ── Tier 1: Standard candles ─────────────────────────────────────────
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
    # adjusted_mclmc
    ("tier1", "mvn_10", "low__adjusted_mclmc__adjusted_mclmc_tuning.json", "e2e"),
    (
        "tier1",
        "mvn_10",
        "low__adjusted_mclmc__adjusted_mclmc_tuning.json",
        "calibrated",
    ),
    # adjusted_mclmc_dynamic
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
    # CI timings: 37.5s, 16.5s, 16.5s, 16.2s — all well under 60s
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
    (
        "tier1",
        "eight_schools_ncp",
        "low__laplace_mhmc__window_adaptation_dense_imm.json",
        "e2e",
    ),
    # ── Tier 2: Interesting geometry (fast cells only) ────────────────────
    # nuts × eight_schools: hierarchical NCP funnel (3.9s e2e, 1.8s calibrated)
    ("tier2", "eight_schools_ncp", "low__nuts__window_adaptation_diag_imm.json", "e2e"),
    (
        "tier2",
        "eight_schools_ncp",
        "low__nuts__window_adaptation_diag_imm.json",
        "calibrated",
    ),
    # nuts × stoch_vol: high-d AR(1) (11.1s e2e, 4.0s calibrated)
    ("tier2", "stoch_vol", "low__nuts__window_adaptation_diag_imm.json", "e2e"),
    ("tier2", "stoch_vol", "low__nuts__window_adaptation_diag_imm.json", "calibrated"),
    # nuts × ill_cond_50: κ=1000 (6.5s e2e, 1.7s calibrated)
    ("tier2", "ill_cond_50", "low__nuts__window_adaptation_dense_imm.json", "e2e"),
    (
        "tier2",
        "ill_cond_50",
        "low__nuts__window_adaptation_dense_imm.json",
        "calibrated",
    ),
    # dmhmc × stoch_vol: high-d (16.2s e2e, 6.5s calibrated)
    ("tier2", "stoch_vol", "low__dmhmc__window_adaptation_dense_imm.json", "e2e"),
    (
        "tier2",
        "stoch_vol",
        "low__dmhmc__window_adaptation_dense_imm.json",
        "calibrated",
    ),
    # mclmc × eight_schools: MCLMC on NCP (6.3s e2e only)
    ("tier2", "eight_schools_ncp", "low__mclmc__mclmc_tuning.json", "e2e"),
    # hmc × lotka_volterra calibrated: stiff ODE calibrated mode (16.9s — fast)
    (
        "tier2",
        "lotka_volterra",
        "low__hmc__window_adaptation_dense_imm__inner_nuts.json",
        "calibrated",
    ),
    # horseshoe × dmhmc calibrated (31.2s — within 60s)
    (
        "tier2",
        "horseshoe",
        "low__dmhmc__window_adaptation_dense_imm.json",
        "calibrated",
    ),
]

# ---------------------------------------------------------------------------
# SLOW cells (>60s in CI) — test_e2e_recipes.py (nightly only)
# ---------------------------------------------------------------------------

SLOW_CELLS: list[tuple[str, str, str, str]] = [
    # hmc × lotka_volterra e2e: stiff ODE, full warmup (204s in CI)
    (
        "tier2",
        "lotka_volterra",
        "low__hmc__window_adaptation_dense_imm__inner_nuts.json",
        "e2e",
    ),
    # horseshoe × dmhmc e2e: extreme geometry, full warmup (75s in CI)
    ("tier2", "horseshoe", "low__dmhmc__window_adaptation_dense_imm.json", "e2e"),
]

# ---------------------------------------------------------------------------
# SPEED-LITE cells — test_speed_lite.py (per-PR wall-clock regression)
#
# Two-axis design for the fast benchmark cells:
#   Horizontal (seed axis):   FAST_CELLS × 3 dated seeds  → seed-CI correctness
#   Vertical (timing axis):   SPEED_LITE_CELLS × today's seed × 5 runs → speed-lite timing
#
# SPEED_LITE_CELLS is *derived from FAST_CELLS* (one source of truth for cell
# definitions).  The filter selects one e2e cell per major sampler family:
# NUTS, HMC, dynamic-HMC, MCLMC, adj-MCLMC, adj-MCLMC-dynamic.
# Excluded: mhmc/dmhmc (subsumed by HMC/dynamic-HMC families), laplace (not
# timing-stable across 5 warm runs at current scale), tier2/slow cells.
#
# To add a cell to speed-lite: add its bench_id to _SPEED_LITE_BENCH_IDS.
# The cell must already exist in FAST_CELLS — add it there first.
# ---------------------------------------------------------------------------


def _bench_id(cell: tuple[str, str, str, str]) -> str:
    """Stable benchmark ID for a cell tuple (mirrors bench_id in _benchmark_helpers)."""
    tier, model, recipe_file, mode = cell
    return f"{tier}-{model}-{recipe_file.replace('.json', '')}-{mode}"


# Speed-lite filter: bench_ids of FAST_CELLS to include in the vertical timing axis.
# One cell per major sampler family, e2e mode on the fastest representative model.
_SPEED_LITE_BENCH_IDS: frozenset[str] = frozenset(
    {
        "tier1-logistic_synthetic-low__nuts__window_adaptation_diag_imm-e2e",
        "tier1-logistic_synthetic-low__hmc__window_adaptation_diag_imm-e2e",
        "tier1-logistic_synthetic-low__dynamic_hmc__window_adaptation_diag_imm-e2e",
        "tier1-logistic_synthetic-low__mclmc__mclmc_tuning-e2e",
        "tier1-mvn_10-low__adjusted_mclmc__adjusted_mclmc_tuning-e2e",
        "tier1-logistic_synthetic-low__adjusted_mclmc_dynamic__adjusted_mclmc_tuning-e2e",
    }
)

# Derived from FAST_CELLS — not a hand-curated duplicate list.
# Preserves the ordering of FAST_CELLS; updates automatically if a cell is renamed.
SPEED_LITE_CELLS: list[tuple[str, str, str, str]] = [
    c for c in FAST_CELLS if _bench_id(c) in _SPEED_LITE_BENCH_IDS
]


# ---------------------------------------------------------------------------
# SPEED_SEED: today's date as YYYYMMDD int, consistent with seed-CI derivation.
# Varies daily so each run samples a representative trajectory length.
# Fixed-L cells (hmc) are fully timing-invariant across seeds.
# Dynamic cells (nuts, dynamic_hmc, mclmc, adjusted_mclmc*) carry seed-induced
# trajectory-length variance on top of runner noise — watch the variance band
# for the first few weeks and pin per-cell if any dynamic cell's seed-variance
# approaches the 200% alert threshold (statistician call).
# ---------------------------------------------------------------------------
def _today_seed() -> int:
    """Return today's YYYYMMDD as an int (date-derived, not a fixed constant)."""
    from datetime import date  # noqa: PLC0415

    return int(date.today().strftime("%Y%m%d"))


SPEED_SEED: int = _today_seed()

# All cells (for nightly full run)
ALL_CELLS: list[tuple[str, str, str, str]] = FAST_CELLS + SLOW_CELLS
