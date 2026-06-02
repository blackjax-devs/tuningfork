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
# ALL_CELLS: union of FAST + SLOW (nightly full run + speed-lite source).
# Defined before SPEED_LITE_CELLS so the speed-lite filter can draw from both.
# ---------------------------------------------------------------------------
ALL_CELLS: list[tuple[str, str, str, str]] = FAST_CELLS + SLOW_CELLS


# ---------------------------------------------------------------------------
# SPEED-LITE cells — test_speed_lite.py (nightly wall-clock regression)
#
# Two-axis design:
#   Horizontal (seed axis):   ALL_CELLS × 3 dated seeds  → seed-CI correctness
#   Vertical (timing axis):   SPEED_LITE_CELLS × pinned/daily seed × 5 runs
#
# SPEED_LITE_CELLS is *derived from ALL_CELLS* (one source of truth for cell
# definitions).  The filter selects representative cells spanning sampler
# families and model topologies.  Slow cells (>60s) are eligible now that
# the per-PR time budget constraint is gone.
#
# To add a cell to speed-lite: add its bench_id to _SPEED_LITE_BENCH_IDS.
# The cell must already exist in FAST_CELLS or SLOW_CELLS — add it there first.
# ---------------------------------------------------------------------------


def _bench_id(cell: tuple[str, str, str, str]) -> str:
    """Stable benchmark ID for a cell tuple (mirrors bench_id in _benchmark_helpers)."""
    tier, model, recipe_file, mode = cell
    return f"{tier}-{model}-{recipe_file.replace('.json', '')}-{mode}"


# Speed-lite filter: bench_ids drawn from ALL_CELLS for the timing axis.
#
# 15 cells (9 original + 6 added in #139).
#
# Original 9 (tier1 logistic/mvn/eight_schools e2e):
#   Covers 7 sampler families on cheap models (logistic_synthetic GLM, mvn_10,
#   eight_schools_ncp NCP funnel) — quick baseline for per-sampler regression.
#
# Added 6 (tier2, nightly budget unlocked):
#   horseshoe-dmhmc-calibrated   : extreme geometry, dense IMM (expK: ~31s, free seed)
#   lotka_volterra-hmc-e2e       : stiff ODE full warmup (expK: ~204s, free seed)
#                                  *** lives in SLOW_CELLS — filter uses ALL_CELLS ***
#   stoch_vol-dmhmc-e2e          : high-d AR(1), dense IMM (expK: ~16s, free seed)
#   stoch_vol-nuts-e2e           : high-d AR(1), diag IMM (expK: ~18s, PIN 20260601)
#   lotka_volterra-hmc-calibrated: stiff ODE, skip_warmup (expK: ~14s, free seed)
#   ill_cond_50-nuts-e2e         : κ=1000 (expK: ~5s, PIN 20260602)
#
# Budget: expK single-run sum ≈ 370s; CI ceiling ≈ 370 × 6 × ~1.8 runner factor
# ≈ 40 min — under 90-min cap with margin.
_SPEED_LITE_BENCH_IDS: frozenset[str] = frozenset(
    {
        # ── Original 9: tier1 sampler families ───────────────────────────────
        # logistic_synthetic — cheap GLM, covers 7 sampler families
        "tier1-logistic_synthetic-low__nuts__window_adaptation_diag_imm-e2e",
        "tier1-logistic_synthetic-low__hmc__window_adaptation_diag_imm-e2e",
        "tier1-logistic_synthetic-low__mhmc__window_adaptation_diag_imm-e2e",
        "tier1-logistic_synthetic-low__dynamic_hmc__window_adaptation_diag_imm-e2e",
        "tier1-logistic_synthetic-low__dmhmc__window_adaptation_diag_imm-e2e",
        "tier1-logistic_synthetic-low__mclmc__mclmc_tuning-e2e",
        "tier1-logistic_synthetic-low__adjusted_mclmc_dynamic__adjusted_mclmc_tuning-e2e",
        # mvn_10 — Gaussian topology, static adjusted_mclmc family
        "tier1-mvn_10-low__adjusted_mclmc__adjusted_mclmc_tuning-e2e",
        # eight_schools_ncp — hierarchical NCP topology diversity
        "tier2-eight_schools_ncp-low__nuts__window_adaptation_diag_imm-e2e",
        # ── Added 6 (#139): heavier tier2 cells, nightly budget ──────────────
        # horseshoe × dmhmc calibrated — extreme geometry, dense IMM (free seed)
        "tier2-horseshoe-low__dmhmc__window_adaptation_dense_imm-calibrated",
        # lotka_volterra × hmc e2e — stiff ODE, full warmup (free seed)
        # NOTE: lives in SLOW_CELLS; reachable only because filter uses ALL_CELLS.
        "tier2-lotka_volterra-low__hmc__window_adaptation_dense_imm__inner_nuts-e2e",
        # stoch_vol × dmhmc e2e — high-d AR(1), dense IMM (free seed)
        "tier2-stoch_vol-low__dmhmc__window_adaptation_dense_imm-e2e",
        # stoch_vol × nuts e2e — high-d AR(1), diag IMM (PIN 20260601)
        # expK phase_b: 20260601=17.85s, 20260602=18.72s, 20260603=16.06s (anomalous low)
        "tier2-stoch_vol-low__nuts__window_adaptation_diag_imm-e2e",
        # lotka_volterra × hmc calibrated — stiff ODE, skip_warmup (free seed)
        # expK phase_b: all three seeds ≈ 14.1-14.2s (seed-stable, free seed fine)
        "tier2-lotka_volterra-low__hmc__window_adaptation_dense_imm__inner_nuts-calibrated",
        # ill_cond_50 × nuts e2e — κ=1000, dense IMM (PIN 20260602)
        # expK phase_b: 20260601=7.09s (anomalous high), 20260602=5.04s, 20260603=5.10s
        "tier2-ill_cond_50-low__nuts__window_adaptation_dense_imm-e2e",
    }
)

# Derived from ALL_CELLS — not a hand-curated duplicate list.
# Preserves the ALL_CELLS ordering; updates automatically if a cell is renamed.
# Invariant: len(SPEED_LITE_CELLS) == 15  (assert in test_speed_lite.py at collection time)
SPEED_LITE_CELLS: list[tuple[str, str, str, str]] = [
    c for c in ALL_CELLS if _bench_id(c) in _SPEED_LITE_BENCH_IDS
]


# ---------------------------------------------------------------------------
# SPEED_SEED: today's date as YYYYMMDD int, consistent with seed-CI derivation.
# Varies daily — representative trajectory length for dynamic cells.
# Fixed-L cells (hmc) are fully timing-invariant across seeds.
# Dynamic cells (nuts, dynamic_hmc, mclmc, adjusted_mclmc*) carry seed-induced
# trajectory-length variance on top of runner noise.
# Per-cell overrides live in PINNED_SEEDS (see below).
# ---------------------------------------------------------------------------
def _today_seed() -> int:
    """Return today's YYYYMMDD as an int (date-derived, not a fixed constant)."""
    from datetime import date  # noqa: PLC0415

    return int(date.today().strftime("%Y%m%d"))


SPEED_SEED: int = _today_seed()

# ---------------------------------------------------------------------------
# PINNED_SEEDS: per-cell seed overrides for dynamic cells whose seed-induced
# trajectory-length variance is anomalous on specific dates.
#
# Rationale (from expK phase_b, experiments/expK_speed_lite_results.json):
#   Each entry names the *stable* date-seed — i.e. the seed whose per-cell mean
#   is representative of the typical regime, not an outlier.
#
# Do NOT tune pinned seeds against config.py inline timing comments; those
# reflect the original CI hardware and expK measured +27% (stoch_vol-dmhmc)
# and +54% (horseshoe) delta vs those comments on current runners.
# ---------------------------------------------------------------------------
PINNED_SEEDS: dict[str, int] = {
    # stoch_vol × nuts e2e:
    #   20260603 is anomalously LOW (16.1s vs ~18s typical — shorter trajectory).
    #   20260601 (17.85s) and 20260602 (18.72s) are both representative; pin 20260601
    #   as the earlier stable date.
    "tier2-stoch_vol-low__nuts__window_adaptation_diag_imm-e2e": 20260601,
    # ill_cond_50 × nuts e2e:
    #   20260601 is anomalously HIGH (7.09s vs ~5.0s typical — longer trajectory).
    #   20260602 (5.04s ≈ 20260603 5.10s) is the stable representative seed.
    "tier2-ill_cond_50-low__nuts__window_adaptation_dense_imm-e2e": 20260602,
}
