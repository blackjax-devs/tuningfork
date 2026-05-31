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

# All cells (for nightly full run)
ALL_CELLS: list[tuple[str, str, str, str]] = FAST_CELLS + SLOW_CELLS
