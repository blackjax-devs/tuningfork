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
"""Speed-lite benchmark: per-sampler-family wall-clock regression detection.

**Purpose (distinct from the seed-CI):**
  seed-CI (nightly)     → correctness regression (ESS/z, 3 seeds × 2 warm runs)
  speed-lite (nightly)  → performance regression (wall-clock trend, actions/cache)

**How it works:**
  Each of 13 cells runs via ``benchmark.pedantic``::

      each round launches a standalone generated program through the
      receipt-backed lifecycle; no cross-subprocess warm-JIT persistence is
      assumed.
      rounds=5        → 5 independent cold measurements → Mean / StdDev

  Seed resolution (per-cell):
    ``PINNED_SEEDS.get(bench_id, SPEED_SEED)``
  Fixed-L cells (``hmc``) are fully timing-invariant across seeds and use the
  floating daily seed.  Dynamic cells (``nuts``, ``dynamic_hmc``, ``mclmc``,
  ``adjusted_mclmc*``) whose seed-induced trajectory-length variance was found
  anomalous on specific dates (see ``PINNED_SEEDS`` in config.py) use a pinned
  stable seed instead.

  ``clear_xla_caches_between_cells`` (conftest) bounds parent-process memory.
  The five measured generated-program rounds are independent subprocesses.

**Workflow:**
  Triggered nightly (23:00 UTC) by ``speed_benchmark.yml`` (#139).
  Trend persisted in ``actions/cache`` (not gh-pages).
  Alert at 200% with ``fail-on-alert: true`` (nightly red = regression signal).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmarks.config import PINNED_SEEDS, SPEED_LITE_CELLS, SPEED_SEED, _bench_id

# Invariant: must be 13 after adjusted_mclmc_dynamic/logistic_synthetic removal
# (2026-06-04, recipe demoted to honest-null; was 15 after #139 cell expansion)
# and lotka_volterra-inner_nuts-e2e removal (2026-06-13, quarantined via
# XFAIL_CELLS — 4/5 seed step-collapse causes 4.2× wall swing + false alerts).
# A silent filter-drop (e.g. a new SLOW cell not yet in ALL_CELLS) would reduce this count.
assert len(SPEED_LITE_CELLS) == 13, (
    f"Expected 13 speed-lite cells, got {len(SPEED_LITE_CELLS)}. "
    "Check _SPEED_LITE_BENCH_IDS against ALL_CELLS in config.py."
)

_CATALOG_ROOT = Path(__file__).resolve().parents[1] / "tuningfork" / "catalog"
_N_SAMPLES = 1000  # same as seed-CI for a comparable workload per round


@pytest.mark.benchmark(group="speed-lite")
@pytest.mark.parametrize(
    "tier,model_name,recipe_file,mode",
    SPEED_LITE_CELLS,
    ids=[_bench_id(c) for c in SPEED_LITE_CELLS],
)
def test_speed_lite(
    benchmark: Any,
    tier: str,
    model_name: str,
    recipe_file: str,
    mode: str,
) -> None:
    """Measure cold generated-program wall-clock for one cell (five rounds).

    Each measured round runs in an independent subprocess. Seed =
    ``PINNED_SEEDS.get(bench_id, SPEED_SEED)`` — pinned for
    dynamic cells with anomalous trajectory-length variance on specific dates,
    floating daily ``SPEED_SEED`` otherwise.  Separate from the seed-CI: no
    seed variation, no correctness assertion.
    """
    from benchmarks._benchmark_helpers import _run_generated  # noqa: PLC0415
    from tuningfork.catalog.inspect import load_recipe  # noqa: PLC0415

    recipe_path = _CATALOG_ROOT / model_name / "recipes" / recipe_file
    if not recipe_path.exists():
        pytest.skip(f"Recipe not found on disk: {recipe_path}")
    recipe = load_recipe(recipe_path)

    # Per-cell seed: use PINNED_SEEDS override for dynamic cells with
    # anomalous trajectory-length variance on specific dates; fall back to
    # the floating daily SPEED_SEED for all other cells.
    cell = (tier, model_name, recipe_file, mode)
    seed = PINNED_SEEDS.get(_bench_id(cell), SPEED_SEED)

    def run_once() -> None:
        _run_generated(recipe, model_name, mode, seed, _N_SAMPLES)

    # pedantic: exact control over rounds.
    # Five independent generated-program measurements for trend comparison.
    # A discarded subprocess cannot warm any measured subprocess, so there is
    # no benchmark-level warmup round.
    benchmark.pedantic(run_once, rounds=5, warmup_rounds=0)
