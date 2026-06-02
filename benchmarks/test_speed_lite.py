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
  seed-CI (nightly)   → correctness regression (ESS/z, 3 seeds × 2 warm runs)
  speed-lite (per-PR) → performance regression (wall-clock trend, actions/cache)

**How it works:**
  Each of 6 cells (one per sampler family) runs via ``benchmark.pedantic``::

      warmup_rounds=1 → first run absorbs XLA JIT cold-start (discarded)
      rounds=5        → 5 warm measurements → stable Mean / StdDev

  Fixed seed ``SPEED_SEED`` — timing is seed-invariant, so cross-run trend
  comparison is valid.

  ``clear_xla_caches_between_cells`` (conftest) fires after each cell, so
  per-cell XLA compile-cache is freed between cells (bounded memory), while
  the 5 measured rounds within a cell all run warm (cache intact).

**Workflow:**
  Triggered per-PR and per-push to main by ``speed_benchmark.yml``.
  Trend persisted in ``actions/cache`` (not gh-pages).
  Alert at 200% (comment-only at rollout; flip to fail-on-alert once stable).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmarks.config import SPEED_LITE_CELLS, SPEED_SEED

_CATALOG_ROOT = Path(__file__).resolve().parents[1] / "tuningfork" / "catalog"
_N_SAMPLES = 1000  # same as seed-CI for a comparable workload per round


def _speed_bench_id(cell: tuple[str, str, str, str]) -> str:
    tier, model, recipe_file, mode = cell
    stem = recipe_file.replace(".json", "")
    return f"{tier}-{model}-{stem}-{mode}"


@pytest.mark.benchmark(group="speed-lite")
@pytest.mark.parametrize(
    "tier,model_name,recipe_file,mode",
    SPEED_LITE_CELLS,
    ids=[_speed_bench_id(c) for c in SPEED_LITE_CELLS],
)
def test_speed_lite(
    benchmark: Any,
    tier: str,
    model_name: str,
    recipe_file: str,
    mode: str,
) -> None:
    """Measure steady-state wall-clock for one sampler family (5 warm rounds).

    ``warmup_rounds=1`` discards the JIT compile; the 5 measured rounds run
    warm.  Fixed seed ``SPEED_SEED`` keeps the cross-PR trend comparable.
    Separate from the seed-CI: no seed-variation, no correctness assertion.
    """
    from tuningfork.catalog.inspect import load_recipe  # noqa: PLC0415
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata  # noqa: PLC0415

    recipe_path = _CATALOG_ROOT / model_name / "recipes" / recipe_file
    if not recipe_path.exists():
        pytest.skip(f"Recipe not found on disk: {recipe_path}")
    recipe = load_recipe(recipe_path)
    skip_warmup = mode == "calibrated"

    def run_once() -> None:
        run_recipe_to_idata(
            recipe,
            skip_warmup=skip_warmup,
            n_samples=_N_SAMPLES,
            force_resample_config={"seed": SPEED_SEED, "n_samples": _N_SAMPLES},
            _suppress_print=True,
        )

    # pedantic: exact control over rounds.
    # warmup_rounds=1 → JIT compile absorbed (discarded from stats)
    # rounds=5        → 5 warm measurements for trend comparison
    benchmark.pedantic(run_once, rounds=5, warmup_rounds=1)
