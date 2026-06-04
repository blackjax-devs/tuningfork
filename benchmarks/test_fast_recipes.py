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
"""Fast recipe benchmarks (≤60s each, ~8 min total nightly).

Runs via:
    make benchmark-fast    # this file only (fast suite, ~8 min)
    make benchmark         # fast + e2e (full nightly suite)

Each benchmark:
  1. Runs a PASS recipe's sampler via ``run_recipe_to_idata``.
  2. Times the run with pytest-benchmark (1 round per cell).
  3. Asserts GT-correctness post-timing: ``max_abs_mean_z < 2.0``.

Cell selection: see ``benchmarks/config.py`` (FAST_CELLS, ≤60s/cell in CI).
Slow e2e cells (>60s): see ``benchmarks/test_e2e_recipes.py``.
"""
from __future__ import annotations

from typing import Any

import pytest

from benchmarks._benchmark_helpers import bench_id, run_benchmark_cell
from benchmarks.config import FAST_CELLS, XFAIL_CELLS


@pytest.mark.benchmark(group="recipes-fast")
@pytest.mark.parametrize(
    "tier,model_name,recipe_file,mode",
    FAST_CELLS,
    ids=[bench_id(c) for c in FAST_CELLS],
)
def test_recipe_perf(
    request: pytest.FixtureRequest,
    benchmark: Any,
    tier: str,
    model_name: str,
    recipe_file: str,
    mode: str,
) -> None:
    """Benchmark a recipe's sampler across 3 date-derived seeds.

    All 3 seeds run in one timed block (3 timing samples + 3 regression inputs).
    JIT is warmed once by the session-scoped conftest fixture before any cell runs.
    Per-seed metrics are stored in ``benchmark.extra_info["per_seed_metrics"]``.
    GT-correctness (max_abs_mean_z < 4.0) is asserted for all seeds.

    Cells in ``XFAIL_CELLS`` are marked ``pytest.mark.xfail``: the cell still
    runs and is timed, but the GT-correctness AssertionError is an *expected*
    failure.  An unexpected pass (XPASS) is surfaced in the report.
    """
    # Apply xfail mark at runtime for known-flaky cells (XFAIL_CELLS).
    # Unlike pytest.xfail() which aborts immediately, request.applymarker lets
    # the full test run (timing is recorded); only a correctness AssertionError
    # becomes an expected failure (xfail) rather than a CI-breaking failure.
    # An unexpected pass (XPASS) is still surfaced in the report.
    cell_id = bench_id((tier, model_name, recipe_file, mode))
    if cell_id in XFAIL_CELLS:
        request.applymarker(
            pytest.mark.xfail(
                reason=XFAIL_CELLS[cell_id],
                strict=False,  # XPASS is OK — means issue resolved
                raises=AssertionError,  # only catch correctness failures
            )
        )
    run_benchmark_cell(benchmark, model_name, recipe_file, mode)
