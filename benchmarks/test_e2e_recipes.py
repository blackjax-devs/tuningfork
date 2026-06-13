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
"""Slow e2e recipe benchmarks (>60s each, nightly only).

These cells run the full warmup + sampling pipeline for models where the
e2e wall time exceeds 60s per CI run (from round-4 CI timings):
  - lotka_volterra × hmc × inner_nuts e2e:  204s (stiff ODE, full warmup)
  - horseshoe × dmhmc × diag_imm e2e:        75s (extreme geometry)

Runs via:
    make benchmark    # fast + e2e (full nightly suite)

DO NOT add these to ``make benchmark-fast`` or per-PR triggers — the wall
time makes them unsuitable for quick local checks.
"""
from __future__ import annotations

from typing import Any

import pytest

from benchmarks._benchmark_helpers import bench_id, run_benchmark_cell
from benchmarks.config import SLOW_CELLS, XFAIL_CELLS


@pytest.mark.benchmark(group="recipes-e2e")
@pytest.mark.parametrize(
    "tier,model_name,recipe_file,mode",
    SLOW_CELLS,
    ids=[bench_id(c) for c in SLOW_CELLS],
)
def test_recipe_e2e_perf(
    request: pytest.FixtureRequest,
    benchmark: Any,
    tier: str,
    model_name: str,
    recipe_file: str,
    mode: str,
) -> None:
    """Benchmark a slow e2e recipe's sampler across 3 date-derived seeds.

    These cells take >60s in CI and are nightly-only. All 3 seeds run in one
    timed block. GT-correctness (max_abs_mean_z < 4.0) is asserted for all seeds.

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
