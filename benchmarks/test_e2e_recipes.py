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
  - horseshoe × dmhmc × dense_imm e2e:       75s (extreme geometry)

Runs via:
    make benchmark    # fast + e2e (full nightly suite)

DO NOT add these to ``make benchmark-fast`` or per-PR triggers — the wall
time makes them unsuitable for quick local checks.
"""
from __future__ import annotations

from typing import Any

import pytest

from benchmarks._benchmark_helpers import bench_id, run_benchmark_cell
from benchmarks.config import SLOW_CELLS


@pytest.mark.benchmark(group="recipes-e2e")
@pytest.mark.parametrize(
    "tier,model_name,recipe_file,mode",
    SLOW_CELLS,
    ids=[bench_id(c) for c in SLOW_CELLS],
)
def test_recipe_e2e_perf(
    benchmark: Any,
    tier: str,
    model_name: str,
    recipe_file: str,
    mode: str,
) -> None:
    """Benchmark a slow e2e recipe's sampler and assert GT-correctness.

    These cells take >60s in CI and are nightly-only. Timing is measured by
    pytest-benchmark (1 timed run per cell). GT-correctness (max_abs_mean_z
    < 2.0) is asserted after the timed run.
    """
    run_benchmark_cell(benchmark, model_name, recipe_file, mode)
