"""Benchmark session fixtures.

JIT warmup pass: runs one throwaway cell at session start to warm the XLA
JIT cache, eliminating the cold-start ESS drift (up to 63% per mock).
"""

from __future__ import annotations

import os

import pytest

from benchmarks._benchmark_helpers import _BENCHMARK_SEED, run_jit_warmup


@pytest.fixture(scope="session", autouse=True)
def jit_warmup_pass():
    """Warm the XLA JIT cache once per session before any benchmark cell runs."""
    seed = int(os.environ.get("BENCHMARK_SEED", str(_BENCHMARK_SEED)))
    run_jit_warmup(seed=seed)
