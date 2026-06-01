"""Benchmark session fixtures.

JIT warmup pass: runs one throwaway cell at session start to warm the XLA
JIT cache, eliminating the cold-start ESS drift (up to 63% per mock).
Runs once for the entire session — all 3 date-derived seeds run warm.
"""

from __future__ import annotations

import pytest

from benchmarks._benchmark_helpers import run_jit_warmup


@pytest.fixture(scope="session", autouse=True)
def jit_warmup_pass():
    """Warm the XLA JIT cache once per session before any benchmark cell runs."""
    run_jit_warmup()  # uses today's date-seed implicitly
