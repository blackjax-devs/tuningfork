"""Benchmark session fixtures.

JIT warmup pass: runs one throwaway cell at session start to warm the XLA
JIT cache, eliminating the cold-start ESS drift (up to 63% per mock).
Runs once for the entire session — all 3 date-derived seeds run warm.
"""

from __future__ import annotations

import pytest

from benchmarks._benchmark_helpers import run_jit_warmup

# ---------------------------------------------------------------------------
# Known crashers: SIGABRT (exit 134) in 7-run mode.
#
# These cells abort the host process on repeated execution, killing all
# subsequent cells in the suite.  They are skipped here so the suite
# completes; each entry has a tracking issue for root-cause diagnosis.
#
# Map: bench_id → skip reason
# ---------------------------------------------------------------------------
_KNOWN_CRASHERS: dict[str, str] = {
    # Confirmed SIGABRT (exit 134) in run 26759837294 (2026-06-01).
    # Likely JAX/XLA internal assert triggered by 7-run repeated execution.
    # Tracking issue: blackjax-devs/tuningfork#137
    "tier1-eight_schools_ncp-low__laplace_hmc__window_adaptation_diag_imm-e2e": (
        "SIGABRT in 7-run mode (exit 134, run 26759837294); "
        "tracking: blackjax-devs/tuningfork#137"
    ),
}


def pytest_collection_modifyitems(items: list, config: pytest.Config) -> None:
    """Skip known-crasher benchmark cells to prevent SIGABRT aborting the suite."""
    for item in items:
        node_id = item.nodeid
        for crasher_id, reason in _KNOWN_CRASHERS.items():
            if crasher_id in node_id:
                item.add_marker(
                    pytest.mark.skip(reason=f"Known crasher — {reason}"),
                    append=False,
                )
                break


@pytest.fixture(scope="session", autouse=True)
def jit_warmup_pass():
    """Warm the XLA JIT cache once per session before any benchmark cell runs."""
    run_jit_warmup()  # uses today's date-seed implicitly
