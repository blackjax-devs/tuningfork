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
    # Confirmed JaxRuntimeError 'Failed to materialize symbols' in run 26761857186
    # (2026-06-01) — scoping dispatch.  The XLA JIT state corruption from this cell
    # cascades to all subsequent cells (11 secondary failures).  Skip at collection
    # so XLA state stays clean for the remaining suite.
    # Tracking issue: blackjax-devs/tuningfork#137 (same root-cause family)
    "tier1-eight_schools_ncp-low__laplace_mhmc__window_adaptation_dense_imm-e2e": (
        "JaxRuntimeError 'Failed to materialize symbols' corrupts XLA JIT state "
        "for subsequent cells (run 26761857186); "
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


@pytest.fixture(autouse=True)
def clear_xla_caches_between_cells():
    """Free XLA compiled executables + Python objects after each benchmark cell.

    Root cause identified from live run 26761857186 (2026-06-01):
    JAX does not free XLA compiled executables between cells.  The benchmark
    process accumulates them across all 31 cells × 7 runs = ~217 compilations.
    By ~cell 18 the runner exhausts memory, the next XLA compile fails with
    ``JaxRuntimeError: INTERNAL: Failed to materialize symbols / Cannot allocate
    memory``, and *every subsequent cell* fails fast (cascade).  The
    ``laplace_hmc`` SIGABRT (#137) is the same root cause — the process crosses
    the OOM threshold with a harder abort rather than a clean exception.

    ``jax.clear_caches()`` frees the XLA compilation cache between cells.
    ``gc.collect()`` releases Python objects holding JAX array references.

    This keeps per-cell memory bounded.  Each cell's own compile-warmup step
    (1st of 7 runs) recompiles the XLA executable from scratch before the 6
    timed seed-runs, so all timed runs remain warm.
    """
    yield  # run the benchmark cell
    import gc  # noqa: PLC0415

    try:
        import jax  # noqa: PLC0415

        jax.clear_caches()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
