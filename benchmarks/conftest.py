"""Benchmark session fixtures.

JIT warmup pass: runs one throwaway cell at session start to warm the XLA
JIT cache, eliminating the cold-start ESS drift (up to 63% per mock).
Runs once for the entire session — all 3 date-derived seeds run warm.
"""

from __future__ import annotations

import jax
import pytest

from benchmarks._benchmark_helpers import run_jit_warmup

# ---------------------------------------------------------------------------
# JAX persistent-cache gate: allow all CPU compilations to be cached
#
# JAX's default ``jax_persistent_cache_min_compile_time_secs=1.0`` silently
# gates out CPU-mode XLA compilations (typically 50–200ms on CI runners and
# local machines).  The result: ``JAX_COMPILATION_CACHE_DIR`` is set but the
# cache stays at ~4 KB (empty) — confirmed locally: a representative compile
# took 76ms, well below the 1.0s threshold.
#
# Effect on the nightly: every night runs cold → all 31 fast + 2 slow cells
# compile from scratch → wall time blows past the 180-min cap.  The
# restore+save split added in PR #185 could not help while the writes were
# being suppressed.
#
# Note: ``jax.clear_caches()`` (called in ``clear_xla_caches_between_cells``
# below) is in-memory only — it does NOT touch the on-disk persistent cache
# (confirmed from JAX 0.10.1 source).  So setting min_compile_time_secs=0
# here is sufficient: compilations hit disk, survive per-cell clear_caches(),
# and are restored the following night via the CI cache step.
# ---------------------------------------------------------------------------
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

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
    # Empty: laplace_hmc and laplace_mhmc were OOM victims, not genuine crashers.
    # jax.clear_caches() (clear_xla_caches_between_cells fixture below) fixes the
    # root cause (XLA compile-cache accumulation → memory exhaustion by ~cell 18).
    # With clear_caches() active both cells run clean — verified 2026-06-01.
    # Issue #137 closed: the SIGABRT was the OOM hitting a harder abort threshold.
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
