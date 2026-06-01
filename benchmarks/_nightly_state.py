"""Session-level nightly benchmark state.

Shared module (not conftest) so both ``_benchmark_helpers.py`` and the
``test_nightly_results.py`` final test can access it without circular imports.

``PER_SEED_METRICS`` is populated by ``run_benchmark_cell`` during the benchmark
session.  ``test_no_regression_vs_prior`` reads it at session-end to run the
regression check and assert no REGRESSION verdict.
"""

from __future__ import annotations

from typing import Any

# {seed_int: {cell_id: metrics_dict}}
# Populated progressively as each benchmark cell completes.
PER_SEED_METRICS: dict[int, dict[str, Any]] = {}
