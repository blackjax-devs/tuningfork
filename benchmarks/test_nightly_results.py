"""Final nightly benchmark test: regression check vs prior night.

Runs last alphabetically (n > f > e) after all ``test_fast_recipes.py`` and
``test_e2e_recipes.py`` cells have populated ``_nightly_state.PER_SEED_METRICS``.

Flow:
1. Skips if PER_SEED_METRICS is empty (dev / non-nightly run).
2. Writes today's per-seed result to ``benchmark_results_dir/nightly_result.json``
   (the CI shell step then pushes this to the ``benchmark-results`` branch via git).
3. Loads the prior-night results from the ``benchmark-results`` branch.
4. Runs ``run_regression_check`` — the test FAILS (and CI reds) on REGRESSION;
   REVIEW / ENVIRONMENT_DRIFT emit ``::warning::`` but the test passes.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

_RESULTS_DIR = Path(__file__).resolve().parents[1] / "benchmark_results_dir"
_RESULT_FILE = _RESULTS_DIR / "nightly_result.json"


def test_no_regression_vs_prior() -> None:
    """Assert the nightly benchmark produces no REGRESSION verdict.

    - Skips silently when no benchmark data was collected (dev / non-nightly run).
    - Writes today's result to ``_RESULT_FILE`` for the CI git-push step.
    - Emits GH Actions ``::warning::`` for REVIEW / ENVIRONMENT_DRIFT but passes.
    - Fails with a clear message on REGRESSION so CI goes red.
    """
    from benchmarks._nightly_state import PER_SEED_METRICS

    if not PER_SEED_METRICS:
        pytest.skip("No per-seed metrics collected — not a nightly benchmark run")

    from benchmarks._benchmark_helpers import get_nightly_seeds
    from benchmarks._regression_check import emit_gha_annotations, run_regression_check
    from benchmarks._result_persistence import (
        get_env_fingerprint,
        load_prior_result,
        load_recent_results,
    )

    run_date = date.today()
    seeds = get_nightly_seeds(run_date)
    env = get_env_fingerprint()

    # Build per-seed result dicts from in-process state
    today_results: list[dict[str, Any]] = [
        {
            "seed": seed,
            "date": run_date.isoformat(),
            "env": env,
            "cells": cells,
        }
        for seed, cells in PER_SEED_METRICS.items()
        if cells
    ]

    if not today_results:
        pytest.skip("PER_SEED_METRICS populated but all cell dicts empty")

    # --- Write result file (CI git-push step consumes this) ---
    _RESULTS_DIR.mkdir(exist_ok=True)
    result_payload = {
        "date": run_date.isoformat(),
        "seeds": list(seeds),
        "env": env,
        "per_seed": {str(s): c for s, c in PER_SEED_METRICS.items()},
    }
    _RESULT_FILE.write_text(json.dumps(result_payload, indent=2) + "\n")

    # --- Load prior results for env-drift triage ---
    overlap_seeds = list(seeds)[:2]  # {date-1, date}
    prior_results: dict[int, dict[str, Any]] = {}
    for seed in overlap_seeds:
        r = load_prior_result(seed)
        if r is not None:
            prior_results[seed] = r

    if not prior_results:
        # Bootstrap night — no priors yet; result written, no regression check
        print("Bootstrap night — no priors; result written, regression check skipped.")
        return

    recent = load_recent_results(n=3)
    reg_result = run_regression_check(today_results, prior_results, recent)

    # Emit GH Actions annotations (warning for REVIEW/ENVIRONMENT_DRIFT)
    emit_gha_annotations(reg_result)

    # FAIL the test (and CI) on REGRESSION; everything else passes
    detail_str = "\n".join(reg_result.details)
    assert (
        reg_result.verdict != "REGRESSION"
    ), f"Benchmark REGRESSION detected:\n{detail_str}"
