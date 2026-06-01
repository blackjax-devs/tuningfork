"""Nightly benchmark runner: parse results, persist, check for regressions.

Entry point for the nightly CI step that previously lived as inline Python
in benchmark.yml.  Invoked as::

    python -m benchmarks.run_nightly [--results-dir DIR] [--dry-run]

Reads the pytest-benchmark JSON written by the benchmark step, extracts
per-seed metrics from ``extra_info["per_seed_metrics"]``, persists to the
``benchmark-results`` branch, runs the regression check, and emits GitHub
Actions annotations with an appropriate exit code.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RESULTS_DIR = _REPO_ROOT / "benchmark_results_dir"


def parse_benchmark_json(
    bench_file: Path,
    seeds: tuple[int, int, int],
) -> dict[int, dict[str, Any]]:
    """Parse pytest-benchmark JSON and extract per-seed cell metrics.

    Each benchmark writes ``extra_info["per_seed_metrics"]`` as a dict
    ``{seed_str: {metric_key: value}}``.  This function aggregates across
    all benchmark entries and returns ``{seed: {cell_id: metrics}}``.

    Parameters
    ----------
    bench_file
        Path to ``bench_results.json`` written by pytest-benchmark.
    seeds
        The 3 seeds that were run (used to initialise the result dict).

    Returns
    -------
    dict mapping seed → {cell_id → metrics}
    """
    raw = json.loads(bench_file.read_text())
    seed_cells: dict[int, dict[str, Any]] = {s: {} for s in seeds}

    for bm in raw.get("benchmarks", []):
        cell_id = bm.get("name", "unknown")
        per_seed = bm.get("extra_info", {}).get("per_seed_metrics", {})
        for seed_str, metrics in per_seed.items():
            try:
                seed = int(seed_str)
            except ValueError:
                continue
            if seed in seed_cells:
                seed_cells[seed][cell_id] = metrics

    return seed_cells


def parse_cell_drift_flags(
    bench_file: Path,
) -> tuple[bool, list[str]]:
    """Read per-cell JAX-drift flags from benchmark JSON and aggregate.

    Each benchmark entry may carry ``extra_info["jax_drift"]["flag"]`` and
    ``extra_info["jax_drift"]["details"]`` written by ``run_benchmark_cell``.
    This function returns ``(any_drifted, all_details)`` across all cells.

    Returns
    -------
    (any_drifted, drift_details)
        any_drifted: True if ≥1 cell reported a JAX-drift signal.
        drift_details: flat list of human-readable detail strings.
    """
    raw = json.loads(bench_file.read_text())
    any_drifted = False
    all_details: list[str] = []

    for bm in raw.get("benchmarks", []):
        jax_drift = bm.get("extra_info", {}).get("jax_drift", {})
        if jax_drift.get("flag"):
            any_drifted = True
            all_details.extend(jax_drift.get("details", []))

    return any_drifted, all_details


def build_today_results(
    seed_cells: dict[int, dict[str, Any]],
    run_date: date,
    env: dict[str, str],
) -> list[dict[str, Any]]:
    """Convert {seed: cells} into the per-seed result dicts used by the check.

    Parameters
    ----------
    seed_cells
        Output of ``parse_benchmark_json``.
    run_date
        The date of the nightly run.
    env
        Environment fingerprint (blackjax_sha, jax_version, etc.).
    """
    results = []
    for seed, cells in seed_cells.items():
        if cells:
            results.append(
                {
                    "seed": seed,
                    "date": run_date.isoformat(),
                    "env": env,
                    "cells": cells,
                }
            )
    return results


def main(argv: list[str] | None = None) -> int:
    """Run the nightly benchmark post-processing.

    Returns exit code: 1 for REGRESSION, 0 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Nightly benchmark runner: parse, persist, regression-check"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_DEFAULT_RESULTS_DIR,
        help=f"Directory with bench_results.json (default: {_DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and check but do not write to benchmark-results branch",
    )
    parser.add_argument(
        "--print-seeds",
        action="store_true",
        help="Print today's seeds and exit (for use in workflow informational step)",
    )
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------ #
    # Seed computation
    # ------------------------------------------------------------------ #
    from benchmarks._benchmark_helpers import get_nightly_seeds  # noqa: PLC0415

    run_date = date.today()
    seeds = get_nightly_seeds(run_date)

    if args.print_seeds:
        print(f"Nightly seeds: {list(seeds)}", flush=True)
        # Also write to GITHUB_OUTPUT if available
        import os  # noqa: PLC0415

        gho = os.environ.get("GITHUB_OUTPUT")
        if gho:
            with open(gho, "a") as f:
                f.write(f"seeds={' '.join(str(s) for s in seeds)}\n")
        return 0

    # ------------------------------------------------------------------ #
    # Parse benchmark results
    # ------------------------------------------------------------------ #
    bench_file = args.results_dir / "bench_results.json"
    if not bench_file.exists():
        print(
            f"ERROR: {bench_file} not found — did the benchmark step run at all?",
            file=sys.stderr,
        )
        return 1

    # Guard: zero benchmark entries → pytest ran but collected no cells.
    # This is as bad as a missing file — the nightly produced no signal.
    import json as _json  # noqa: PLC0415

    _raw = _json.loads(bench_file.read_text())
    if not _raw.get("benchmarks"):
        print(
            "ERROR: bench_results.json has 0 benchmark entries — no cells ran "
            "(check pytest collection, -m benchmark filter, and suite selection).",
            file=sys.stderr,
        )
        return 1

    seed_cells = parse_benchmark_json(bench_file, seeds)

    # ------------------------------------------------------------------ #
    # Persistence: load prior, store today's results
    # ------------------------------------------------------------------ #
    from benchmarks._result_persistence import (  # noqa: PLC0415
        get_env_fingerprint,
        load_prior_result,
        load_recent_results,
        store_result,
    )

    env = get_env_fingerprint()
    today_results = build_today_results(seed_cells, run_date, env)

    if not today_results:
        print(
            "ERROR: no per-seed metrics extracted from benchmark JSON "
            "(cells ran but extra_info['per_seed_metrics'] is absent or empty).",
            file=sys.stderr,
        )
        return 1

    # Load-before-store: compare against prior THEN overwrite
    overlap_seeds = list(seeds)[:2]  # {date-1, date}
    prior_results: dict[int, dict[str, Any]] = {}
    for seed in overlap_seeds:
        r = load_prior_result(seed)
        if r is not None:
            prior_results[seed] = r

    recent = load_recent_results(n=3)

    if not args.dry_run:
        for result in today_results:
            ok = store_result(result["seed"], result["cells"], run_date)
            status = "ok" if ok else "WARN: push failed"
            n = len(result["cells"])
            print(f"Stored seed={result['seed']} ({n} cells) [{status}]", flush=True)

    # ------------------------------------------------------------------ #
    # Bootstrap night: no priors yet — store only
    # ------------------------------------------------------------------ #
    if not prior_results:
        print("Bootstrap night — no priors; storing results only.", flush=True)
        return 0

    # ------------------------------------------------------------------ #
    # Regression check
    # ------------------------------------------------------------------ #
    from benchmarks._regression_check import (  # noqa: PLC0415
        exit_with_verdict,
        run_regression_check,
    )

    reg_result = run_regression_check(today_results, prior_results, recent)

    # Augment with JAX-drift signal (additive, non-blocking — verdict unchanged)
    jax_drift_flag, jax_drift_details = parse_cell_drift_flags(bench_file)
    reg_result.jax_drift_flag = jax_drift_flag
    reg_result.jax_drift_details = jax_drift_details

    return exit_with_verdict(reg_result)


if __name__ == "__main__":
    raise SystemExit(main())
