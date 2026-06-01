"""Cross-date regression check for the nightly benchmark suite.

Implements the locked criterion from worklog/threads/benchmark-regression-criterion.md:

Primary signal (correctness):
  ANY cell at ANY seed: max_abs_mean_z >= 4.0 → immediate REGRESSION FAIL

Secondary signal (ESS trend):
  min_bulk_ess < 50% of prior-3-night median for ≥ 2/3 seeds → REVIEW flag

Environment disambiguation:
  If env fingerprint changed (jax_version OR runner_image) between today and prior:
  → label deviation as ENVIRONMENT_DRIFT, not REGRESSION

Verdict:
  0/2 overlap seeds deviate  → GREEN  (no action)
  1/2 deviate                → REVIEW (GH Actions annotation ⚠️)
  2/2 same-env deviate       → REGRESSION (GH step FAIL + auto-issue)
  2/2 env-changed            → ENVIRONMENT_DRIFT (annotation only)
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass, field
from typing import Any

# Tolerance thresholds (from criterion spec)
_EXACT_TOL = 1e-4  # 0.01% relative tolerance for exact metrics
_RUNTIME_TOL = 0.30  # 30% relative tolerance for soft runtime flag
_ESS_TREND_THRESHOLD = 0.50  # <50% of 3-night median → ESS trend flag
_Z_THRESHOLD = 4.0  # primary correctness gate (same as _benchmark_helpers)


@dataclass
class RegressionResult:
    """Outcome of a single-night regression check."""

    verdict: str  # "GREEN", "REVIEW", "REGRESSION", "ENVIRONMENT_DRIFT"
    details: list[str] = field(default_factory=list)
    correctness_fail: bool = False  # any z ≥ 4.0 (single-seed)
    seeds_deviated: int = 0
    env_drifted: bool = False


def _env_changed(today_env: dict, prior_env: dict) -> bool:
    """Return True if a meaningful env component changed (JAX or runner image)."""
    for key in ("jax_version", "runner_image"):
        if today_env.get(key) != prior_env.get(key):
            return True
    return False


def _metric_deviated(today_val: Any, prior_val: Any, tol: float = _EXACT_TOL) -> bool:
    """Return True if the metric deviated beyond tolerance."""
    if today_val is None or prior_val is None:
        return False
    try:
        denom = max(1.0, abs(float(prior_val)))
        return abs(float(today_val) - float(prior_val)) / denom > tol
    except (TypeError, ValueError):
        return False


def check_seed_pair(
    today_result: dict[str, Any],
    prior_result: dict[str, Any],
) -> tuple[bool, bool, list[str]]:
    """Check one overlapping seed for deviations.

    Returns (seed_deviated, env_changed, details_lines).
    """
    details: list[str] = []
    deviated = False

    today_env = today_result.get("env", {})
    prior_env = prior_result.get("env", {})
    env_changed = _env_changed(today_env, prior_env)

    today_cells = today_result.get("cells", {})
    prior_cells = prior_result.get("cells", {})

    for cell_id in today_cells:
        if cell_id not in prior_cells:
            continue
        tc = today_cells[cell_id]
        pc = prior_cells[cell_id]

        # z-score: any deviation → flag (directional signal, not just threshold)
        if _metric_deviated(tc.get("max_abs_mean_z"), pc.get("max_abs_mean_z")):
            deviated = True
            details.append(
                f"  {cell_id}: max_abs_mean_z "
                f"{pc.get('max_abs_mean_z')!r} → {tc.get('max_abs_mean_z')!r}"
            )

        # ESS: any meaningful drop
        if _metric_deviated(tc.get("min_bulk_ess"), pc.get("min_bulk_ess")):
            deviated = True
            details.append(
                f"  {cell_id}: min_bulk_ess "
                f"{pc.get('min_bulk_ess')!r} → {tc.get('min_bulk_ess')!r}"
            )

    return deviated, env_changed, details


def check_correctness(
    today_result: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Primary signal: any cell with max_abs_mean_z >= 4.0 → immediate FAIL.

    Returns (correctness_failed, details).
    """
    cells = today_result.get("cells", {})
    failed = False
    details: list[str] = []
    for cell_id, metrics in cells.items():
        z = metrics.get("max_abs_mean_z")
        if z is not None and z >= _Z_THRESHOLD:
            failed = True
            details.append(
                f"  CORRECTNESS FAIL: {cell_id} max_abs_mean_z={z:.3f} ≥ {_Z_THRESHOLD}"
            )
    return failed, details


def check_ess_trend(
    today_results: list[dict[str, Any]],
    recent_results: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Secondary signal: ESS < 50% of 3-night median for ≥ 2/3 seeds → REVIEW.

    ``today_results`` is a list of dicts for the 3 seeds run today.
    ``recent_results`` is the last 3 stored results (for the median baseline).

    Returns (ess_trend_flagged, details).
    """
    if len(recent_results) < 2:
        return False, ["  ESS trend: too few prior nights (< 2); skipping"]

    # Compute per-cell 3-night median ESS baseline
    all_cell_ids: set[str] = set()
    for r in recent_results:
        all_cell_ids.update(r.get("cells", {}).keys())

    ess_medians: dict[str, float] = {}
    for cell_id in all_cell_ids:
        prior_ess = [
            r["cells"][cell_id].get("min_bulk_ess")
            for r in recent_results
            if cell_id in r.get("cells", {})
            and r["cells"][cell_id].get("min_bulk_ess") is not None
        ]
        if prior_ess:
            ess_medians[cell_id] = statistics.median(prior_ess)

    if not ess_medians:
        return False, []

    # Check how many of today's seeds are below 50% threshold
    seeds_below = 0
    details: list[str] = []
    for result in today_results:
        cells = result.get("cells", {})
        any_below = False
        for cell_id, metrics in cells.items():
            ess = metrics.get("min_bulk_ess")
            median = ess_medians.get(cell_id)
            if (
                ess is not None
                and median is not None
                and ess < median * _ESS_TREND_THRESHOLD
            ):
                any_below = True
                details.append(
                    f"  ESS trend: {cell_id} ess={ess:.1f} < 50% of median={median:.1f}"
                )
        if any_below:
            seeds_below += 1

    flagged = seeds_below >= max(2, len(today_results) * 2 // 3)
    return flagged, details


def run_regression_check(
    today_results: list[dict[str, Any]],
    prior_results_by_seed: dict[int, dict[str, Any]],
    recent_results: list[dict[str, Any]] | None = None,
) -> RegressionResult:
    """Full regression check: correctness + exact-metric comparison + ESS trend.

    Parameters
    ----------
    today_results
        List of result dicts for the 3 seeds run today.
    prior_results_by_seed
        Dict mapping seed → prior result dict (from the benchmark-results branch).
    recent_results
        Last 3 stored results for ESS-trend baseline.  If None, trend check skipped.

    Returns
    -------
    RegressionResult
    """
    result = RegressionResult(verdict="GREEN")

    # Primary: correctness check (any z ≥ 4.0 on any seed → immediate FAIL)
    for tr in today_results:
        failed, c_details = check_correctness(tr)
        if failed:
            result.correctness_fail = True
            result.details.extend(c_details)
            result.verdict = "REGRESSION"

    # Exact-metric cross-seed comparison
    seeds_deviated = 0
    any_env_changed = False
    for tr in today_results:
        seed = tr.get("seed")
        if seed is None:
            continue
        pr = prior_results_by_seed.get(int(seed))
        if pr is None:
            continue  # no prior for this seed (bootstrap night)
        deviated, env_chg, details = check_seed_pair(tr, pr)
        if deviated:
            seeds_deviated += 1
            result.details.extend(details)
        if env_chg:
            any_env_changed = True

    result.seeds_deviated = seeds_deviated
    result.env_drifted = any_env_changed

    # Set verdict (don't downgrade from REGRESSION)
    if result.verdict != "REGRESSION":
        if seeds_deviated >= 2:
            if any_env_changed:
                result.verdict = "ENVIRONMENT_DRIFT"
                result.details.insert(
                    0,
                    "  2/2 seeds deviated but env changed → ENVIRONMENT_DRIFT (not REGRESSION)",
                )
            else:
                result.verdict = "REGRESSION"
        elif seeds_deviated == 1:
            result.verdict = "REVIEW"

    # Secondary: ESS trend (additive flag — doesn't upgrade GREEN beyond REVIEW)
    if recent_results is not None:
        ess_flagged, ess_details = check_ess_trend(today_results, recent_results)
        if ess_flagged:
            result.details.extend(ess_details)
            if result.verdict == "GREEN":
                result.verdict = "REVIEW"

    return result


def emit_gha_annotations(result: RegressionResult) -> None:
    """Write GitHub Actions annotations to stdout based on verdict."""
    if result.verdict == "GREEN":
        return

    detail_text = "\n".join(result.details[:20]) if result.details else "(no details)"

    if result.verdict == "REGRESSION":
        print(f"::error title=Benchmark Regression::{detail_text}", flush=True)
        if result.correctness_fail:
            print(
                "::error title=Correctness Gate FAILED::"
                f"max_abs_mean_z ≥ {_Z_THRESHOLD} — sampler correctness compromised",
                flush=True,
            )
    elif result.verdict in ("REVIEW", "ENVIRONMENT_DRIFT"):
        print(
            f"::warning title=Benchmark {result.verdict}::{detail_text}",
            flush=True,
        )


def exit_with_verdict(result: RegressionResult) -> int:
    """Return exit code: 1 for REGRESSION, 0 for everything else."""
    emit_gha_annotations(result)
    if result.verdict == "REGRESSION":
        print(
            f"\nBENCHMARK REGRESSION DETECTED (seeds_deviated={result.seeds_deviated}, "
            f"correctness_fail={result.correctness_fail})",
            file=sys.stderr,
        )
        return 1
    print(
        f"\nBenchmark check: {result.verdict} "
        f"(seeds_deviated={result.seeds_deviated})",
        file=sys.stderr,
    )
    return 0
