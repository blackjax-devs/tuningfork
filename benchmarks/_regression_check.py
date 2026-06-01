"""Cross-date regression check for the nightly benchmark suite.

Implements the revised criterion from worklog/threads/benchmark-regression-criterion.md
(§ MOCK VALIDATION RESULTS — mock superseded the pre-mock 0/1/2 exact-comparison scheme).

**Revised criterion:**

Primary signal (correctness):
  ANY cell at ANY seed: max_abs_mean_z >= 4.0 → REGRESSION or ENVIRONMENT_DRIFT
  (Single seed is sufficient — correctness is binary; if the sampler targets the
  wrong distribution, even one seed proves it.)

  Environment triage:
    - If z≥4.0 AND today's env fingerprint matches prior night → REGRESSION
    - If z≥4.0 AND env changed (jax_version or runner_image) → ENVIRONMENT_DRIFT

Secondary signal (ESS trend):
  min_bulk_ess < 50% of prior-3-night median for ≥ 2/3 seeds → REVIEW flag

**DROPPED from 1-to-1 comparison (mock proved unreliable):**
  n_divergences, total_grad_evals, min_bulk_ess, max_abs_mean_z direct prev-vs-today
  comparison — these drift 5-63% and ±1.0 abs between same-seed runs due to
  JIT non-determinism + NUTS U-turn variance. Exact comparison produces a false
  REGRESSION every night.

Verdict:
  GREEN           — no signal (correctness passes, ESS stable)
  REVIEW          — ESS-trend drop ≥2/3 seeds (secondary signal only)
  REGRESSION      — z≥4.0 on any seed, same env as prior night
  ENVIRONMENT_DRIFT — z≥4.0 on any seed, but env fingerprint changed
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass, field
from typing import Any

# Correctness gate (matches _benchmark_helpers._Z_THRESHOLD)
_Z_THRESHOLD = 4.0
# ESS-trend secondary signal
_ESS_TREND_THRESHOLD = 0.50  # <50% of 3-night median → flag


@dataclass
class RegressionResult:
    """Outcome of a single-night regression check."""

    verdict: str  # "GREEN", "REVIEW", "REGRESSION", "ENVIRONMENT_DRIFT"
    details: list[str] = field(default_factory=list)
    correctness_fail: bool = False  # any z ≥ 4.0 (single-seed)
    env_drifted: bool = False
    # JAX-drift signal: additive, non-blocking — never changes the verdict.
    # Populated by run_nightly.py from per-cell extra_info["jax_drift"].
    jax_drift_flag: bool = False
    jax_drift_details: list[str] = field(default_factory=list)


def _env_changed(today_env: dict, prior_env: dict) -> bool:
    """Return True if a meaningful env component changed (JAX or runner image)."""
    for key in ("jax_version", "runner_image"):
        if today_env.get(key) != prior_env.get(key):
            return True
    return False


def check_correctness(
    today_result: dict[str, Any],
    prior_result: dict[str, Any] | None = None,
) -> tuple[bool, bool, list[str]]:
    """Primary signal: any cell with max_abs_mean_z >= 4.0 → correctness fail.

    Also performs ENVIRONMENT_DRIFT triage: if z≥4 fires AND env changed vs
    prior night, it's environment drift — not a blackjax regression.

    Parameters
    ----------
    today_result
        Result dict for one seed run today.
    prior_result
        Prior-night result for the SAME seed (from benchmark-results branch).
        If None (bootstrap night), env triage is skipped.

    Returns
    -------
    (correctness_failed, env_drifted, details)
        correctness_failed: True if any cell z ≥ 4.0
        env_drifted: True if z≥4 BUT env fingerprint changed vs prior
        details: list of human-readable strings
    """
    cells = today_result.get("cells", {})
    correctness_failed = False
    env_drifted_on_fail = False
    details: list[str] = []

    today_env = today_result.get("env", {})
    prior_env = prior_result.get("env", {}) if prior_result is not None else None

    for cell_id, metrics in cells.items():
        z = metrics.get("max_abs_mean_z")
        if z is not None and z >= _Z_THRESHOLD:
            correctness_failed = True

            # Env triage: classify REGRESSION vs ENVIRONMENT_DRIFT
            if prior_env is not None and _env_changed(today_env, prior_env):
                env_drifted_on_fail = True
                details.append(
                    f"  z≥{_Z_THRESHOLD} on {cell_id} (z={z:.3f}) BUT env changed "
                    f"jax {prior_env.get('jax_version')!r}→{today_env.get('jax_version')!r} "
                    f"runner {prior_env.get('runner_image')!r}→{today_env.get('runner_image')!r} "
                    f"→ ENVIRONMENT_DRIFT"
                )
            else:
                details.append(
                    f"  CORRECTNESS FAIL: {cell_id} max_abs_mean_z={z:.3f} ≥ {_Z_THRESHOLD}"
                )

    return correctness_failed, env_drifted_on_fail, details


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
    """Full regression check: correctness (primary) + ESS trend (secondary).

    The pre-mock 0/2/1/2/2/2 exact-metric comparison (``check_seed_pair``) was
    removed: ESS and z drift 5-63% between same-seed runs due to JIT
    non-determinism and NUTS U-turn variance, producing false REGRESSIONs nightly.
    Only the correctness gate (z≥4 per-seed) and ESS trend are reliable signals.

    Parameters
    ----------
    today_results
        List of result dicts for the 3 seeds run today.
    prior_results_by_seed
        Dict mapping seed → prior result dict (from the benchmark-results branch).
        Used ONLY for env-fingerprint triage of correctness failures.
    recent_results
        Last 3 stored results for ESS-trend baseline.  If None, trend check skipped.

    Returns
    -------
    RegressionResult
    """
    result = RegressionResult(verdict="GREEN")
    any_env_drifted = False

    # Primary: correctness check with env-drift triage per seed
    for tr in today_results:
        seed = tr.get("seed")
        prior = prior_results_by_seed.get(int(seed)) if seed is not None else None
        failed, env_drifted, c_details = check_correctness(tr, prior)
        if failed:
            result.correctness_fail = True
            result.details.extend(c_details)
            if env_drifted:
                any_env_drifted = True
            else:
                # At least one correctness fail with same env → REGRESSION
                result.verdict = "REGRESSION"

    # If correctness fired but ALL failures were env-drifted → ENVIRONMENT_DRIFT
    if result.correctness_fail and result.verdict != "REGRESSION":
        result.verdict = "ENVIRONMENT_DRIFT"
        result.env_drifted = True
    elif result.env_drifted is False and any_env_drifted:
        result.env_drifted = True

    # Secondary: ESS trend (additive — only upgrades GREEN to REVIEW)
    if recent_results is not None and result.verdict == "GREEN":
        ess_flagged, ess_details = check_ess_trend(today_results, recent_results)
        if ess_flagged:
            result.details.extend(ess_details)
            result.verdict = "REVIEW"

    return result


def emit_gha_annotations(result: RegressionResult) -> None:
    """Write GitHub Actions annotations to stdout based on verdict + drift flag."""
    # Correctness / regression annotations
    if result.verdict != "GREEN":
        detail_text = (
            "\n".join(result.details[:20]) if result.details else "(no details)"
        )
        if result.verdict == "REGRESSION":
            print(f"::error title=Benchmark Regression::{detail_text}", flush=True)
            if result.correctness_fail:
                print(
                    "::error title=Correctness Gate FAILED::"
                    f"max_abs_mean_z >= {_Z_THRESHOLD} — sampler correctness compromised",
                    flush=True,
                )
        elif result.verdict in ("REVIEW", "ENVIRONMENT_DRIFT"):
            print(
                f"::warning title=Benchmark {result.verdict}::{detail_text}",
                flush=True,
            )

    # JAX-drift annotation: additive, non-blocking, always emitted when present.
    # Surfaces "which blackjax change moved the numerics" context for engineers.
    if result.jax_drift_flag:
        drift_text = (
            "\n".join(result.jax_drift_details[:20])
            if result.jax_drift_details
            else "(no details)"
        )
        print(f"::warning title=JAX Numeric Drift::{drift_text}", flush=True)


def exit_with_verdict(result: RegressionResult) -> int:
    """Return exit code: 1 for REGRESSION, 0 for everything else."""
    emit_gha_annotations(result)
    if result.verdict == "REGRESSION":
        print(
            f"\nBENCHMARK REGRESSION DETECTED (correctness_fail={result.correctness_fail})",
            file=sys.stderr,
        )
        return 1
    print(f"\nBenchmark check: {result.verdict}", file=sys.stderr)
    return 0
