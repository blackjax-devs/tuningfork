"""Tests for benchmark seed-scheme, metric extraction, and regression check.

All tests are @fast (pure logic, no JAX trace).
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from benchmarks._benchmark_helpers import (
    _Z_THRESHOLD,
    compute_max_abs_mean_z,
    get_nightly_seeds,
)
from benchmarks._regression_check import (
    check_correctness,
    check_ess_trend,
    run_regression_check,
)

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# P0 fixes
# ---------------------------------------------------------------------------


def test_z_threshold_is_4() -> None:
    """_Z_THRESHOLD must be 4.0 (mock showed 2.0 false-positives on hard geometries)."""
    assert _Z_THRESHOLD == 4.0


def test_compute_max_abs_mean_z_dict_orientation(tmp_path) -> None:
    """compute_max_abs_mean_z must NOT return None when GT summary is present.

    P0 bug: the function was passing raw summary.json ({"mean":{"x":0}}) to auto_gate
    which expects ({"x":{"mean":0}}).  z was always None.  Verify the fix by mocking
    auto_gate so the restructured dict is what it receives.
    """
    import json

    import numpy as np

    # Write a minimal summary.json
    model_dir = tmp_path / "test_model" / "reference"
    model_dir.mkdir(parents=True)
    summary = {"mean": {"x": 0.0}, "std": {"x": 1.0}}
    (model_dir / "summary.json").write_text(json.dumps(summary))

    # Mock idata with 'x' draws
    idata = MagicMock()
    idata.posterior.data_vars = ["x"]
    idata.posterior.__getitem__ = lambda self, k: MagicMock(values=np.zeros((4, 100)))

    # Mock auto_gate to record what it was called with and return a valid result
    captured: dict = {}

    class _FakeVerdict:
        max_abs_mean_z = 0.3

    def mock_auto_gate(samples, info, *, ground_truth_summaries=None, **kw):
        captured["gt"] = ground_truth_summaries
        return _FakeVerdict()

    with (
        patch(
            "tuningfork.calibration.statistician_gate.auto_gate",
            side_effect=mock_auto_gate,
        ),
        patch("benchmarks._benchmark_helpers._CATALOG_ROOT", tmp_path),
    ):
        result = compute_max_abs_mean_z(idata, "test_model")

    # The fix restructures dict; auto_gate should receive {"x": {"mean": 0.0, "std": 1.0}}
    assert captured.get("gt") == {"x": {"mean": 0.0, "std": 1.0}}
    assert result == 0.3  # non-None confirms the bug is fixed


# ---------------------------------------------------------------------------
# Seed scheme
# ---------------------------------------------------------------------------


def test_get_nightly_seeds_values() -> None:
    """Seeds are int(YYYYMMDD) for {date-1, date, date+1}."""
    d = date(2026, 6, 1)
    seeds = get_nightly_seeds(d)
    assert seeds == (20260531, 20260601, 20260602)


def test_get_nightly_seeds_overlap() -> None:
    """Night D and D+1 share seeds {D, D+1} (2-seed overlap)."""
    d = date(2026, 6, 1)
    d_next = d + timedelta(days=1)
    seeds_d = set(get_nightly_seeds(d))
    seeds_d1 = set(get_nightly_seeds(d_next))
    overlap = seeds_d & seeds_d1
    assert len(overlap) == 2


def test_get_nightly_seeds_year_boundary() -> None:
    """Year boundary: Dec 31 → Jan 1 wraps correctly."""
    d = date(2026, 12, 31)
    seeds = get_nightly_seeds(d)
    assert seeds[0] == 20261230
    assert seeds[1] == 20261231
    assert seeds[2] == 20270101


# ---------------------------------------------------------------------------
# Correctness primary signal with env-drift triage
# ---------------------------------------------------------------------------


def _result(seed: int, cells: dict, jax_ver: str = "0.10.1") -> dict:
    return {
        "seed": seed,
        "date": "2026-06-01",
        "env": {"jax_version": jax_ver, "runner_image": "ubuntu-24.04"},
        "cells": cells,
    }


def test_correctness_pass() -> None:
    r = _result(1, {"c1": {"max_abs_mean_z": 3.9}})
    failed, env_drift, _ = check_correctness(r)
    assert not failed


def test_correctness_fail_at_threshold() -> None:
    """z >= 4.0 fires (inclusive)."""
    r = _result(1, {"c1": {"max_abs_mean_z": 4.0}})
    failed, _, details = check_correctness(r)
    assert failed
    assert "CORRECTNESS FAIL" in details[0]


def test_correctness_fail_above_threshold() -> None:
    r = _result(1, {"c1": {"max_abs_mean_z": 4.5}})
    failed, _, _ = check_correctness(r)
    assert failed


def test_correctness_env_drift_triage() -> None:
    """z≥4 but env changed → env_drifted=True, NOT a regression label."""
    today = _result(1, {"c1": {"max_abs_mean_z": 4.2}}, jax_ver="0.10.2")
    prior = _result(1, {"c1": {"max_abs_mean_z": 0.3}}, jax_ver="0.10.1")
    failed, env_drifted, details = check_correctness(today, prior)
    assert failed
    assert env_drifted
    assert "ENVIRONMENT_DRIFT" in details[0]


def test_correctness_same_env_no_drift() -> None:
    """z≥4 with same env → not env_drifted → caller emits REGRESSION."""
    today = _result(1, {"c1": {"max_abs_mean_z": 4.2}}, jax_ver="0.10.1")
    prior = _result(1, {"c1": {"max_abs_mean_z": 0.3}}, jax_ver="0.10.1")
    failed, env_drifted, details = check_correctness(today, prior)
    assert failed
    assert not env_drifted
    assert "CORRECTNESS FAIL" in details[0]


def test_correctness_no_prior_same_env_assumed() -> None:
    """Bootstrap night (no prior) → z≥4 → no env triage → REGRESSION."""
    today = _result(1, {"c1": {"max_abs_mean_z": 4.2}})
    failed, env_drifted, _ = check_correctness(today, None)
    assert failed
    assert not env_drifted  # no prior → no env comparison → treat as REGRESSION


# ---------------------------------------------------------------------------
# run_regression_check — full verdicts
# ---------------------------------------------------------------------------

_PASS_CELLS = {"c1": {"max_abs_mean_z": 0.5, "min_bulk_ess": 2000.0}}
_FAIL_CELLS = {"c1": {"max_abs_mean_z": 4.5, "min_bulk_ess": 2000.0}}


def test_run_regression_check_green() -> None:
    """All z < 4.0, no ESS trend → GREEN."""
    today = [_result(s, _PASS_CELLS) for s in [20260531, 20260601, 20260602]]
    result = run_regression_check(today, {})
    assert result.verdict == "GREEN"
    assert not result.correctness_fail


def test_run_regression_check_regression_same_env() -> None:
    """z≥4, same env as prior → REGRESSION."""
    today = [_result(20260601, _FAIL_CELLS)]
    prior = {20260601: _result(20260601, _PASS_CELLS)}
    result = run_regression_check(today, prior)
    assert result.verdict == "REGRESSION"
    assert result.correctness_fail
    assert not result.env_drifted


def test_run_regression_check_environment_drift() -> None:
    """z≥4 on ALL seeds, but env changed → ENVIRONMENT_DRIFT (not REGRESSION)."""
    today = [_result(s, _FAIL_CELLS, jax_ver="0.10.2") for s in [20260601, 20260602]]
    prior = {
        20260601: _result(20260601, _PASS_CELLS, jax_ver="0.10.1"),
        20260602: _result(20260602, _PASS_CELLS, jax_ver="0.10.1"),
    }
    result = run_regression_check(today, prior)
    assert result.verdict == "ENVIRONMENT_DRIFT"
    assert result.correctness_fail
    assert result.env_drifted


def test_run_regression_check_mixed_env() -> None:
    """z≥4 on one seed (same env) + z≥4 on another (env changed) → REGRESSION.

    A single same-env correctness failure is sufficient for REGRESSION regardless
    of other seeds' env status.
    """
    today = [
        _result(20260531, _FAIL_CELLS, jax_ver="0.10.1"),  # same env → REGRESSION
        _result(20260601, _FAIL_CELLS, jax_ver="0.10.2"),  # env changed
    ]
    prior = {
        20260531: _result(20260531, _PASS_CELLS, jax_ver="0.10.1"),
        20260601: _result(20260601, _PASS_CELLS, jax_ver="0.10.1"),
    }
    result = run_regression_check(today, prior)
    assert result.verdict == "REGRESSION"


def test_run_regression_check_bootstrap_no_priors() -> None:
    """Bootstrap night (no priors, z passes) → GREEN."""
    today = [_result(s, _PASS_CELLS) for s in [20260531, 20260601, 20260602]]
    result = run_regression_check(today, {})
    assert result.verdict == "GREEN"


def test_run_regression_check_bootstrap_z_fail_no_prior() -> None:
    """Bootstrap night (no priors, z fails) → REGRESSION (no env to compare)."""
    today = [_result(20260601, _FAIL_CELLS)]
    result = run_regression_check(today, {})
    assert result.verdict == "REGRESSION"
    assert result.correctness_fail


# ---------------------------------------------------------------------------
# ESS trend check
# ---------------------------------------------------------------------------


def test_ess_trend_skips_with_few_prior_nights() -> None:
    """Fewer than 2 prior nights → trend check skipped."""
    today = [_result(s, {"c": {"min_bulk_ess": 100.0}}) for s in range(3)]
    flagged, details = check_ess_trend(today, [])
    assert not flagged
    assert any("too few" in d for d in details)


def test_ess_trend_flags_large_drop() -> None:
    """ESS drops to <50% of 3-night median for 2+ seeds → flagged."""
    high_ess = {"c1": {"min_bulk_ess": 2000.0}}
    low_ess = {"c1": {"min_bulk_ess": 100.0}}
    recent = [_result(s, high_ess) for s in [20260528, 20260529, 20260530]]
    today = [_result(s, low_ess) for s in [20260531, 20260601, 20260602]]
    flagged, _ = check_ess_trend(today, recent)
    assert flagged


def test_ess_trend_stable_not_flagged() -> None:
    """ESS stable → not flagged."""
    cells = {"c1": {"min_bulk_ess": 2000.0}}
    recent = [_result(s, cells) for s in [20260528, 20260529, 20260530]]
    today = [_result(s, cells) for s in [20260531, 20260601, 20260602]]
    flagged, _ = check_ess_trend(today, recent)
    assert not flagged


def test_ess_trend_review_only_when_green() -> None:
    """ESS trend adds REVIEW only when verdict would otherwise be GREEN."""
    low_ess = {"c1": {"min_bulk_ess": 100.0}}
    high_ess = {"c1": {"min_bulk_ess": 2000.0}}
    recent = [_result(s, high_ess) for s in [20260528, 20260529, 20260530]]
    today = [_result(s, low_ess) for s in [20260531, 20260601, 20260602]]
    result = run_regression_check(today, {}, recent_results=recent)
    assert result.verdict == "REVIEW"
