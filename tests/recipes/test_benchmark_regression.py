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
# Regression check — correctness primary signal
# ---------------------------------------------------------------------------


def _make_result(seed: int, cells: dict) -> dict:
    return {
        "seed": seed,
        "date": "2026-06-01",
        "env": {"jax_version": "0.10.1", "runner_image": "ubuntu-24.04"},
        "cells": cells,
    }


def test_correctness_pass() -> None:
    r = _make_result(20260601, {"cell1": {"max_abs_mean_z": 3.5}})
    failed, _ = check_correctness(r)
    assert not failed


def test_correctness_fail_z_above_threshold() -> None:
    r = _make_result(20260601, {"cell1": {"max_abs_mean_z": 4.1}})
    failed, details = check_correctness(r)
    assert failed
    assert "CORRECTNESS FAIL" in details[0]


def test_correctness_exact_threshold() -> None:
    """z >= 4.0 is a FAIL (criterion: max_abs_mean_z >= 4.0 → REGRESSION)."""
    r_at = _make_result(1, {"c": {"max_abs_mean_z": 4.0}})  # exactly at threshold
    r_above = _make_result(1, {"c": {"max_abs_mean_z": 4.001}})  # above threshold
    r_below = _make_result(1, {"c": {"max_abs_mean_z": 3.999}})  # below threshold
    assert check_correctness(r_at)[0], "z == 4.0 must trigger FAIL (>= threshold)"
    assert check_correctness(r_above)[0], "z > 4.0 must trigger FAIL"
    assert not check_correctness(r_below)[0], "z < 4.0 must NOT trigger FAIL"


# ---------------------------------------------------------------------------
# Regression check — 0/2 / 1/2 / 2/2 verdicts
# ---------------------------------------------------------------------------

_STABLE_CELLS = {
    "cell1": {"max_abs_mean_z": 0.5, "min_bulk_ess": 2000.0},
}
_DEVIATED_CELLS = {
    "cell1": {"max_abs_mean_z": 1.5, "min_bulk_ess": 500.0},  # significant change
}


def test_regression_check_green_0_of_2() -> None:
    """0/2 seeds deviate → GREEN."""
    today = [_make_result(s, _STABLE_CELLS) for s in [20260531, 20260601, 20260602]]
    priors = {
        20260531: _make_result(20260531, _STABLE_CELLS),
        20260601: _make_result(20260601, _STABLE_CELLS),
    }
    result = run_regression_check(today, priors)
    assert result.verdict == "GREEN"
    assert result.seeds_deviated == 0


def test_regression_check_review_1_of_2() -> None:
    """1/2 overlapping seeds deviate → REVIEW."""
    today_cells_1 = {"cell1": {"max_abs_mean_z": 1.5, "min_bulk_ess": 300.0}}
    today = [
        _make_result(20260531, today_cells_1),  # overlapping seed deviated
        _make_result(20260601, _STABLE_CELLS),  # overlapping seed stable
        _make_result(20260602, _STABLE_CELLS),
    ]
    priors = {
        20260531: _make_result(20260531, _STABLE_CELLS),
        20260601: _make_result(20260601, _STABLE_CELLS),
    }
    result = run_regression_check(today, priors)
    assert result.verdict == "REVIEW"
    assert result.seeds_deviated == 1


def test_regression_check_regression_2_of_2_same_env() -> None:
    """2/2 seeds deviate with same env → REGRESSION."""
    today = [_make_result(s, _DEVIATED_CELLS) for s in [20260531, 20260601, 20260602]]
    priors = {
        20260531: _make_result(20260531, _STABLE_CELLS),
        20260601: _make_result(20260601, _STABLE_CELLS),
    }
    result = run_regression_check(today, priors)
    assert result.verdict == "REGRESSION"
    assert result.seeds_deviated == 2
    assert not result.env_drifted


def test_regression_check_env_drift_not_regression() -> None:
    """2/2 seeds deviate but env changed → ENVIRONMENT_DRIFT (not REGRESSION)."""

    def make_env_result(seed, cells, jax_ver):
        r = _make_result(seed, cells)
        r["env"]["jax_version"] = jax_ver
        return r

    today = [
        make_env_result(s, _DEVIATED_CELLS, "0.10.2")
        for s in [20260531, 20260601, 20260602]
    ]
    priors = {
        20260531: make_env_result(20260531, _STABLE_CELLS, "0.10.1"),
        20260601: make_env_result(20260601, _STABLE_CELLS, "0.10.1"),
    }
    result = run_regression_check(today, priors)
    assert result.verdict == "ENVIRONMENT_DRIFT"
    assert result.env_drifted


def test_regression_check_bootstrap_night_no_priors() -> None:
    """Bootstrap night (no priors) → GREEN."""
    today = [_make_result(s, _STABLE_CELLS) for s in [20260531, 20260601, 20260602]]
    result = run_regression_check(today, {})
    assert result.verdict == "GREEN"
    assert result.seeds_deviated == 0


# ---------------------------------------------------------------------------
# ESS trend check
# ---------------------------------------------------------------------------


def test_ess_trend_skips_with_few_prior_nights() -> None:
    """Fewer than 2 prior nights → trend check skipped."""
    today = [_make_result(s, {"c": {"min_bulk_ess": 100.0}}) for s in range(3)]
    flagged, details = check_ess_trend(today, [])  # empty recent
    assert not flagged
    assert any("too few" in d for d in details)


def test_ess_trend_flags_large_drop() -> None:
    """ESS drops to <50% of 3-night median for 2+ seeds → flagged."""
    high_ess_cells = {"c1": {"min_bulk_ess": 2000.0}}
    low_ess_cells = {"c1": {"min_bulk_ess": 100.0}}  # ~5% of 2000

    recent = [_make_result(s, high_ess_cells) for s in [20260528, 20260529, 20260530]]
    today = [_make_result(s, low_ess_cells) for s in [20260531, 20260601, 20260602]]
    flagged, details = check_ess_trend(today, recent)
    assert flagged


def test_ess_trend_stable_not_flagged() -> None:
    """ESS stable at same level → not flagged."""
    cells = {"c1": {"min_bulk_ess": 2000.0}}
    recent = [_make_result(s, cells) for s in [20260528, 20260529, 20260530]]
    today = [_make_result(s, cells) for s in [20260531, 20260601, 20260602]]
    flagged, _ = check_ess_trend(today, recent)
    assert not flagged
