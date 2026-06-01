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


# ---------------------------------------------------------------------------
# 3-seed internal loop (single invocation)
# ---------------------------------------------------------------------------


def test_get_nightly_seeds_returns_3() -> None:
    """get_nightly_seeds returns exactly 3 distinct integer seeds."""
    seeds = get_nightly_seeds(date(2026, 6, 1))
    assert len(seeds) == 3
    assert len(set(seeds)) == 3  # all distinct
    assert all(isinstance(s, int) for s in seeds)


# ---------------------------------------------------------------------------
# run_nightly.py entry-point: parse → persist → check
# ---------------------------------------------------------------------------


def test_parse_benchmark_json_extracts_per_seed_metrics(tmp_path) -> None:
    """parse_benchmark_json reads extra_info.per_seed_metrics into {seed: cells}."""
    import json

    from benchmarks.run_nightly import parse_benchmark_json

    seeds = (20260531, 20260601, 20260602)
    bench_data = {
        "benchmarks": [
            {
                "name": "tier1-mvn_10-low__nuts__window_adaptation_diag_imm-calibrated",
                "extra_info": {
                    "per_seed_metrics": {
                        "20260531": {"min_bulk_ess": 1800.0, "max_abs_mean_z": 0.4},
                        "20260601": {"min_bulk_ess": 1850.0, "max_abs_mean_z": 0.3},
                        "20260602": {"min_bulk_ess": 1700.0, "max_abs_mean_z": 0.5},
                    }
                },
            }
        ]
    }
    bench_file = tmp_path / "bench_results.json"
    bench_file.write_text(json.dumps(bench_data))

    result = parse_benchmark_json(bench_file, seeds)

    assert set(result.keys()) == set(seeds)
    assert (
        result[20260601][
            "tier1-mvn_10-low__nuts__window_adaptation_diag_imm-calibrated"
        ]["min_bulk_ess"]
        == 1850.0
    )


def test_parse_benchmark_json_unknown_seeds_ignored(tmp_path) -> None:
    """Seeds not in the seed tuple are silently ignored."""
    import json

    from benchmarks.run_nightly import parse_benchmark_json

    seeds = (20260531, 20260601, 20260602)
    bench_data = {
        "benchmarks": [
            {
                "name": "cell1",
                "extra_info": {
                    "per_seed_metrics": {
                        "20260601": {"min_bulk_ess": 1800.0, "max_abs_mean_z": 0.4},
                        "99999999": {"min_bulk_ess": 1000.0},  # unknown seed
                    }
                },
            }
        ]
    }
    bench_file = tmp_path / "bench_results.json"
    bench_file.write_text(json.dumps(bench_data))

    result = parse_benchmark_json(bench_file, seeds)
    # Only known seeds in output
    assert 99999999 not in result
    assert 20260601 in result


def test_run_nightly_main_green(tmp_path) -> None:
    """main() returns 0 (GREEN) when all z < 4.0 and no ESS trend."""
    import json
    from unittest.mock import patch

    from benchmarks.run_nightly import main

    seeds = (20260531, 20260601, 20260602)
    bench_data = {
        "benchmarks": [
            {
                "name": "cell1",
                "extra_info": {
                    "per_seed_metrics": {
                        str(s): {"min_bulk_ess": 2000.0, "max_abs_mean_z": 0.3}
                        for s in seeds
                    }
                },
            }
        ]
    }
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "bench_results.json").write_text(json.dumps(bench_data))

    with (
        patch("benchmarks._benchmark_helpers.get_nightly_seeds", return_value=seeds),
        patch(
            "benchmarks._result_persistence.get_env_fingerprint",
            return_value={"jax_version": "0.10.1", "runner_image": "ubuntu-24.04"},
        ),
        patch("benchmarks._result_persistence.load_prior_result", return_value=None),
        patch("benchmarks._result_persistence.load_recent_results", return_value=[]),
        patch("benchmarks._result_persistence.store_result", return_value=True),
    ):
        exit_code = main(["--results-dir", str(results_dir), "--dry-run"])

    assert exit_code == 0  # GREEN


def test_run_nightly_main_regression(tmp_path) -> None:
    """main() returns 1 (REGRESSION) when z >= 4.0 with same env as prior."""
    import json
    from unittest.mock import patch

    from benchmarks.run_nightly import main

    seeds = (20260531, 20260601, 20260602)
    bench_data = {
        "benchmarks": [
            {
                "name": "cell1",
                "extra_info": {
                    "per_seed_metrics": {
                        str(s): {"min_bulk_ess": 1800.0, "max_abs_mean_z": 4.5}
                        for s in seeds
                    }
                },
            }
        ]
    }
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "bench_results.json").write_text(json.dumps(bench_data))

    same_env = {"jax_version": "0.10.1", "runner_image": "ubuntu-24.04"}
    prior = {
        "seed": 20260531,
        "date": "2026-05-31",
        "env": same_env,
        "cells": {"cell1": {"max_abs_mean_z": 0.3}},
    }

    with (
        patch("benchmarks._benchmark_helpers.get_nightly_seeds", return_value=seeds),
        patch(
            "benchmarks._result_persistence.get_env_fingerprint", return_value=same_env
        ),
        patch("benchmarks._result_persistence.load_prior_result", return_value=prior),
        patch("benchmarks._result_persistence.load_recent_results", return_value=[]),
        patch("benchmarks._result_persistence.store_result", return_value=True),
    ):
        exit_code = main(["--results-dir", str(results_dir), "--dry-run"])

    assert exit_code == 1  # REGRESSION


def test_run_nightly_main_env_drift_overlap_seeds_only(tmp_path) -> None:
    """Overlap seeds with z≥4.0 + env changed → ENVIRONMENT_DRIFT (not REGRESSION).

    The 3rd seed (date+1) has no prior; only overlap seeds {date-1, date} are
    checked for env drift.  When ALL overlap-seed z≥4 fires have env changed,
    and the 3rd seed either passes OR fires z≥4 without a prior, the verdict
    is ENVIRONMENT_DRIFT only if no same-env z≥4 fires.  This test uses only
    2 seeds (both overlap) so z≥4 on both → ENVIRONMENT_DRIFT.
    """
    import json
    from unittest.mock import patch

    from benchmarks.run_nightly import main

    # Use only 2 seeds — simulates a 2-seed run (both overlap) to avoid the
    # "no prior for 3rd seed → REGRESSION" edge case
    seeds = (20260531, 20260601)
    bench_data = {
        "benchmarks": [
            {
                "name": "cell1",
                "extra_info": {
                    "per_seed_metrics": {
                        str(s): {"min_bulk_ess": 1800.0, "max_abs_mean_z": 4.5}
                        for s in seeds
                    }
                },
            }
        ]
    }
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "bench_results.json").write_text(json.dumps(bench_data))

    today_env = {"jax_version": "0.10.2", "runner_image": "ubuntu-24.04"}
    prior_env = {"jax_version": "0.10.1", "runner_image": "ubuntu-24.04"}

    def make_prior(seed):
        return {
            "seed": seed,
            "date": "2026-05-31",
            "env": prior_env,
            "cells": {"cell1": {"max_abs_mean_z": 0.3}},
        }

    with (
        patch("benchmarks._benchmark_helpers.get_nightly_seeds", return_value=seeds),
        patch(
            "benchmarks._result_persistence.get_env_fingerprint", return_value=today_env
        ),
        patch(
            "benchmarks._result_persistence.load_prior_result",
            side_effect=[make_prior(20260531), make_prior(20260601)],
        ),
        patch("benchmarks._result_persistence.load_recent_results", return_value=[]),
        patch("benchmarks._result_persistence.store_result", return_value=True),
    ):
        exit_code = main(["--results-dir", str(results_dir), "--dry-run"])

    assert exit_code == 0  # ENVIRONMENT_DRIFT (not REGRESSION — env changed)


def test_per_seed_metrics_all_3_seeds_captured(tmp_path) -> None:
    """run_benchmark_cell must capture metrics for all 3 date-derived seeds.

    We mock the JAX-level run so this stays @fast while confirming the loop
    over seeds is wired correctly.
    """
    from datetime import date
    from unittest.mock import MagicMock, patch

    import numpy as np

    from benchmarks._benchmark_helpers import get_nightly_seeds, run_benchmark_cell

    # Build a minimal catalog with a fake recipe
    catalog = tmp_path / "catalog"
    model_dir = catalog / "mvn_10"
    recipe_path = model_dir / "recipes" / "low__nuts__window_adaptation_diag_imm.json"
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(
        '{"model_name":"mvn_10","base_method_name":"nuts","warmup_name":"window_adaptation_diag_imm",'
        '"warmups":[{"name":"window_adaptation_diag_imm","params":{"n_warmup":100,"num_chains":4}}],'
        '"base_method_params":{"step_size":0.3},'
        '"calibration_budget":{"n_warmup":100,"n_samples":100,"num_chains":4},'
        '"gate_evidence":{"auto":{"verdict":"PASS"}},'
        '"effort":"LOW","tuning_seed":20260601}'
    )

    # Mock idata returned by run_recipe_to_idata
    mock_idata = MagicMock()
    mock_idata.posterior.data_vars = []
    mock_idata.sample_stats.ds = {"diverging": MagicMock(values=np.zeros((4, 100)))}

    captured_seeds: list[int] = []

    def mock_run_recipe(
        recipe, *, skip_warmup, n_samples, force_resample_config, _suppress_print
    ):
        if force_resample_config:
            captured_seeds.append(force_resample_config.get("seed", -1))
        return mock_idata

    mock_benchmark = MagicMock()

    # Make benchmark() call the function it receives
    def call_fn(fn):
        fn()
        return None

    mock_benchmark.side_effect = call_fn
    mock_benchmark.extra_info = {}

    run_date = date(2026, 6, 1)
    expected_seeds = set(get_nightly_seeds(run_date))

    mock_recipe = MagicMock()
    _dummy_metrics = {
        "min_bulk_ess": 1800.0,
        "max_abs_mean_z": 0.4,
        "n_divergences": 0,
        "runtime_warmup_s": 1.0,
        "runtime_sample_s": 0.0,
        "correctness_passed": True,
    }
    with (
        patch("benchmarks._benchmark_helpers._CATALOG_ROOT", catalog),
        patch("tuningfork.catalog.inspect.load_recipe", return_value=mock_recipe),
        patch(
            "tuningfork.recipes._recipe_runner.run_recipe_to_idata",
            side_effect=mock_run_recipe,
        ),
        patch(
            "benchmarks._benchmark_helpers.extract_cell_metrics",
            return_value=_dummy_metrics,
        ),
        patch("benchmarks._benchmark_helpers.compute_max_abs_mean_z", return_value=0.5),
    ):
        per_seed = run_benchmark_cell(
            mock_benchmark,
            "mvn_10",
            "low__nuts__window_adaptation_diag_imm.json",
            "calibrated",
            run_date=run_date,
        )

    # All 3 seeds must be present in the returned dict
    assert (
        set(per_seed.keys()) == expected_seeds
    ), f"Expected seeds {expected_seeds}, got {set(per_seed.keys())}"

    # per_seed_metrics must be stored in extra_info
    assert "per_seed_metrics" in mock_benchmark.extra_info
    assert {
        int(k) for k in mock_benchmark.extra_info["per_seed_metrics"].keys()
    } == expected_seeds
