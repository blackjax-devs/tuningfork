# Copyright 2026- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for bjx_bench.reference._posteriordb_xcheck.

Covers:
1. XCheckResult schema — constructs and serialises to JSON correctly.
2. Mock-pass — our_summaries identical to synthetic stan_draws → passed=True.
3. Mock-fail — our_summaries differ by 0.5 std from synthetic → passed=False,
   failed_dims non-empty.
4. Posteriordb unavailable — invalid posteriordb_id → not-checked result with
   error string in failed_dims.

Tests are marked ``fast`` (no MCMC chains).  The posteriordb Python client
is available in this venv (checked at import time) but we do NOT require a
live posteriordb database for tests 1–3; those tests use the
``posteriordb_root`` argument to point at a synthetic in-memory-like path or
monkeypatch the client.  Test 4 relies on the graceful-error path.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bjx_bench.reference._posteriordb_xcheck import (
    XCheckResult,
    cross_check_against_posteriordb,
)

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Helpers: synthetic stan_draws fixture
# ---------------------------------------------------------------------------


def _make_stan_draws(
    params: dict[str, list[float]],
    n_chains: int = 2,
) -> dict[str, dict[str, list[float]]]:
    """Build a synthetic stan_draws dict with ``n_chains`` identical chains.

    Parameters
    ----------
    params
        Dict of {param_name: [draw_0, draw_1, ...]}.
    n_chains
        Number of chains (each carries the same draws; for testing only).

    Returns
    -------
    dict like ``{"chain:1": {...}, "chain:2": {...}}``
    """
    return {f"chain:{i + 1}": dict(params) for i in range(n_chains)}


def _make_our_summaries(
    means: dict[str, float],
    stds: dict[str, float],
) -> dict[str, dict[str, object]]:
    """Build a synthetic our_summaries dict from scalar mean/std pairs."""
    return {
        site: {
            "mean": np.array([mean]),
            "std": np.array([std]),
            "q05": np.array([mean - 1.645 * std]),
            "q95": np.array([mean + 1.645 * std]),
        }
        for site, (mean, std) in {
            site: (means[site], stds[site]) for site in means
        }.items()
    }


# ---------------------------------------------------------------------------
# Test 1: XCheckResult schema
# ---------------------------------------------------------------------------


class TestXCheckResultSchema:
    """XCheckResult constructs and serialises correctly."""

    def _make_result(self, **overrides: object) -> XCheckResult:
        defaults: dict[str, object] = dict(
            model_name="test_model",
            posteriordb_id="test-posterior",
            passed=True,
            n_dims_compared=3,
            failed_dims=(),
            max_abs_mean_z=0.5,
            max_std_ratio_dev=0.02,
        )
        defaults.update(overrides)
        return XCheckResult(**defaults)  # type: ignore[arg-type]

    def test_constructs_passed(self) -> None:
        """XCheckResult constructs with passed=True and empty failed_dims."""
        r = self._make_result()
        assert r.passed is True
        assert r.failed_dims == ()
        assert r.n_dims_compared == 3
        assert r.max_abs_mean_z == pytest.approx(0.5)
        assert r.max_std_ratio_dev == pytest.approx(0.02)

    def test_constructs_failed(self) -> None:
        """XCheckResult constructs with passed=False and non-empty failed_dims."""
        r = self._make_result(
            passed=False,
            failed_dims=("mu[0]", "theta[2]"),
            max_abs_mean_z=3.7,
            max_std_ratio_dev=0.12,
        )
        assert r.passed is False
        assert "mu[0]" in r.failed_dims
        assert r.max_abs_mean_z == pytest.approx(3.7)

    def test_frozen(self) -> None:
        """XCheckResult is frozen — mutation raises FrozenInstanceError."""
        r = self._make_result()
        with pytest.raises(Exception):
            r.passed = False  # type: ignore[misc]

    def test_save_writes_json(self, tmp_path: Path) -> None:
        """XCheckResult.save writes valid JSON with correct content."""
        r = self._make_result(
            passed=False,
            failed_dims=("x[0]",),
            max_abs_mean_z=2.5,
        )
        out = tmp_path / "result.json"
        r.save(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["model_name"] == "test_model"
        assert data["passed"] is False
        assert "x[0]" in data["failed_dims"]
        assert isinstance(data["failed_dims"], list)  # tuple → list for JSON
        assert data["max_abs_mean_z"] == pytest.approx(2.5)

    def test_save_ends_with_newline_and_is_idempotent(self, tmp_path: Path) -> None:
        """XCheckResult.save writes a POSIX-clean file ending in '\\n' and is idempotent.

        Without the trailing newline, the `fix end of files` pre-commit hook
        rewrites the file on the next pass and every test run leaves a
        cosmetic 1-byte diff. Regression test for bug found pre-Phase-5
        cleanup (see _posteriordb_xcheck.py::XCheckResult.save).
        """
        r = self._make_result()
        out = tmp_path / "newline_check.json"
        r.save(out)
        text_first = out.read_text()
        assert text_first.endswith("\n"), "save() output must end with newline"
        # Idempotency: saving again produces the same bytes.
        r.save(out)
        text_second = out.read_text()
        assert text_first == text_second

    def test_save_nan_fields(self, tmp_path: Path) -> None:
        """XCheckResult.save handles math.nan fields (JSON stores as null via Python)."""
        r = self._make_result(
            passed=False,
            n_dims_compared=0,
            failed_dims=("no-common-params",),
            max_abs_mean_z=math.nan,
            max_std_ratio_dev=math.nan,
        )
        out = tmp_path / "nan_result.json"
        r.save(out)
        out.read_text()  # smoke check the file is readable
        # NaN in Python json.dumps becomes null or NaN depending on allow_nan
        # We do NOT restrict this; just check the file is readable.
        # Python's json module writes NaN as "NaN" by default (non-standard JSON).
        assert out.exists()

    def test_all_fields_accessible(self) -> None:
        """All seven fields are readable on XCheckResult."""
        r = self._make_result()
        _ = r.model_name
        _ = r.posteriordb_id
        _ = r.passed
        _ = r.n_dims_compared
        _ = r.failed_dims
        _ = r.max_abs_mean_z
        _ = r.max_std_ratio_dev


# ---------------------------------------------------------------------------
# Test 2: Mock-pass — identical summaries
# ---------------------------------------------------------------------------


class TestMockPass:
    """Mock-pass: our_summaries identical to synthetic stan_draws → passed=True."""

    def test_scalar_param_passes(self) -> None:
        """Single scalar param, identical mean/std → passed=True, no failed dims."""
        # Stan draws: 1000 samples from N(2.0, 0.5)
        rng = np.random.default_rng(0)
        samples = rng.normal(loc=2.0, scale=0.5, size=1000)
        stan_draws = _make_stan_draws({"mu": samples.tolist()})

        # Our summaries: exact same mean/std (no discrepancy)
        our_summaries = {
            "mu": {
                "mean": np.array([np.mean(samples)]),
                "std": np.array([np.std(samples, ddof=1)]),
                "q05": np.array([np.quantile(samples, 0.05)]),
                "q95": np.array([np.quantile(samples, 0.95)]),
            }
        }

        with patch("posteriordb.PosteriorDatabase") as MockPDB:
            mock_pdb = MagicMock()
            MockPDB.return_value = mock_pdb
            mock_posterior = MagicMock()
            mock_pdb.posterior.return_value = mock_posterior
            mock_posterior.reference_draws.return_value = stan_draws

            result = cross_check_against_posteriordb(
                model_name="test_model",
                posteriordb_id="test-posterior",
                our_summaries=our_summaries,
                n_samples_ours=len(samples),
            )

        assert result.passed is True, (
            f"Expected passed=True; failed_dims={result.failed_dims}, "
            f"max_abs_mean_z={result.max_abs_mean_z:.3f}"
        )
        assert result.failed_dims == ()
        assert result.n_dims_compared == 1
        assert math.isfinite(result.max_abs_mean_z)

    def test_vector_param_passes(self) -> None:
        """Vector param (3 dims), identical summaries → passed=True."""
        rng = np.random.default_rng(1)
        n = 500
        # 3-dim parameter
        samples = rng.normal(loc=[0.0, 1.0, -1.0], scale=[0.3, 0.5, 0.7], size=(n, 3))
        # Flatten to list-of-lists for posteriordb format
        stan_draws = _make_stan_draws({"theta": samples.tolist()})

        our_summaries = {
            "theta": {
                "mean": np.mean(samples, axis=0),
                "std": np.std(samples, axis=0, ddof=1),
                "q05": np.quantile(samples, 0.05, axis=0),
                "q95": np.quantile(samples, 0.95, axis=0),
            }
        }

        with patch("posteriordb.PosteriorDatabase") as MockPDB:
            mock_pdb = MagicMock()
            MockPDB.return_value = mock_pdb
            mock_posterior = MagicMock()
            mock_pdb.posterior.return_value = mock_posterior
            mock_posterior.reference_draws.return_value = stan_draws

            result = cross_check_against_posteriordb(
                model_name="test_model",
                posteriordb_id="test-posterior",
                our_summaries=our_summaries,
                n_samples_ours=n,
            )

        assert (
            result.passed is True
        ), f"failed_dims={result.failed_dims}, max_z={result.max_abs_mean_z:.3f}"
        assert result.n_dims_compared == 3


# ---------------------------------------------------------------------------
# Test 3: Mock-fail — our_summaries differ by 0.5 std
# ---------------------------------------------------------------------------


class TestMockFail:
    """Mock-fail: summaries differ by 0.5 std → passed=False, failed_dims non-empty."""

    def test_mean_shift_fails(self) -> None:
        """A mean shift of 3×SE triggers a failure (|Δmean|/SE ≥ 2)."""
        rng = np.random.default_rng(2)
        n_stan = 5000  # large N → tiny SE_stan
        n_ours = 5000  # large N → tiny SE_ours
        true_std = 1.0

        stan_samples = rng.normal(loc=0.0, scale=true_std, size=n_stan)
        stan_draws = _make_stan_draws({"mu": stan_samples.tolist()})

        # Our mean is shifted by 0.5 (much larger than SE for large N)
        our_mean = np.mean(stan_samples) + 3.0 * true_std / math.sqrt(n_ours)
        # This gives |Δmean| / SE = 3.0 > 2 → should fail
        our_std = true_std

        our_summaries = {
            "mu": {
                "mean": np.array([our_mean]),
                "std": np.array([our_std]),
                "q05": np.array([our_mean - 1.645 * our_std]),
                "q95": np.array([our_mean + 1.645 * our_std]),
            }
        }

        with patch("posteriordb.PosteriorDatabase") as MockPDB:
            mock_pdb = MagicMock()
            MockPDB.return_value = mock_pdb
            mock_posterior = MagicMock()
            mock_pdb.posterior.return_value = mock_posterior
            mock_posterior.reference_draws.return_value = stan_draws

            result = cross_check_against_posteriordb(
                model_name="test_model",
                posteriordb_id="test-posterior",
                our_summaries=our_summaries,
                n_samples_ours=n_ours,
            )

        assert (
            result.passed is False
        ), f"Expected passed=False; max_z={result.max_abs_mean_z:.3f}"
        assert len(result.failed_dims) > 0
        assert result.max_abs_mean_z >= 2.0

    def test_std_ratio_fails(self) -> None:
        """A std ratio of 1.2 (20% off) triggers a failure (|ratio-1| ≥ 0.05)."""
        rng = np.random.default_rng(3)
        n = 2000
        true_std = 1.0

        stan_samples = rng.normal(loc=0.0, scale=true_std, size=n)
        stan_draws = _make_stan_draws({"sigma": stan_samples.tolist()})

        # Our std is 20% larger than Stan's
        our_std = true_std * 1.20

        our_summaries = {
            "sigma": {
                "mean": np.array([np.mean(stan_samples)]),  # same mean
                "std": np.array([our_std]),
                "q05": np.array([-1.645 * our_std]),
                "q95": np.array([1.645 * our_std]),
            }
        }

        with patch("posteriordb.PosteriorDatabase") as MockPDB:
            mock_pdb = MagicMock()
            MockPDB.return_value = mock_pdb
            mock_posterior = MagicMock()
            mock_pdb.posterior.return_value = mock_posterior
            mock_posterior.reference_draws.return_value = stan_draws

            result = cross_check_against_posteriordb(
                model_name="test_model",
                posteriordb_id="test-posterior",
                our_summaries=our_summaries,
                n_samples_ours=n,
            )

        assert result.passed is False
        assert (
            result.max_std_ratio_dev >= 0.05
        ), f"Expected max_std_ratio_dev≥0.05; got {result.max_std_ratio_dev:.4f}"

    def test_failed_dims_contains_param_name(self) -> None:
        """When 'mu' fails, 'mu' (or 'mu[0]') appears in failed_dims."""
        rng = np.random.default_rng(4)
        n = 3000
        stan_samples = rng.normal(loc=0.0, scale=1.0, size=n)
        stan_draws = _make_stan_draws({"mu": stan_samples.tolist()})

        # Large mean shift → definite fail
        our_summaries = {
            "mu": {
                "mean": np.array([100.0]),  # massively wrong
                "std": np.array([1.0]),
                "q05": np.array([98.355]),
                "q95": np.array([101.645]),
            }
        }

        with patch("posteriordb.PosteriorDatabase") as MockPDB:
            mock_pdb = MagicMock()
            MockPDB.return_value = mock_pdb
            mock_posterior = MagicMock()
            mock_pdb.posterior.return_value = mock_posterior
            mock_posterior.reference_draws.return_value = stan_draws

            result = cross_check_against_posteriordb(
                model_name="test_model",
                posteriordb_id="test-posterior",
                our_summaries=our_summaries,
                n_samples_ours=n,
            )

        assert result.passed is False
        # "mu" should appear in some form in failed_dims (could be "mu" or "mu[0]")
        assert any(
            "mu" in d for d in result.failed_dims
        ), f"Expected 'mu' in failed_dims; got {result.failed_dims}"


# ---------------------------------------------------------------------------
# Test 4: Posteriordb unavailable
# ---------------------------------------------------------------------------


class TestPosteriordbunavailable:
    """When posteriordb raises, returns a not-checked result with error string."""

    def test_unknown_posterior_id(self) -> None:
        """An unknown posteriordb_id triggers the graceful-error path."""
        # Patch PosteriorDatabase to raise an exception (simulating unknown ID)
        with patch("posteriordb.PosteriorDatabase") as MockPDB:
            MockPDB.side_effect = Exception("Database not found")

            result = cross_check_against_posteriordb(
                model_name="test_model",
                posteriordb_id="this-id-does-not-exist",
                our_summaries={
                    "mu": {"mean": [0.0], "std": [1.0], "q05": [-1.6], "q95": [1.6]}
                },
                n_samples_ours=100,
            )

        assert result.passed is False
        assert result.n_dims_compared == 0
        assert len(result.failed_dims) == 1
        assert "posteriordb-error" in result.failed_dims[0]
        assert math.isnan(result.max_abs_mean_z)
        assert math.isnan(result.max_std_ratio_dev)

    def test_reference_draws_raises(self) -> None:
        """When reference_draws() raises, returns a graceful not-checked result."""
        with patch("posteriordb.PosteriorDatabase") as MockPDB:
            mock_pdb = MagicMock()
            MockPDB.return_value = mock_pdb
            mock_posterior = MagicMock()
            mock_pdb.posterior.return_value = mock_posterior
            mock_posterior.reference_draws.side_effect = FileNotFoundError(
                "Reference draws not found"
            )

            result = cross_check_against_posteriordb(
                model_name="test_model",
                posteriordb_id="some-id",
                our_summaries={
                    "mu": {"mean": [0.0], "std": [1.0], "q05": [-1.6], "q95": [1.6]}
                },
                n_samples_ours=100,
            )

        assert result.passed is False
        assert result.n_dims_compared == 0
        assert any("posteriordb-error" in d for d in result.failed_dims)

    def test_no_common_params(self) -> None:
        """When parameter names don't match, returns a no-common-params result."""
        # Stan draws has "mu", our summaries has "sigma" → no overlap
        rng = np.random.default_rng(5)
        stan_samples = rng.normal(size=100)
        stan_draws = _make_stan_draws({"mu": stan_samples.tolist()})

        our_summaries = {
            "sigma": {  # different name
                "mean": np.array([0.0]),
                "std": np.array([1.0]),
                "q05": np.array([-1.6]),
                "q95": np.array([1.6]),
            }
        }

        with patch("posteriordb.PosteriorDatabase") as MockPDB:
            mock_pdb = MagicMock()
            MockPDB.return_value = mock_pdb
            mock_posterior = MagicMock()
            mock_pdb.posterior.return_value = mock_posterior
            mock_posterior.reference_draws.return_value = stan_draws

            result = cross_check_against_posteriordb(
                model_name="test_model",
                posteriordb_id="test-posterior",
                our_summaries=our_summaries,
                n_samples_ours=100,
            )

        assert result.passed is False
        assert result.n_dims_compared == 0
        assert any("no-common-params" in d for d in result.failed_dims)
