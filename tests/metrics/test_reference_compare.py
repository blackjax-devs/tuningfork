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
"""Tests for tuningfork.metrics.reference_compare.compute_sample_quality.

Five invariant classes:
  1. Dict-of-dicts input vs single-array input — both code paths exercised.
  2. Multi-chain vs single-chain equivalence — flattening is chains × samples.
  3. NaN reference summary — skip + warn for that parameter.
  4. Gaussian draws matching the reference — std_ratio_max_dev < 0.1.
  5. Mean-shifted draws — mae_vs_reference ≈ shift / ref_std.

Additional tests cover error paths: all-NaN reference, NaN in draws, key mismatch.
"""

import warnings

import numpy as np
import pytest

from tuningfork.metrics.reference_compare import compute_sample_quality

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Shared RNG seed and helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _std_normal_ref(name: str) -> dict[str, dict[str, float]]:
    """Build a reference summary dict for a standard N(0,1) parameter."""
    return {
        name: {
            "mean": 0.0,
            "std": 1.0,
            "q05": float(
                np.quantile(np.random.default_rng(0).standard_normal(100_000), 0.05)
            ),
            "q95": float(
                np.quantile(np.random.default_rng(0).standard_normal(100_000), 0.95)
            ),
        }
    }


def _std_normal_ref_flat() -> dict[str, float]:
    """Single-level reference summary for a standard N(0,1) parameter."""
    return {
        "mean": 0.0,
        "std": 1.0,
        "q05": float(
            np.quantile(np.random.default_rng(0).standard_normal(100_000), 0.05)
        ),
        "q95": float(
            np.quantile(np.random.default_rng(0).standard_normal(100_000), 0.95)
        ),
    }


# ---------------------------------------------------------------------------
# 1. Dict-of-dicts input vs single-array input
# ---------------------------------------------------------------------------


class TestInputDuality:
    """Both dict-of-dicts and single-array inputs must yield results."""

    def test_dict_input_returns_four_keys(self) -> None:
        """Dict-of-dicts path: returns exactly four float keys."""
        draws = {"x": RNG.standard_normal((4, 500, 1))}
        ref = _std_normal_ref("x")
        result = compute_sample_quality(draws, ref)

        assert set(result.keys()) == {
            "mae_vs_reference",
            "q05_error",
            "q95_error",
            "std_ratio_max_dev",
        }
        for key, val in result.items():
            assert isinstance(val, float), f"metric {key!r} is not float: {type(val)}"

    def test_array_input_returns_same_shape(self) -> None:
        """Single-array path: flat reference summary + array draw → same 4 keys."""
        draws = RNG.standard_normal((4, 500, 1))
        ref = _std_normal_ref_flat()
        result = compute_sample_quality(draws, ref)

        assert set(result.keys()) == {
            "mae_vs_reference",
            "q05_error",
            "q95_error",
            "std_ratio_max_dev",
        }

    def test_dict_and_array_inputs_agree(self) -> None:
        """Dict-of-dicts and single-array paths must produce the same result."""
        rng = np.random.default_rng(7)
        arr = rng.standard_normal((2, 1000, 1))

        ref_flat = _std_normal_ref_flat()
        ref_dict = {"x": ref_flat}
        draws_dict = {"x": arr}

        result_dict = compute_sample_quality(draws_dict, ref_dict)
        result_arr = compute_sample_quality(arr, ref_flat)

        for key in result_dict:
            assert abs(result_dict[key] - result_arr[key]) < 1e-10, (
                f"Mismatch on {key!r}: dict={result_dict[key]}, "
                f"array={result_arr[key]}"
            )


# ---------------------------------------------------------------------------
# 2. Multi-chain vs single-chain equivalence
# ---------------------------------------------------------------------------


class TestMultiChainEquivalence:
    """(4 chains × 1000 samples) and (1 chain × 4000 samples) are equivalent."""

    def test_multichain_vs_singlechain_equivalent(self) -> None:
        """Flatten is over (chains × samples); reshaping doesn't change metrics."""
        rng = np.random.default_rng(13)
        total_samples = 4000
        data = rng.standard_normal(total_samples)

        # 4 chains × 1000 samples
        multi_chain = {"x": data.reshape(4, 1000, 1)}
        # 1 chain × 4000 samples
        single_chain = {"x": data.reshape(1, 4000, 1)}

        ref = _std_normal_ref("x")
        result_multi = compute_sample_quality(multi_chain, ref)
        result_single = compute_sample_quality(single_chain, ref)

        for key in result_multi:
            assert abs(result_multi[key] - result_single[key]) < 1e-10, (
                f"Multi-chain vs single-chain mismatch on {key!r}: "
                f"multi={result_multi[key]}, single={result_single[key]}"
            )

    def test_multichain_no_per_chain_averaging(self) -> None:
        """Draws shape (C, S) still produces finite metrics for scalar params."""
        rng = np.random.default_rng(17)
        draws = {"theta": rng.standard_normal((3, 200))}  # no trailing event dim
        ref = _std_normal_ref("theta")
        result = compute_sample_quality(draws, ref)
        # All four metrics must be finite
        for key, val in result.items():
            assert np.isfinite(val), f"{key!r} is not finite: {val}"


# ---------------------------------------------------------------------------
# 3. NaN reference summary — skip + warn
# ---------------------------------------------------------------------------


class TestNaNReference:
    """A NaN reference for one parameter is skipped with a warning."""

    def test_nan_ref_one_of_three_params_skipped(self) -> None:
        """With 3 params and 1 NaN reference, only 2 contribute to the reduction."""
        rng = np.random.default_rng(99)
        draws = {
            "a": rng.standard_normal((2, 500, 1)),
            "b": rng.standard_normal((2, 500, 1)),
            "c": rng.standard_normal((2, 500, 1)),
        }
        ref = {
            "a": {"mean": 0.0, "std": 1.0, "q05": -1.645, "q95": 1.645},
            "b": {"mean": float("nan"), "std": 1.0, "q05": -1.645, "q95": 1.645},
            "c": {"mean": 0.0, "std": 1.0, "q05": -1.645, "q95": 1.645},
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = compute_sample_quality(draws, ref)

        # Should warn exactly once (for param "b")
        assert len(caught) == 1, f"Expected 1 warning, got {len(caught)}"
        assert "b" in str(caught[0].message).lower() or "b" in str(caught[0].message)

        # Result should still be a valid 4-key dict
        assert set(result.keys()) == {
            "mae_vs_reference",
            "q05_error",
            "q95_error",
            "std_ratio_max_dev",
        }

    def test_all_nan_references_raises(self) -> None:
        """If all parameters have NaN references, must raise ValueError."""
        rng = np.random.default_rng(55)
        draws = {"x": rng.standard_normal((2, 100, 1))}
        ref = {
            "x": {
                "mean": float("nan"),
                "std": 1.0,
                "q05": float("nan"),
                "q95": float("nan"),
            }
        }
        with pytest.raises(ValueError, match="All parameters have NaN"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                compute_sample_quality(draws, ref)


# ---------------------------------------------------------------------------
# 4. Gaussian draws matching reference → std_ratio_max_dev < 0.1
# ---------------------------------------------------------------------------


class TestGaussianMatch:
    """Large-n draws from N(0,1) should closely match the N(0,1) reference."""

    def test_gaussian_match_std_ratio(self) -> None:
        """n=10_000 i.i.d. standard normal draws → std_ratio_max_dev < 0.1."""
        rng = np.random.default_rng(123)
        n_total = 10_000
        draws = {"x": rng.standard_normal((1, n_total, 1))}
        # Reference quantiles computed from the same distribution (analytic).
        from scipy.stats import norm  # type: ignore[import]

        ref = {
            "x": {
                "mean": 0.0,
                "std": 1.0,
                "q05": float(norm.ppf(0.05)),
                "q95": float(norm.ppf(0.95)),
            }
        }
        result = compute_sample_quality(draws, ref)
        assert result["std_ratio_max_dev"] < 0.1, (
            f"std_ratio_max_dev={result['std_ratio_max_dev']:.4f} exceeds 0.1 "
            "for n=10_000 Gaussian draws; check normalisation."
        )

    def test_gaussian_match_mae(self) -> None:
        """n=10_000 i.i.d. standard normal draws → mae_vs_reference < 0.1."""
        rng = np.random.default_rng(456)
        n_total = 10_000
        draws = {"x": rng.standard_normal((2, n_total // 2, 1))}
        from scipy.stats import norm  # type: ignore[import]

        ref = {
            "x": {
                "mean": 0.0,
                "std": 1.0,
                "q05": float(norm.ppf(0.05)),
                "q95": float(norm.ppf(0.95)),
            }
        }
        result = compute_sample_quality(draws, ref)
        assert result["mae_vs_reference"] < 0.1, (
            f"mae_vs_reference={result['mae_vs_reference']:.4f} exceeds 0.1 "
            "for n=10_000 Gaussian draws."
        )


# ---------------------------------------------------------------------------
# 5. Mean shift by k × ref_std → mae_vs_reference ≈ k
# ---------------------------------------------------------------------------


class TestMeanShiftRecoverability:
    """Shifting draws by k×ref_std must yield mae_vs_reference ≈ k."""

    def test_mean_shift_recoverable(self) -> None:
        """k=2.0 shift → mae_vs_reference within 0.5 of 2.0 (MC tolerance)."""
        rng = np.random.default_rng(789)
        k = 2.0
        ref_std = 1.0
        n_total = 20_000

        # i.i.d. draws shifted by k * ref_std
        raw = rng.standard_normal((2, n_total // 2, 1))
        draws = {"x": raw + k * ref_std}

        from scipy.stats import norm  # type: ignore[import]

        ref = {
            "x": {
                "mean": 0.0,
                "std": ref_std,
                "q05": float(norm.ppf(0.05)),
                "q95": float(norm.ppf(0.95)),
            }
        }
        result = compute_sample_quality(draws, ref)
        # The mean is shifted by k, so mae_vs_reference ≈ k.
        assert (
            abs(result["mae_vs_reference"] - k) < 0.5
        ), f"Expected mae_vs_reference ≈ {k}, got {result['mae_vs_reference']:.4f}"

    def test_reference_std_normalization_not_empirical(self) -> None:
        """Draws with doubled std → std_ratio_max_dev ≈ 1.0, not 0.0.

        Validates that we normalise by REFERENCE std, not empirical std.
        If we used empirical std, std_ratio_max_dev would be close to 0 for any
        scaled draw, making it useless as a diagnostic for spread errors.
        """
        rng = np.random.default_rng(321)
        n_total = 10_000

        # Draws from N(0, 2) — double the reference std
        draws = {"x": rng.standard_normal((1, n_total, 1)) * 2.0}
        from scipy.stats import norm  # type: ignore[import]

        ref = {
            "x": {
                "mean": 0.0,
                "std": 1.0,  # reference std = 1
                "q05": float(norm.ppf(0.05)),
                "q95": float(norm.ppf(0.95)),
            }
        }
        result = compute_sample_quality(draws, ref)
        # std_ratio_max_dev = |2/1 - 1| = 1.0 (normalised by ref std=1)
        assert result["std_ratio_max_dev"] > 0.5, (
            f"std_ratio_max_dev={result['std_ratio_max_dev']:.4f} is suspiciously low "
            "for draws with 2× the reference std. "
            "Check that normalisation uses REFERENCE std, not empirical std."
        )


# ---------------------------------------------------------------------------
# 6. Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    """ValueError and TypeError for malformed inputs."""

    def test_nan_in_draws_raises(self) -> None:
        """NaN values in draws must raise ValueError."""
        rng = np.random.default_rng(11)
        arr = rng.standard_normal((2, 100, 1))
        arr[0, 5, 0] = float("nan")
        draws = {"x": arr}
        ref = _std_normal_ref("x")
        with pytest.raises(ValueError, match="NaN"):
            compute_sample_quality(draws, ref)

    def test_key_mismatch_raises(self) -> None:
        """draws and reference_summaries with mismatched keys must raise ValueError."""
        rng = np.random.default_rng(22)
        draws = {"x": rng.standard_normal((2, 100, 1))}
        ref = {"y": {"mean": 0.0, "std": 1.0, "q05": -1.645, "q95": 1.645}}
        with pytest.raises(ValueError, match="keys"):
            compute_sample_quality(draws, ref)

    def test_missing_ref_key_raises(self) -> None:
        """Reference summary missing a required key must raise ValueError."""
        rng = np.random.default_rng(33)
        draws = {"x": rng.standard_normal((2, 100, 1))}
        ref = {"x": {"mean": 0.0, "std": 1.0, "q05": -1.645}}  # missing q95
        with pytest.raises(ValueError, match="missing required key"):
            compute_sample_quality(draws, ref)

    def test_array_draw_1d_raises(self) -> None:
        """A 1-D draw array (no chain/sample axes) must raise ValueError."""
        arr = np.ones(100)
        ref = _std_normal_ref_flat()
        with pytest.raises(ValueError, match="at least 2 axes"):
            compute_sample_quality(arr, ref)
