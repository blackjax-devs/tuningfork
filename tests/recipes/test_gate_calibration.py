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
"""Tests for the PR #245 calibrated marginal-z gate — benchmark path.

Covers the four properties guaranteed by the pooled-SE denominator,
dimension-aware Bonferroni threshold, and materiality co-primary gate,
both at the shared-helper level (``marginal_z_verdict``) and through the
benchmark gate path (``_compute_gt_compare``).

All tests are pure-logic (no JAX trace, no chain runs) → ``fast`` marker.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tuningfork.calibration._gate.marginal_z import (
    _DEFAULT_NU,
    _SE_FLOOR,
    _TAU_SCI,
    bonferroni_z_crit,
    marginal_z_verdict,
)

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# bonferroni_z_crit
# ---------------------------------------------------------------------------


def test_bonferroni_z_crit_degenerate_zero_dims() -> None:
    """D_total=0 returns inf (no dims to test)."""
    assert math.isinf(bonferroni_z_crit(0, nu=18))


def test_bonferroni_z_crit_spot_values() -> None:
    """Pin z_crit for two known (D_total, nu) pairs matching _verify.py's test."""
    from scipy import stats

    alpha = 0.05
    # D=390, nu=18 (radon-sized, 10-chain GT) → z_crit ≈ 4.8514
    z = bonferroni_z_crit(390, nu=18, alpha=alpha)
    expected = float(stats.t.ppf(1.0 - alpha / (2 * 390), 18))
    assert z == pytest.approx(expected, rel=1e-6)
    assert z == pytest.approx(4.8514, rel=1e-3)

    # D=26, nu=18 (german_credit-sized) → z_crit ≈ 3.6281
    z2 = bonferroni_z_crit(26, nu=18, alpha=alpha)
    expected2 = float(stats.t.ppf(1.0 - alpha / (2 * 26), 18))
    assert z2 == pytest.approx(expected2, rel=1e-6)
    assert z2 == pytest.approx(3.6281, rel=1e-3)


# ---------------------------------------------------------------------------
# marginal_z_verdict: pooled SE denominator (decision 1)
# ---------------------------------------------------------------------------


def test_pooled_se_halves_z_vs_old_max_at_equal_se() -> None:
    """Pooled SE denom = sqrt(2)×max() at equal SE → z is divided by sqrt(2).

    Old formula: denom = max(se_a, se_b) = se (at equal SE).
    New formula: denom = sqrt(se_a² + se_b²) = se·sqrt(2).
    Ratio: new_z = old_z / sqrt(2).

    This is the PR #245 fix-1 property: the old gate over-penalised by √2.
    """
    se = 0.1
    delta = 10.0
    # std_b large so materiality is tiny (REVIEW, not FAIL)
    mean_a = np.array([delta])
    std_a = np.array([1.0])
    se_a = np.array([se])
    mean_b = np.array([0.0])
    std_b = np.array([1000.0])
    se_b = np.array([se])

    _, vdicts, meta = marginal_z_verdict(mean_a, std_a, se_a, mean_b, std_b, se_b,
                                         D_total=1, n_chains=10)
    new_z = vdicts[0]["z"]
    old_z = delta / se  # old max() denominator
    expected_new_z = old_z / math.sqrt(2)
    assert new_z == pytest.approx(expected_new_z, rel=1e-5), (
        f"new_z={new_z:.4f} should equal old_z/sqrt(2)={expected_new_z:.4f}. "
        "Pooled SE formula not applied correctly."
    )


def test_se_floor_prevents_zero_division() -> None:
    """Near-zero SE uses _SE_FLOOR to avoid division by zero."""
    mean_a = np.array([1.0])
    std_a = np.array([1.0])
    se_a = np.array([0.0])
    mean_b = np.array([1.0])
    std_b = np.array([1.0])
    se_b = np.array([0.0])

    all_pass, vdicts, meta = marginal_z_verdict(mean_a, std_a, se_a, mean_b, std_b, se_b,
                                                D_total=1, n_chains=10)
    assert all_pass
    assert math.isfinite(meta["max_z"])
    assert vdicts[0]["z"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# marginal_z_verdict: dimension-aware Bonferroni (decision 2)
# ---------------------------------------------------------------------------


def test_high_d_review_pass_immaterial() -> None:
    """High-D model (D=1500): max_z=7 but mat << TAU_SCI → PASS via REVIEW.

    With D=1500, n_chains=10 → nu=18:
        z_crit = t.ppf(1 - 0.05/3000, 18) ≈ 5.48

    max_z=7 > z_crit but materiality = delta/std_b << 0.05 → REVIEW (counts as PASS).
    Under the OLD fixed threshold of 4.0, z=7 would be a hard FAIL.
    """
    from scipy import stats

    D = 1500
    se = 0.01
    n_chains = 10
    std_b_val = 100.0  # large std_b → tiny materiality

    # Compute the expected z_crit to verify
    nu = 2 * (n_chains - 1)  # = 18
    z_crit_expected = float(stats.t.ppf(1.0 - 0.05 / (2 * D), nu))

    # One hot dim with z = 7 > z_crit (≈5.48)
    se_denom = math.sqrt(2) * se  # equal se_a == se_b
    delta_hot = 7.0 * se_denom  # z = 7

    mean_a = np.zeros(D)
    mean_a[0] = delta_hot
    std_a = np.ones(D)
    se_a = np.full(D, se)
    mean_b = np.zeros(D)
    std_b = np.full(D, std_b_val)
    se_b = np.full(D, se)

    all_pass, vdicts, meta = marginal_z_verdict(mean_a, std_a, se_a, mean_b, std_b, se_b,
                                                D_total=D, n_chains=n_chains)

    assert all_pass, (
        f"D=1500 immaterial case should PASS under calibrated gate. "
        f"z_crit={meta['z_crit']:.3f}, max_z={meta['max_z']:.3f}, "
        f"n_fail={meta['n_fail']}, n_review={meta['n_review']}"
    )
    assert meta["n_review"] >= 1, "hot dim should be REVIEW (z > z_crit but mat < TAU_SCI)"
    assert meta["n_fail"] == 0
    # z_crit matches expected Bonferroni value
    assert meta["z_crit"] == pytest.approx(z_crit_expected, rel=1e-5)
    # hot dim materiality is truly immaterial
    hot_mat = vdicts[0]["mat"]
    assert hot_mat < _TAU_SCI, f"hot_mat={hot_mat:.5f} should be < TAU_SCI={_TAU_SCI}"


# ---------------------------------------------------------------------------
# marginal_z_verdict: materiality co-primary (decision 3)
# ---------------------------------------------------------------------------


def test_materiality_hard_fail() -> None:
    """Large z AND large |Δμ|/std_b (> TAU_SCI) → hard FAIL."""
    se = 0.01
    delta = 1.0  # |Δμ|/std_b = 1.0 >> TAU_SCI = 0.05
    std_b_val = 1.0

    mean_a = np.array([delta])
    std_a = np.array([1.0])
    se_a = np.array([se])
    mean_b = np.array([0.0])
    std_b = np.array([std_b_val])
    se_b = np.array([se])

    all_pass, vdicts, meta = marginal_z_verdict(mean_a, std_a, se_a, mean_b, std_b, se_b,
                                                D_total=1, n_chains=10)
    assert not all_pass, "Large z + large mat should hard-FAIL"
    assert vdicts[0]["verdict"] == "FAIL"
    assert meta["n_fail"] == 1
    assert meta["n_review"] == 0


def test_materiality_review_pass() -> None:
    """Large z but |Δμ|/std_b <= TAU_SCI → REVIEW (counts as PASS)."""
    # se=1e-6 → se_denom ≈ sqrt(2)*1e-6; delta=0.001 → z ≈ 707 >> z_crit
    # std_b=100 → mat = 0.001/100 = 1e-5 << TAU_SCI = 0.05
    se = 1e-6
    delta = 0.001
    std_b_val = 100.0

    mean_a = np.array([delta])
    std_a = np.array([1.0])
    se_a = np.array([se])
    mean_b = np.array([0.0])
    std_b = np.array([std_b_val])
    se_b = np.array([se])

    all_pass, vdicts, meta = marginal_z_verdict(mean_a, std_a, se_a, mean_b, std_b, se_b,
                                                D_total=1, n_chains=10)
    assert all_pass, "Large z but immaterial should be REVIEW (counts as PASS)"
    assert vdicts[0]["verdict"] == "REVIEW"
    assert meta["n_review"] == 1
    assert meta["n_fail"] == 0


def test_materiality_boundary_strict_gt() -> None:
    """mat exactly at TAU_SCI boundary → REVIEW (strict >, not FAIL)."""
    se = 1e-6
    std_b_val = 1.0
    delta_at = _TAU_SCI * std_b_val  # mat = exactly TAU_SCI → strict > is False

    mean_a = np.array([delta_at])
    std_a = np.array([1.0])
    se_a = np.array([se])
    mean_b = np.array([0.0])
    std_b = np.array([std_b_val])
    se_b = np.array([se])

    all_pass, vdicts, _ = marginal_z_verdict(mean_a, std_a, se_a, mean_b, std_b, se_b,
                                             D_total=1, n_chains=10)
    assert all_pass, f"mat exactly at boundary ({_TAU_SCI}) should be REVIEW, not FAIL"
    assert vdicts[0]["verdict"] == "REVIEW"

    # Just above the boundary → FAIL
    delta_above = _TAU_SCI * std_b_val + 1e-9
    mean_a_above = np.array([delta_above])
    all_pass2, vdicts2, _ = marginal_z_verdict(mean_a_above, std_a, se_a, mean_b, std_b, se_b,
                                               D_total=1, n_chains=10)
    assert not all_pass2, "mat just above boundary should hard-FAIL"
    assert vdicts2[0]["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# Benchmark gate integration: _compute_gt_compare calibrated verdict
# ---------------------------------------------------------------------------


def _make_mc_samples(n_chains: int, n_draws: int, D: int,
                     mean: float = 0.0) -> dict[str, np.ndarray]:
    """Synthetic mc_samples dict (n_chains, n_draws, D) with given mean."""
    rng = np.random.default_rng(42)
    arr = rng.normal(mean, 1.0, (n_chains, n_draws, D))
    return {"x": arr}


def _make_gt_summary(D: int, gt_mean: float = 0.0,
                     n_samples: int = 40000) -> dict[str, dict]:
    """Synthetic ground_truth_summaries dict (legacy summary.json format)."""
    return {
        "x": {
            "mean": [gt_mean] * D,
            "std": [1.0] * D,
            "n_samples": n_samples,
        }
    }


def test_gt_compare_calibrated_pass_no_bias() -> None:
    """Zero-bias benchmark run → calibrated_pass=True."""
    from tuningfork.calibration._gate.gt_compare import _compute_gt_compare

    mc = _make_mc_samples(n_chains=1, n_draws=1000, D=10, mean=0.0)
    gt = _make_gt_summary(D=10, gt_mean=0.0)

    result = _compute_gt_compare(mc, gt, min_bulk_ess=None)

    assert result.calibrated_pass is True
    assert result.calibrated_n_fail == 0
    assert result.calibrated_D_total == 10
    # With n_chains=1 (benchmark path), nu falls back to _DEFAULT_NU
    assert result.calibrated_nu == _DEFAULT_NU


def test_gt_compare_calibrated_fail_large_bias() -> None:
    """Large bias benchmark run → calibrated_pass=False (hard FAIL dim)."""
    from tuningfork.calibration._gate.gt_compare import _compute_gt_compare

    # shift of 50 std → z >> z_crit and mat = 50 >> TAU_SCI
    mc = _make_mc_samples(n_chains=1, n_draws=1000, D=10, mean=50.0)
    gt = _make_gt_summary(D=10, gt_mean=0.0)

    result = _compute_gt_compare(mc, gt, min_bulk_ess=None)

    assert result.calibrated_pass is False
    assert result.calibrated_n_fail is not None and result.calibrated_n_fail >= 1


def test_gt_compare_calibrated_nu_from_mc_n_chains() -> None:
    """When mc n_chains > 1, nu = 2*(n_chains-1) (NOT _DEFAULT_NU)."""
    from tuningfork.calibration._gate.gt_compare import _compute_gt_compare

    # n_chunks=4 in auto_gate → mc_samples arrives with n_chains=4
    mc = _make_mc_samples(n_chains=4, n_draws=250, D=5)
    gt = _make_gt_summary(D=5)

    result = _compute_gt_compare(mc, gt, min_bulk_ess=None)

    assert result.calibrated_nu == 2 * (4 - 1), (
        f"Expected nu=6 (2*(4-1)), got {result.calibrated_nu}. "
        "nu should be derived from mc n_chains when > 1."
    )


def test_gt_compare_calibrated_review_immaterial_high_d() -> None:
    """High-D benchmark (D=503, stoch_vol-sized): z=4.335 < z_crit → PASS.

    With D=503 and nu=_DEFAULT_NU=9:
        z_crit = t.ppf(1 - 0.05/(2*503), 9) ≈ 6.60

    stoch_vol's reported z=4.335 (x86_64, JAX 0.10) is well below z_crit
    and passes the calibrated gate even without the materiality co-primary.

    This test uses a synthetic run with max_abs_mean_z ≈ 4.335 to confirm
    the calibrated gate classifies it as PASS (not FAIL).
    """
    from tuningfork.calibration._gate.gt_compare import _compute_gt_compare
    from tuningfork.calibration._gate.marginal_z import bonferroni_z_crit

    D = 503
    n_draws = 1000
    rng = np.random.default_rng(0)

    # Design: sample mean deviates from GT mean enough to give z ≈ 4.335.
    # se_sample ≈ 1.0 / sqrt(n_draws) ≈ 0.0316; se_gt ≈ 1.0/sqrt(40000) ≈ 0.005.
    # Pooled SE denom ≈ sqrt(0.0316^2 + 0.005^2) ≈ 0.0320.
    # To get z ≈ 4.335 in one hot dim: delta = 4.335 * 0.0320 ≈ 0.139.
    se_sample_approx = 1.0 / math.sqrt(n_draws)
    se_gt_approx = 1.0 / math.sqrt(40000)
    denom_approx = math.sqrt(se_sample_approx**2 + se_gt_approx**2)
    target_z = 4.335
    delta = target_z * denom_approx

    # Build a run where dim 0 has the target shift; other dims are zero
    arr = rng.normal(0.0, 1.0, (1, n_draws, D))
    arr[:, :, 0] += delta  # shift dim 0
    mc = {"x": arr}
    gt = _make_gt_summary(D=D, gt_mean=0.0)

    result = _compute_gt_compare(mc, gt, min_bulk_ess=None)

    # Confirm calibrated z_crit >> 4.335 for D=503, nu=9
    z_crit_expected = bonferroni_z_crit(D, _DEFAULT_NU)
    assert z_crit_expected > 6.0, f"z_crit={z_crit_expected:.3f} expected > 6 for D=503, nu=9"

    assert result.calibrated_pass is True, (
        f"stoch_vol-sized D=503 with max_abs_mean_z≈{result.max_abs_mean_z:.3f} should PASS "
        f"under calibrated gate (z_crit={result.calibrated_z_crit:.3f}, "
        f"n_fail={result.calibrated_n_fail})"
    )


def test_gt_compare_max_abs_mean_z_preserved() -> None:
    """max_abs_mean_z is still present in _GtCompareResult (trend tracking)."""
    from tuningfork.calibration._gate.gt_compare import _compute_gt_compare

    mc = _make_mc_samples(n_chains=1, n_draws=500, D=5)
    gt = _make_gt_summary(D=5)

    result = _compute_gt_compare(mc, gt, min_bulk_ess=None)

    # max_abs_mean_z is the trend-tracking field; must not be removed
    assert result.max_abs_mean_z is not None
    assert math.isfinite(result.max_abs_mean_z)


def test_gt_compare_calibrated_se_pooled_vs_old_max() -> None:
    """Calibrated z is lower than the old max() z at equal SE.

    At equal se_sample = se_gt = se:
        old denom = max(se, se) = se       → old_z = delta/se
        new denom = sqrt(se^2+se^2) = se*sqrt(2) → new_z = delta/(se*sqrt(2))

    new_z = old_z / sqrt(2) ≈ 0.707 × old_z.
    This ensures the calibrated gate is strictly LESS aggressive at equal SE.
    """
    from tuningfork.calibration._gate.gt_compare import _compute_gt_compare

    D = 1
    n_draws = 10000
    delta = 0.5  # noticeable shift

    rng = np.random.default_rng(123)
    arr = rng.normal(delta, 1.0, (1, n_draws, D))
    mc = {"x": arr}

    # GT with same SE as sample (set n_samples = n_draws so se_gt ≈ se_sample)
    gt = {
        "x": {
            "mean": [0.0],
            "std": [1.0],
            "n_samples": n_draws,  # matches benchmark run
        }
    }

    result = _compute_gt_compare(mc, gt, min_bulk_ess=None)

    # With pooled SE: z ≈ delta / (std * sqrt(2/n_draws))
    # Both se_sample and se_gt ≈ 1/sqrt(n_draws) when well-mixed
    se_approx = 1.0 / math.sqrt(n_draws)
    old_z_approx = delta / se_approx
    new_z_approx = delta / (se_approx * math.sqrt(2))

    # The actual z should be closer to new_z_approx than old_z_approx
    actual_z = result.max_abs_mean_z
    assert actual_z is not None
    # Actual z ≈ new_z_approx (within 20% due to ESS != n_draws exactly)
    assert actual_z < old_z_approx * 0.9, (
        f"Pooled SE gate: actual_z={actual_z:.3f} should be < 90% of "
        f"old_z_approx={old_z_approx:.3f}"
    )
    assert actual_z == pytest.approx(new_z_approx, rel=0.2), (
        f"actual_z={actual_z:.3f} should be close to new_z_approx={new_z_approx:.3f} "
        f"(within 20%)"
    )
