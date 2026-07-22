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
    _TAU_SCI,
    _TAU_SCI_BENCHMARK,
    bonferroni_z_crit,
    bonferroni_z_crit_normal,
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
# bonferroni_z_crit_normal (benchmark regime)
# ---------------------------------------------------------------------------


def test_bonferroni_z_crit_normal_degenerate_zero_dims() -> None:
    """D_total=0 returns inf (no dims to test)."""
    assert math.isinf(bonferroni_z_crit_normal(0))


def test_bonferroni_z_crit_normal_spot_values() -> None:
    """Pin normal-Bonferroni z_crit at key model sizes (benchmark regime).

    These are the values the benchmark gate uses; they differ materially from
    the t-df form (_verify.py's coherence regime).  The stoch_vol value (D=503)
    is load-bearing: z_crit≈3.892 < the CI-observed max_z≈4.335, so the
    materiality co-primary gate is what prevents a false hard-FAIL there.
    """
    from scipy import stats

    alpha = 0.05
    # D=503 (stoch_vol) → ≈ 3.892  (< old fixed threshold 4.0)
    z503 = bonferroni_z_crit_normal(503, alpha=alpha)
    assert z503 == pytest.approx(float(stats.norm.ppf(1.0 - alpha / 1006)), rel=1e-6)
    assert z503 == pytest.approx(3.892, rel=1e-3)
    assert z503 < 4.0, "normal z_crit at D=503 must be below the old fixed 4.0 gate"

    # D=1 (single-dim) → 1.96 (two-tailed α/2 = 0.025)
    z1 = bonferroni_z_crit_normal(1, alpha=alpha)
    assert z1 == pytest.approx(1.96, rel=1e-3)

    # D=1500 (high-D) → ≈ 4.149 (more conservative as D grows)
    z1500 = bonferroni_z_crit_normal(1500, alpha=alpha)
    assert z1500 > z503, "z_crit should grow with D (larger Bonferroni penalty)"
    assert z1500 == pytest.approx(4.149, rel=1e-3)


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

    _, vdicts, meta = marginal_z_verdict(
        mean_a, std_a, se_a, mean_b, std_b, se_b, D_total=1, n_chains=10
    )
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

    all_pass, vdicts, meta = marginal_z_verdict(
        mean_a, std_a, se_a, mean_b, std_b, se_b, D_total=1, n_chains=10
    )
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

    all_pass, vdicts, meta = marginal_z_verdict(
        mean_a, std_a, se_a, mean_b, std_b, se_b, D_total=D, n_chains=n_chains
    )

    assert all_pass, (
        f"D=1500 immaterial case should PASS under calibrated gate. "
        f"z_crit={meta['z_crit']:.3f}, max_z={meta['max_z']:.3f}, "
        f"n_fail={meta['n_fail']}, n_review={meta['n_review']}"
    )
    assert (
        meta["n_review"] >= 1
    ), "hot dim should be REVIEW (z > z_crit but mat < TAU_SCI)"
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

    all_pass, vdicts, meta = marginal_z_verdict(
        mean_a, std_a, se_a, mean_b, std_b, se_b, D_total=1, n_chains=10
    )
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

    all_pass, vdicts, meta = marginal_z_verdict(
        mean_a, std_a, se_a, mean_b, std_b, se_b, D_total=1, n_chains=10
    )
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

    all_pass, vdicts, _ = marginal_z_verdict(
        mean_a, std_a, se_a, mean_b, std_b, se_b, D_total=1, n_chains=10
    )
    assert all_pass, f"mat exactly at boundary ({_TAU_SCI}) should be REVIEW, not FAIL"
    assert vdicts[0]["verdict"] == "REVIEW"

    # Just above the boundary → FAIL
    delta_above = _TAU_SCI * std_b_val + 1e-9
    mean_a_above = np.array([delta_above])
    all_pass2, vdicts2, _ = marginal_z_verdict(
        mean_a_above, std_a, se_a, mean_b, std_b, se_b, D_total=1, n_chains=10
    )
    assert not all_pass2, "mat just above boundary should hard-FAIL"
    assert vdicts2[0]["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# _TAU_SCI_BENCHMARK: benchmark-regime materiality bar = 0.15
# ---------------------------------------------------------------------------


def test_tau_sci_benchmark_value() -> None:
    """_TAU_SCI_BENCHMARK is pinned at 0.15 (GT-correctness regime).

    Regression guard: this value was chosen so that stoch_vol seed-18's
    per-seed worst-marginal bias (0.085σ) is a REVIEW, not a FAIL, while
    genuine ≥0.15σ biases are still caught.  Any change to this constant
    must be a deliberate TL decision.
    """
    assert _TAU_SCI_BENCHMARK == pytest.approx(0.15)
    # The benchmark bar must be strictly looser than the coherence bar.
    assert (
        _TAU_SCI_BENCHMARK > _TAU_SCI
    ), "_TAU_SCI_BENCHMARK (correctness) must be looser than _TAU_SCI (coherence)"


def test_benchmark_tau_sci_review_at_seed18_mat() -> None:
    """mat=0.085σ (seed-18 worst-dim) → REVIEW at _TAU_SCI_BENCHMARK=0.15.

    Seed-18 (stoch_vol nightly) has bias_sigma_at_argmax_z=0.085.  Under the
    benchmark gate this is a REVIEW (mat > 0.05 but ≤ 0.15), not a hard FAIL.
    Verifies the `gt_compare.py` calibrated verdict at TAU_SCI_BENCHMARK=0.15.
    """
    from tuningfork.calibration._gate.gt_compare import _compute_gt_compare

    D = 503
    n_draws = 5000
    rng = np.random.default_rng(18)  # seed-18 analogy

    # Construct one dim with z > z_crit and mat = 0.085 (between 0.05 and 0.15).
    # At D=503, z_crit_normal ≈ 3.892.
    # denom ≈ sqrt((1/sqrt(n_draws))^2 + (1/sqrt(40000))^2) ≈ 0.01443
    se_sample = 1.0 / math.sqrt(n_draws)  # ≈ 0.01414
    se_gt = 1.0 / math.sqrt(40000)  # = 0.005
    denom = math.sqrt(se_sample**2 + se_gt**2)  # ≈ 0.01484

    target_z = 5.0  # > z_crit=3.892 with comfortable margin
    target_mat = 0.085  # between _TAU_SCI=0.05 and _TAU_SCI_BENCHMARK=0.15

    # std_b such that delta/std_b = 0.085 → std_b = delta/0.085
    delta = target_z * denom
    std_b_hot = delta / target_mat  # ≈ 0.083 / 0.085 ≈ 0.975 ≈ 1.0

    arr = rng.normal(0.0, 1.0, (1, n_draws, D))
    arr[:, :, 0] += delta
    mc = {"x": arr}
    gt = {
        "x": {
            "mean": [0.0] * D,
            "std": [std_b_hot] + [1.0] * (D - 1),
            "n_samples": 40000,
        }
    }

    result = _compute_gt_compare(mc, gt, min_bulk_ess=None)

    # At TAU_SCI_BENCHMARK=0.15: mat=0.085 ≤ 0.15 → REVIEW, not FAIL.
    assert result.calibrated_pass is True, (
        f"mat≈0.085 < TAU_SCI_BENCHMARK=0.15 should be REVIEW (PASS). "
        f"z_crit={result.calibrated_z_crit:.3f}, n_fail={result.calibrated_n_fail}, "
        f"n_review={result.calibrated_n_review}"
    )
    assert result.calibrated_n_fail == 0
    assert (
        result.calibrated_n_review is not None and result.calibrated_n_review >= 1
    ), "hot dim (mat≈0.085) should be REVIEW under TAU_SCI_BENCHMARK=0.15"


def test_benchmark_tau_sci_hard_fail_at_genuine_bias() -> None:
    """mat=0.2σ (genuine bias) → hard FAIL at _TAU_SCI_BENCHMARK=0.15.

    Confirms the benchmark gate still catches real biases above 0.15σ,
    so relaxing TAU_SCI_BENCHMARK from 0.05 to 0.15 is not a blanket amnesty.
    """
    from tuningfork.calibration._gate.gt_compare import _compute_gt_compare

    D = 503
    n_draws = 5000
    rng = np.random.default_rng(0)

    se_sample = 1.0 / math.sqrt(n_draws)
    se_gt = 1.0 / math.sqrt(40000)
    denom = math.sqrt(se_sample**2 + se_gt**2)

    target_z = 6.0  # > z_crit=3.892
    target_mat = 0.2  # > _TAU_SCI_BENCHMARK=0.15 → hard FAIL

    delta = target_z * denom
    std_b_hot = delta / target_mat  # std_b calibrated so mat = 0.20

    arr = rng.normal(0.0, 1.0, (1, n_draws, D))
    arr[:, :, 0] += delta
    mc = {"x": arr}
    gt = {
        "x": {
            "mean": [0.0] * D,
            "std": [std_b_hot] + [1.0] * (D - 1),
            "n_samples": 40000,
        }
    }

    result = _compute_gt_compare(mc, gt, min_bulk_ess=None)

    # At TAU_SCI_BENCHMARK=0.15: mat=0.2 > 0.15 → hard FAIL.
    assert result.calibrated_pass is False, (
        f"mat≈0.20 > TAU_SCI_BENCHMARK=0.15 should hard-FAIL. "
        f"z_crit={result.calibrated_z_crit:.3f}, n_fail={result.calibrated_n_fail}"
    )
    assert result.calibrated_n_fail is not None and result.calibrated_n_fail >= 1


# ---------------------------------------------------------------------------
# Benchmark gate integration: _compute_gt_compare calibrated verdict
# ---------------------------------------------------------------------------


def _make_mc_samples(
    n_chains: int, n_draws: int, D: int, mean: float = 0.0
) -> dict[str, np.ndarray]:
    """Synthetic mc_samples dict (n_chains, n_draws, D) with given mean."""
    rng = np.random.default_rng(42)
    arr = rng.normal(mean, 1.0, (n_chains, n_draws, D))
    return {"x": arr}


def _make_gt_summary(
    D: int, gt_mean: float = 0.0, n_samples: int = 40000
) -> dict[str, dict]:
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
    # Benchmark path always uses normal-Bonferroni (df → ∞) → calibrated_nu=None
    assert result.calibrated_nu is None


def test_gt_compare_calibrated_fail_large_bias() -> None:
    """Large bias benchmark run → calibrated_pass=False (hard FAIL dim)."""
    from tuningfork.calibration._gate.gt_compare import _compute_gt_compare

    # shift of 50 std → z >> z_crit and mat = 50 >> TAU_SCI
    mc = _make_mc_samples(n_chains=1, n_draws=1000, D=10, mean=50.0)
    gt = _make_gt_summary(D=10, gt_mean=0.0)

    result = _compute_gt_compare(mc, gt, min_bulk_ess=None)

    assert result.calibrated_pass is False
    assert result.calibrated_n_fail is not None and result.calibrated_n_fail >= 1


def test_gt_compare_calibrated_review_immaterial_high_d() -> None:
    """High-D benchmark (D=503, stoch_vol-sized): z>z_crit but mat<TAU_SCI → REVIEW (PASS).

    Under normal-Bonferroni at D=503 (benchmark regime, large df):
        z_crit = norm.ppf(1 - 0.05/(2*503)) ≈ 3.892

    Constructs one "hot" dim with z≈5.5 > z_crit but |Δμ|/std_b ≈ 0.032 < TAU_SCI=0.05.
    This dim is classified REVIEW (not FAIL) → calibrated_pass=True.

    This is the stoch_vol scenario: a 503-dim model crosses the dimension-aware
    threshold by chance, but the per-dim bias is immaterial relative to the
    posterior spread.  Under the OLD fixed gate (z<4.0), z=4.335 was a hard
    FAIL with no materiality escape.

    Design notes:
    - n_draws=5000 makes background max_z ≈ 2.9 << z_crit=3.892 (very safe margin).
    - std_b_bg=2.0: if any background dim trips z_crit by chance, its
      mat ≈ 0.034 < TAU_SCI → REVIEW (not FAIL) regardless.
    - denom_hot uses std_b_hot for se_gt (the actual denom, not std=1 shortcut).
    """
    from tuningfork.calibration._gate.gt_compare import _compute_gt_compare

    D = 503
    n_draws = 5000
    std_b_hot = 5.0  # hot dim: large GT spread → immaterial high-z bias
    std_b_bg = 2.0  # background: mat < TAU_SCI even if z > z_crit by accident
    rng = np.random.default_rng(0)

    # Compute denom using actual se_gt for the hot dim (std_b_hot, not std=1).
    se_sample_approx = 1.0 / math.sqrt(n_draws)  # ≈ 0.01414
    se_gt_hot = std_b_hot / math.sqrt(40000)  # = 0.025
    denom_hot = math.sqrt(se_sample_approx**2 + se_gt_hot**2)  # ≈ 0.02874

    # target_z=5.5 >> z_crit≈3.892; 3σ lower bound ≈ 4.03 > 3.89
    target_z = 5.5
    delta = target_z * denom_hot  # ≈ 0.158; mat = 0.158/5.0 = 0.032 < 0.05

    arr = rng.normal(0.0, 1.0, (1, n_draws, D))
    arr[:, :, 0] += delta  # shift only the hot dim
    mc = {"x": arr}
    gt = {
        "x": {
            "mean": [0.0] * D,
            "std": [std_b_hot] + [std_b_bg] * (D - 1),
            "n_samples": 40000,
        }
    }

    result = _compute_gt_compare(mc, gt, min_bulk_ess=None)

    # Normal z_crit at D=503 ≈ 3.892
    assert result.calibrated_z_crit is not None
    assert result.calibrated_z_crit == pytest.approx(3.892, rel=1e-3)

    # Immaterial high-z dim(s) → REVIEW → calibrated PASS
    assert result.calibrated_pass is True, (
        f"D=503 immaterial case should PASS. "
        f"z_crit={result.calibrated_z_crit:.3f}, max_z={result.max_abs_mean_z:.3f}, "
        f"n_fail={result.calibrated_n_fail}, n_review={result.calibrated_n_review}"
    )
    assert result.calibrated_n_fail == 0
    # Dim 0 is designed to trip z_crit but remain immaterial → REVIEW
    assert result.calibrated_n_review is not None and result.calibrated_n_review >= 1, (
        f"Hot dim (target_z={target_z}) should be REVIEW (z>z_crit, mat<TAU_SCI). "
        f"Got n_review={result.calibrated_n_review}, max_z={result.max_abs_mean_z:.3f}"
    )
    # Benchmark path always uses normal-approx (no finite nu)
    assert result.calibrated_nu is None


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
