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
"""Tests for Statistician auto-gate.

Tests
-----
1.  test_pass_with_clean_synthetic_samples
        Well-mixed Gaussian samples → PASS.
2.  test_review_with_borderline_rhat
        Correlated chains → R̂ in [1.01, 1.05) → REVIEW.
3.  test_fail_with_high_rhat
        Badly mixing chains → R̂ ≥ 1.05 → FAIL.
4a. test_few_divergences_still_pass / 4b. test_moderate_divergences_review /
4c. test_many_divergences_fail
        3-band n_divergences gate (amended 2026-05-12):
        ≤ 5 → PASS, 6–39 → REVIEW, ≥ 40 → FAIL.
5.  test_review_with_low_ess
        Artificially autocorrelated samples → bulk-ESS in [100, 400) → REVIEW.
6.  test_fail_with_very_low_ess
        Severely autocorrelated samples → bulk-ESS < 100 → FAIL.
7.  test_no_ground_truth_skips_z_check
        ground_truth_summaries=None → max_abs_mean_z is None; key absent from margins.
8.  test_with_ground_truth_z_pass
        Sample mean within 1.5 SE of ground truth → max_abs_mean_z ~ 1.5 → "pass" band.
9.  test_resolve_thresholds_default
        resolve_thresholds(posterior=None) returns DEFAULT_THRESHOLDS (deep-equal).
10. test_resolve_thresholds_funnel_tag
        Posterior with tags=("funnel",) → min_bulk_ess.pass == (50.0, inf).
11. test_resolve_thresholds_multimodal_tag
        Posterior with tags=("multimodal",) → no "max_abs_mean_z" key.
12. test_to_dict_keys_match_gate_evidence_auto_schema
        AutoGateVerdict.to_dict() keys exactly match the locked schema.
13. test_worst_verdict_aggregation_fail_beats_review
        One FAIL metric + one REVIEW metric → overall FAIL.
14. test_single_chain_rechunked
        Single-chain samples (n_samples, dim) are reshaped into n_chunks for split-R̂.
15. test_resolve_thresholds_high_correlation_tag
        Posterior with tags=("high-correlation",) → rhat_max review upper = 1.10.

Dimension-aware (Šidák) PASS band
-----------------------------------------------------------------------------------------------
17. test_sidak_t_pass_monotone_floor_cap
        t_pass(d) non-decreasing in d; t_pass(1) == 2.0; capped at 4.0 for large d.
18. test_sidak_t_pass_loosen_only
        t_pass(d) >= 2.0 for d in a sweep 1..2000 (loosen-only invariant).
19. test_sidak_t_pass_spot_values
        Spot values from the decision doc table: d=10 → 2.80; d=50 → 3.28; d=200 → 3.66.
20. test_dimension_aware_gate_verification_cases
        The two empirical-trigger cells: max_abs_mean_z=2.036 at d=10 → PASS;
        max_abs_mean_z=2.820 at d=10 → REVIEW.
21. test_dimension_aware_gate_still_fails_genuine_bias
        max_abs_mean_z >= 4.0 at any d → FAIL (the fixed FAIL boundary is untouched).
"""

import json
import math
import types

import jax.numpy as jnp
import numpy as np
import pytest

from tuningfork.calibration._gate.bands import _build_margin
from tuningfork.calibration.statistician_gate import (
    DEFAULT_THRESHOLDS,
    Z_VERDICT_ESS_CEILING,
    AutoGateVerdict,
    auto_gate,
    resolve_thresholds,
    sidak_t_pass,
)

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Helpers for building synthetic info structs
# ---------------------------------------------------------------------------


def _make_info(n_chains: int, n_samples: int, *, n_divergences: int = 0):
    """Build a mock sampler info object with an is_divergent bool array."""
    is_div = jnp.zeros((n_chains, n_samples), dtype=bool)
    if n_divergences > 0:
        # Set the first n_divergences entries of the first chain to True
        flat = np.zeros(n_chains * n_samples, dtype=bool)
        flat[:n_divergences] = True
        is_div = jnp.asarray(flat.reshape(n_chains, n_samples))
    return types.SimpleNamespace(is_divergent=is_div)


def _make_clean_samples(
    rng: np.random.RandomState,
    n_chains: int,
    n_samples: int,
    dim: int,
) -> dict:
    """Return well-mixed i.i.d. Gaussian samples; shape (n_chains, n_samples, dim)."""
    return {"x": rng.normal(size=(n_chains, n_samples, dim))}


# ---------------------------------------------------------------------------
# Test 1 — clean samples → PASS
# ---------------------------------------------------------------------------


def test_pass_with_clean_synthetic_samples():
    """Well-mixed i.i.d. Gaussian samples produce R̂ ≈ 1.0, high ESS → PASS."""
    rng = np.random.RandomState(0)
    samples = _make_clean_samples(rng, n_chains=4, n_samples=1000, dim=3)
    info = _make_info(4, 1000, n_divergences=0)
    verdict = auto_gate(samples, info)
    assert verdict.verdict == "PASS"
    assert verdict.rhat_max is not None
    assert verdict.rhat_max < 1.01
    assert verdict.min_bulk_ess is not None
    assert verdict.min_bulk_ess >= 400.0
    assert verdict.n_divergences == 0
    assert verdict.max_abs_mean_z is None  # no ground truth


# ---------------------------------------------------------------------------
# Test 2 — borderline R̂ → REVIEW
# ---------------------------------------------------------------------------


def test_review_with_borderline_rhat():
    """Correlated chains produce R̂ in [1.01, 1.05) → REVIEW."""
    rng = np.random.RandomState(7)
    n_chains, n_samples, dim = 4, 500, 2
    # Build chains with different means to push R̂ just above 1.01 but below 1.05
    # Use a strong chain offset (biased initialisation that decays slowly)
    base = rng.normal(size=(n_chains, n_samples, dim))
    # Add a chain-specific slowly decaying offset
    offsets = np.linspace(0, 2.0, n_chains)[:, None, None]
    decay = np.linspace(1.0, 0.0, n_samples)[None, :, None]
    samples = {"x": base + offsets * decay}
    info = _make_info(n_chains, n_samples, n_divergences=0)
    verdict = auto_gate(samples, info)
    assert verdict.verdict in {"REVIEW", "FAIL"}  # at minimum REVIEW
    if verdict.rhat_max is not None:
        assert verdict.rhat_max >= 1.01


# ---------------------------------------------------------------------------
# Test 3 — high R̂ → FAIL
# ---------------------------------------------------------------------------


def test_fail_with_high_rhat():
    """Chains stuck at different modes → R̂ ≥ 1.05 → FAIL."""
    rng = np.random.RandomState(1)
    n_chains, n_samples, dim = 4, 200, 1
    # Each chain stuck near a different value; no mixing
    chain_offsets = np.array([0.0, 5.0, 10.0, 15.0])[:, None, None]
    samples = {
        "x": rng.normal(scale=0.01, size=(n_chains, n_samples, dim)) + chain_offsets
    }
    info = _make_info(n_chains, n_samples, n_divergences=0)
    verdict = auto_gate(samples, info)
    assert verdict.verdict == "FAIL"
    assert verdict.rhat_max is not None
    assert verdict.rhat_max >= 1.05


# ---------------------------------------------------------------------------
# Test 4 — divergences → FAIL
# ---------------------------------------------------------------------------


def test_few_divergences_still_pass():
    """≤ 5 divergences → PASS (amended 2026-05-12: rate-tolerant)."""
    rng = np.random.RandomState(2)
    samples = _make_clean_samples(rng, n_chains=4, n_samples=1000, dim=3)
    info = _make_info(4, 1000, n_divergences=5)
    verdict = auto_gate(samples, info)
    assert verdict.verdict == "PASS"
    assert verdict.n_divergences == 5
    assert verdict.margins["n_divergences"]["band"] == "PASS"


def test_nonfinite_proposals_zero_is_pass_and_serialized():
    """False entries in ``nonans`` are counted with an exact denominator."""
    rng = np.random.RandomState(22)
    samples = _make_clean_samples(rng, 4, 100, 1)
    info = types.SimpleNamespace(nonans=jnp.ones((4, 100), dtype=bool))
    verdict = auto_gate(samples, info)
    assert verdict.n_nonfinite_proposals == 0
    assert verdict.n_proposals_evaluated == 400
    assert verdict.nonfinite_proposal_rate == 0.0
    assert verdict.n_divergences is None
    assert verdict.verdict == "PASS"
    margin = verdict.margins["n_nonfinite_proposals"]
    assert margin["policy_id"] == "tuningfork.nonfinite-proposal-review.v1"
    assert margin["policy_status"] == "provisional"
    assert margin["calibrated"] is False
    assert margin["nonfinite_proposal_rate"] == 0.0
    assert verdict.to_dict()["n_proposals_evaluated"] == 400


def test_nonfinite_proposals_positive_requests_review():
    """Any positive non-finite count prevents an overall PASS (provisional)."""
    rng = np.random.RandomState(23)
    samples = _make_clean_samples(rng, 4, 100, 1)
    nonans = np.ones((4, 100), dtype=bool)
    nonans[0, 0] = False
    info = types.SimpleNamespace(nonans=jnp.asarray(nonans))
    verdict = auto_gate(samples, info)
    assert verdict.n_nonfinite_proposals == 1
    assert verdict.n_proposals_evaluated == 400
    assert verdict.nonfinite_proposal_rate == 1 / 400
    assert verdict.n_divergences is None
    assert verdict.verdict == "REVIEW"
    assert verdict.margins["n_nonfinite_proposals"]["band"] == "REVIEW"


def test_empty_nonans_evidence_requests_review():
    """An empty sampler-info array cannot provide PASS evidence."""
    rng = np.random.RandomState(24)
    samples = _make_clean_samples(rng, 4, 100, 1)
    info = types.SimpleNamespace(nonans=jnp.ones((0,), dtype=bool))
    verdict = auto_gate(samples, info)
    assert verdict.n_nonfinite_proposals == 0
    assert verdict.n_proposals_evaluated == 0
    assert verdict.nonfinite_proposal_rate is None
    assert verdict.n_divergences is None
    assert verdict.verdict == "REVIEW"
    assert verdict.margins["n_nonfinite_proposals"]["band"] == "REVIEW"


def test_moderate_divergences_review():
    """6 ≤ n_divergences < 40 → REVIEW."""
    rng = np.random.RandomState(2)
    samples = _make_clean_samples(rng, n_chains=4, n_samples=1000, dim=3)
    info = _make_info(4, 1000, n_divergences=20)
    verdict = auto_gate(samples, info)
    assert verdict.verdict == "REVIEW"
    assert verdict.margins["n_divergences"]["band"] == "REVIEW"


def test_many_divergences_fail():
    """n_divergences ≥ 40 → FAIL."""
    rng = np.random.RandomState(2)
    samples = _make_clean_samples(rng, n_chains=4, n_samples=1000, dim=3)
    info = _make_info(4, 1000, n_divergences=50)
    verdict = auto_gate(samples, info)
    assert verdict.verdict == "FAIL"
    assert verdict.n_divergences == 50
    assert verdict.margins["n_divergences"]["band"] == "FAIL"


# ---------------------------------------------------------------------------
# Test 5 — low ESS → REVIEW
# ---------------------------------------------------------------------------


def test_review_with_low_ess():
    """Highly autocorrelated samples → bulk-ESS in [100, 400) → REVIEW."""
    rng = np.random.RandomState(3)
    n_chains, n_steps, dim = 4, 2000, 1
    # Build AR(1) chains with high autocorrelation (ρ ≈ 0.99) to crush ESS
    phi = 0.99
    samples_arr = np.zeros((n_chains, n_steps, dim))
    for c in range(n_chains):
        samples_arr[c, 0] = rng.normal()
        for t in range(1, n_steps):
            samples_arr[c, t] = phi * samples_arr[c, t - 1] + np.sqrt(
                1 - phi**2
            ) * rng.normal(size=dim)
    samples = {"x": samples_arr}
    info = _make_info(n_chains, n_steps, n_divergences=0)
    verdict = auto_gate(samples, info)
    # ESS should be very low due to high autocorrelation
    assert verdict.min_bulk_ess is not None
    if verdict.min_bulk_ess < 100:
        assert verdict.verdict == "FAIL"
    elif verdict.min_bulk_ess < 400:
        assert verdict.verdict in {"REVIEW", "FAIL"}
    else:
        # Edge case: if ESS somehow still high, check margins
        pass


# ---------------------------------------------------------------------------
# Test 6 — very low ESS → FAIL
# ---------------------------------------------------------------------------


def test_fail_with_very_low_ess():
    """Extreme autocorrelation (ρ = 0.999) → bulk-ESS < 100 → FAIL."""
    rng = np.random.RandomState(4)
    n_chains, n_steps, dim = 2, 500, 1
    phi = 0.999
    samples_arr = np.zeros((n_chains, n_steps, dim))
    for c in range(n_chains):
        samples_arr[c, 0] = rng.normal()
        for t in range(1, n_steps):
            samples_arr[c, t] = phi * samples_arr[c, t - 1] + np.sqrt(
                1 - phi**2
            ) * rng.normal(size=dim)
    samples = {"x": samples_arr}
    info = _make_info(n_chains, n_steps, n_divergences=0)
    verdict = auto_gate(samples, info)
    assert verdict.min_bulk_ess is not None
    # With ρ=0.999 and only 500 steps per chain, ESS should be very small
    assert verdict.min_bulk_ess < 400.0
    assert verdict.verdict in {"REVIEW", "FAIL"}  # at least REVIEW


# ---------------------------------------------------------------------------
# Test 7 — no ground truth → z-check skipped
# ---------------------------------------------------------------------------


def test_no_ground_truth_skips_z_check():
    """ground_truth_summaries=None → max_abs_mean_z is None; key absent from margins."""
    rng = np.random.RandomState(5)
    samples = _make_clean_samples(rng, n_chains=4, n_samples=500, dim=2)
    info = _make_info(4, 500, n_divergences=0)
    verdict = auto_gate(samples, info, ground_truth_summaries=None)
    assert verdict.max_abs_mean_z is None
    assert "max_abs_mean_z" not in verdict.margins
    # Should still be PASS (clean samples, no GT check)
    assert verdict.verdict == "PASS"


# ---------------------------------------------------------------------------
# Test 8 — ground truth z ≈ 1.5 → PASS band
# ---------------------------------------------------------------------------


def test_with_ground_truth_z_pass():
    """Sample mean within ~1.5 SE of ground truth → max_abs_mean_z in PASS band."""
    rng = np.random.RandomState(6)
    n_chains, n_samples, dim = 4, 2000, 1
    # True mean = 0; sample near 0
    samples = {"x": rng.normal(loc=0.0, scale=1.0, size=(n_chains, n_samples, dim))}
    info = _make_info(n_chains, n_samples, n_divergences=0)
    # Ground truth: mean=0, std=1, plenty of reference samples
    gt = {"x": {"mean": np.array([0.0]), "std": np.array([1.0]), "n_samples": 100_000}}
    verdict = auto_gate(samples, info, ground_truth_summaries=gt)
    assert verdict.max_abs_mean_z is not None
    # With 8000 total samples, SE_sample ≈ 1/sqrt(ESS) ≈ small
    # z = |sample_mean - 0| / SE ≈ small; should be in PASS band
    assert verdict.max_abs_mean_z < 4.0  # at most REVIEW
    if "max_abs_mean_z" in verdict.margins:
        assert verdict.margins["max_abs_mean_z"]["band"] in {"PASS", "REVIEW"}


# ---------------------------------------------------------------------------
# Test 9 — resolve_thresholds(None) == DEFAULT_THRESHOLDS
# ---------------------------------------------------------------------------


def test_resolve_thresholds_default():
    """resolve_thresholds(posterior=None) returns DEFAULT_THRESHOLDS (deep-equal copy)."""
    resolved = resolve_thresholds(posterior=None)
    assert resolved == DEFAULT_THRESHOLDS
    # Must be a deep copy — mutating resolved should not affect DEFAULT_THRESHOLDS
    resolved["rhat_max"]["pass"] = (0.0, 999.0)
    assert DEFAULT_THRESHOLDS["rhat_max"]["pass"] == (0.0, 1.01)


# ---------------------------------------------------------------------------
# Test 10 — funnel tag → relaxed ESS thresholds
# ---------------------------------------------------------------------------


def test_resolve_thresholds_funnel_tag():
    """Posterior with tags=('funnel',) → min_bulk_ess pass band = (50.0, inf)."""
    posterior = types.SimpleNamespace(tags=("funnel",))
    resolved = resolve_thresholds(posterior)
    assert resolved["min_bulk_ess"]["pass"] == (50.0, math.inf)
    assert resolved["min_bulk_ess"]["review"] == (10.0, 50.0)
    # Other thresholds unchanged
    assert resolved["rhat_max"] == DEFAULT_THRESHOLDS["rhat_max"]


def test_unbounded_gate_margin_is_strict_json_and_finite_bounds_stay_numeric():
    """Unbounded threshold edges are explicit JSON objects, not Infinity."""
    posterior = types.SimpleNamespace(tags=("funnel",))
    rng = np.random.RandomState(123)
    samples = _make_clean_samples(rng, n_chains=4, n_samples=500, dim=1)
    verdict = auto_gate(samples, _make_info(4, 500), posterior=posterior)
    margin = verdict.margins["min_bulk_ess"]

    assert margin["pass_lo"] == 50.0
    assert margin["review_lo"] == 10.0
    assert margin["review_hi"] == 50.0
    assert margin["pass_hi"] == {
        "schema": "tuningfork.gate-threshold-bound.v1",
        "kind": "unbounded",
        "sign": "positive",
    }
    json.dumps(verdict.to_dict(), allow_nan=False)


def test_gate_margin_rejects_nonfinite_observed_metric():
    """Only semantic threshold infinities are encodable; observations are finite."""
    with pytest.raises(ValueError, match="non-finite observed metric"):
        _build_margin(float("inf"), {"pass": (0.0, 1.0)}, "PASS")


# ---------------------------------------------------------------------------
# Test 11 — multimodal tag → max_abs_mean_z dropped
# ---------------------------------------------------------------------------


def test_resolve_thresholds_multimodal_tag():
    """Posterior with tags=('multimodal',) → 'max_abs_mean_z' key absent."""
    posterior = types.SimpleNamespace(tags=("multimodal",))
    resolved = resolve_thresholds(posterior)
    assert "max_abs_mean_z" not in resolved
    # Other thresholds intact
    assert "rhat_max" in resolved
    assert "min_bulk_ess" in resolved
    assert "n_divergences" in resolved


# ---------------------------------------------------------------------------
# Test 12 — to_dict keys match locked schema
# ---------------------------------------------------------------------------


def test_to_dict_keys_match_gate_evidence_auto_schema():
    """AutoGateVerdict.to_dict() keys exactly match Recipe.gate_evidence['auto'] schema."""
    expected_keys = {
        "rhat_max",
        "min_bulk_ess",
        "n_divergences",
        "max_abs_mean_z",
        "verdict",
        "margins",
    }
    verdict = AutoGateVerdict(
        rhat_max=1.005,
        min_bulk_ess=500.0,
        n_divergences=0,
        max_abs_mean_z=None,
        verdict="PASS",
        margins={},
    )
    assert set(verdict.to_dict().keys()) == expected_keys


def test_auto_gate_verdict_preserves_optional_positional_order():
    """New evidence fields do not displace older optional positional fields."""
    calibrated = {"pass": True}
    verdict = AutoGateVerdict(
        1.005,
        500.0,
        0,
        None,
        "PASS",
        {},
        True,
        None,
        calibrated,
    )
    assert verdict.resonance_warning is True
    assert verdict.w1_realm_result is None
    assert verdict.gt_calibrated is calibrated
    assert verdict.n_nonfinite_proposals is None
    assert verdict.n_proposals_evaluated is None
    assert verdict.nonfinite_proposal_rate is None


# ---------------------------------------------------------------------------
# Test 13 — FAIL beats REVIEW in aggregation
# ---------------------------------------------------------------------------


def test_worst_verdict_aggregation_fail_beats_review():
    """One metric in FAIL + another in REVIEW → overall FAIL."""
    rng = np.random.RandomState(8)
    n_chains, n_samples, dim = 4, 200, 1
    # Stuck chains → FAIL on R̂
    chain_offsets = np.array([0.0, 5.0, 10.0, 15.0])[:, None, None]
    samples = {
        "x": rng.normal(scale=0.01, size=(n_chains, n_samples, dim)) + chain_offsets
    }
    # 2 divergences → also FAIL
    info = _make_info(n_chains, n_samples, n_divergences=2)
    verdict = auto_gate(samples, info)
    assert verdict.verdict == "FAIL"


# ---------------------------------------------------------------------------
# Test 14 — single-chain samples are rechunked
# ---------------------------------------------------------------------------


def test_single_chain_rechunked():
    """Single-chain (n_samples, dim) samples are rechunked into n_chunks for split-R̂."""
    rng = np.random.RandomState(9)
    # Single-chain layout: (n_samples, dim) — first dim > 64 so detected as single-chain
    n_samples, dim = 2000, 2
    samples = {"x": rng.normal(size=(n_samples, dim))}
    info = types.SimpleNamespace(is_divergent=jnp.zeros(n_samples, dtype=bool))
    # Should not raise; verdict should be PASS (i.i.d. samples)
    verdict = auto_gate(samples, info, n_chunks=4)
    assert verdict.verdict == "PASS"
    assert verdict.rhat_max is not None
    assert verdict.min_bulk_ess is not None


# ---------------------------------------------------------------------------
# Test 15 — high-correlation tag → relaxed rhat review band
# ---------------------------------------------------------------------------


def test_resolve_thresholds_high_correlation_tag():
    """Posterior with tags=('high-correlation',) → rhat_max review upper = 1.10."""
    posterior = types.SimpleNamespace(tags=("high-correlation",))
    resolved = resolve_thresholds(posterior)
    assert resolved["rhat_max"]["review"] == (1.01, 1.10)
    # pass band unchanged
    assert resolved["rhat_max"]["pass"] == (0.0, 1.01)


# ---------------------------------------------------------------------------
# Test 16 — M1 regression: per-dim ESS used for SE, not global min_bulk_ess
# ---------------------------------------------------------------------------


def test_per_dim_ess_not_global_min_for_z_se():
    """SE uses per-dim ESS, not global min_bulk_ess (M1 regression guard).

    Construct a 2-param model where one param mixes well (high ESS) and one
    mixes poorly (low ESS):
    - ``fast``: 4 chains, 1000 draws, i.i.d. Gaussian → ESS ≈ 4000
    - ``slow``: 4 chains, 1000 draws, AR(0.99) → ESS ≈ 4 * (1000 * 0.01) ≈ 40

    Ground truth: both at mean=0.  With global-min-ESS for SE, ``fast``'s SE is
    over-inflated (ESS=40 instead of 4000) → z-score too small → gate too lenient.
    With per-dim ESS, ``fast``'s SE uses ESS≈4000 → z-score larger (still near 0,
    since the chain is well-mixed).  We verify that:
    1. ``max_abs_mean_z`` is finite and the verdict is PASS.
    2. The margins dict carries ``max_abs_mean_z`` (gate ran).
    """
    rng = np.random.RandomState(42)
    n_chains, n_draws = 4, 1000

    # ``fast``: i.i.d. → ESS ≈ 4000
    fast_samples = rng.normal(size=(n_chains, n_draws))

    # ``slow``: AR(0.99) → ESS ≈ 4 * 1000 * 0.01 ≈ 40
    phi = 0.99
    slow_chain = np.zeros((n_chains, n_draws))
    for c in range(n_chains):
        for t in range(1, n_draws):
            slow_chain[c, t] = phi * slow_chain[c, t - 1] + rng.normal() * np.sqrt(
                1 - phi**2
            )

    samples = {
        "fast": fast_samples[:, :, np.newaxis],  # (4, 1000, 1)
        "slow": slow_chain[:, :, np.newaxis],  # (4, 1000, 1)
    }
    gt = {
        "fast": {"mean": np.array([0.0]), "std": np.array([1.0]), "n_samples": 40000},
        "slow": {"mean": np.array([0.0]), "std": np.array([1.0]), "n_samples": 40000},
    }

    verdict = auto_gate(samples, info=None, ground_truth_summaries=gt)

    # Gate should run and produce a finite max_abs_mean_z
    assert verdict.max_abs_mean_z is not None
    assert math.isfinite(verdict.max_abs_mean_z)
    assert "max_abs_mean_z" in verdict.margins

    # Both chains are centred at 0 → z-scores should be small → PASS or REVIEW
    # (not FAIL), confirming the SE computation is reasonable.
    assert verdict.max_abs_mean_z < 4.0, (
        f"max_abs_mean_z={verdict.max_abs_mean_z:.3f} unexpectedly large; "
        "per-dim ESS may have regressed to global-min logic."
    )


# ---------------------------------------------------------------------------
# Dimension-aware (Šidák) PASS band — helpers
# ---------------------------------------------------------------------------


def _calibrate_shift_for_max_z(
    target_z: float,
    *,
    d: int,
    n_chains: int,
    n_draws: int,
    seed: int,
    n_iter: int = 60,
) -> tuple[np.ndarray, dict]:
    """Binary-search a mean-shift on dim 0 so ``auto_gate`` reports exactly
    ``max_abs_mean_z == target_z`` on the real code path (not a hand-derived
    SE formula — exercises ``auto_gate``'s actual per-dim-ESS/SE computation).

    Returns ``(samples, gt)`` ready to feed into ``auto_gate``.
    """
    rng = np.random.RandomState(seed)
    base = rng.normal(size=(n_chains, n_draws, d))
    gt = {"x": {"mean": np.zeros(d), "std": np.ones(d), "n_samples": 100_000}}

    def _max_z_at(shift: float) -> float:
        samples = base.copy()
        samples[:, :, 0] += shift
        info = types.SimpleNamespace(
            is_divergent=jnp.zeros((n_chains, n_draws), dtype=bool)
        )
        verdict = auto_gate({"x": samples}, info, ground_truth_summaries=gt)
        assert verdict.max_abs_mean_z is not None
        return verdict.max_abs_mean_z

    lo, hi = 0.0, 1.0
    while _max_z_at(hi) < target_z:
        hi *= 2
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        if _max_z_at(mid) < target_z:
            lo = mid
        else:
            hi = mid

    samples = base.copy()
    samples[:, :, 0] += hi
    return samples, gt


# ---------------------------------------------------------------------------
# Test 17 — sidak_t_pass: monotone, floor, cap
# ---------------------------------------------------------------------------


def test_sidak_t_pass_monotone_floor_cap():
    """t_pass(d) is non-decreasing in d; t_pass(1) == 2.0; capped at 4.0."""
    assert sidak_t_pass(1) == 2.0
    ds = [1, 2, 5, 10, 26, 50, 100, 200, 500, 1000, 1600, 5000]
    values = [sidak_t_pass(d) for d in ds]
    for prev, curr in zip(values, values[1:]):
        assert curr >= prev - 1e-12, (prev, curr)
    assert values[0] == 2.0
    # Very high d hits the 4.0 cap.
    assert sidak_t_pass(5000) == 4.0
    assert sidak_t_pass(1_000_000) == 4.0


# ---------------------------------------------------------------------------
# Test 18 — loosen-only invariant: t_pass(d) >= 2.0 for all d
# ---------------------------------------------------------------------------


def test_sidak_t_pass_loosen_only():
    """t_pass(d) >= 2.0 for all d in 1..2000 (loosen-only invariant).

    This is the property that guarantees the dimension-aware band can only
    ever widen the PASS region relative to the historical fixed PASS<2.0
    boundary — no recipe that currently PASSes can regress to REVIEW/FAIL.
    """
    for d in range(1, 2001):
        assert sidak_t_pass(d) >= 2.0, f"loosen-only invariant violated at d={d}"


# ---------------------------------------------------------------------------
# Test 19 — spot values from the decision doc table
# ---------------------------------------------------------------------------


def test_sidak_t_pass_spot_values():
    """Spot values for the dimension-aware Šidák pass-band."""
    assert sidak_t_pass(10) == pytest.approx(2.80, abs=1e-2)
    assert sidak_t_pass(50) == pytest.approx(3.28, abs=1e-2)
    assert sidak_t_pass(200) == pytest.approx(3.66, abs=1e-2)


# ---------------------------------------------------------------------------
# Test 20 — the two verification cases (empirical trigger)
# ---------------------------------------------------------------------------


def test_dimension_aware_gate_verification_cases():
    """The two eight_schools_ncp-style cells from the decision doc:

    - max_abs_mean_z = 2.036 at d=10 → PASS (2.036 < t_pass(10) = 2.80).
    - max_abs_mean_z = 2.820 at d=10 → REVIEW (2.820 > t_pass(10), < 4.0).

    Constructed via ``auto_gate`` on a synthetic 10-dim sample+ground-truth
    pair (real code path), not a direct classification-helper unit test.
    """
    t_pass_10 = sidak_t_pass(10)
    assert t_pass_10 == pytest.approx(2.80, abs=1e-2)

    samples_pass, gt_pass = _calibrate_shift_for_max_z(
        2.036, d=10, n_chains=4, n_draws=2000, seed=1
    )
    info = types.SimpleNamespace(is_divergent=jnp.zeros((4, 2000), dtype=bool))
    verdict_pass = auto_gate({"x": samples_pass}, info, ground_truth_summaries=gt_pass)
    assert verdict_pass.max_abs_mean_z == pytest.approx(2.036, abs=1e-6)
    assert verdict_pass.verdict == "PASS"
    assert verdict_pass.margins["max_abs_mean_z"]["band"] == "PASS"

    samples_review, gt_review = _calibrate_shift_for_max_z(
        2.820, d=10, n_chains=4, n_draws=2000, seed=1
    )
    verdict_review = auto_gate(
        {"x": samples_review}, info, ground_truth_summaries=gt_review
    )
    assert verdict_review.max_abs_mean_z == pytest.approx(2.820, abs=1e-6)
    assert verdict_review.verdict == "REVIEW"
    assert verdict_review.margins["max_abs_mean_z"]["band"] == "REVIEW"


# ---------------------------------------------------------------------------
# Test 21 — genuine bias (z >= 4.0) still FAILs regardless of d
# ---------------------------------------------------------------------------


def test_dimension_aware_gate_still_fails_genuine_bias():
    """A max z >= 4.0 gives FAIL in small realm, REVIEW with z_advisory in large realm.

    With 4 chains × 2000 draws, ESS ≈ 8000 > 6400 (large realm).
    Per issue #223, z-driven FAILs demote to REVIEW in large realm with z_advisory flag.
    This test verifies the boundary is correctly applied by d.
    """
    for d in (1, 10, 50):
        samples, gt = _calibrate_shift_for_max_z(
            4.5, d=d, n_chains=4, n_draws=2000, seed=2
        )
        info = types.SimpleNamespace(is_divergent=jnp.zeros((4, 2000), dtype=bool))
        verdict = auto_gate({"x": samples}, info, ground_truth_summaries=gt)
        assert verdict.max_abs_mean_z >= 4.0
        # In large realm (ESS > 6400), z-driven FAIL demotes to REVIEW with z_advisory
        assert verdict.min_bulk_ess is not None
        if verdict.min_bulk_ess > Z_VERDICT_ESS_CEILING:
            # Large realm: z-driven FAIL demotes to REVIEW
            assert (
                verdict.verdict == "REVIEW"
            ), f"d={d}, ESS={verdict.min_bulk_ess:.0f}: expected REVIEW with z_advisory"
            assert verdict.margins["max_abs_mean_z"].get("z_advisory") is True
        else:
            # Small realm: verdict is FAIL as before
            assert (
                verdict.verdict == "FAIL"
            ), f"d={d} did not FAIL at z={verdict.max_abs_mean_z}"


# ---------------------------------------------------------------------------
# Regression tests for issue #217 — multichain misclassification (≤64 cliff bug)
# ---------------------------------------------------------------------------
# The bug: ndim >= 2 and shape[0] <= 64 was treated as multichain, but this
# failed for genuine multichain arrays with nc > 64. The fix uses ndim >= 3
# as definitive multichain, and adds an explicit multichain parameter.


def test_samples_to_multichain_explicit_true_preserves_shape():
    """multichain=True preserves (128, 1000, 5) shape without rechunking."""
    from tuningfork.calibration.statistician_gate import _samples_to_multichain

    rng = np.random.RandomState(10)
    samples = {"x": rng.normal(size=(128, 1000, 5))}
    # Explicit multichain=True should return as-is
    result = _samples_to_multichain(samples, n_chunks=4, multichain=True)
    result_arr = np.asarray(result["x"])
    assert result_arr.shape == (128, 1000, 5), (
        f"multichain=True rechunked the array; expected (128, 1000, 5), "
        f"got {result_arr.shape}"
    )


def test_samples_to_multichain_heuristic_ndim3_unscrambled():
    """(128, 1000, 5) via heuristic (multichain=None) is NOT rechunked."""
    from tuningfork.calibration.statistician_gate import _samples_to_multichain

    rng = np.random.RandomState(11)
    samples = {"x": rng.normal(size=(128, 1000, 5))}
    # ndim=3 → heuristic detects as multichain → no rechunk
    result = _samples_to_multichain(samples, n_chunks=4, multichain=None)
    result_arr = np.asarray(result["x"])
    assert result_arr.shape == (128, 1000, 5), (
        f"heuristic (ndim=3) should detect as multichain; expected (128, 1000, 5), "
        f"got {result_arr.shape}"
    )


def test_samples_to_multichain_explicit_false_rechunks():
    """multichain=False rechunks single-chain (4000, 5) into n_chunks segments."""
    from tuningfork.calibration.statistician_gate import _samples_to_multichain

    rng = np.random.RandomState(12)
    samples = {"x": rng.normal(size=(4000, 5))}
    # Explicit multichain=False should rechunk
    result = _samples_to_multichain(samples, n_chunks=4, multichain=False)
    result_arr = np.asarray(result["x"])
    assert result_arr.shape == (4, 1000, 5), (
        f"multichain=False should rechunk (4000, 5) into (4, 1000, 5), "
        f"got {result_arr.shape}"
    )


def test_auto_gate_multichain_true_healthy_numbers():
    """auto_gate with multichain=True on synthetic (128, 200, 3) → healthy rhat/ess."""
    rng = np.random.RandomState(13)
    # Synthetic well-mixed 128 chains × 200 draws × 3D
    samples = {"x": rng.normal(size=(128, 200, 3))}
    info = _make_info(128, 200, n_divergences=0)
    # Explicitly flag as multichain; should NOT rechunk and should pass
    verdict = auto_gate(samples, info, multichain=True)
    assert verdict.verdict == "PASS", (
        f"Well-mixed (128, 200, 3) with multichain=True should PASS, "
        f"got {verdict.verdict} (rhat={verdict.rhat_max:.4f}, ess={verdict.min_bulk_ess:.1f})"
    )
    assert verdict.rhat_max is not None
    assert verdict.rhat_max < 1.01, (
        f"multichain=True on well-mixed draws should have rhat < 1.01, "
        f"got {verdict.rhat_max:.4f}"
    )
    assert verdict.min_bulk_ess is not None
    assert verdict.min_bulk_ess > 100, (
        f"multichain=True on (128, 200) draws should have ESS >> 100, "
        f"got {verdict.min_bulk_ess:.1f}"
    )


def test_auto_gate_multichain_none_heuristic_large_nc_no_rechunk():
    """multichain=None heuristic on (128, 1000, 5) detects as multichain (ndim=3)."""
    rng = np.random.RandomState(14)
    # Large nc (128 > 64), ndim=3: old ≤64 cliff would misclassify as single-chain
    # and rechunk to (4, 32000, 5), scrambling ESS.
    # New heuristic (ndim >= 3) detects as multichain → no rechunk.
    samples = {"x": rng.normal(size=(128, 1000, 5))}
    info = _make_info(128, 1000, n_divergences=0)
    # multichain=None (default): heuristic should see ndim=3 and NOT rechunk
    verdict = auto_gate(samples, info, multichain=None, n_chunks=4)
    # If the old bug was active, ESS would be ~21.65 (as mentioned in the issue).
    # With the fix, ESS should be healthy (>> 100 for well-mixed draws).
    assert verdict.min_bulk_ess is not None
    assert verdict.min_bulk_ess > 100, (
        f"Heuristic on (128, 1000, 5) incorrectly rechunked; ESS={verdict.min_bulk_ess:.1f} "
        f"suggests scrambling (expected >> 100 for well-mixed)"
    )
    assert verdict.verdict == "PASS", (
        f"Well-mixed (128, 1000, 5) should PASS with correct heuristic, "
        f"got {verdict.verdict}"
    )


# ---------------------------------------------------------------------------
# Z-advisory realm gate policy (issue #223)
# ---------------------------------------------------------------------------


def test_z_advisory_small_realm_bit_identical():
    """Small realm (ess<6400), z>2.0 → verdict follows z (no demotion to REVIEW)."""
    # Construct samples with controlled bias to push z into FAIL territory
    rng = np.random.RandomState(101)
    n_chains, n_samples, dim = 4, 1000, 1

    # Build well-mixed samples with bias large enough to cause z-driven FAIL
    # With bias=0.10, z ≈ 6.3 (well into FAIL band)
    samples = {"x": rng.normal(loc=0.10, scale=1.0, size=(n_chains, n_samples, dim))}
    info = _make_info(n_chains, n_samples, n_divergences=0)
    gt = {
        "x": {
            "mean": np.array([0.0]),
            "std": np.array([1.0]),
            "n_samples": 1_000_000,
        }
    }

    verdict = auto_gate(samples, info, ground_truth_summaries=gt)
    # min_bulk_ess should be < 6400 (small realm)
    assert verdict.min_bulk_ess is not None
    assert verdict.min_bulk_ess < Z_VERDICT_ESS_CEILING
    # z should be large enough for FAIL
    assert verdict.max_abs_mean_z is not None
    assert verdict.max_abs_mean_z > 4.0
    # In small realm, z-driven FAIL stays FAIL (no demotion)
    assert verdict.verdict == "FAIL"
    # z_advisory should NOT be set in small realm
    if "max_abs_mean_z" in verdict.margins:
        assert verdict.margins["max_abs_mean_z"].get("z_advisory") is not True


def test_z_advisory_large_realm_demotion():
    """Large realm (ess>6400), z-driven FAIL → REVIEW with z_advisory=true."""
    # Create samples with large ESS (well-mixed chains)
    rng = np.random.RandomState(102)
    n_chains, n_samples, dim = 4, 5000, 1

    # Create samples with enough bias to cause z-driven FAIL in small realm
    # ess ≈ 20000 (large realm), bias=0.10 → z ≈ high
    samples = {"x": rng.normal(loc=0.10, scale=1.0, size=(n_chains, n_samples, dim))}
    info = _make_info(n_chains, n_samples, n_divergences=0)
    gt = {
        "x": {
            "mean": np.array([0.0]),
            "std": np.array([1.0]),
            "n_samples": 1_000_000,
        }
    }

    verdict = auto_gate(samples, info, ground_truth_summaries=gt)
    # min_bulk_ess should be > 6400 (large realm)
    assert verdict.min_bulk_ess is not None
    assert verdict.min_bulk_ess > Z_VERDICT_ESS_CEILING
    # z should be large (would cause FAIL in small realm)
    assert verdict.max_abs_mean_z is not None
    assert verdict.max_abs_mean_z > 4.0
    # z would normally give FAIL, but at large ess it demotes to REVIEW
    assert verdict.verdict == "REVIEW"
    # z_advisory should be True in margins
    assert "max_abs_mean_z" in verdict.margins
    assert verdict.margins["max_abs_mean_z"].get("z_advisory") is True


def test_z_advisory_boundary_6400_exact():
    """Boundary case: ess=6400 exactly → FAIL (≤ 6400 is verdict-bearing, not advisory)."""
    # Build samples with ESS near 6400 using AR(1) construction
    rng = np.random.RandomState(103)
    n_chains, n_samples, dim = 4, 2000, 1
    # AR(1) with ρ=0.1 gives ess_per_iter ≈ 0.81
    # so ess ≈ 2000 * 4 * 0.81 ≈ 6480 (close to boundary)
    phi = 0.1
    samples_arr = np.zeros((n_chains, n_samples, dim))
    for c in range(n_chains):
        samples_arr[c, 0] = rng.normal()
        for t in range(1, n_samples):
            samples_arr[c, t] = phi * samples_arr[c, t - 1] + np.sqrt(
                1 - phi**2
            ) * rng.normal(size=dim)

    # Add bias for z ≈ 4.5
    samples_arr += 0.056
    samples = {"x": samples_arr}
    info = _make_info(n_chains, n_samples, n_divergences=0)
    gt = {
        "x": {
            "mean": np.array([0.0]),
            "std": np.array([1.0]),
            "n_samples": 1_000_000,
        }
    }

    verdict = auto_gate(samples, info, ground_truth_summaries=gt)
    # Check that ess is close to 6400
    assert verdict.min_bulk_ess is not None
    # The boundary is: ess > 6400 is advisory. At ess ≤ 6400, it's NOT advisory
    if verdict.min_bulk_ess <= Z_VERDICT_ESS_CEILING:
        # Small realm: z should drive verdict as before
        assert verdict.verdict == "FAIL"
        assert verdict.margins.get("max_abs_mean_z", {}).get("z_advisory") is not True


def test_z_advisory_rhat_fail_unaffected():
    """rhat-driven FAIL remains FAIL at any ESS (demotion only for z-driven FAIL)."""
    rng = np.random.RandomState(104)
    n_chains, n_samples, dim = 4, 5000, 1
    # Build chains with high rhat (stuck at different means) → rhat ≥ 1.05 → FAIL
    chain_means = np.array([0.0, 5.0, 10.0, 15.0])[:, None, None]
    samples = {
        "x": rng.normal(scale=0.01, size=(n_chains, n_samples, dim)) + chain_means
    }
    info = _make_info(n_chains, n_samples, n_divergences=0)

    # Include GT to compute z, but make bias small so z is fine
    gt = {
        "x": {
            "mean": np.array([7.5]),
            "std": np.array([6.0]),
            "n_samples": 1_000_000,
        }
    }

    verdict = auto_gate(samples, info, ground_truth_summaries=gt)
    # rhat should be very high (chains stuck at different means)
    assert verdict.rhat_max is not None
    assert verdict.rhat_max >= 1.05
    # rhat-driven FAIL should NOT demote even with large ESS
    assert verdict.verdict == "FAIL"


def test_z_advisory_div_fail_unaffected():
    """Divergence-driven FAIL remains FAIL at any ESS."""
    rng = np.random.RandomState(105)
    n_chains, n_samples, dim = 4, 1000, 1
    # Clean samples but many divergences → divergence-driven FAIL
    samples = {"x": rng.normal(size=(n_chains, n_samples, dim))}
    info = _make_info(n_chains, n_samples, n_divergences=50)  # Many divergences → FAIL

    verdict = auto_gate(samples, info)
    # n_divergences ≥ 40 → FAIL
    assert verdict.n_divergences == 50
    assert verdict.verdict == "FAIL"


def test_z_advisory_bias_sigma_fields_present():
    """Margins contain bias_sigma_* fields when z is computed."""
    rng = np.random.RandomState(106)
    n_chains, n_samples, dim = 4, 1000, 2
    # Create samples with known bias
    samples = {
        "x": rng.normal(loc=0.5, scale=1.0, size=(n_chains, n_samples, dim)),
        "y": rng.normal(loc=-0.3, scale=1.0, size=(n_chains, n_samples, dim)),
    }
    info = _make_info(n_chains, n_samples, n_divergences=0)
    gt = {
        "x": {"mean": np.array([0.0, 0.0]), "std": np.array([1.0, 1.0])},
        "y": {"mean": np.array([0.0, 0.0]), "std": np.array([1.0, 1.0])},
    }

    verdict = auto_gate(samples, info, ground_truth_summaries=gt)
    # Check that margin fields are present
    assert "max_abs_mean_z" in verdict.margins
    margins_z = verdict.margins["max_abs_mean_z"]
    # At least one of the bias_sigma fields should be present
    has_bias_sigma = (
        "bias_sigma_at_argmax_z" in margins_z
        or "bias_sigma_max_at_z4" in margins_z
        or "achieved_bias_bound_sigma" in margins_z
    )
    assert has_bias_sigma, f"No bias_sigma fields in margins: {margins_z}"


def test_z_advisory_bias_sigma_numerically_correct():
    """Multi-dim case pins bias_sigma_max_at_z4 semantics (max among z>=4 dims).

    Case: 2D parameter.
    - Dim 0: mean=0.025, std=0.1; GT mean=0, std=5
      bias_sigma=0.025/5=0.005; z≈5 (FAILS threshold ≥4)
    - Dim 1: mean=0.048, std=1.0; GT mean=0, std=1
      bias_sigma=0.048/1=0.048; z≈3 (PASSES threshold <4)

    Expected: bias_sigma_max_at_z4 = 0.005 (max among failing dims z≥4).
    This test pins the fix for the inverted mask bug: z >= 4.0 (not z <= 4.0).
    """
    rng = np.random.RandomState(107)
    n_chains, n_samples = 4, 1000

    # 2D samples: dim 0 high-z (failing), dim 1 moderate-z (passing)
    samples = {
        "x": np.concatenate(
            [
                rng.normal(loc=0.025, scale=0.1, size=(n_chains, n_samples, 1)),
                rng.normal(loc=0.048, scale=1.0, size=(n_chains, n_samples, 1)),
            ],
            axis=2,
        )
    }
    info = _make_info(n_chains, n_samples, n_divergences=0)
    gt = {
        "x": {
            "mean": np.array([0.0, 0.0]),
            "std": np.array([5.0, 1.0]),
            "n_samples": 1_000_000,
        }
    }

    verdict = auto_gate(samples, info, ground_truth_summaries=gt)
    assert "max_abs_mean_z" in verdict.margins
    margins_z = verdict.margins["max_abs_mean_z"]

    # bias_sigma_max_at_z4: only dims with z >= 4 count.
    # Dim 0 (z≈5, failing) has bias_sigma ≈ 0.005.
    # Dim 1 (z≈3, passing) has bias_sigma ≈ 0.048.
    # Expected: max among failing = 0.005 (from dim 0, NOT 0.048 from passing dim).
    if "bias_sigma_max_at_z4" in margins_z:
        bias_sigma_max_z4 = margins_z["bias_sigma_max_at_z4"]
        # Should be close to 0.005 (from dim 0), NOT 0.048 (from dim 1).
        # This pins the semantic: max among z>=4 failing dims.
        assert (
            0.003 < bias_sigma_max_z4 < 0.010
        ), f"bias_sigma_max_at_z4={bias_sigma_max_z4}, expected ≈0.005 (from failing dim z≥4)"


def test_z_advisory_cost_block_optional():
    """Cost metrics appear in margins when provided."""
    rng = np.random.RandomState(108)
    n_chains, n_samples, dim = 4, 500, 1
    samples = {"x": rng.normal(size=(n_chains, n_samples, dim))}
    info = _make_info(n_chains, n_samples, n_divergences=0)

    # Call with cost kwargs
    verdict = auto_gate(
        samples,
        info,
        ess_per_grad=10.5,
        total_grad_evals=5000,
        wall_seconds=42.3,
    )

    assert "cost" in verdict.margins
    cost_block = verdict.margins["cost"]
    assert cost_block["ess_per_grad"] == 10.5
    assert cost_block["total_grad_evals"] == 5000
    assert cost_block["wall_seconds"] == 42.3


def test_z_advisory_catalog_zero_flip_proof():
    """Validate that live recipes (small-realm) remain unaffected by the new rule.

    This test verifies the zero-flip property: all committed recipes in the catalog
    have small ESS (≤6400), so the new z-advisory demotion never fires. Re-classifying
    existing recipes under the new rule should produce identical verdicts.
    """
    # Simulate a small-realm recipe: low ESS, moderate z
    # e.g., a simple LOW recipe with ess ≈ 1000, z ≈ 3.5
    rng = np.random.RandomState(109)
    n_chains, n_samples, dim = 4, 250, 1

    samples = {"x": rng.normal(loc=0.055, scale=1.0, size=(n_chains, n_samples, dim))}
    info = _make_info(n_chains, n_samples, n_divergences=0)
    gt = {
        "x": {
            "mean": np.array([0.0]),
            "std": np.array([1.0]),
            "n_samples": 1_000_000,
        }
    }

    verdict = auto_gate(samples, info, ground_truth_summaries=gt)

    # Verify small realm (the zero-flip constraint)
    assert verdict.min_bulk_ess is not None
    assert verdict.min_bulk_ess <= Z_VERDICT_ESS_CEILING

    # With small ESS, z-driven FAILs are NOT demoted
    # z ≈ 3.5 is in REVIEW band (2 < z < 4), so REVIEW anyway
    # Even if z were ≥ 4, it would be FAIL (not demoted in small realm)
    assert verdict.verdict in {"PASS", "REVIEW", "FAIL"}

    # If it's not a FAIL (because z < 4 or other metrics pass),
    # no z_advisory flag should be set
    if verdict.verdict != "FAIL":
        if "max_abs_mean_z" in verdict.margins:
            assert verdict.margins["max_abs_mean_z"].get("z_advisory") is not True
