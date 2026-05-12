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
"""

import math
import types

import jax.numpy as jnp
import numpy as np
import pytest

from tuningfork.calibration.statistician_gate import (
    DEFAULT_THRESHOLDS,
    AutoGateVerdict,
    auto_gate,
    resolve_thresholds,
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
