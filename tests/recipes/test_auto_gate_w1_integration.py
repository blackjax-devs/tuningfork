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
"""Integration tests for the W1/σ two-prong gate wired into auto_gate().

Tests
-----
1.  test_w1_realm_runs_when_stage1_passes
        Stage-3 PASS (good R̂/ESS/div) + gt_draws provided → W1 realm runs,
        w1_realm_result is not None, margins["w1_realm"] present.

2.  test_w1_realm_skipped_when_rhat_fails
        Stage-3 FAIL (high R̂) + gt_draws provided → W1 realm skipped,
        w1_realm_result is None, no "w1_realm" key in margins.

3.  test_w1_realm_skipped_when_gt_draws_absent
        Stage-3 PASS + gt_draws=None → W1 realm skipped even when
        ground_truth_summaries is provided.

4.  test_w1_realm_skipped_in_vi_mode
        vi_sampler_mode=True + gt_draws provided → W1 realm skipped.

5.  test_w1_null_case_does_not_add_top_level_to_dict_key
        to_dict() keys unchanged (schema test) even when W1 was run; W1
        scalars live under margins["w1_realm"], not as a new top-level key.

6.  test_w1_verdict_propagates_to_overall
        Inject a large W1 shift → overall verdict is FAIL if W1 prong fails.
"""

import types

import numpy as np
import pytest

from tuningfork.calibration.statistician_gate import auto_gate

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared constants — small enough for sub-second W1 bootstrap (D=3, n=200)
# ---------------------------------------------------------------------------

_D = 3
_N_CHAINS = 4
_N_DRAWS = 200
_N_GT_CHAINS = 2
_N_GT_DRAWS = 500


def _make_gt(
    rng: np.random.Generator,
    *,
    n_gt_chains: int = _N_GT_CHAINS,
    n_gt_draws: int = _N_GT_DRAWS,
    dim: int = _D,
    mean_shift: float = 0.0,
) -> tuple[dict, dict]:
    """Synthetic GT draws + matching summaries.  Returns (gt_draws, gt_summaries)."""
    gt_arr = rng.normal(size=(n_gt_chains, n_gt_draws, dim))
    gt_arr += mean_shift
    gt_flat = gt_arr.reshape(-1, dim)
    gt_draws = {"x": gt_arr}
    gt_summaries = {
        "x": {
            "mean": gt_flat.mean(axis=0).tolist(),
            "std": np.maximum(gt_flat.std(axis=0), 0.1).tolist(),
            "bulk_ess": [float(n_gt_chains * n_gt_draws)] * dim,
            "tail_ess": [float(n_gt_chains * n_gt_draws)] * dim,
        }
    }
    return gt_draws, gt_summaries


def _make_gen(
    rng: np.random.Generator,
    *,
    n_chains: int = _N_CHAINS,
    n_draws: int = _N_DRAWS,
    dim: int = _D,
    mean_shift: float = 0.0,
) -> dict:
    """Well-mixed i.i.d. N(mean_shift, 1) samples, shape (n_chains, n_draws, dim)."""
    return {"x": rng.normal(size=(n_chains, n_draws, dim)) + mean_shift}


def _make_info(n_chains: int, n_samples: int, *, n_divergences: int = 0):
    """Mock sampler info with is_divergent bool array."""
    flat = np.zeros(n_chains * n_samples, dtype=bool)
    flat[:n_divergences] = True
    return types.SimpleNamespace(is_divergent=flat.reshape(n_chains, n_samples))


# ---------------------------------------------------------------------------
# Test 1 — Stage-3 PASS + gt_draws → W1 runs
# ---------------------------------------------------------------------------


def test_w1_realm_runs_when_stage1_passes():
    """W1 realm activates when R̂/ESS/div pass and gt_draws is provided."""
    rng = np.random.default_rng(1)
    gt_draws, gt_summaries = _make_gt(rng)
    gen = _make_gen(rng)
    info = _make_info(_N_CHAINS, _N_DRAWS)

    verdict = auto_gate(
        gen,
        info,
        ground_truth_summaries=gt_summaries,
        gt_draws=gt_draws,
    )

    # Stage-3 metrics should be clean
    assert verdict.rhat_max is not None and verdict.rhat_max < 1.01
    assert verdict.min_bulk_ess is not None and verdict.min_bulk_ess >= 400.0
    assert verdict.n_divergences == 0

    # W1 realm must have run
    assert verdict.w1_realm_result is not None, "Expected W1 realm result, got None"
    assert "w1_realm" in verdict.margins, "Expected 'w1_realm' key in margins"

    # Scalar stats in margins["w1_realm"]
    w1_m = verdict.margins["w1_realm"]
    for key in ("verdict", "max_w1_sigma", "floor_of_max", "frac_failing_dims"):
        assert key in w1_m, f"Expected key '{key}' in margins['w1_realm']"

    # Under null (gen ~ GT), the null max W1 should be small and W1 should PASS
    assert w1_m["verdict"] in {
        "PASS",
        "REVIEW",
    }, f"Expected PASS/REVIEW for null case, got {w1_m['verdict']}"


# ---------------------------------------------------------------------------
# Test 2 — Stage-3 FAIL (high R̂) → W1 skipped
# ---------------------------------------------------------------------------


def test_w1_realm_skipped_when_rhat_fails():
    """W1 realm is not run when R̂ FAILs (stage-3 gate)."""
    rng = np.random.default_rng(2)
    gt_draws, gt_summaries = _make_gt(rng)

    # Build chains with wildly different means → very high R̂
    n_chains, n_draws, dim = 4, 200, _D
    offsets = np.arange(n_chains)[:, None, None] * 10.0
    gen = {"x": rng.normal(size=(n_chains, n_draws, dim)) + offsets}
    info = _make_info(n_chains, n_draws)

    verdict = auto_gate(
        gen,
        info,
        ground_truth_summaries=gt_summaries,
        gt_draws=gt_draws,
    )

    # R̂ must be in FAIL territory
    assert verdict.rhat_max is not None and verdict.rhat_max >= 1.05

    # W1 realm must have been skipped
    assert (
        verdict.w1_realm_result is None
    ), "Expected W1 realm to be skipped when R̂ FAILs"
    assert (
        "w1_realm" not in verdict.margins
    ), "Expected no 'w1_realm' key in margins when W1 was skipped"

    assert verdict.verdict == "FAIL"


# ---------------------------------------------------------------------------
# Test 3 — gt_draws absent → W1 skipped even if stage-3 passes
# ---------------------------------------------------------------------------


def test_w1_realm_skipped_when_gt_draws_absent():
    """W1 realm is not run when gt_draws is None, even with ground_truth_summaries."""
    rng = np.random.default_rng(3)
    _, gt_summaries = _make_gt(rng)
    gen = _make_gen(rng)
    info = _make_info(_N_CHAINS, _N_DRAWS)

    verdict = auto_gate(
        gen,
        info,
        ground_truth_summaries=gt_summaries,
        gt_draws=None,
    )

    assert verdict.w1_realm_result is None
    assert "w1_realm" not in verdict.margins


# ---------------------------------------------------------------------------
# Test 4 — vi_sampler_mode → W1 skipped
# ---------------------------------------------------------------------------


def test_w1_realm_skipped_in_vi_mode():
    """W1 realm is not run when vi_sampler_mode=True."""
    rng = np.random.default_rng(4)
    gt_draws, gt_summaries = _make_gt(rng)
    gen = _make_gen(rng)
    info = _make_info(_N_CHAINS, _N_DRAWS)

    verdict = auto_gate(
        gen,
        info,
        ground_truth_summaries=gt_summaries,
        gt_draws=gt_draws,
        vi_sampler_mode=True,
    )

    assert verdict.w1_realm_result is None, "W1 realm must not run in VI mode"
    assert "w1_realm" not in verdict.margins


# ---------------------------------------------------------------------------
# Test 5 — to_dict() schema unchanged even with W1 run
# ---------------------------------------------------------------------------


def test_w1_null_case_does_not_add_top_level_to_dict_key():
    """to_dict() top-level keys unchanged; W1 lives in margins['w1_realm']."""
    rng = np.random.default_rng(5)
    gt_draws, gt_summaries = _make_gt(rng)
    gen = _make_gen(rng)
    info = _make_info(_N_CHAINS, _N_DRAWS)

    verdict = auto_gate(
        gen,
        info,
        ground_truth_summaries=gt_summaries,
        gt_draws=gt_draws,
    )
    assert verdict.w1_realm_result is not None  # W1 must have run

    d = verdict.to_dict()
    expected_top_level = {
        "rhat_max",
        "min_bulk_ess",
        "n_divergences",
        "max_abs_mean_z",
        "verdict",
        "margins",
    }
    assert (
        set(d.keys()) == expected_top_level
    ), f"Unexpected top-level keys in to_dict(): {set(d.keys()) - expected_top_level}"
    # W1 lives under margins
    assert "w1_realm" in d["margins"]


# ---------------------------------------------------------------------------
# Test 6 — large W1 shift propagates FAIL to overall verdict
# ---------------------------------------------------------------------------


def test_w1_verdict_propagates_to_overall():
    """A large mean-shift in gen vs GT drives the W1 max prong to FAIL."""
    rng = np.random.default_rng(6)
    gt_draws, gt_summaries = _make_gt(rng)

    # Shift generated samples by +5σ on all dims → W1/σ >> floor
    gen = _make_gen(rng, mean_shift=5.0)
    info = _make_info(_N_CHAINS, _N_DRAWS)

    verdict = auto_gate(
        gen,
        info,
        ground_truth_summaries=gt_summaries,
        gt_draws=gt_draws,
    )

    assert verdict.w1_realm_result is not None
    w1_m = verdict.margins["w1_realm"]
    assert (
        w1_m["max_prong_verdict"] == "FAIL"
    ), f"Expected max prong FAIL with +5σ shift, got {w1_m['max_prong_verdict']}"
    # Overall verdict must propagate the W1 FAIL
    assert (
        verdict.verdict == "FAIL"
    ), f"Expected overall FAIL from W1 prong, got {verdict.verdict}"
