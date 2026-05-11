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
"""Integration tests for the BO tuning loop body.

Tests the actual ``tune_algorithm`` implementation for mass-matrix gradient
kernels (NUTS, HMC).  The NotImplementedError paths for MALA and MCLMC are
also exercised to confirm the dispatch logic.

Test parameters chosen to keep total runtime < 60s on CPU:
- n_warmup=200, n_samples=200: enough for a sane mass matrix on 10-D MVN
- n_trials=3: enough to exercise BO improvement without being slow
- n_seeds=1, n_chains=1: keeps the test matrix simple; structure is correct

Empirical findings (recorded after the first successful run):
1. Window adaptation at n_warmup=200 on MVN-10: observed in test output.
2. Per-trial JIT compile: each trial with a new concrete step_size value
   may trigger a fresh JAX JIT compilation.  The per-trial wall-time is
   recorded in result.history and accessible for inspection.
3. NUTS default vs best on MVN-10 (3 trials): result.difficulty.default_works
   is determined empirically; the test verifies the logical invariants
   rather than a fixed numeric outcome.
"""

import math

import jax
import pytest

from bjx_bench.calibration.tune import (
    TuningDifficulty,
    TuningResult,
    default_params_for,
    tune_algorithm,
)
from bjx_bench.inference.base_method import BASE_METHODS
from bjx_bench.model import MODELS

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

_N_WARMUP = 200
_N_SAMPLES = 200
_N_TRIALS = 3
_N_SEEDS = 1
_N_CHAINS = 1

_MVN_ENTRY = MODELS["mvn_10"]
_NUTS_ENTRY = BASE_METHODS["nuts"]
_HMC_ENTRY = BASE_METHODS["hmc"]
_MALA_ENTRY = BASE_METHODS["mala"]
_MCLMC_ENTRY = BASE_METHODS["mclmc"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_nuts_mvn(
    rng_key: jax.Array,
    n_trials: int = _N_TRIALS,
    n_seeds: int = _N_SEEDS,
    n_chains: int = _N_CHAINS,
) -> TuningResult:
    """Convenience wrapper for NUTS + MVN-10 runs."""
    return tune_algorithm(
        _MVN_ENTRY,
        _NUTS_ENTRY,
        n_trials=n_trials,
        n_seeds=n_seeds,
        n_chains=n_chains,
        n_samples=_N_SAMPLES,
        n_warmup=_N_WARMUP,
        rng_key=rng_key,
    )


# ---------------------------------------------------------------------------
# 1. NUTS + MVN-10 smoke test (n_trials=3)
# ---------------------------------------------------------------------------


class TestNutsMvnSmoke:
    """Smoke test: NUTS on MVN-10 with 3 Optuna trials."""

    def test_result_base_method_name(self) -> None:
        result = _run_nuts_mvn(jax.random.key(0))
        assert result.base_method_name == "nuts"

    def test_result_posterior_name(self) -> None:
        result = _run_nuts_mvn(jax.random.key(1))
        assert result.posterior_name == "mvn_10"

    def test_n_trials_completed(self) -> None:
        result = _run_nuts_mvn(jax.random.key(2))
        assert result.n_trials_completed == _N_TRIALS

    def test_history_length(self) -> None:
        result = _run_nuts_mvn(jax.random.key(3))
        assert len(result.history) == _N_TRIALS

    def test_best_score_gte_default_score(self) -> None:
        """BO can only improve: best_score >= default_score (trial 0)."""
        result = _run_nuts_mvn(jax.random.key(4))
        assert result.best_score >= result.difficulty.default_score, (
            f"best_score={result.best_score:.4f} < "
            f"default_score={result.difficulty.default_score:.4f}: "
            "BO should never produce a result worse than the best seen trial"
        )

    def test_n_trials_to_best_in_range(self) -> None:
        """n_trials_to_best is between 0 and n_trials-1 inclusive."""
        result = _run_nuts_mvn(jax.random.key(5))
        assert 0 <= result.difficulty.n_trials_to_best < _N_TRIALS, (
            f"n_trials_to_best={result.difficulty.n_trials_to_best} "
            f"should be in [0, {_N_TRIALS - 1}]"
        )

    def test_best_score_is_finite(self) -> None:
        """At least the default config should produce a finite score on MVN-10."""
        result = _run_nuts_mvn(jax.random.key(6))
        assert math.isfinite(result.best_score), (
            f"best_score={result.best_score} is non-finite; "
            "NUTS on a 10-D MVN should never diverge with default HPs"
        )

    def test_result_is_tuning_result(self) -> None:
        result = _run_nuts_mvn(jax.random.key(7))
        assert isinstance(result, TuningResult)

    def test_difficulty_is_tuning_difficulty(self) -> None:
        result = _run_nuts_mvn(jax.random.key(8))
        assert isinstance(result.difficulty, TuningDifficulty)

    def test_n_seeds_recorded(self) -> None:
        result = _run_nuts_mvn(jax.random.key(9))
        assert result.n_seeds == _N_SEEDS


# ---------------------------------------------------------------------------
# 2. HMC + MVN-10 smoke test (n_trials=3)
# ---------------------------------------------------------------------------


class TestHmcMvnSmoke:
    """HMC on MVN-10: both step_size and num_integration_steps are tuned."""

    def _run(self, seed: int) -> TuningResult:
        return tune_algorithm(
            _MVN_ENTRY,
            _HMC_ENTRY,
            n_trials=_N_TRIALS,
            n_seeds=_N_SEEDS,
            n_chains=_N_CHAINS,
            n_samples=_N_SAMPLES,
            n_warmup=_N_WARMUP,
            rng_key=jax.random.key(seed),
        )

    def test_base_method_name(self) -> None:
        result = self._run(10)
        assert result.base_method_name == "hmc"

    def test_best_params_has_step_size(self) -> None:
        """HMC best_params must contain step_size."""
        result = self._run(11)
        assert (
            "step_size" in result.best_params
        ), f"best_params keys: {list(result.best_params.keys())}"

    def test_best_params_has_num_integration_steps(self) -> None:
        """HMC best_params must contain num_integration_steps."""
        result = self._run(12)
        assert (
            "num_integration_steps" in result.best_params
        ), f"best_params keys: {list(result.best_params.keys())}"

    def test_step_size_in_bounds(self) -> None:
        """step_size must be within [1e-3, 1.0]."""
        result = self._run(13)
        ss = result.best_params["step_size"]
        assert 1e-3 <= ss <= 1.0, f"step_size={ss} out of [1e-3, 1.0]"

    def test_num_integration_steps_in_bounds(self) -> None:
        """num_integration_steps must be in [1, 128]."""
        result = self._run(14)
        ns = result.best_params["num_integration_steps"]
        assert 1 <= ns <= 128, f"num_integration_steps={ns} out of [1, 128]"

    def test_best_score_is_finite(self) -> None:
        result = self._run(15)
        assert math.isfinite(
            result.best_score
        ), f"HMC best_score={result.best_score} is non-finite on MVN-10"

    def test_n_trials_completed(self) -> None:
        result = self._run(16)
        assert result.n_trials_completed == _N_TRIALS


# ---------------------------------------------------------------------------
# 3. Trial 0 = default convention
# ---------------------------------------------------------------------------


class TestTrialZeroIsDefault:
    """history[0]['params'] must equal default_params_for(algorithm_entry)."""

    def test_nuts_trial_0_params_equal_defaults(self) -> None:
        """Trial 0 for NUTS must use the deterministic default step_size."""
        result = _run_nuts_mvn(jax.random.key(20))
        expected = default_params_for(_NUTS_ENTRY)
        actual = result.history[0]["params"]
        assert set(actual.keys()) == set(expected.keys()), (
            f"Trial 0 params keys {set(actual.keys())} != "
            f"expected keys {set(expected.keys())}"
        )
        for key, exp_val in expected.items():
            act_val = actual[key]
            if isinstance(exp_val, float):
                assert act_val == pytest.approx(
                    exp_val, rel=1e-6
                ), f"Trial 0 param {key!r}: got {act_val}, expected {exp_val}"
            else:
                assert (
                    act_val == exp_val
                ), f"Trial 0 param {key!r}: got {act_val}, expected {exp_val}"

    def test_history_0_has_all_keys(self) -> None:
        """Each history record must carry trial, params, score, certified, wall_seconds."""
        result = _run_nuts_mvn(jax.random.key(21))
        rec = result.history[0]
        assert "trial" in rec
        assert "params" in rec
        assert "score" in rec
        assert "certified" in rec
        assert "wall_seconds" in rec

    def test_history_0_trial_index_is_0(self) -> None:
        result = _run_nuts_mvn(jax.random.key(22))
        assert result.history[0]["trial"] == 0


# ---------------------------------------------------------------------------
# 4. TuningDifficulty invariants
# ---------------------------------------------------------------------------


class TestTuningDifficultyInvariants:
    """Structural invariants of the difficulty profile from a real run."""

    def test_threshold_formula(self) -> None:
        """threshold_score == max(default_score, 0.5 * best_score)."""
        result = _run_nuts_mvn(jax.random.key(30))
        d = result.difficulty
        expected_threshold = max(d.default_score, 0.5 * d.best_score)
        assert d.threshold_score == pytest.approx(expected_threshold, rel=1e-6), (
            f"threshold_score={d.threshold_score:.4f}, "
            f"expected max({d.default_score:.4f}, {0.5 * d.best_score:.4f})="
            f"{expected_threshold:.4f}"
        )

    def test_default_works_flag(self) -> None:
        """default_works is True iff default_score >= 0.5 * best_score."""
        result = _run_nuts_mvn(jax.random.key(31))
        d = result.difficulty
        expected_default_works = d.default_score >= 0.5 * d.best_score
        assert d.default_works == expected_default_works

    def test_n_trials_to_threshold_zero_iff_default_works(self) -> None:
        """n_trials_to_threshold == 0 iff default_works is True."""
        result = _run_nuts_mvn(jax.random.key(32))
        d = result.difficulty
        if d.default_works:
            assert d.n_trials_to_threshold == 0, (
                f"default_works=True but n_trials_to_threshold="
                f"{d.n_trials_to_threshold} (expected 0)"
            )
        else:
            assert d.n_trials_to_threshold > 0, (
                "default_works=False but n_trials_to_threshold=0 "
                "(should indicate at least one non-default trial needed)"
            )

    def test_wall_seconds_to_best_positive(self) -> None:
        """wall_seconds_to_best must be > 0 (at least one trial was run)."""
        result = _run_nuts_mvn(jax.random.key(33))
        assert result.difficulty.wall_seconds_to_best > 0.0


# ---------------------------------------------------------------------------
# MALA dispatch: these paths are wired — smoke-check they run
# ---------------------------------------------------------------------------


class TestMalaDispatchWired:
    """Sanity check: tune_algorithm no longer raises NotImplementedError for MALA.

    These were previously NotImplementedError tests; the dispatch now wires MALA/RWM
    dispatch (no warmup), so these tests confirm the dispatch is active.
    Full coverage lives in tests/test_tier_b_dispatch.py.
    """

    def test_does_not_raise_not_implemented(self) -> None:
        # Should complete without raising NotImplementedError.
        result = tune_algorithm(
            _MVN_ENTRY,
            _MALA_ENTRY,
            n_trials=1,
            n_seeds=1,
            n_chains=1,
            n_samples=50,
            n_warmup=50,
            rng_key=jax.random.key(0),
        )
        assert isinstance(result, TuningResult)
        assert result.base_method_name == "mala"


# ---------------------------------------------------------------------------
# MCLMC dispatch: these paths are wired — smoke-check they run
# ---------------------------------------------------------------------------


class TestMclmcDispatchWired:
    """Sanity check: tune_algorithm no longer raises NotImplementedError for MCLMC.

    These were previously NotImplementedError tests; the dispatch now wires MCLMC dispatch
    via mclmc_find_L_and_step_size, so these tests confirm the dispatch is
    active.  Full coverage lives in tests/test_tier_b_dispatch.py.
    """

    def test_does_not_raise_not_implemented(self) -> None:
        # Should complete without raising NotImplementedError.
        result = tune_algorithm(
            _MVN_ENTRY,
            _MCLMC_ENTRY,
            n_trials=1,
            n_seeds=1,
            n_chains=1,
            n_samples=50,
            n_warmup=200,
            rng_key=jax.random.key(0),
        )
        assert isinstance(result, TuningResult)
        assert result.base_method_name == "mclmc"
