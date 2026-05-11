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
"""Dispatch tests for BO tuning extensions.

Covers:
  (i)   sampler="tpe"|"random" argument routing through Optuna samplers.
  (ii)  MALA + RWM dispatch (no warmup; trial params flow directly).
  (iii) MCLMC dispatch via mclmc_find_L_and_step_size.
  (iv)  best_trial robustness guard (all-diverged fallback to trial-0).

Test parameters chosen for fast CI on CPU (total target < 90 s):
  n_warmup=200, n_samples=200, n_trials=3, n_seeds=1, n_chains=1
  MCLMC: n_warmup=200 (adaptation needs enough steps for ESS estimates)
"""

import math
from unittest.mock import patch

import jax
import pytest

from bjx_bench.calibration.tune import TuningResult, default_params_for, tune_algorithm
from bjx_bench.inference.base_method import BASE_METHODS
from bjx_bench.model import MODELS

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_N_WARMUP = 200
_N_SAMPLES = 200
_N_TRIALS = 3
_N_SEEDS = 1
_N_CHAINS = 1

_MVN_ENTRY = MODELS["mvn_10"]
_NUTS_ENTRY = BASE_METHODS["nuts"]
_MALA_ENTRY = BASE_METHODS["mala"]
_RWM_ENTRY = BASE_METHODS["rwm"]
_MCLMC_ENTRY = BASE_METHODS["mclmc"]


# ---------------------------------------------------------------------------
# 1. MALA + MVN-10 with sampler="tpe"
# ---------------------------------------------------------------------------


class TestMalaMvnSmoke:
    """MALA on MVN-10: no warmup; step_size tuned by BO."""

    def _run(self, seed: int, **kwargs) -> TuningResult:
        return tune_algorithm(
            _MVN_ENTRY,
            _MALA_ENTRY,
            n_trials=_N_TRIALS,
            n_seeds=_N_SEEDS,
            n_chains=_N_CHAINS,
            n_samples=_N_SAMPLES,
            n_warmup=_N_WARMUP,
            rng_key=jax.random.key(seed),
            sampler="tpe",
            **kwargs,
        )

    def test_base_method_name(self) -> None:
        result = self._run(40)
        assert result.base_method_name == "mala"

    def test_n_trials_completed(self) -> None:
        result = self._run(41)
        assert result.n_trials_completed == _N_TRIALS

    def test_history_length(self) -> None:
        result = self._run(42)
        assert len(result.history) == _N_TRIALS

    def test_result_is_tuning_result(self) -> None:
        result = self._run(43)
        assert isinstance(result, TuningResult)

    def test_best_params_has_step_size(self) -> None:
        result = self._run(44)
        assert (
            "step_size" in result.best_params
        ), f"best_params keys: {list(result.best_params.keys())}"

    def test_best_score_is_finite(self) -> None:
        """MALA on MVN-10 with default step_size should converge."""
        result = self._run(45)
        assert math.isfinite(
            result.best_score
        ), f"MALA best_score={result.best_score} is non-finite on MVN-10"


# ---------------------------------------------------------------------------
# 2. RWM + MVN-10 with sampler="tpe"
# ---------------------------------------------------------------------------


class TestRwmMvnSmoke:
    """RWM on MVN-10: no warmup; sigma tuned by BO."""

    def _run(self, seed: int) -> TuningResult:
        return tune_algorithm(
            _MVN_ENTRY,
            _RWM_ENTRY,
            n_trials=_N_TRIALS,
            n_seeds=_N_SEEDS,
            n_chains=_N_CHAINS,
            n_samples=_N_SAMPLES,
            n_warmup=_N_WARMUP,
            rng_key=jax.random.key(seed),
            sampler="tpe",
        )

    def test_base_method_name(self) -> None:
        result = self._run(50)
        assert result.base_method_name == "rwm"

    def test_n_trials_completed(self) -> None:
        result = self._run(51)
        assert result.n_trials_completed == _N_TRIALS

    def test_best_params_has_sigma(self) -> None:
        result = self._run(52)
        assert (
            "sigma" in result.best_params
        ), f"best_params keys: {list(result.best_params.keys())}"

    def test_result_is_tuning_result(self) -> None:
        result = self._run(53)
        assert isinstance(result, TuningResult)

    def test_history_has_all_keys(self) -> None:
        result = self._run(54)
        for rec in result.history:
            for key in ("trial", "params", "score", "certified", "wall_seconds"):
                assert key in rec, f"history record missing key {key!r}: {rec}"


# ---------------------------------------------------------------------------
# 3. MCLMC + MVN-10 with sampler="tpe"
# ---------------------------------------------------------------------------


class TestMclmcMvnSmoke:
    """MCLMC on MVN-10: mclmc_find_L_and_step_size warmup; step_size + L BO-tuned."""

    def _run(self, seed: int) -> TuningResult:
        return tune_algorithm(
            _MVN_ENTRY,
            _MCLMC_ENTRY,
            n_trials=_N_TRIALS,
            n_seeds=_N_SEEDS,
            n_chains=_N_CHAINS,
            n_samples=_N_SAMPLES,
            n_warmup=_N_WARMUP,
            rng_key=jax.random.key(seed),
            sampler="tpe",
        )

    def test_base_method_name(self) -> None:
        result = self._run(60)
        assert result.base_method_name == "mclmc"

    def test_n_trials_completed(self) -> None:
        result = self._run(61)
        assert result.n_trials_completed == _N_TRIALS

    def test_best_params_has_step_size(self) -> None:
        result = self._run(62)
        assert (
            "step_size" in result.best_params
        ), f"best_params keys: {list(result.best_params.keys())}"

    def test_best_params_has_L(self) -> None:
        result = self._run(63)
        assert (
            "L" in result.best_params
        ), f"best_params keys: {list(result.best_params.keys())}"

    def test_result_is_tuning_result(self) -> None:
        result = self._run(64)
        assert isinstance(result, TuningResult)

    def test_mclmc_params_keys_in_history(self) -> None:
        """history[0]['params'] (trial-0 default) must contain step_size and L.

        This confirms that the MCLMC entry's default_hp_space matches the BO
        search space correctly (step_size, L).
        """
        result = self._run(65)
        trial0_params = result.history[0]["params"]
        assert (
            "step_size" in trial0_params
        ), f"MCLMC trial-0 params keys: {list(trial0_params.keys())}"
        assert (
            "L" in trial0_params
        ), f"MCLMC trial-0 params keys: {list(trial0_params.keys())}"


# ---------------------------------------------------------------------------
# 4. NUTS + MVN-10 with sampler="random"
# ---------------------------------------------------------------------------


class TestNutsRandomSamplerSmoke:
    """NUTS on MVN-10 with Optuna RandomSampler: same result shape as TPE."""

    def _run(self, seed: int) -> TuningResult:
        return tune_algorithm(
            _MVN_ENTRY,
            _NUTS_ENTRY,
            n_trials=_N_TRIALS,
            n_seeds=_N_SEEDS,
            n_chains=_N_CHAINS,
            n_samples=_N_SAMPLES,
            n_warmup=_N_WARMUP,
            rng_key=jax.random.key(seed),
            sampler="random",
        )

    def test_base_method_name(self) -> None:
        result = self._run(70)
        assert result.base_method_name == "nuts"

    def test_n_trials_completed(self) -> None:
        result = self._run(71)
        assert result.n_trials_completed == _N_TRIALS

    def test_history_length(self) -> None:
        result = self._run(72)
        assert len(result.history) == _N_TRIALS

    def test_result_fields_match_tpe(self) -> None:
        """RandomSampler result must have the same dataclass fields as TPE result."""
        result_random = tune_algorithm(
            _MVN_ENTRY,
            _NUTS_ENTRY,
            n_trials=_N_TRIALS,
            n_seeds=_N_SEEDS,
            n_chains=_N_CHAINS,
            n_samples=_N_SAMPLES,
            n_warmup=_N_WARMUP,
            rng_key=jax.random.key(73),
            sampler="random",
        )
        result_tpe = tune_algorithm(
            _MVN_ENTRY,
            _NUTS_ENTRY,
            n_trials=_N_TRIALS,
            n_seeds=_N_SEEDS,
            n_chains=_N_CHAINS,
            n_samples=_N_SAMPLES,
            n_warmup=_N_WARMUP,
            rng_key=jax.random.key(73),
            sampler="tpe",
        )
        # Same set of fields, same types
        assert set(vars(result_random).keys()) == set(
            vars(result_tpe).keys()
        ), "Random and TPE results have different fields"
        # Both history tuples have the same inner dict keys
        for i, (r_rec, t_rec) in enumerate(
            zip(result_random.history, result_tpe.history, strict=False)
        ):
            assert set(r_rec.keys()) == set(
                t_rec.keys()
            ), f"History record {i} has different keys between random and tpe"


# ---------------------------------------------------------------------------
# 5. TPE-vs-random qualitative smoke check (NUTS + MVN-10, n_trials=10)
# ---------------------------------------------------------------------------


class TestTpeVsRandomQualitative:
    """TPE should not catastrophically underperform random on MVN-10.

    This is a smoke check, not a rigorous statistical test.  The assertion
    is conservative: allow random to beat TPE by up to 0.1 in score units
    before flagging as surprising.  If random consistently beats TPE here
    we should investigate (could be signal on the dogfood comparison).
    """

    def test_tpe_not_catastrophically_worse_than_random(self) -> None:
        n_trials = 10
        rng_key = jax.random.key(80)
        shared_kwargs = dict(
            n_trials=n_trials,
            n_seeds=1,
            n_chains=1,
            n_samples=_N_SAMPLES,
            n_warmup=_N_WARMUP,
            rng_key=rng_key,
        )
        result_tpe = tune_algorithm(
            _MVN_ENTRY, _NUTS_ENTRY, sampler="tpe", **shared_kwargs
        )
        result_random = tune_algorithm(
            _MVN_ENTRY, _NUTS_ENTRY, sampler="random", **shared_kwargs
        )
        # Allow random to beat TPE by at most 0.1 in score
        assert result_tpe.best_score > result_random.best_score - 0.1, (
            f"TPE best_score={result_tpe.best_score:.4f} catastrophically worse "
            f"than random best_score={result_random.best_score:.4f} (gap > 0.1). "
            "This is surprising and may warrant investigation."
        )


# ---------------------------------------------------------------------------
# 6. sampler="invalid" raises ValueError
# ---------------------------------------------------------------------------


class TestInvalidSampler:
    """Passing an unsupported sampler name raises ValueError immediately."""

    def test_invalid_sampler_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="sampler must be 'tpe' or 'random'"):
            tune_algorithm(
                _MVN_ENTRY,
                _NUTS_ENTRY,
                n_trials=1,
                n_seeds=1,
                n_chains=1,
                n_samples=10,
                n_warmup=10,
                rng_key=jax.random.key(0),
                sampler="bogus",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# 7. best_trial fallback: all trials diverge → trial-0 params returned
# ---------------------------------------------------------------------------


class TestBestTrialFallback:
    """If all trials return -inf, tune_algorithm returns trial-0 params (not raises)."""

    def test_fallback_to_trial_zero_params(self) -> None:
        """Monkeypatch _run_trial to always return -inf."""
        from bjx_bench.calibration import tune

        with patch.object(tune, "_run_trial", return_value=float("-inf")):
            result = tune_algorithm(
                _MVN_ENTRY,
                _NUTS_ENTRY,
                n_trials=3,
                n_seeds=1,
                n_chains=1,
                n_samples=10,
                n_warmup=50,
                rng_key=jax.random.key(90),
                sampler="tpe",
            )

        # Must not raise; result should be a TuningResult
        assert isinstance(result, TuningResult)
        # best_score should be -inf (all diverged)
        assert result.best_score == float("-inf") or not math.isfinite(
            result.best_score
        ), f"Expected -inf best_score, got {result.best_score}"
        # best_params should be the trial-0 defaults (not empty)
        expected_defaults = default_params_for(_NUTS_ENTRY)
        assert set(result.best_params.keys()) == set(expected_defaults.keys()), (
            f"best_params keys {set(result.best_params.keys())} != "
            f"expected defaults keys {set(expected_defaults.keys())}"
        )

    def test_result_does_not_raise_when_all_diverge(self) -> None:
        """Confirm no ValueError is raised from study.best_trial."""
        from bjx_bench.calibration import tune

        with patch.object(tune, "_run_trial", return_value=float("-inf")):
            # Should complete without raising ValueError
            result = tune_algorithm(
                _MVN_ENTRY,
                _NUTS_ENTRY,
                n_trials=2,
                n_seeds=1,
                n_chains=1,
                n_samples=10,
                n_warmup=50,
                rng_key=jax.random.key(91),
                sampler="random",
            )
        assert isinstance(result, TuningResult)


# ---------------------------------------------------------------------------
# 8. MCLMC mclmc_params keys verification via history
# ---------------------------------------------------------------------------


class TestMclmcParamsKeysVerification:
    """Verify MCLMC trial-0 params contain step_size and L.

    Confirms the integration layer correctly wires the MCLMC dispatch:
    history[0]['params'] should have exactly the keys from
    default_params_for(mclmc_entry), which are {step_size, L}.
    """

    def test_mclmc_default_params_have_step_size_and_L(self) -> None:
        defaults = default_params_for(_MCLMC_ENTRY)
        assert (
            "step_size" in defaults
        ), f"MCLMC default_params missing 'step_size'; keys={list(defaults.keys())}"
        assert (
            "L" in defaults
        ), f"MCLMC default_params missing 'L'; keys={list(defaults.keys())}"

    def test_mclmc_history_trial0_params_match_defaults(self) -> None:
        result = tune_algorithm(
            _MVN_ENTRY,
            _MCLMC_ENTRY,
            n_trials=_N_TRIALS,
            n_seeds=_N_SEEDS,
            n_chains=_N_CHAINS,
            n_samples=_N_SAMPLES,
            n_warmup=_N_WARMUP,
            rng_key=jax.random.key(100),
            sampler="tpe",
        )
        trial0_params = result.history[0]["params"]
        expected_keys = set(default_params_for(_MCLMC_ENTRY).keys())
        actual_keys = set(trial0_params.keys())
        assert (
            actual_keys == expected_keys
        ), f"MCLMC trial-0 params keys {actual_keys} != expected {expected_keys}"
