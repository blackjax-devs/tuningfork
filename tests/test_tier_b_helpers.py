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
"""Tests for bjx_bench.calibration.tier_b foundation layer (T2.6a).

Covers:
- ``default_value_for_space``: one test per kind (loguniform, uniform, int,
  categorical) plus bad-kind ValueError.
- ``default_params_for``: round-trip on the HMC entry from BASE_METHODS.
- ``optuna_distribution_for_space``: kind-level type and attribute checks;
  Optuna enqueue_trial round-trip smoke.
- ``optuna_distributions_for``: on HMC entry, both keys present with correct
  distribution types.
- ``TuningDifficulty`` and ``TuningResult`` construction and invariants.
- ``tune_algorithm`` stub: raises NotImplementedError with T2.6b mention.
- Optuna integration smoke: enqueue default params, ask study, verify round-trip.

Empirical findings documented here:
1. **Optuna enqueue_trial trial-index**: enqueuing before the first
   ``study.optimize`` call gives the enqueued trial index 0. Optuna
   guarantees that enqueued trials are dequeued before TPE is asked.
   ``study.trials[0].params`` equals the enqueued dict exactly.
   Verified in ``TestOptunaIntegrationSmoke.test_enqueue_trial_is_trial_0``.
2. **int midpoint convention**: ``(low + high) // 2`` (integer division).
   For low=1, high=128: ``(1 + 128) // 2 = 64``.  Agrees with
   ``int((1 + 128) / 2) = 64`` for this pair (129 // 2 = 64, int(64.5) = 64
   in Python 3 truncation).  For any non-negative integer pair the two
   formulas produce the same result because Python's ``//`` truncates toward
   zero and the sum is non-negative.
3. **history as tuple**: ``TuningResult.history`` is ``tuple[dict, ...]``.
   The frozen dataclass demands hashable *field types* only at the
   declaration level (Python does not enforce deep hashability for frozen
   dataclasses); the ``tuple`` wrapper is still valuable because it signals
   immutable sequence to readers and prevents accidental in-place mutation
   (``list.append``).  The individual dicts inside remain mutable, but
   callers are expected not to mutate them after ``TuningResult`` creation.
"""

from typing import Any

import optuna
import optuna.distributions as D
import pytest

from bjx_bench.calibration.tier_b import (
    TuningDifficulty,
    TuningResult,
    default_params_for,
    default_value_for_space,
    optuna_distribution_for_space,
    optuna_distributions_for,
)
from bjx_bench.inference.base_method import BASE_METHODS
from bjx_bench.inference.base_method._base import BaseMethod, HyperparamSpace

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

HMC_ENTRY: BaseMethod = BASE_METHODS["hmc"]

# HMC default_hp_space:
#   HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0)
#   HyperparamSpace("num_integration_steps", "int", low=1, high=128)


# ---------------------------------------------------------------------------
# 1. default_value_for_space
# ---------------------------------------------------------------------------


class TestDefaultValueForSpace:
    """One test per kind + bad-kind error."""

    def test_loguniform(self) -> None:
        """70th-percentile on log-scale: low * (high/low)**0.7.

        For [1e-3, 1.0]: 1e-3 * 1000**0.7 ≈ 0.1259 (P4.0 tweak from sqrt).
        """
        space = HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0)
        result = default_value_for_space(space)
        expected = 1e-3 * (1.0 / 1e-3) ** 0.7
        assert result == pytest.approx(
            expected, rel=1e-9
        ), f"loguniform default: expected {expected}, got {result}"

    def test_loguniform_symmetric(self) -> None:
        """For symmetric log-range e.g. [0.01, 100], default == 0.01 * 10000**0.7."""
        space = HyperparamSpace("lr", "loguniform", low=0.01, high=100.0)
        result = default_value_for_space(space)
        expected = 0.01 * (100.0 / 0.01) ** 0.7
        assert result == pytest.approx(expected, rel=1e-9)

    def test_uniform(self) -> None:
        """Midpoint of [0, 10] is 5.0."""
        space = HyperparamSpace("alpha", "uniform", low=0.0, high=10.0)
        result = default_value_for_space(space)
        assert result == pytest.approx(5.0, rel=1e-9)

    def test_uniform_asymmetric(self) -> None:
        """Midpoint of [3.0, 7.0] is 5.0."""
        space = HyperparamSpace("beta", "uniform", low=3.0, high=7.0)
        result = default_value_for_space(space)
        assert result == pytest.approx(5.0, rel=1e-9)

    def test_int(self) -> None:
        """Integer midpoint of [1, 128] is (1 + 128) // 2 = 64.

        Convention: ``(low + high) // 2``.  For 1+128=129 → 129//2 = 64.
        This equals int(64.5) = 64 in Python 3 (truncation toward zero for
        non-negative values).
        """
        space = HyperparamSpace("num_leapfrog", "int", low=1, high=128)
        result = default_value_for_space(space)
        assert result == 64, f"Expected 64, got {result}"
        assert isinstance(
            result, int
        ), f"int kind must return int, got {type(result).__name__}"

    def test_int_even_range(self) -> None:
        """Integer midpoint of [0, 10] is 5."""
        space = HyperparamSpace("n", "int", low=0, high=10)
        result = default_value_for_space(space)
        assert result == 5
        assert isinstance(result, int)

    def test_int_single_value(self) -> None:
        """Integer midpoint of [7, 7] is 7."""
        space = HyperparamSpace("n", "int", low=7, high=7)
        assert default_value_for_space(space) == 7

    def test_categorical_first_choice(self) -> None:
        """First choice of ('a', 'b', 'c') is 'a'."""
        space = HyperparamSpace("method", "categorical", choices=("a", "b", "c"))
        result = default_value_for_space(space)
        assert result == "a"

    def test_categorical_single_choice(self) -> None:
        """Categorical with one choice returns that choice."""
        space = HyperparamSpace("integrator", "categorical", choices=("verlet",))
        result = default_value_for_space(space)
        assert result == "verlet"

    def test_bad_kind_raises_value_error(self) -> None:
        """A HyperparamSpace with an invalid kind (bypassed post_init) raises ValueError."""
        # Bypass __post_init__ by using object.__setattr__ on a frozen instance.
        space = HyperparamSpace("x", "loguniform", low=1e-3, high=1.0)
        # Manually override kind field (frozen dataclass hack for testing).
        object.__setattr__(space, "kind", "exponential")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unknown kind"):
            default_value_for_space(space)


# ---------------------------------------------------------------------------
# 2. default_params_for
# ---------------------------------------------------------------------------


class TestDefaultParamsFor:
    """Round-trip on the HMC entry."""

    def test_hmc_keys(self) -> None:
        """HMC default params must contain step_size and num_integration_steps."""
        params = default_params_for(HMC_ENTRY)
        assert set(params.keys()) == {
            "step_size",
            "num_integration_steps",
        }, f"Unexpected keys: {set(params.keys())}"

    def test_hmc_step_size_value(self) -> None:
        """step_size default = 1e-3 * 1000**0.7 (70th-percentile on log-scale, P4.0)."""
        params = default_params_for(HMC_ENTRY)
        expected = 1e-3 * (1.0 / 1e-3) ** 0.7
        assert params["step_size"] == pytest.approx(expected, rel=1e-9)

    def test_hmc_num_integration_steps_value(self) -> None:
        """num_integration_steps default = (1 + 128) // 2 = 64."""
        params = default_params_for(HMC_ENTRY)
        assert params["num_integration_steps"] == 64
        assert isinstance(params["num_integration_steps"], int)

    def test_result_is_dict(self) -> None:
        """Return type is a plain Python dict."""
        params = default_params_for(HMC_ENTRY)
        assert isinstance(params, dict)

    def test_values_match_per_space(self) -> None:
        """Each value in default_params_for == default_value_for_space per space."""
        params = default_params_for(HMC_ENTRY)
        for space in HMC_ENTRY.default_hp_space:
            expected = default_value_for_space(space)
            assert params[space.name] == pytest.approx(
                expected, rel=1e-9
            ), f"Mismatch for {space.name}: expected {expected}, got {params[space.name]}"

    def test_multi_space_entry(self) -> None:
        """Custom entry with 3 spaces produces a dict with 3 keys."""
        entry = BaseMethod(
            name="custom",
            family="mcmc",
            factory=lambda fn, **kw: None,
            grad_count_per_step=lambda info: 1,
            default_hp_space=(
                HyperparamSpace("a", "loguniform", low=1e-4, high=1.0),
                HyperparamSpace("b", "uniform", low=0.0, high=2.0),
                HyperparamSpace("c", "int", low=1, high=9),
            ),
        )
        params = default_params_for(entry)
        assert set(params.keys()) == {"a", "b", "c"}
        assert params["a"] == pytest.approx(1e-4 * (1.0 / 1e-4) ** 0.7, rel=1e-9)
        assert params["b"] == pytest.approx(1.0, rel=1e-9)
        assert params["c"] == 5  # (1+9)//2 = 5


# ---------------------------------------------------------------------------
# 3. optuna_distribution_for_space
# ---------------------------------------------------------------------------


class TestOptunaDistributionForSpace:
    """Round-trip type and attribute checks per kind."""

    def test_loguniform_type(self) -> None:
        """loguniform → FloatDistribution."""
        space = HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0)
        dist = optuna_distribution_for_space(space)
        assert isinstance(dist, D.FloatDistribution)

    def test_loguniform_log_flag(self) -> None:
        """loguniform → FloatDistribution with log=True."""
        space = HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0)
        dist = optuna_distribution_for_space(space)
        assert dist.log is True  # type: ignore[union-attr]

    def test_loguniform_bounds(self) -> None:
        """loguniform → correct low and high on the distribution."""
        space = HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0)
        dist = optuna_distribution_for_space(space)
        assert dist.low == pytest.approx(1e-3)  # type: ignore[union-attr]
        assert dist.high == pytest.approx(1.0)  # type: ignore[union-attr]

    def test_uniform_type(self) -> None:
        """uniform → FloatDistribution."""
        space = HyperparamSpace("alpha", "uniform", low=0.0, high=10.0)
        dist = optuna_distribution_for_space(space)
        assert isinstance(dist, D.FloatDistribution)

    def test_uniform_log_flag(self) -> None:
        """uniform → FloatDistribution with log=False."""
        space = HyperparamSpace("alpha", "uniform", low=0.0, high=10.0)
        dist = optuna_distribution_for_space(space)
        assert dist.log is False  # type: ignore[union-attr]

    def test_uniform_bounds(self) -> None:
        """uniform → correct low and high."""
        space = HyperparamSpace("alpha", "uniform", low=0.0, high=10.0)
        dist = optuna_distribution_for_space(space)
        assert dist.low == pytest.approx(0.0)  # type: ignore[union-attr]
        assert dist.high == pytest.approx(10.0)  # type: ignore[union-attr]

    def test_int_type(self) -> None:
        """int → IntDistribution."""
        space = HyperparamSpace("n", "int", low=1, high=128)
        dist = optuna_distribution_for_space(space)
        assert isinstance(dist, D.IntDistribution)

    def test_int_bounds(self) -> None:
        """int → correct low and high on IntDistribution."""
        space = HyperparamSpace("n", "int", low=1, high=128)
        dist = optuna_distribution_for_space(space)
        assert dist.low == 1  # type: ignore[union-attr]
        assert dist.high == 128  # type: ignore[union-attr]

    def test_categorical_type(self) -> None:
        """categorical → CategoricalDistribution."""
        space = HyperparamSpace("method", "categorical", choices=("a", "b"))
        dist = optuna_distribution_for_space(space)
        assert isinstance(dist, D.CategoricalDistribution)

    def test_categorical_choices(self) -> None:
        """categorical → CategoricalDistribution with matching choices."""
        space = HyperparamSpace("method", "categorical", choices=("a", "b", "c"))
        dist = optuna_distribution_for_space(space)
        assert set(dist.choices) == {"a", "b", "c"}  # type: ignore[union-attr]

    def test_bad_kind_raises(self) -> None:
        """A space with a bad kind (bypassed post_init) raises ValueError."""
        space = HyperparamSpace("x", "loguniform", low=1e-3, high=1.0)
        object.__setattr__(space, "kind", "exponential")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unknown kind"):
            optuna_distribution_for_space(space)

    def test_default_value_accepted_by_distribution(self) -> None:
        """The default value must be within the distribution's internal range.

        Optuna distributions expose an ``_contains`` or ``internal_repr``
        method in newer versions; we instead do a softer check by verifying
        the default is between low and high for numeric kinds.
        """
        for space in HMC_ENTRY.default_hp_space:
            default = default_value_for_space(space)
            dist = optuna_distribution_for_space(space)
            if isinstance(dist, (D.FloatDistribution, D.IntDistribution)):
                assert dist.low <= default <= dist.high, (  # type: ignore[union-attr]
                    f"{space.name}: default {default} outside [{dist.low}, {dist.high}]"  # type: ignore[union-attr]
                )
            elif isinstance(dist, D.CategoricalDistribution):
                assert default in dist.choices  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 4. optuna_distributions_for
# ---------------------------------------------------------------------------


class TestOptunaDistributionsFor:
    """optuna_distributions_for on the HMC entry."""

    def test_hmc_keys(self) -> None:
        """Returns dict with step_size and num_integration_steps keys."""
        dists = optuna_distributions_for(HMC_ENTRY)
        assert set(dists.keys()) == {"step_size", "num_integration_steps"}

    def test_step_size_is_float_log(self) -> None:
        """step_size → FloatDistribution(log=True)."""
        dists = optuna_distributions_for(HMC_ENTRY)
        dist = dists["step_size"]
        assert isinstance(dist, D.FloatDistribution)
        assert dist.log is True

    def test_num_integration_steps_is_int(self) -> None:
        """num_integration_steps → IntDistribution."""
        dists = optuna_distributions_for(HMC_ENTRY)
        dist = dists["num_integration_steps"]
        assert isinstance(dist, D.IntDistribution)

    def test_result_is_dict(self) -> None:
        """Return type is a plain Python dict."""
        dists = optuna_distributions_for(HMC_ENTRY)
        assert isinstance(dists, dict)


# ---------------------------------------------------------------------------
# 5. TuningDifficulty and TuningResult construction
# ---------------------------------------------------------------------------


class TestTuningDifficultyConstruction:
    """TuningDifficulty dataclass construction and invariants."""

    def _make_difficulty(self, **kw: Any) -> TuningDifficulty:
        defaults: dict[str, Any] = dict(
            default_score=0.05,
            best_score=0.10,
            threshold_score=0.05,  # max(0.05, 0.5*0.10) = max(0.05, 0.05) = 0.05
            default_works=True,  # 0.05 >= 0.5*0.10 = 0.05 → True
            n_trials_to_threshold=0,
            n_trials_to_best=3,
            wall_seconds_to_threshold=0.0,
            wall_seconds_to_best=12.5,
        )
        defaults.update(kw)
        return TuningDifficulty(**defaults)  # type: ignore[arg-type]

    def test_default_works_threshold_edge_case(self) -> None:
        """default_score=0.05, best_score=0.10: threshold=0.05, default_works=True.

        Threshold formula: max(default_score, 0.5 * best_score)
                         = max(0.05, 0.5 * 0.10)
                         = max(0.05, 0.05) = 0.05
        default_works = default_score >= threshold_score = 0.05 >= 0.05 = True.
        n_trials_to_threshold = 0 (iff default_works).
        """
        d = self._make_difficulty()
        assert d.threshold_score == pytest.approx(0.05)
        assert d.default_works is True
        assert d.n_trials_to_threshold == 0
        assert d.wall_seconds_to_threshold == 0.0

    def test_default_does_not_work(self) -> None:
        """default_score=0.03, best_score=0.10: threshold=0.05, default_works=False."""
        # threshold = max(0.03, 0.5*0.10) = max(0.03, 0.05) = 0.05
        # default_works = 0.03 >= 0.05 = False
        d = self._make_difficulty(
            default_score=0.03,
            best_score=0.10,
            threshold_score=0.05,
            default_works=False,
            n_trials_to_threshold=7,
            wall_seconds_to_threshold=5.2,
        )
        assert d.threshold_score == pytest.approx(0.05)
        assert d.default_works is False
        assert d.n_trials_to_threshold == 7
        assert d.wall_seconds_to_threshold == pytest.approx(5.2)

    def test_default_beats_best(self) -> None:
        """default_score=0.20 > best_score=0.10 (rare but possible if BO explores badly).

        threshold = max(0.20, 0.5*0.10) = max(0.20, 0.05) = 0.20
        default_works = 0.20 >= 0.20 = True
        n_trials_to_threshold = 0.
        """
        d = self._make_difficulty(
            default_score=0.20,
            best_score=0.10,
            threshold_score=0.20,  # max(0.20, 0.05) = 0.20
            default_works=True,
            n_trials_to_threshold=0,
            n_trials_to_best=0,  # trial 0 is the best
            wall_seconds_to_threshold=0.0,
            wall_seconds_to_best=1.3,
        )
        assert d.default_works is True
        assert d.n_trials_to_threshold == 0

    def test_frozen(self) -> None:
        """TuningDifficulty is frozen — mutation raises FrozenInstanceError."""
        d = self._make_difficulty()
        with pytest.raises(Exception):
            d.default_score = 0.99  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All eight fields are readable."""
        d = self._make_difficulty()
        _ = d.default_score
        _ = d.best_score
        _ = d.threshold_score
        _ = d.default_works
        _ = d.n_trials_to_threshold
        _ = d.n_trials_to_best
        _ = d.wall_seconds_to_threshold
        _ = d.wall_seconds_to_best


class TestTuningResultConstruction:
    """TuningResult dataclass construction."""

    def _make_difficulty(self) -> TuningDifficulty:
        return TuningDifficulty(
            default_score=0.05,
            best_score=0.10,
            threshold_score=0.05,
            default_works=True,
            n_trials_to_threshold=0,
            n_trials_to_best=3,
            wall_seconds_to_threshold=0.0,
            wall_seconds_to_best=12.5,
        )

    def test_minimal_construction(self) -> None:
        """Construct a TuningResult with all required fields."""
        history = (
            {
                "trial": 0,
                "params": {"step_size": 0.03},
                "score": 0.05,
                "certified": True,
                "wall_seconds": 1.0,
            },
        )
        result = TuningResult(
            base_method_name="hmc",
            posterior_name="mvn_10",
            best_params={"step_size": 0.03},
            best_score=0.10,
            n_trials_completed=1,
            n_seeds=5,
            history=history,
            difficulty=self._make_difficulty(),
        )
        assert result.base_method_name == "hmc"
        assert result.posterior_name == "mvn_10"
        assert result.best_score == pytest.approx(0.10)
        assert result.n_trials_completed == 1
        assert result.n_seeds == 5
        assert len(result.history) == 1

    def test_history_is_tuple(self) -> None:
        """history field must be a tuple (not a list)."""
        history: tuple[dict, ...] = tuple()
        result = TuningResult(
            base_method_name="mala",
            posterior_name="funnel",
            best_params={},
            best_score=0.0,
            n_trials_completed=0,
            n_seeds=1,
            history=history,
            difficulty=self._make_difficulty(),
        )
        assert isinstance(
            result.history, tuple
        ), f"history type: expected tuple, got {type(result.history).__name__}"

    def test_frozen(self) -> None:
        """TuningResult is frozen — mutation raises FrozenInstanceError."""
        result = TuningResult(
            base_method_name="hmc",
            posterior_name="mvn_10",
            best_params={},
            best_score=0.0,
            n_trials_completed=0,
            n_seeds=1,
            history=(),
            difficulty=self._make_difficulty(),
        )
        with pytest.raises(Exception):
            result.best_score = 99.9  # type: ignore[misc]

    def test_history_contents(self) -> None:
        """Per-trial dict keys are accessible on history elements."""
        rec = {
            "trial": 0,
            "params": {"step_size": 0.03162},
            "score": 0.05,
            "certified": True,
            "wall_seconds": 2.1,
        }
        result = TuningResult(
            base_method_name="hmc",
            posterior_name="mvn_10",
            best_params=rec["params"],  # type: ignore[arg-type]
            best_score=rec["score"],  # type: ignore[arg-type]
            n_trials_completed=1,
            n_seeds=5,
            history=(rec,),
            difficulty=self._make_difficulty(),
        )
        assert result.history[0]["trial"] == 0
        assert result.history[0]["certified"] is True


# ---------------------------------------------------------------------------
# 6. tune_algorithm stub (T2.6a tests — retired in T2.6b)
# ---------------------------------------------------------------------------
# The three stub tests that verified tune_algorithm raised NotImplementedError
# with a "T2.6b" message have been REMOVED here.  T2.6b replaces the stub
# body with the real Optuna BO loop; those tests were specifically exercising
# the now-obsolete stub.  The new behaviour (real BO loop, NotImplementedError
# only for non-MM kernels) is covered by tests/test_tier_b_bo_loop.py.


# ---------------------------------------------------------------------------
# 7. Optuna integration smoke
# ---------------------------------------------------------------------------


class TestOptunaIntegrationSmoke:
    """Verify Optuna enqueue_trial round-trip and study interaction.

    Empirical finding (finding #1 in this module's docstring):
    ``study.enqueue_trial(params)`` guarantees the enqueued params are
    used in the *next* ``study.ask()`` or the next ``study.optimize()``
    call.  The first completed trial in ``study.trials`` therefore holds
    the enqueued params exactly.  This pins the T2.6b convention: the BO
    loop injects default params via ``enqueue_trial`` before calling
    ``optimize``, so ``study.trials[0].params == default_params_for(entry)``.
    """

    def test_enqueue_trial_is_trial_0(self) -> None:
        """Enqueued params appear as trial 0 in study.trials after optimize.

        Strategy: create a silent study (no logging), enqueue default
        params for HMC, run optimize for 3 trials with an objective that
        returns the step_size value, then check study.trials[0].
        """
        import logging

        optuna.logging.set_verbosity(logging.WARNING)
        study = optuna.create_study(direction="maximize")

        default_params = default_params_for(HMC_ENTRY)
        study.enqueue_trial(default_params)

        def objective(trial: optuna.Trial) -> float:
            # Use suggest_* so Optuna records values correctly
            step_size = trial.suggest_float(
                "step_size",
                low=1e-3,
                high=1.0,
                log=True,
            )
            _ = trial.suggest_int("num_integration_steps", low=1, high=128)
            return float(step_size)

        study.optimize(objective, n_trials=3)

        # Empirical finding: trial 0 holds the enqueued params exactly.
        assert len(study.trials) == 3
        trial0_step_size = study.trials[0].params["step_size"]
        expected_step_size = default_params["step_size"]
        assert trial0_step_size == pytest.approx(expected_step_size, rel=1e-9), (
            f"Trial 0 step_size {trial0_step_size} != enqueued {expected_step_size}. "
            "Optuna did NOT use the enqueued trial first."
        )

    def test_enqueue_round_trip_ask(self) -> None:
        """study.ask() after enqueue_trial returns a trial with the enqueued params.

        This is the lower-level API used by T2.6b's loop.
        Optuna's frozen-trial → add-trial path is NOT tested here because
        it requires an already-completed trial; we test the more common
        ask-then-tell path instead.
        """
        import logging

        optuna.logging.set_verbosity(logging.WARNING)
        study = optuna.create_study(direction="maximize")

        default_params = default_params_for(HMC_ENTRY)
        dists = optuna_distributions_for(HMC_ENTRY)

        study.enqueue_trial(default_params)
        trial = study.ask(fixed_distributions=dists)

        # The trial returned by ask() should have the enqueued step_size.
        # Note: suggest_* already records params on the trial object
        # when passed via fixed_distributions.
        assert trial.params["step_size"] == pytest.approx(
            default_params["step_size"], rel=1e-9
        ), (
            f"ask() step_size {trial.params['step_size']} != "
            f"enqueued {default_params['step_size']}"
        )
        assert (
            trial.params["num_integration_steps"]
            == default_params["num_integration_steps"]
        )

    def test_study_respects_distribution_bounds(self) -> None:
        """Post-enqueue TPE trials respect distribution bounds.

        Run 10 trials; check all non-enqueued trials are within bounds.
        """
        import logging

        optuna.logging.set_verbosity(logging.WARNING)
        study = optuna.create_study(direction="maximize")
        default_params = default_params_for(HMC_ENTRY)
        study.enqueue_trial(default_params)

        def objective(trial: optuna.Trial) -> float:
            step_size = trial.suggest_float("step_size", 1e-3, 1.0, log=True)
            n_steps = trial.suggest_int("num_integration_steps", 1, 128)
            return step_size + n_steps

        study.optimize(objective, n_trials=10)

        for t in study.trials:
            ss = t.params["step_size"]
            ns = t.params["num_integration_steps"]
            assert 1e-3 <= ss <= 1.0, f"step_size {ss} out of [1e-3, 1.0]"
            assert 1 <= ns <= 128, f"num_integration_steps {ns} out of [1, 128]"
