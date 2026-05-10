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
"""Tests for the P5.15 orbital_hmc base method registry entry.

Covers:
  1. ENTRY field correctness (name, family, default_hp_space, etc.).
  2. factory returns a SamplingAlgorithm with .init and .step.
  3. End-to-end smoke test on mvn_5d_logdensity:
     - PeriodicOrbitalState shape assertion.
     - positions array has leading dim == period.
     - No NaN in positions, weights, logdensities.
  4. grad_count_per_step callable.
  5. HP space: step_size (loguniform), period (int) present.
"""

import jax
import jax.numpy as jnp
import pytest
from blackjax.mcmc.periodic_orbital import PeriodicOrbitalState

from bjx_bench.inference.base_method.orbital_hmc import ENTRY
from tests.fixtures import mvn_5d_init, mvn_5d_logdensity

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_D = 5
_SEED = 42
_PERIOD = 4  # number of orbit steps; must be in [2, 20]
_STEP_SIZE = 0.1
_INVERSE_MASS_MATRIX = jnp.ones(_D)

# ---------------------------------------------------------------------------
# 1. ENTRY field correctness
# ---------------------------------------------------------------------------


class TestOrbitalHmcEntryFields:
    """ENTRY field validation for orbital_hmc."""

    def test_name(self) -> None:
        assert ENTRY.name == "orbital_hmc"

    def test_family(self) -> None:
        assert ENTRY.family == "mcmc"

    def test_needs_mass_matrix(self) -> None:
        """inverse_mass_matrix comes from warmup adaptation."""
        assert ENTRY.needs_mass_matrix is True

    def test_target_acceptance_rate_none(self) -> None:
        """No MH step; orbital weights replace rejection."""
        assert ENTRY.target_acceptance_rate is None

    def test_extra_required_kwargs_empty(self) -> None:
        """orbital_hmc is NOT specialised — standard factory."""
        assert ENTRY.extra_required_kwargs == ()

    def test_factory_callable(self) -> None:
        assert callable(ENTRY.factory)

    def test_grad_count_callable(self) -> None:
        assert callable(ENTRY.grad_count_per_step)

    def test_default_hp_space_nonempty(self) -> None:
        assert len(ENTRY.default_hp_space) > 0


# ---------------------------------------------------------------------------
# 2. Factory: returns SamplingAlgorithm with .init and .step
# ---------------------------------------------------------------------------


class TestOrbitalHmcFactory:
    """Factory invocation correctness."""

    def test_factory_returns_algorithm_with_init_step(self) -> None:
        algo = ENTRY.factory(
            mvn_5d_logdensity,
            step_size=_STEP_SIZE,
            period=_PERIOD,
            inverse_mass_matrix=_INVERSE_MASS_MATRIX,
        )
        assert hasattr(algo, "init"), "factory result must have .init"
        assert hasattr(algo, "step"), "factory result must have .step"

    def test_factory_init_returns_periodic_orbital_state(self) -> None:
        algo = ENTRY.factory(
            mvn_5d_logdensity,
            step_size=_STEP_SIZE,
            period=_PERIOD,
            inverse_mass_matrix=_INVERSE_MASS_MATRIX,
        )
        init_pos = mvn_5d_init()
        state = algo.init(init_pos)
        assert isinstance(
            state, PeriodicOrbitalState
        ), f"Expected PeriodicOrbitalState, got {type(state)}"


# ---------------------------------------------------------------------------
# 3. End-to-end smoke test on mvn_5d_logdensity
# ---------------------------------------------------------------------------


class TestOrbitalHmcEndToEnd:
    """End-to-end smoke test: PeriodicOrbitalState shape, period, no NaN."""

    def _build_algo(self, period: int = _PERIOD):
        return ENTRY.factory(
            mvn_5d_logdensity,
            step_size=_STEP_SIZE,
            period=period,
            inverse_mass_matrix=_INVERSE_MASS_MATRIX,
        )

    def test_positions_leading_dim_equals_period(self) -> None:
        """state.positions has shape (period, D)."""
        period = _PERIOD
        algo = self._build_algo(period=period)
        state = algo.init(mvn_5d_init())

        # positions is a tree-leaf; for flat arrays it has shape (period, D)
        positions = state.positions
        assert positions.shape == (period, _D), (
            f"Expected positions shape ({period}, {_D}), got {positions.shape}. "
            f"PeriodicOrbitalState.positions must have leading dim == period."
        )

    def test_weights_shape_equals_period(self) -> None:
        """state.weights has shape (period,)."""
        period = _PERIOD
        algo = self._build_algo(period=period)
        state = algo.init(mvn_5d_init())
        assert state.weights.shape == (
            period,
        ), f"Expected weights shape ({period},), got {state.weights.shape}."

    def test_weights_sum_to_one(self) -> None:
        """After init, weights are uniform (sum to 1.0)."""
        algo = self._build_algo()
        state = algo.init(mvn_5d_init())
        weight_sum = float(jnp.sum(state.weights))
        assert (
            abs(weight_sum - 1.0) < 1e-5
        ), f"Weights do not sum to 1.0: sum={weight_sum}"

    def test_no_nan_in_positions_after_init(self) -> None:
        """No NaN in state.positions after init."""
        algo = self._build_algo()
        state = algo.init(mvn_5d_init())
        assert jnp.all(
            jnp.isfinite(state.positions)
        ), "NaN or Inf found in state.positions after init."

    def test_no_nan_in_logdensities_after_init(self) -> None:
        """No NaN in state.logdensities after init."""
        algo = self._build_algo()
        state = algo.init(mvn_5d_init())
        assert jnp.all(
            jnp.isfinite(state.logdensities)
        ), "NaN or Inf found in state.logdensities after init."

    def test_step_preserves_state_type(self) -> None:
        """algo.step returns a PeriodicOrbitalState."""
        algo = self._build_algo()
        state = algo.init(mvn_5d_init())
        key = jax.random.key(_SEED)
        new_state, info = algo.step(key, state)
        assert isinstance(
            new_state, PeriodicOrbitalState
        ), f"Expected PeriodicOrbitalState after step, got {type(new_state)}"

    def test_step_no_nan_positions(self) -> None:
        """No NaN in state.positions after one step."""
        algo = self._build_algo()
        state = algo.init(mvn_5d_init())
        key = jax.random.key(_SEED)
        new_state, _ = algo.step(key, state)
        assert jnp.all(
            jnp.isfinite(new_state.positions)
        ), "NaN or Inf found in state.positions after one step."

    def test_step_no_nan_weights(self) -> None:
        """No NaN in state.weights after one step."""
        algo = self._build_algo()
        state = algo.init(mvn_5d_init())
        key = jax.random.key(_SEED)
        new_state, _ = algo.step(key, state)
        assert jnp.all(
            jnp.isfinite(new_state.weights)
        ), "NaN or Inf found in state.weights after one step."

    def test_scan_10_steps_no_nan(self) -> None:
        """10 steps via lax.scan: no NaN, shape preserved."""
        period = _PERIOD
        algo = self._build_algo(period=period)
        state = algo.init(mvn_5d_init())

        def one_step(carry, key):
            new_state, info = algo.step(key, carry)
            return new_state, new_state

        keys = jax.random.split(jax.random.key(_SEED), 10)
        final_state, _ = jax.lax.scan(one_step, state, keys)

        assert jnp.all(
            jnp.isfinite(final_state.positions)
        ), "NaN or Inf in positions after 10-step scan."
        assert final_state.positions.shape == (
            period,
            _D,
        ), f"positions shape changed after scan: {final_state.positions.shape}"

    def test_period_hp_range_low_and_high(self) -> None:
        """period=2 and period=20 (hp-space bounds) both work without error."""
        for period in (2, 20):
            algo = self._build_algo(period=period)
            state = algo.init(mvn_5d_init())
            key = jax.random.key(_SEED)
            new_state, _ = algo.step(key, state)
            assert new_state.positions.shape[0] == period, (
                f"period={period}: positions leading dim should be {period}, "
                f"got {new_state.positions.shape[0]}"
            )


# ---------------------------------------------------------------------------
# 4. grad_count_per_step callable
# ---------------------------------------------------------------------------


class TestOrbitalHmcGradCount:
    """grad_count_per_step is a callable that accepts info."""

    def test_grad_count_callable(self) -> None:
        assert callable(ENTRY.grad_count_per_step)

    def test_grad_count_with_none_info(self) -> None:
        """grad_count_per_step(None) returns a JAX array >= 1."""
        result = ENTRY.grad_count_per_step(None)
        assert isinstance(
            result, jax.Array
        ), f"Expected JAX Array from grad_count_per_step, got {type(result)}"
        assert int(result) >= 1, f"grad_count >= 1 expected, got {int(result)}"


# ---------------------------------------------------------------------------
# 5. HP space correctness
# ---------------------------------------------------------------------------


class TestOrbitalHmcHpSpace:
    """HP space: step_size (loguniform), period (int) present."""

    def _hp_names(self) -> set[str]:
        return {hp.name for hp in ENTRY.default_hp_space}

    def test_hp_space_has_step_size(self) -> None:
        assert (
            "step_size" in self._hp_names()
        ), f"step_size missing from orbital_hmc hp_space; got {self._hp_names()}"

    def test_hp_space_has_period(self) -> None:
        assert (
            "period" in self._hp_names()
        ), f"period missing from orbital_hmc hp_space; got {self._hp_names()}"

    def test_step_size_is_loguniform(self) -> None:
        for hp in ENTRY.default_hp_space:
            if hp.name == "step_size":
                assert (
                    hp.kind == "loguniform"
                ), f"step_size kind should be 'loguniform', got {hp.kind!r}"
                break

    def test_period_is_int(self) -> None:
        for hp in ENTRY.default_hp_space:
            if hp.name == "period":
                assert hp.kind == "int", f"period kind should be 'int', got {hp.kind!r}"
                assert hp.low == 2, f"period low should be 2, got {hp.low}"
                assert hp.high == 20, f"period high should be 20, got {hp.high}"
                break
