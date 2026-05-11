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
"""Tests for the NUTS and HMC algorithm wrappers.

Covers:
- Both entries are registered in BASE_METHODS under their expected names.
- factory is callable and returns a SamplingAlgorithm-shaped object
  (has .init and .step attributes).
- .init(position) returns a state with finite logdensity.
- .step(rng_key, state) returns (new_state, info) where
  info.num_integration_steps is a non-negative scalar.
- grad_count_per_step(info) returns a non-negative integer array.
- default_hp_space is non-empty; each HyperparamSpace has consistent
  low/high (or choices).
- 5-step end-to-end chain smoke test for each algorithm.
"""

import jax
import jax.numpy as jnp
import pytest

from tuningfork.inference.base_method import BASE_METHODS
from tuningfork.inference.base_method._base import BaseMethod, HyperparamSpace

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_DIM = 10
_LOGDENSITY_FN = lambda x: -0.5 * jnp.sum(x["x"] ** 2)
_POSITION = {"x": jnp.zeros(_DIM)}
_IMM = jnp.ones(_DIM)  # diagonal identity mass matrix


def _make_nuts_params() -> dict:
    return {"step_size": 0.1, "inverse_mass_matrix": _IMM}


def _make_hmc_params() -> dict:
    return {"step_size": 0.1, "inverse_mass_matrix": _IMM, "num_integration_steps": 10}


# ===========================================================================
# Registry tests
# ===========================================================================


class TestAlgorithmRegistry:
    def test_nuts_registered(self) -> None:
        assert "nuts" in BASE_METHODS, "BASE_METHODS must contain 'nuts'"

    def test_hmc_registered(self) -> None:
        assert "hmc" in BASE_METHODS, "BASE_METHODS must contain 'hmc'"

    def test_nuts_is_algorithm_entry(self) -> None:
        assert isinstance(BASE_METHODS["nuts"], BaseMethod)

    def test_hmc_is_algorithm_entry(self) -> None:
        assert isinstance(BASE_METHODS["hmc"], BaseMethod)

    def test_nuts_family(self) -> None:
        assert BASE_METHODS["nuts"].family == "mcmc"

    def test_hmc_family(self) -> None:
        assert BASE_METHODS["hmc"].family == "mcmc"

    def test_nuts_needs_mass_matrix(self) -> None:
        assert BASE_METHODS["nuts"].needs_mass_matrix is True

    def test_hmc_needs_mass_matrix(self) -> None:
        assert BASE_METHODS["hmc"].needs_mass_matrix is True

    def test_nuts_target_acceptance(self) -> None:
        assert BASE_METHODS["nuts"].target_acceptance_rate == pytest.approx(0.80)

    def test_hmc_target_acceptance(self) -> None:
        assert BASE_METHODS["hmc"].target_acceptance_rate == pytest.approx(0.65)


# ===========================================================================
# HyperparamSpace sanity tests
# ===========================================================================


class TestHyperparamSpaceSanity:
    @pytest.mark.parametrize("name", ["nuts", "hmc"])
    def test_default_hp_space_non_empty(self, name: str) -> None:
        entry = BASE_METHODS[name]
        assert len(entry.default_hp_space) >= 1

    @pytest.mark.parametrize("name", ["nuts", "hmc"])
    def test_all_hp_are_hyperparam_space(self, name: str) -> None:
        for hp in BASE_METHODS[name].default_hp_space:
            assert isinstance(hp, HyperparamSpace)

    @pytest.mark.parametrize("name", ["nuts", "hmc"])
    def test_hp_bounds_consistent(self, name: str) -> None:
        for hp in BASE_METHODS[name].default_hp_space:
            if hp.kind in ("loguniform", "uniform", "int"):
                assert hp.low is not None
                assert hp.high is not None
                assert hp.low < hp.high
            elif hp.kind == "categorical":
                assert hp.choices is not None and len(hp.choices) > 0

    def test_nuts_has_step_size_hp(self) -> None:
        names = [hp.name for hp in BASE_METHODS["nuts"].default_hp_space]
        assert "step_size" in names

    def test_hmc_has_step_size_and_num_steps(self) -> None:
        names = [hp.name for hp in BASE_METHODS["hmc"].default_hp_space]
        assert "step_size" in names
        assert "num_integration_steps" in names

    def test_nuts_has_no_mass_matrix_hp(self) -> None:
        """inverse_mass_matrix must NOT appear in the NUTS BO search space."""
        names = [hp.name for hp in BASE_METHODS["nuts"].default_hp_space]
        assert "inverse_mass_matrix" not in names

    def test_hmc_has_no_mass_matrix_hp(self) -> None:
        """inverse_mass_matrix must NOT appear in the HMC BO search space."""
        names = [hp.name for hp in BASE_METHODS["hmc"].default_hp_space]
        assert "inverse_mass_matrix" not in names


# ===========================================================================
# Factory → init → step pipeline tests
# ===========================================================================


class TestNutsFactory:
    def test_factory_callable(self) -> None:
        entry = BASE_METHODS["nuts"]
        assert callable(entry.factory)

    def test_factory_returns_sampling_algorithm(self) -> None:
        entry = BASE_METHODS["nuts"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_nuts_params())
        assert hasattr(kernel, "init"), "kernel must have .init"
        assert hasattr(kernel, "step"), "kernel must have .step"

    def test_init_returns_finite_logdensity(self) -> None:
        entry = BASE_METHODS["nuts"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_nuts_params())
        state = kernel.init(_POSITION)
        assert jnp.isfinite(state.logdensity)

    def test_step_returns_new_state_and_info(self) -> None:
        key = jax.random.key(1)
        entry = BASE_METHODS["nuts"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_nuts_params())
        state = kernel.init(_POSITION)
        new_state, info = kernel.step(key, state)
        assert jnp.isfinite(new_state.logdensity)

    def test_step_info_has_num_integration_steps(self) -> None:
        key = jax.random.key(2)
        entry = BASE_METHODS["nuts"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_nuts_params())
        state = kernel.init(_POSITION)
        _, info = kernel.step(key, state)
        assert hasattr(info, "num_integration_steps")
        assert info.num_integration_steps >= 0

    def test_grad_count_non_negative(self) -> None:
        key = jax.random.key(3)
        entry = BASE_METHODS["nuts"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_nuts_params())
        state = kernel.init(_POSITION)
        _, info = kernel.step(key, state)
        count = entry.grad_count_per_step(info)
        assert int(jnp.asarray(count)) >= 0

    def test_grad_count_matches_info(self) -> None:
        key = jax.random.key(4)
        entry = BASE_METHODS["nuts"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_nuts_params())
        state = kernel.init(_POSITION)
        _, info = kernel.step(key, state)
        count = entry.grad_count_per_step(info)
        assert int(jnp.asarray(count)) == int(info.num_integration_steps)


class TestHmcFactory:
    def test_factory_callable(self) -> None:
        entry = BASE_METHODS["hmc"]
        assert callable(entry.factory)

    def test_factory_returns_sampling_algorithm(self) -> None:
        entry = BASE_METHODS["hmc"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_hmc_params())
        assert hasattr(kernel, "init"), "kernel must have .init"
        assert hasattr(kernel, "step"), "kernel must have .step"

    def test_init_returns_finite_logdensity(self) -> None:
        entry = BASE_METHODS["hmc"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_hmc_params())
        state = kernel.init(_POSITION)
        assert jnp.isfinite(state.logdensity)

    def test_step_returns_new_state_and_info(self) -> None:
        key = jax.random.key(5)
        entry = BASE_METHODS["hmc"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_hmc_params())
        state = kernel.init(_POSITION)
        new_state, info = kernel.step(key, state)
        assert jnp.isfinite(new_state.logdensity)

    def test_step_info_has_num_integration_steps(self) -> None:
        key = jax.random.key(6)
        entry = BASE_METHODS["hmc"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_hmc_params())
        state = kernel.init(_POSITION)
        _, info = kernel.step(key, state)
        assert hasattr(info, "num_integration_steps")
        assert int(info.num_integration_steps) == 10  # fixed-L HMC: should equal param

    def test_grad_count_non_negative(self) -> None:
        key = jax.random.key(7)
        entry = BASE_METHODS["hmc"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_hmc_params())
        state = kernel.init(_POSITION)
        _, info = kernel.step(key, state)
        count = entry.grad_count_per_step(info)
        assert int(jnp.asarray(count)) >= 0

    def test_grad_count_equals_num_integration_steps(self) -> None:
        key = jax.random.key(8)
        entry = BASE_METHODS["hmc"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_hmc_params())
        state = kernel.init(_POSITION)
        _, info = kernel.step(key, state)
        count = entry.grad_count_per_step(info)
        assert int(jnp.asarray(count)) == int(info.num_integration_steps)


# ===========================================================================
# 5-step end-to-end chain smoke tests
# ===========================================================================


class TestEndToEndChain:
    def _run_chain(self, entry: BaseMethod, params: dict, n_steps: int = 5) -> None:
        """Run n_steps of a chain and assert all states have finite logdensity."""
        key = jax.random.key(42)
        kernel = entry.factory(_LOGDENSITY_FN, **params)
        state = kernel.init(_POSITION)
        total_grads = 0
        for i in range(n_steps):
            key, subkey = jax.random.split(key)
            state, info = kernel.step(subkey, state)
            assert jnp.isfinite(
                state.logdensity
            ), f"Non-finite logdensity at step {i}: {state.logdensity}"
            total_grads += int(jnp.asarray(entry.grad_count_per_step(info)))
        assert total_grads > 0, "Expected at least one gradient evaluation"

    def test_nuts_5_step_chain(self) -> None:
        self._run_chain(BASE_METHODS["nuts"], _make_nuts_params())

    def test_hmc_5_step_chain(self) -> None:
        self._run_chain(BASE_METHODS["hmc"], _make_hmc_params())

    def test_nuts_position_changes(self) -> None:
        """At least one step of NUTS should move the position."""
        key = jax.random.key(99)
        entry = BASE_METHODS["nuts"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_nuts_params())
        state = kernel.init(_POSITION)
        any_moved = False
        for _ in range(5):
            key, subkey = jax.random.split(key)
            new_state, _ = kernel.step(subkey, state)
            if not jnp.allclose(new_state.position["x"], state.position["x"]):
                any_moved = True
                break
            state = new_state
        assert any_moved, "NUTS position never moved in 5 steps — likely a bug"

    def test_hmc_position_changes(self) -> None:
        """At least one step of HMC should move the position."""
        key = jax.random.key(100)
        entry = BASE_METHODS["hmc"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_hmc_params())
        state = kernel.init(_POSITION)
        any_moved = False
        for _ in range(5):
            key, subkey = jax.random.split(key)
            new_state, _ = kernel.step(subkey, state)
            if not jnp.allclose(new_state.position["x"], state.position["x"]):
                any_moved = True
                break
            state = new_state
        assert any_moved, "HMC position never moved in 5 steps — likely a bug"


# ===========================================================================
# Window adaptation smoke test for HMC (empirical question #2)
# ===========================================================================


class TestHmcWindowAdaptation:
    def test_window_adaptation_hmc_runs(self) -> None:
        """Confirm blackjax.window_adaptation works with blackjax.hmc.

        This verifies that the BO tuning runner can use the same warmup path for
        HMC as for NUTS.  Empirical question #2 in the design spec.
        """
        import blackjax

        key = jax.random.key(0)
        warmup = blackjax.window_adaptation(
            blackjax.hmc,
            _LOGDENSITY_FN,
            target_acceptance_rate=0.65,
            num_integration_steps=10,
        )
        (adapted_state, adapted_params), _ = warmup.run(key, _POSITION, 100)
        assert "step_size" in adapted_params
        assert "inverse_mass_matrix" in adapted_params
        assert jnp.isfinite(adapted_state.logdensity)
        assert float(adapted_params["step_size"]) > 0


# ===========================================================================
# Optuna distribution round-trip (empirical question #3)
# ===========================================================================


class TestOptunaRoundTrip:
    def test_loguniform_distribution(self) -> None:
        """HyperparamSpace 'loguniform' → FloatDistribution(log=True) round-trip."""
        import optuna
        from optuna.distributions import FloatDistribution

        hp = BASE_METHODS["nuts"].default_hp_space[0]  # step_size, loguniform
        assert hp.kind == "loguniform"
        dist = FloatDistribution(hp.low, hp.high, log=True)
        study = optuna.create_study()
        trial = study.ask({"step_size": dist})
        val = trial.params["step_size"]
        assert hp.low <= val <= hp.high

    def test_int_distribution(self) -> None:
        """HyperparamSpace 'int' → IntDistribution round-trip."""
        import optuna
        from optuna.distributions import IntDistribution

        # num_integration_steps is the second HP of HMC
        hp = BASE_METHODS["hmc"].default_hp_space[1]  # num_integration_steps, int
        assert hp.kind == "int"
        dist = IntDistribution(hp.low, hp.high)
        study = optuna.create_study()
        trial = study.ask({"num_integration_steps": dist})
        val = trial.params["num_integration_steps"]
        assert hp.low <= val <= hp.high
