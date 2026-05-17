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
"""Tests for the dynamic_hmc base method registry entry.

Covers:
  1. dynamic_hmc entry exists in BASE_METHODS registry.
  2. dynamic_hmc factory is callable and returns a SamplingAlgorithm.
  3. dynamic_hmc kernel step advances position (state is mutated by one step).
  4. dynamic_hmc grad_count_per_step reads info.num_integration_steps.
  5. dynamic_hmc default_hp_space has exactly 1 BO-tunable HP (step_size);
     inverse_mass_matrix is NOT in the HP space (warmup-derived).
  6. needs_mass_matrix=True confirmed.
  7. target_acceptance_rate=0.651 (CHEES default, not 0.65).
  8. blackjax.dhmc is blackjax.dynamic_hmc alias confirmed.
"""

import blackjax
import jax
import jax.numpy as jnp
import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.model import MODELS

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_MVN = MODELS["mvn_10"]
_D = 10  # MVN-10 has 10 dimensions
_SEED = 42
_RNG_KEY = jax.random.key(_SEED)


def _build_logdensity(posterior_entry, key):
    from tuningfork.model._numpyro import build_logdensity_fn

    init_position, logdensity_fn, _ = build_logdensity_fn(key, posterior_entry)
    return init_position, logdensity_fn


# ---------------------------------------------------------------------------
# 1. Registry: dynamic_hmc entry exists
# ---------------------------------------------------------------------------


class TestDynamicHmcRegistry:
    """dynamic_hmc entry exists in BASE_METHODS and is correctly named."""

    def test_dynamic_hmc_entry_in_registry(self) -> None:
        assert (
            "dynamic_hmc" in BASE_METHODS
        ), f"'dynamic_hmc' not found in BASE_METHODS; registered: {sorted(BASE_METHODS)}"

    def test_dynamic_hmc_name_field_matches_key(self) -> None:
        entry = BASE_METHODS["dynamic_hmc"]
        assert (
            entry.name == "dynamic_hmc"
        ), f"Expected name='dynamic_hmc', got {entry.name!r}"

    def test_dynamic_hmc_family_is_mcmc(self) -> None:
        entry = BASE_METHODS["dynamic_hmc"]
        assert entry.family == "mcmc", f"Expected family='mcmc', got {entry.family!r}"

    def test_dynamic_hmc_needs_mass_matrix_true(self) -> None:
        """inverse_mass_matrix comes from CHEES warmup, not BO."""
        entry = BASE_METHODS["dynamic_hmc"]
        assert (
            entry.needs_mass_matrix is True
        ), "dynamic_hmc.needs_mass_matrix must be True (inverse_mass_matrix from CHEES warmup)"

    def test_dynamic_hmc_target_acceptance_rate_is_chees_default(self) -> None:
        """CHEES default target_acceptance_rate=0.651 (slightly above HMC 0.65)."""
        entry = BASE_METHODS["dynamic_hmc"]
        assert entry.target_acceptance_rate == pytest.approx(
            0.651
        ), f"Expected target_acceptance_rate=0.651 (CHEES default), got {entry.target_acceptance_rate}"


# ---------------------------------------------------------------------------
# 2. Factory: dynamic_hmc factory returns a SamplingAlgorithm
# ---------------------------------------------------------------------------


class TestDynamicHmcFactory:
    """dynamic_hmc factory produces a valid BlackJAX SamplingAlgorithm."""

    def test_dynamic_hmc_factory_callable(self) -> None:
        entry = BASE_METHODS["dynamic_hmc"]
        assert callable(entry.factory), "dynamic_hmc.factory must be callable"

    def test_dynamic_hmc_factory_returns_sampling_algorithm(self) -> None:
        """factory(logdensity_fn, **valid_hps) returns object with .init and .step."""
        key = jax.random.key(101)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["dynamic_hmc"]
        inverse_mass_matrix = jnp.ones(_D)
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=inverse_mass_matrix,
        )
        assert hasattr(algo, "init"), "factory result must have .init method"
        assert hasattr(algo, "step"), "factory result must have .step method"

    def test_dynamic_hmc_factory_init_returns_state_with_position(self) -> None:
        """kernel.init(position, rng_key) returns DynamicHMCState with position field."""
        key = jax.random.key(102)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["dynamic_hmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(_D),
        )
        init_key, _ = jax.random.split(key)
        # dynamic_hmc.init requires rng_key for random_generator_arg
        state = algo.init(init_pos, rng_key=init_key)
        assert hasattr(state, "position"), "DynamicHMCState must have 'position' field"
        assert hasattr(
            state, "random_generator_arg"
        ), "DynamicHMCState must have 'random_generator_arg' field"


# ---------------------------------------------------------------------------
# 3. Kernel step advances position
# ---------------------------------------------------------------------------


class TestDynamicHmcKernelStep:
    """dynamic_hmc kernel step produces a new state (position changes on typical step)."""

    def test_dynamic_hmc_kernel_step_advances_position(self) -> None:
        """One kernel.step call from typical init should change position."""
        key = jax.random.key(201)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["dynamic_hmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(_D),
        )
        init_key, step_key = jax.random.split(key)
        init_state = algo.init(init_pos, rng_key=init_key)
        new_state, info = algo.step(step_key, init_state)

        # With step_size=0.1, position should change.
        assert not jnp.allclose(
            jax.tree.leaves(init_state.position)[0],
            jax.tree.leaves(new_state.position)[0],
            atol=1e-8,
        ), "Position did not change after one dynamic_hmc step (possible kernel bug)"

    def test_dynamic_hmc_kernel_step_returns_info_with_acceptance_rate(self) -> None:
        """dynamic_hmc step info must include acceptance_rate (HMCInfo)."""
        key = jax.random.key(202)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["dynamic_hmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(_D),
        )
        init_key, step_key = jax.random.split(key)
        init_state = algo.init(init_pos, rng_key=init_key)
        _, info = algo.step(step_key, init_state)

        assert hasattr(
            info, "acceptance_rate"
        ), f"dynamic_hmc info must have acceptance_rate; got fields: {getattr(info, '_fields', dir(info))}"
        ar = float(info.acceptance_rate)
        assert 0.0 <= ar <= 1.0, f"acceptance_rate={ar} out of [0, 1]"

    def test_dynamic_hmc_kernel_step_info_has_num_integration_steps(self) -> None:
        """dynamic_hmc step info must include num_integration_steps (random each step)."""
        key = jax.random.key(203)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["dynamic_hmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(_D),
        )
        init_key, step_key = jax.random.split(key)
        init_state = algo.init(init_pos, rng_key=init_key)
        _, info = algo.step(step_key, init_state)

        assert hasattr(
            info, "num_integration_steps"
        ), f"dynamic_hmc info must have num_integration_steps; got: {getattr(info, '_fields', dir(info))}"
        n_steps = int(info.num_integration_steps)
        assert n_steps >= 1, f"num_integration_steps={n_steps} must be >= 1"


# ---------------------------------------------------------------------------
# 4. Grad count reads info.num_integration_steps
# ---------------------------------------------------------------------------


class TestDynamicHmcGradCount:
    """dynamic_hmc grad_count_per_step reads info.num_integration_steps (dynamic)."""

    def test_dynamic_hmc_grad_count_uses_num_integration_steps(self) -> None:
        """grad_count_per_step(real_info) == info.num_integration_steps."""
        key = jax.random.key(301)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["dynamic_hmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(_D),
        )
        init_key, step_key = jax.random.split(key)
        init_state = algo.init(init_pos, rng_key=init_key)
        _, info = algo.step(step_key, init_state)

        grad_count = entry.grad_count_per_step(info)
        expected = int(info.num_integration_steps)
        assert (
            int(grad_count) == expected
        ), f"grad_count_per_step should return {expected} (from info), got {grad_count}"

    def test_dynamic_hmc_grad_count_returns_array(self) -> None:
        """grad_count_per_step returns a JAX array (not a Python int)."""
        key = jax.random.key(302)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["dynamic_hmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(_D),
        )
        init_key, step_key = jax.random.split(key)
        init_state = algo.init(init_pos, rng_key=init_key)
        _, info = algo.step(step_key, init_state)

        grad_count = entry.grad_count_per_step(info)
        assert isinstance(
            grad_count, jax.Array
        ), f"grad_count_per_step should return a JAX Array, got {type(grad_count)}"

    def test_dynamic_hmc_grad_count_positive(self) -> None:
        """grad_count_per_step result must be >= 1 (at least 1 leapfrog step)."""
        key = jax.random.key(303)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["dynamic_hmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(_D),
        )
        init_key, step_key = jax.random.split(key)
        init_state = algo.init(init_pos, rng_key=init_key)
        _, info = algo.step(step_key, init_state)

        grad_count = entry.grad_count_per_step(info)
        assert int(grad_count) >= 1, f"grad_count must be >= 1, got {grad_count}"


# ---------------------------------------------------------------------------
# 5. HP space: only step_size; inverse_mass_matrix absent
# ---------------------------------------------------------------------------


class TestDynamicHmcHpSpace:
    """dynamic_hmc default_hp_space has exactly 1 BO-tunable HP (step_size)."""

    def _hp_names(self) -> set[str]:
        return {hp.name for hp in BASE_METHODS["dynamic_hmc"].default_hp_space}

    def test_dynamic_hmc_hp_space_has_only_step_size(self) -> None:
        """Only step_size is BO-tunable; trajectory length is CHEES-adapted."""
        hp_names = self._hp_names()
        assert (
            "step_size" in hp_names
        ), f"step_size missing from hp_space; got {hp_names}"
        assert len(hp_names) == 1, (
            f"Expected exactly 1 BO-tunable HP (step_size), got {len(hp_names)}: {hp_names}. "
            f"inverse_mass_matrix and callable trajectory-length params are NOT BO-tunable."
        )

    def test_dynamic_hmc_hp_space_no_inverse_mass_matrix(self) -> None:
        """inverse_mass_matrix must NOT be in HP space (warmup-derived from CHEES)."""
        assert (
            "inverse_mass_matrix" not in self._hp_names()
        ), "inverse_mass_matrix must NOT be in dynamic_hmc hp_space (it's CHEES warmup-derived)"

    def test_dynamic_hmc_hp_space_no_num_integration_steps(self) -> None:
        """num_integration_steps is NOT BO-tunable (dynamic; adapted by CHEES)."""
        assert (
            "num_integration_steps" not in self._hp_names()
        ), "num_integration_steps must NOT be in dynamic_hmc hp_space (adapted by CHEES)"

    def test_dynamic_hmc_step_size_is_loguniform(self) -> None:
        for hp in BASE_METHODS["dynamic_hmc"].default_hp_space:
            if hp.name == "step_size":
                assert (
                    hp.kind == "loguniform"
                ), f"step_size kind should be 'loguniform', got {hp.kind!r}"
                assert hp.low == pytest.approx(1e-3)
                assert hp.high == pytest.approx(1.0)
                break


# ---------------------------------------------------------------------------
# 6. Alias: blackjax.dhmc is blackjax.dynamic_hmc
# ---------------------------------------------------------------------------


class TestDhmcAlias:
    """blackjax.dhmc is blackjax.dynamic_hmc (confirmed alias, 2026-05-09)."""

    def test_dhmc_alias_confirmed(self) -> None:
        assert (
            blackjax.dhmc is blackjax.dynamic_hmc
        ), "blackjax.dhmc alias broken: dhmc is not dynamic_hmc"
