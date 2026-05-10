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
"""Tests for the P5.13 mhmc base method registry entry.

Covers:
  1. mhmc entry exists in BASE_METHODS registry.
  2. mhmc factory is callable and returns a SamplingAlgorithm.
  3. mhmc kernel step advances position (state is mutated by one step).
  4. mhmc grad_count_per_step reads info.num_integration_steps.
  5. mhmc default_hp_space has step_size + num_integration_steps (identical to HMC).
  6. needs_mass_matrix=True confirmed.
  7. target_acceptance_rate=0.65 (same as HMC, Beskos et al.).
  8. blackjax.multinomial_hmc is blackjax.mhmc alias confirmed.
"""

import blackjax
import jax
import jax.numpy as jnp
import pytest

from bjx_bench.inference.base_method import BASE_METHODS
from bjx_bench.model import MODELS

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_MVN = MODELS["mvn_10"]
_D = 10  # MVN-10 has 10 dimensions
_SEED = 42
_RNG_KEY = jax.random.key(_SEED)


def _build_logdensity(posterior_entry, key):
    from bjx_bench.model._numpyro import build_logdensity_fn

    init_position, logdensity_fn, _ = build_logdensity_fn(key, posterior_entry)
    return init_position, logdensity_fn


# ---------------------------------------------------------------------------
# 1. Registry: mhmc entry exists
# ---------------------------------------------------------------------------


class TestMhmcRegistry:
    """mhmc entry exists in BASE_METHODS and is correctly named."""

    def test_mhmc_entry_in_registry(self) -> None:
        assert (
            "mhmc" in BASE_METHODS
        ), f"'mhmc' not found in BASE_METHODS; registered: {sorted(BASE_METHODS)}"

    def test_mhmc_name_field_matches_key(self) -> None:
        entry = BASE_METHODS["mhmc"]
        assert entry.name == "mhmc", f"Expected name='mhmc', got {entry.name!r}"

    def test_mhmc_family_is_mcmc(self) -> None:
        entry = BASE_METHODS["mhmc"]
        assert entry.family == "mcmc", f"Expected family='mcmc', got {entry.family!r}"

    def test_mhmc_needs_mass_matrix_true(self) -> None:
        """inverse_mass_matrix comes from window adaptation warmup, not BO."""
        entry = BASE_METHODS["mhmc"]
        assert (
            entry.needs_mass_matrix is True
        ), "mhmc.needs_mass_matrix must be True (inverse_mass_matrix from warmup)"

    def test_mhmc_target_acceptance_rate_is_065(self) -> None:
        """Optimal accept rate ≈ 0.65 for fixed-L HMC (Beskos et al. 2013)."""
        entry = BASE_METHODS["mhmc"]
        assert entry.target_acceptance_rate == pytest.approx(
            0.65
        ), f"Expected target_acceptance_rate=0.65, got {entry.target_acceptance_rate}"


# ---------------------------------------------------------------------------
# 2. Factory: mhmc factory returns a SamplingAlgorithm
# ---------------------------------------------------------------------------


class TestMhmcFactory:
    """mhmc factory produces a valid BlackJAX SamplingAlgorithm."""

    def test_mhmc_factory_callable(self) -> None:
        entry = BASE_METHODS["mhmc"]
        assert callable(entry.factory), "mhmc.factory must be callable"

    def test_mhmc_factory_returns_sampling_algorithm(self) -> None:
        """factory(logdensity_fn, **valid_hps) returns object with .init and .step."""
        key = jax.random.key(101)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["mhmc"]
        inverse_mass_matrix = jnp.ones(_D)
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=inverse_mass_matrix,
            num_integration_steps=10,
        )
        assert hasattr(algo, "init"), "factory result must have .init method"
        assert hasattr(algo, "step"), "factory result must have .step method"

    def test_mhmc_factory_init_returns_state_with_position(self) -> None:
        """kernel.init(position) returns HMCState with position field."""
        key = jax.random.key(102)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["mhmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(_D),
            num_integration_steps=10,
        )
        state = algo.init(init_pos)
        assert hasattr(state, "position"), "HMCState must have 'position' field"


# ---------------------------------------------------------------------------
# 3. Kernel step advances position
# ---------------------------------------------------------------------------


class TestMhmcKernelStep:
    """mhmc kernel step produces a new state (position changes on typical step)."""

    def test_mhmc_kernel_step_advances_position(self) -> None:
        """One kernel.step call from typical init should change position."""
        key = jax.random.key(201)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["mhmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(_D),
            num_integration_steps=10,
        )
        init_state = algo.init(init_pos)
        new_state, info = algo.step(key, init_state)

        # With step_size=0.1, position should change.
        assert not jnp.allclose(
            jax.tree.leaves(init_state.position)[0],
            jax.tree.leaves(new_state.position)[0],
            atol=1e-8,
        ), "Position did not change after one mhmc step (possible kernel bug)"

    def test_mhmc_kernel_step_returns_info_with_acceptance_rate(self) -> None:
        """mhmc step info must include acceptance_rate (HMCInfo)."""
        key = jax.random.key(202)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["mhmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(_D),
            num_integration_steps=10,
        )
        init_state = algo.init(init_pos)
        _, info = algo.step(key, init_state)

        assert hasattr(
            info, "acceptance_rate"
        ), f"mhmc info must have acceptance_rate; got fields: {getattr(info, '_fields', dir(info))}"
        ar = float(info.acceptance_rate)
        assert 0.0 <= ar <= 1.0, f"acceptance_rate={ar} out of [0, 1]"

    def test_mhmc_kernel_step_info_has_num_integration_steps(self) -> None:
        """mhmc step info must include num_integration_steps."""
        key = jax.random.key(203)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["mhmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(_D),
            num_integration_steps=10,
        )
        init_state = algo.init(init_pos)
        _, info = algo.step(key, init_state)

        assert hasattr(
            info, "num_integration_steps"
        ), f"mhmc info must have num_integration_steps; got: {getattr(info, '_fields', dir(info))}"
        n_steps = int(info.num_integration_steps)
        assert (
            n_steps == 10
        ), f"num_integration_steps={n_steps} must equal configured 10"


# ---------------------------------------------------------------------------
# 4. Grad count reads info.num_integration_steps
# ---------------------------------------------------------------------------


class TestMhmcGradCount:
    """mhmc grad_count_per_step reads info.num_integration_steps."""

    def test_mhmc_grad_count_uses_num_integration_steps(self) -> None:
        """grad_count_per_step(real_info) == info.num_integration_steps."""
        key = jax.random.key(301)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["mhmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(_D),
            num_integration_steps=10,
        )
        init_state = algo.init(init_pos)
        _, info = algo.step(key, init_state)

        grad_count = entry.grad_count_per_step(info)
        expected = int(info.num_integration_steps)
        assert (
            int(grad_count) == expected
        ), f"grad_count_per_step should return {expected} (from info), got {grad_count}"

    def test_mhmc_grad_count_returns_array(self) -> None:
        """grad_count_per_step returns a JAX array (not a Python int)."""
        key = jax.random.key(302)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["mhmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(_D),
            num_integration_steps=5,
        )
        init_state = algo.init(init_pos)
        _, info = algo.step(key, init_state)

        grad_count = entry.grad_count_per_step(info)
        assert isinstance(
            grad_count, jax.Array
        ), f"grad_count_per_step should return a JAX Array, got {type(grad_count)}"

    def test_mhmc_grad_count_positive(self) -> None:
        """grad_count_per_step result must be >= 1 (at least 1 leapfrog step)."""
        key = jax.random.key(303)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["mhmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(_D),
            num_integration_steps=3,
        )
        init_state = algo.init(init_pos)
        _, info = algo.step(key, init_state)

        grad_count = entry.grad_count_per_step(info)
        assert int(grad_count) >= 1, f"grad_count must be >= 1, got {grad_count}"


# ---------------------------------------------------------------------------
# 5. HP space: step_size + num_integration_steps (identical to HMC)
# ---------------------------------------------------------------------------


class TestMhmcHpSpace:
    """mhmc default_hp_space has step_size + num_integration_steps (identical to HMC)."""

    def _hp_names(self) -> set[str]:
        return {hp.name for hp in BASE_METHODS["mhmc"].default_hp_space}

    def test_mhmc_hp_space_has_step_size_and_num_integration_steps(self) -> None:
        """Both step_size and num_integration_steps are BO-tunable."""
        hp_names = self._hp_names()
        assert (
            "step_size" in hp_names
        ), f"step_size missing from hp_space; got {hp_names}"
        assert "num_integration_steps" in hp_names, (
            f"num_integration_steps missing from hp_space; got {hp_names}. "
            f"mhmc is static-L like HMC, so num_integration_steps IS BO-tunable."
        )
        assert len(hp_names) == 2, (
            f"Expected exactly 2 BO-tunable HPs (step_size + num_integration_steps), "
            f"got {len(hp_names)}: {hp_names}"
        )

    def test_mhmc_hp_space_no_inverse_mass_matrix(self) -> None:
        """inverse_mass_matrix must NOT be in HP space (warmup-derived)."""
        assert (
            "inverse_mass_matrix" not in self._hp_names()
        ), "inverse_mass_matrix must NOT be in mhmc hp_space (it's warmup-derived)"

    def test_mhmc_step_size_is_loguniform(self) -> None:
        for hp in BASE_METHODS["mhmc"].default_hp_space:
            if hp.name == "step_size":
                assert (
                    hp.kind == "loguniform"
                ), f"step_size kind should be 'loguniform', got {hp.kind!r}"
                assert hp.low == pytest.approx(1e-3)
                assert hp.high == pytest.approx(1.0)
                break

    def test_mhmc_num_integration_steps_is_int(self) -> None:
        for hp in BASE_METHODS["mhmc"].default_hp_space:
            if hp.name == "num_integration_steps":
                assert (
                    hp.kind == "int"
                ), f"num_integration_steps kind should be 'int', got {hp.kind!r}"
                assert hp.low == 1
                assert hp.high == 128
                break


# ---------------------------------------------------------------------------
# 6. Alias: blackjax.multinomial_hmc is blackjax.mhmc
# ---------------------------------------------------------------------------


class TestMultinomialHmcAlias:
    """blackjax.multinomial_hmc is blackjax.mhmc (backward-compat alias)."""

    def test_multinomial_hmc_alias_confirmed(self) -> None:
        assert (
            blackjax.multinomial_hmc is blackjax.mhmc
        ), "blackjax.multinomial_hmc alias broken: multinomial_hmc is not mhmc"
