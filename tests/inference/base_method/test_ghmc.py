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
"""Tests for the GHMC base method registry entry.

Covers:
  1. GHMC entry exists in BASE_METHODS registry.
  2. GHMC factory is callable and returns a SamplingAlgorithm.
  3. GHMC kernel step advances position (state is mutated by one step).
  4. GHMC grad_count_per_step always returns 1 (constant; independent of info).
  5. GHMC default_hp_space has exactly the 3 BO-tunable HPs (step_size, alpha,
     delta); momentum_inverse_scale is NOT in the HP space (warmup-derived).
"""

import jax
import jax.numpy as jnp
import pytest

from bjx_bench.inference.base_method import BASE_METHODS
from bjx_bench.model import MODELS

pytestmark = pytest.mark.fast

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
# 1. Registry: GHMC entry exists
# ---------------------------------------------------------------------------


class TestGhmcRegistry:
    """GHMC entry exists in BASE_METHODS and is correctly named."""

    def test_ghmc_entry_in_registry(self) -> None:
        assert (
            "ghmc" in BASE_METHODS
        ), f"'ghmc' not found in BASE_METHODS; registered: {sorted(BASE_METHODS)}"

    def test_ghmc_name_field_matches_key(self) -> None:
        entry = BASE_METHODS["ghmc"]
        assert entry.name == "ghmc", f"Expected name='ghmc', got {entry.name!r}"

    def test_ghmc_family_is_mcmc(self) -> None:
        entry = BASE_METHODS["ghmc"]
        assert entry.family == "mcmc", f"Expected family='mcmc', got {entry.family!r}"

    def test_ghmc_needs_mass_matrix(self) -> None:
        """momentum_inverse_scale comes from MEADS warmup, not BO."""
        entry = BASE_METHODS["ghmc"]
        assert (
            entry.needs_mass_matrix is True
        ), "ghmc.needs_mass_matrix must be True (momentum_inverse_scale from warmup)"

    def test_ghmc_target_acceptance_rate_is_0_65(self) -> None:
        entry = BASE_METHODS["ghmc"]
        assert entry.target_acceptance_rate == pytest.approx(
            0.65
        ), f"Expected target_acceptance_rate=0.65, got {entry.target_acceptance_rate}"


# ---------------------------------------------------------------------------
# 2. Factory: GHMC factory returns a SamplingAlgorithm
# ---------------------------------------------------------------------------


class TestGhmcFactory:
    """GHMC factory produces a valid BlackJAX SamplingAlgorithm."""

    def test_ghmc_factory_callable(self) -> None:
        entry = BASE_METHODS["ghmc"]
        assert callable(entry.factory), "ghmc.factory must be callable"

    def test_ghmc_factory_returns_sampling_algorithm(self) -> None:
        """factory(logdensity_fn, **valid_hps) returns object with .init and .step."""
        key = jax.random.key(101)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["ghmc"]
        momentum_inverse_scale = jnp.ones(_D)
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            momentum_inverse_scale=momentum_inverse_scale,
            alpha=0.5,
            delta=0.1,
        )
        assert hasattr(algo, "init"), "factory result must have .init method"
        assert hasattr(algo, "step"), "factory result must have .step method"

    def test_ghmc_factory_init_returns_state_with_position(self) -> None:
        """kernel.init(position, rng_key) returns GHMCState with position field."""
        key = jax.random.key(102)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["ghmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            momentum_inverse_scale=jnp.ones(_D),
            alpha=0.5,
            delta=0.1,
        )
        init_key, _ = jax.random.split(key)
        state = algo.init(init_pos, rng_key=init_key)
        assert hasattr(state, "position"), "GHMCState must have 'position' field"
        assert hasattr(state, "momentum"), "GHMCState must have 'momentum' field"
        assert hasattr(state, "slice"), "GHMCState must have 'slice' field"


# ---------------------------------------------------------------------------
# 3. Kernel step advances position
# ---------------------------------------------------------------------------


class TestGhmcKernelStep:
    """GHMC kernel step produces a new state (position changes on typical step)."""

    def test_ghmc_kernel_step_advances_position(self) -> None:
        """One kernel.step call from typical init should change position."""
        key = jax.random.key(201)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["ghmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            momentum_inverse_scale=jnp.ones(_D),
            alpha=0.5,
            delta=0.1,
        )
        init_key, step_key = jax.random.split(key)
        init_state = algo.init(init_pos, rng_key=init_key)
        new_state, info = algo.step(step_key, init_state)

        # With step_size=0.1 and reasonable alpha/delta, position should change.
        assert not jnp.allclose(
            jax.tree.leaves(init_state.position)[0],
            jax.tree.leaves(new_state.position)[0],
            atol=1e-8,
        ), "Position did not change after one GHMC step (possible kernel bug)"

    def test_ghmc_kernel_step_returns_info_with_acceptance_rate(self) -> None:
        """GHMC step info must include acceptance_rate (HMCInfo)."""
        key = jax.random.key(202)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["ghmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            momentum_inverse_scale=jnp.ones(_D),
            alpha=0.5,
            delta=0.1,
        )
        init_key, step_key = jax.random.split(key)
        init_state = algo.init(init_pos, rng_key=init_key)
        _, info = algo.step(step_key, init_state)

        assert hasattr(
            info, "acceptance_rate"
        ), f"GHMC info must have acceptance_rate; got fields: {getattr(info, '_fields', dir(info))}"
        # Acceptance rate should be in [0, 1]
        ar = float(info.acceptance_rate)
        assert 0.0 <= ar <= 1.0, f"acceptance_rate={ar} out of [0, 1]"


# ---------------------------------------------------------------------------
# 4. Grad count per step is always 1
# ---------------------------------------------------------------------------


class TestGhmcGradCount:
    """GHMC grad_count_per_step always returns 1 (constant; 1 leapfrog per step)."""

    def test_ghmc_grad_count_is_one_with_real_info(self) -> None:
        """grad_count_per_step(real_info) == 1."""
        key = jax.random.key(301)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        entry = BASE_METHODS["ghmc"]
        algo = entry.factory(
            logdensity_fn,
            step_size=0.1,
            momentum_inverse_scale=jnp.ones(_D),
            alpha=0.5,
            delta=0.1,
        )
        init_key, step_key = jax.random.split(key)
        init_state = algo.init(init_pos, rng_key=init_key)
        _, info = algo.step(step_key, init_state)

        grad_count = entry.grad_count_per_step(info)
        assert (
            int(grad_count) == 1
        ), f"grad_count_per_step should return 1, got {grad_count}"

    def test_ghmc_grad_count_is_one_with_none_info(self) -> None:
        """grad_count_per_step(None) == 1 (constant; doesn't read info)."""
        entry = BASE_METHODS["ghmc"]
        grad_count = entry.grad_count_per_step(None)
        assert (
            int(grad_count) == 1
        ), f"grad_count_per_step(None) should return 1, got {grad_count}"

    def test_ghmc_grad_count_returns_array(self) -> None:
        """grad_count_per_step returns a JAX array (not a Python int)."""
        import jax

        entry = BASE_METHODS["ghmc"]
        grad_count = entry.grad_count_per_step(None)
        assert isinstance(
            grad_count, jax.Array
        ), f"grad_count_per_step should return a JAX Array, got {type(grad_count)}"


# ---------------------------------------------------------------------------
# 5. HP space: step_size, alpha, delta present; momentum_inverse_scale absent
# ---------------------------------------------------------------------------


class TestGhmcHpSpace:
    """GHMC default_hp_space has exactly the 3 BO-tunable HPs."""

    def _hp_names(self) -> set[str]:
        return {hp.name for hp in BASE_METHODS["ghmc"].default_hp_space}

    def test_hp_space_has_step_size(self) -> None:
        assert (
            "step_size" in self._hp_names()
        ), f"step_size missing from ghmc hp_space; got {self._hp_names()}"

    def test_hp_space_has_alpha(self) -> None:
        assert (
            "alpha" in self._hp_names()
        ), f"alpha missing from ghmc hp_space; got {self._hp_names()}"

    def test_hp_space_has_delta(self) -> None:
        assert (
            "delta" in self._hp_names()
        ), f"delta missing from ghmc hp_space; got {self._hp_names()}"

    def test_hp_space_does_not_have_momentum_inverse_scale(self) -> None:
        """momentum_inverse_scale comes from MEADS warmup, not BO."""
        assert (
            "momentum_inverse_scale" not in self._hp_names()
        ), "momentum_inverse_scale must NOT be in ghmc hp_space (it's warmup-derived)"

    def test_hp_space_has_exactly_three_entries(self) -> None:
        """Exactly 3 BO-tunable HPs: step_size, alpha, delta."""
        hp_names = self._hp_names()
        assert (
            len(hp_names) == 3
        ), f"Expected exactly 3 BO-tunable HPs, got {len(hp_names)}: {hp_names}"

    def test_step_size_is_loguniform(self) -> None:
        for hp in BASE_METHODS["ghmc"].default_hp_space:
            if hp.name == "step_size":
                assert (
                    hp.kind == "loguniform"
                ), f"step_size kind should be 'loguniform', got {hp.kind!r}"
                break

    def test_alpha_is_uniform(self) -> None:
        for hp in BASE_METHODS["ghmc"].default_hp_space:
            if hp.name == "alpha":
                assert (
                    hp.kind == "uniform"
                ), f"alpha kind should be 'uniform', got {hp.kind!r}"
                assert hp.low == pytest.approx(0.0)
                assert hp.high == pytest.approx(1.0)
                break

    def test_delta_is_uniform(self) -> None:
        for hp in BASE_METHODS["ghmc"].default_hp_space:
            if hp.name == "delta":
                assert (
                    hp.kind == "uniform"
                ), f"delta kind should be 'uniform', got {hp.kind!r}"
                assert hp.low == pytest.approx(0.0)
                assert hp.high == pytest.approx(1.0)
                break
