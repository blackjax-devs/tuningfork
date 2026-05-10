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
"""Tests for the meanfield_vi base method (sampler mode).

Covers:
  1. ENTRY field correctness (name, family, hp_space, target_acceptance_rate).
  2. factory() returns an object with .init and .step.
  3. init() preserves pytree shape and returns MFVISamplerState.
  4. step() returns (MFVISamplerState, MFVIInfo) with correct shapes.
  5. JIT compatibility: jax.jit(algo.step)(key, state) runs without error.
  6. End-to-end: 5-D std normal target; ELBO converges; sample mean/std in
     atol=0.15 of (0, 1) after 200 draws.

Single seed make_rng(42) per kickoff decision — no parametrized fixture.
"""

import math

import jax
import jax.numpy as jnp
import optax
import pytest

from bjx_bench.inference.base_method.meanfield_vi import ENTRY, MFVISamplerState

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_D = 5
_SEED = 42


def _logdensity_fn(x: jax.Array) -> jax.Array:
    """Standard 5-D isotropic Gaussian log-density (without normalising const)."""
    return -0.5 * jnp.sum(x**2)


# ---------------------------------------------------------------------------
# 1. ENTRY field correctness
# ---------------------------------------------------------------------------


class TestMFVIEntryFields:
    """ENTRY field validation for meanfield_vi."""

    def test_name(self) -> None:
        assert ENTRY.name == "meanfield_vi"

    def test_family(self) -> None:
        assert ENTRY.family == "vi"

    def test_default_hp_space_has_num_optimization_steps(self) -> None:
        """num_optimization_steps is listed to satisfy BaseMethod validation.
        It is recipe-time by default but can be BO-tuned if desired."""
        names = {s.name for s in ENTRY.default_hp_space}
        assert (
            "num_optimization_steps" in names
        ), f"Expected 'num_optimization_steps' in hp_space; got {names}"

    def test_target_acceptance_rate_none(self) -> None:
        """VI is not a MH sampler; no target acceptance rate."""
        assert ENTRY.target_acceptance_rate is None

    def test_needs_mass_matrix_false(self) -> None:
        assert ENTRY.needs_mass_matrix is False

    def test_factory_callable(self) -> None:
        assert callable(ENTRY.factory)

    def test_grad_count_callable(self) -> None:
        assert callable(ENTRY.grad_count_per_step)

    def test_grad_count_returns_one(self) -> None:
        """grad_count_per_step is approximately 1 per step."""
        from blackjax.vi.meanfield_vi import MFVIInfo

        fake_info = MFVIInfo(elbo=jnp.asarray(-3.0))
        result = ENTRY.grad_count_per_step(fake_info)
        assert int(result) == 1, f"Expected grad_count=1, got {result}"


# ---------------------------------------------------------------------------
# 2. factory() returns an object with .init and .step
# ---------------------------------------------------------------------------


class TestMFVIFactory:
    """Factory invocation and interface contract."""

    def test_factory_returns_algo_with_init(self) -> None:
        algo = ENTRY.factory(_logdensity_fn, num_optimization_steps=50)
        assert hasattr(algo, "init"), "factory result must have .init"

    def test_factory_returns_algo_with_step(self) -> None:
        algo = ENTRY.factory(_logdensity_fn, num_optimization_steps=50)
        assert hasattr(algo, "step"), "factory result must have .step"

    def test_factory_accepts_optimizer_kwarg(self) -> None:
        custom_optimizer = optax.adam(5e-3)
        algo = ENTRY.factory(
            _logdensity_fn,
            num_optimization_steps=50,
            optimizer=custom_optimizer,
        )
        assert hasattr(algo, "init")


# ---------------------------------------------------------------------------
# 3. init() returns MFVISamplerState with correct shape
# ---------------------------------------------------------------------------


class TestMFVIInit:
    """init() correctness tests."""

    def test_init_returns_mfvi_sampler_state(self) -> None:
        algo = ENTRY.factory(_logdensity_fn, num_optimization_steps=50)
        init_pos = jnp.zeros(_D)
        state = algo.init(init_pos)
        assert isinstance(
            state, MFVISamplerState
        ), f"Expected MFVISamplerState, got {type(state)}"

    def test_init_position_shape(self) -> None:
        algo = ENTRY.factory(_logdensity_fn, num_optimization_steps=50)
        init_pos = jnp.zeros(_D)
        state = algo.init(init_pos)
        assert state.position.shape == (
            _D,
        ), f"Expected position shape ({_D},), got {state.position.shape}"

    def test_init_vi_state_has_mu_rho(self) -> None:
        algo = ENTRY.factory(_logdensity_fn, num_optimization_steps=50)
        state = algo.init(jnp.zeros(_D))
        assert hasattr(state.vi_state, "mu"), "vi_state must have 'mu'"
        assert hasattr(state.vi_state, "rho"), "vi_state must have 'rho'"
        assert hasattr(state.vi_state, "opt_state"), "vi_state must have 'opt_state'"

    def test_init_mu_shape(self) -> None:
        algo = ENTRY.factory(_logdensity_fn, num_optimization_steps=50)
        state = algo.init(jnp.zeros(_D))
        mu_flat, _ = jax.flatten_util.ravel_pytree(state.vi_state.mu)
        assert mu_flat.shape == (_D,), f"Expected mu shape ({_D},), got {mu_flat.shape}"


# ---------------------------------------------------------------------------
# 4. step() returns (MFVISamplerState, MFVIInfo) with correct shapes
# ---------------------------------------------------------------------------


class TestMFVIStep:
    """step() correctness tests."""

    def test_step_returns_two_tuple(self) -> None:
        algo = ENTRY.factory(_logdensity_fn, num_optimization_steps=50)
        state = algo.init(jnp.zeros(_D))
        key = jax.random.key(_SEED)
        result = algo.step(key, state)
        assert len(result) == 2, f"Expected 2-tuple from step, got len={len(result)}"

    def test_step_new_state_is_mfvi_sampler_state(self) -> None:
        algo = ENTRY.factory(_logdensity_fn, num_optimization_steps=50)
        state = algo.init(jnp.zeros(_D))
        new_state, _info = algo.step(jax.random.key(_SEED), state)
        assert isinstance(new_state, MFVISamplerState)

    def test_step_position_shape(self) -> None:
        algo = ENTRY.factory(_logdensity_fn, num_optimization_steps=50)
        state = algo.init(jnp.zeros(_D))
        new_state, _info = algo.step(jax.random.key(_SEED), state)
        assert new_state.position.shape == (
            _D,
        ), f"Expected position shape ({_D},), got {new_state.position.shape}"

    def test_step_vi_state_unchanged(self) -> None:
        """The vi_state must be identical after step (fit is frozen)."""
        algo = ENTRY.factory(_logdensity_fn, num_optimization_steps=50)
        state = algo.init(jnp.zeros(_D))
        new_state, _info = algo.step(jax.random.key(_SEED), state)
        # mu should be identical (frozen)
        assert jnp.allclose(
            jax.flatten_util.ravel_pytree(new_state.vi_state.mu)[0],
            jax.flatten_util.ravel_pytree(state.vi_state.mu)[0],
        ), "vi_state.mu should be unchanged after step"

    def test_step_info_has_elbo_field(self) -> None:
        algo = ENTRY.factory(_logdensity_fn, num_optimization_steps=50)
        state = algo.init(jnp.zeros(_D))
        _new_state, info = algo.step(jax.random.key(_SEED), state)
        assert hasattr(info, "elbo"), "MFVIInfo must have 'elbo' field"


# ---------------------------------------------------------------------------
# 5. JIT compatibility
# ---------------------------------------------------------------------------


class TestMFVIJIT:
    """JIT compatibility tests."""

    def test_step_is_jittable(self) -> None:
        algo = ENTRY.factory(_logdensity_fn, num_optimization_steps=50)
        state = algo.init(jnp.zeros(_D))
        key = jax.random.key(_SEED)

        jit_step = jax.jit(algo.step)
        new_state, info = jit_step(key, state)
        assert new_state.position.shape == (
            _D,
        ), f"JIT step position shape wrong: {new_state.position.shape}"

    def test_scan_over_step(self) -> None:
        """Verify step is compatible with jax.lax.scan (tracer-safe)."""
        algo = ENTRY.factory(_logdensity_fn, num_optimization_steps=50)
        state = algo.init(jnp.zeros(_D))

        def one_step(carry, key):
            new_state, info = algo.step(key, carry)
            return new_state, new_state.position

        keys = jax.random.split(jax.random.key(_SEED), 10)
        final_state, positions = jax.lax.scan(one_step, state, keys)
        assert positions.shape == (
            10,
            _D,
        ), f"Expected scan positions shape (10, {_D}), got {positions.shape}"


# ---------------------------------------------------------------------------
# 6. End-to-end: 5-D std normal target
# ---------------------------------------------------------------------------


class TestMFVIEndToEnd:
    """End-to-end convergence test on 5-D standard normal target.

    Uses 2_000 optimisation steps (test default from kickoff) and collects
    200 samples via jax.lax.scan over step().
    """

    def test_elbo_converges_on_5d_std_normal(self) -> None:
        """Final ELBO > -0.5*5*(1+log(2π)) - 1.0 after 2_000 steps."""
        from blackjax.vi.meanfield_vi import init, step

        optimizer = optax.adam(1e-2)
        vi_state = init(jnp.zeros(_D), optimizer)
        num_steps = 2_000

        def one_step(carry, key):
            new_state, info = step(key, carry, _logdensity_fn, optimizer, 5)
            return new_state, info

        keys = jax.random.split(jax.random.key(_SEED), num_steps)
        final_state, infos = jax.lax.scan(one_step, vi_state, keys)

        final_elbo = float(infos.elbo[-1])
        threshold = -0.5 * _D * (1 + math.log(2 * math.pi)) - 1.0
        assert final_elbo > threshold, (
            f"ELBO did not converge: final_elbo={final_elbo:.4f}, "
            f"threshold={threshold:.4f}"
        )

    def test_samples_mean_close_to_zero(self) -> None:
        """Sample mean within atol=0.15 of 0 after 200 draws."""
        algo = ENTRY.factory(_logdensity_fn, num_optimization_steps=2_000)
        state = algo.init(jnp.zeros(_D))

        def one_step(carry, key):
            new_state, _info = algo.step(key, carry)
            return new_state, new_state.position

        keys = jax.random.split(jax.random.key(_SEED), 200)
        _final_state, positions = jax.lax.scan(one_step, state, keys)
        # positions: (200, D)
        sample_mean = jnp.mean(positions, axis=0)
        assert jnp.allclose(
            sample_mean, jnp.zeros(_D), atol=0.15
        ), f"Sample mean too far from 0: {sample_mean}"

    def test_samples_std_close_to_one(self) -> None:
        """Sample std within atol=0.15 of 1 after 200 draws."""
        algo = ENTRY.factory(_logdensity_fn, num_optimization_steps=2_000)
        state = algo.init(jnp.zeros(_D))

        def one_step(carry, key):
            new_state, _info = algo.step(key, carry)
            return new_state, new_state.position

        keys = jax.random.split(jax.random.key(_SEED), 200)
        _final_state, positions = jax.lax.scan(one_step, state, keys)
        # positions: (200, D)
        sample_std = jnp.std(positions, axis=0)
        assert jnp.allclose(
            sample_std, jnp.ones(_D), atol=0.15
        ), f"Sample std too far from 1: {sample_std}"
