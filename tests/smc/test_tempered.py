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
"""Tests for the tempered_smc SMC registry entry.

Covers:
  1. ENTRY field correctness (name, family, default_inner_method,
     num_particles_default, step_kwargs_schema, HP space).
  2. _COMPATIBLE_INNER excludes mclmc family; includes MH-based methods.
  3. HP space: num_mcmc_steps int [1, 50].
  4. End-to-end factory test with RWM inner kernel on synthetic 5-D Gaussian.
  5. State type: TemperedSMCState._fields = ('particles', 'weights', 'tempering_param').
  6. Step-signature contract: step_fn(key, state, tempering_param) — 3-arg.
  7. Schema-validation negative test: missing required kwargs raises TypeError.

Finding: step_fn uses 'tempering_param' (not 'lmbda' as in
persistent_sampling). The state field is also 'tempering_param'. This is the
correct upstream spelling from blackjax/smc/tempered.py:TemperedSMCState.

Comparison with adaptive_tempered_smc:
  - adaptive_tempered_smc: step_fn(key, state) — 2-arg; auto-selects lmbda.
  - tempered_smc: step_fn(key, state, tempering_param) — 3-arg; caller drives schedule.
"""

import functools

import blackjax.mcmc.random_walk as _rw
import jax
import jax.numpy as jnp
import pytest
from blackjax.base import SamplingAlgorithm
from blackjax.smc.tempered import TemperedSMCState

from tuningfork.smc.tempered import ENTRY

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DIM = 5
_NUM_PARTICLES = 200  # small for test speed
_SEED = 42


def _logprior_fn(x):
    return -0.5 * jnp.sum(x**2)


def _loglikelihood_fn(x):
    return -0.5 * jnp.sum((x - 1.0) ** 2)


def _make_rwm_inner_kernel(dim: int, sigma: float) -> SamplingAlgorithm:
    """Build a RWM inner kernel suitable for tempered SMC.

    Finding: mcmc_parameters must contain ONLY JAX arrays.
    Non-array params (random_step callable for RWM) must be bound via
    functools.partial BEFORE passing as mcmc_step_fn.
    """
    sigma_arr = jnp.full(dim, sigma)
    step_fn = functools.partial(
        _rw.build_additive_step(), random_step=_rw.normal(sigma_arr)
    )
    return SamplingAlgorithm(init=_rw.init, step=step_fn)


# ===========================================================================
# 1. ENTRY field correctness
# ===========================================================================


class TestTemperedSMCEntry:
    def test_name(self) -> None:
        assert ENTRY.name == "tempered_smc"

    def test_family(self) -> None:
        assert ENTRY.family == "smc"

    def test_default_inner_method(self) -> None:
        assert ENTRY.default_inner_method == "rwm"

    def test_num_particles_default(self) -> None:
        assert ENTRY.num_particles_default == 1000

    def test_factory_is_callable(self) -> None:
        assert callable(ENTRY.factory)

    def test_step_kwargs_schema(self) -> None:
        """Non-standard step: step_fn(key, state, tempering_param)."""
        assert ENTRY.step_kwargs_schema == ("tempering_param",), (
            f"Expected step_kwargs_schema=('tempering_param',), "
            f"got {ENTRY.step_kwargs_schema}"
        )

    def test_notes_non_empty(self) -> None:
        assert len(ENTRY.notes) > 0

    def test_compatible_inner_methods_non_empty(self) -> None:
        assert len(ENTRY.compatible_inner_methods) > 0

    def test_default_inner_in_compatible(self) -> None:
        assert ENTRY.default_inner_method in ENTRY.compatible_inner_methods

    def test_hp_space_has_num_mcmc_steps(self) -> None:
        names = {hp.name for hp in ENTRY.default_hp_space}
        assert (
            "num_mcmc_steps" in names
        ), f"num_mcmc_steps not in default_hp_space: {names}"


# ===========================================================================
# 2. _COMPATIBLE_INNER excludes mclmc family
# ===========================================================================


class TestTemperedSMCCompatibleInner:
    def test_mclmc_excluded(self) -> None:
        """mclmc microcanonical invariance is violated by tempering."""
        assert "mclmc" not in ENTRY.compatible_inner_methods

    def test_adjusted_mclmc_excluded(self) -> None:
        assert "adjusted_mclmc" not in ENTRY.compatible_inner_methods

    def test_adjusted_mclmc_dynamic_excluded(self) -> None:
        assert "adjusted_mclmc_dynamic" not in ENTRY.compatible_inner_methods

    def test_rwm_included(self) -> None:
        assert "rwm" in ENTRY.compatible_inner_methods

    def test_nuts_included(self) -> None:
        assert "nuts" in ENTRY.compatible_inner_methods

    def test_mala_included(self) -> None:
        assert "mala" in ENTRY.compatible_inner_methods

    def test_hmc_included(self) -> None:
        assert "hmc" in ENTRY.compatible_inner_methods


# ===========================================================================
# 3. HP space default values
# ===========================================================================


class TestTemperedSMCHpSpace:
    def _hp_by_name(self, name: str):
        for hp in ENTRY.default_hp_space:
            if hp.name == name:
                return hp
        raise KeyError(f"HP '{name}' not found in ENTRY.default_hp_space")

    def test_num_mcmc_steps_is_int(self) -> None:
        hp = self._hp_by_name("num_mcmc_steps")
        assert hp.kind == "int"

    def test_num_mcmc_steps_low(self) -> None:
        hp = self._hp_by_name("num_mcmc_steps")
        assert hp.low == 1

    def test_num_mcmc_steps_high(self) -> None:
        hp = self._hp_by_name("num_mcmc_steps")
        assert hp.high == 50


# ===========================================================================
# 4. End-to-end factory test with RWM inner kernel
# ===========================================================================


class TestTemperedSMCEndToEnd:
    def test_factory_returns_sampling_algorithm(self) -> None:
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            _logprior_fn,
            _loglikelihood_fn,
            inner_kernel=inner,
            mcmc_parameters={},
            num_mcmc_steps=5,
        )
        assert hasattr(smc_alg, "init"), "factory result must have .init"
        assert hasattr(smc_alg, "step"), "factory result must have .step"

    def test_init_returns_state_with_correct_particle_shape(self) -> None:
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            _logprior_fn,
            _loglikelihood_fn,
            inner_kernel=inner,
            mcmc_parameters={},
            num_mcmc_steps=5,
        )
        key = jax.random.key(_SEED)
        initial_particles = jax.random.normal(key, (_NUM_PARTICLES, _DIM))
        state = smc_alg.init(initial_particles)
        assert state.particles.shape == (_NUM_PARTICLES, _DIM)
        assert float(state.tempering_param) == 0.0

    def test_step_advances_tempering_param(self) -> None:
        """step_fn(key, state, tempering_param=0.5) sets state.tempering_param=0.5."""
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            _logprior_fn,
            _loglikelihood_fn,
            inner_kernel=inner,
            mcmc_parameters={},
            num_mcmc_steps=5,
        )
        key = jax.random.key(_SEED)
        init_key, step_key = jax.random.split(key)
        initial_particles = jax.random.normal(init_key, (_NUM_PARTICLES, _DIM))
        state = smc_alg.init(initial_particles)
        # Caller supplies tempering_param explicitly
        new_state, _ = jax.jit(smc_alg.step)(step_key, state, 0.5)
        assert float(new_state.tempering_param) == pytest.approx(
            0.5, abs=1e-6
        ), f"Expected tempering_param=0.5 after step, got {new_state.tempering_param}"

    def test_three_steps_shape_and_finite(self) -> None:
        """Run 3 SMC steps with manual schedule; verify shape and finiteness."""
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            _logprior_fn,
            _loglikelihood_fn,
            inner_kernel=inner,
            mcmc_parameters={},
            num_mcmc_steps=5,
        )
        key = jax.random.key(_SEED)
        init_key, smc_key = jax.random.split(key)
        initial_particles = jax.random.normal(init_key, (_NUM_PARTICLES, _DIM))
        state = smc_alg.init(initial_particles)

        # Manual tempering schedule: 0 → 0.3 → 0.6 → 1.0
        schedule = [0.3, 0.6, 1.0]
        for lmbda in schedule:
            smc_key, step_key = jax.random.split(smc_key)
            state, info = jax.jit(smc_alg.step)(step_key, state, lmbda)

        assert state.particles.shape == (_NUM_PARTICLES, _DIM), (
            f"Particle shape changed: expected ({_NUM_PARTICLES}, {_DIM}), "
            f"got {state.particles.shape}"
        )
        assert float(state.tempering_param) == pytest.approx(1.0, abs=1e-6)
        assert jnp.all(
            jnp.isfinite(state.particles)
        ), "Particles contain non-finite values after 3 SMC steps"


# ===========================================================================
# 5. State type: TemperedSMCState._fields
# ===========================================================================


class TestTemperedSMCStateType:
    """TemperedSMCState from blackjax.smc.tempered (not lmbda — tempering_param)."""

    def test_init_returns_tempered_smc_state(self) -> None:
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            _logprior_fn,
            _loglikelihood_fn,
            inner_kernel=inner,
            mcmc_parameters={},
            num_mcmc_steps=5,
        )
        key = jax.random.key(_SEED)
        particles = jax.random.normal(key, (_NUM_PARTICLES, _DIM))
        state = smc_alg.init(particles)
        assert isinstance(
            state, TemperedSMCState
        ), f"Expected TemperedSMCState, got {type(state).__name__}"

    def test_state_fields(self) -> None:
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            _logprior_fn,
            _loglikelihood_fn,
            inner_kernel=inner,
            mcmc_parameters={},
            num_mcmc_steps=5,
        )
        key = jax.random.key(_SEED)
        particles = jax.random.normal(key, (_NUM_PARTICLES, _DIM))
        state = smc_alg.init(particles)
        expected = ("particles", "weights", "tempering_param")
        assert state._fields == expected, (
            f"TemperedSMCState._fields changed: expected {expected}, "
            f"got {state._fields}. Note: upstream uses 'tempering_param', "
            f"NOT 'lmbda' (unlike persistent_sampling)."
        )


# ===========================================================================
# 6. Step-signature contract: 3-arg (key, state, tempering_param)
# ===========================================================================


class TestTemperedSMCStepSignature:
    """tempered_smc.step_fn is 3-arg: (key, state, tempering_param)."""

    def test_step_without_tempering_param_raises(self) -> None:
        """step_fn(key, state) without tempering_param should raise TypeError."""
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            _logprior_fn,
            _loglikelihood_fn,
            inner_kernel=inner,
            mcmc_parameters={},
            num_mcmc_steps=2,
        )
        key = jax.random.key(_SEED)
        particles = jax.random.normal(key, (_NUM_PARTICLES, _DIM))
        state = smc_alg.init(particles)
        with pytest.raises(TypeError):
            # Missing required positional arg tempering_param
            smc_alg.step(key, state)

    def test_step_with_tempering_param_zero_is_noop(self) -> None:
        """step_fn with tempering_param == state.tempering_param == 0 is a no-op reweight."""
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            _logprior_fn,
            _loglikelihood_fn,
            inner_kernel=inner,
            mcmc_parameters={},
            num_mcmc_steps=2,
        )
        key = jax.random.key(_SEED)
        init_key, step_key = jax.random.split(key)
        particles = jax.random.normal(init_key, (_NUM_PARTICLES, _DIM))
        state = smc_alg.init(particles)
        # delta = 0 − 0 = 0: weights unchanged (all equal), no resampling
        new_state, _ = jax.jit(smc_alg.step)(step_key, state, 0.0)
        assert float(new_state.tempering_param) == pytest.approx(0.0, abs=1e-6)
        assert new_state.particles.shape == (_NUM_PARTICLES, _DIM)


# ===========================================================================
# 7. Schema-validation negative test
# ===========================================================================


class TestTemperedSMCFactoryNegative:
    def test_missing_inner_kernel_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            ENTRY.factory(_logprior_fn, _loglikelihood_fn)

    def test_missing_mcmc_parameters_raises_type_error(self) -> None:
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        with pytest.raises(TypeError):
            ENTRY.factory(
                _logprior_fn,
                _loglikelihood_fn,
                inner_kernel=inner,
                # mcmc_parameters intentionally omitted
            )
