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
"""Tests for the persistent_sampling_smc SMC registry entry.

Covers:
1. ENTRY field correctness (name, family, default_inner_method,
   num_particles_default, HP space, step_kwargs_schema).
2. _COMPATIBLE_INNER excludes mclmc, adjusted_mclmc, adjusted_mclmc_dynamic.
3. HP space default values: n_schedule int [5, 50], num_mcmc_steps int [1, 50].
4. End-to-end factory test with RWM inner kernel on synthetic 5-D Gaussian:
   - factory returns SamplingAlgorithm with .init and .step
   - init returns PersistentSMCState with correct particle shape
   - step (3-arg: rng_key, state, lmbda) advances state correctly
   - multiple steps produce finite particles
5. Schema-validation negative test: missing required kwargs raises TypeError.

Step-arity finding: persistent_sampling_smc uses a 3-arg step
  step_fn(rng_key, state, lmbda) — caller must supply lmbda. This differs
  from adaptive_tempered_smc's standard 2-arg step_fn(rng_key, state).
  step_kwargs_schema = ("lmbda",) reflects this.
"""

import functools

import blackjax.mcmc.random_walk as _rw
import jax
import jax.numpy as jnp
import pytest
from blackjax.base import SamplingAlgorithm

from tuningfork.smc.persistent_sampling import ENTRY

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DIM = 5
_NUM_PARTICLES = 200  # small for test speed
_SEED = 42
_N_SCHEDULE = 10  # number of tempering steps for preallocation


def _logprior_fn(x):
    return -0.5 * jnp.sum(x**2)


def _loglikelihood_fn(x):
    return -0.5 * jnp.sum((x - 1.0) ** 2)


# ===========================================================================
# 1. ENTRY field correctness
# ===========================================================================


class TestPersistentSamplingSMCEntry:
    def test_name(self) -> None:
        assert ENTRY.name == "persistent_sampling_smc"

    def test_family(self) -> None:
        assert ENTRY.family == "smc"

    def test_default_inner_method(self) -> None:
        assert ENTRY.default_inner_method == "rwm"

    def test_num_particles_default(self) -> None:
        assert ENTRY.num_particles_default == 1000

    def test_factory_is_callable(self) -> None:
        assert callable(ENTRY.factory)

    def test_step_kwargs_schema_has_lmbda(self) -> None:
        """persistent_sampling step_fn(key, state, lmbda) requires extra lmbda arg."""
        assert ENTRY.step_kwargs_schema == ("lmbda",)

    def test_notes_non_empty(self) -> None:
        assert len(ENTRY.notes) > 0

    def test_compatible_inner_methods_non_empty(self) -> None:
        assert len(ENTRY.compatible_inner_methods) > 0

    def test_default_inner_in_compatible(self) -> None:
        assert ENTRY.default_inner_method in ENTRY.compatible_inner_methods

    def test_hp_space_has_two_entries(self) -> None:
        assert len(ENTRY.default_hp_space) == 2

    def test_hp_space_names(self) -> None:
        names = {hp.name for hp in ENTRY.default_hp_space}
        assert "n_schedule" in names
        assert "num_mcmc_steps" in names


# ===========================================================================
# 2. _COMPATIBLE_INNER excludes mclmc family
# ===========================================================================


class TestPersistentSamplingCompatibleInner:
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


class TestPersistentSamplingHpSpace:
    def _hp_by_name(self, name: str):
        for hp in ENTRY.default_hp_space:
            if hp.name == name:
                return hp
        raise KeyError(f"HP '{name}' not found in ENTRY.default_hp_space")

    def test_n_schedule_is_int(self) -> None:
        hp = self._hp_by_name("n_schedule")
        assert hp.kind == "int"

    def test_n_schedule_low(self) -> None:
        hp = self._hp_by_name("n_schedule")
        assert hp.low == 5

    def test_n_schedule_high(self) -> None:
        hp = self._hp_by_name("n_schedule")
        assert hp.high == 50

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


def _make_rwm_inner_kernel(dim: int, sigma: float) -> SamplingAlgorithm:
    """Build a RWM inner kernel suitable for SMC.

    Key finding: blackjax SMC's from_mcmc.unshared_parameters_and_step_fn
    calls .shape on every value in mcmc_parameters, so mcmc_parameters must
    contain ONLY JAX arrays. The random_step callable for RWM must be bound via
    functools.partial BEFORE passing as mcmc_step_fn.
    """
    sigma_arr = jnp.full(dim, sigma)
    step_fn = functools.partial(
        _rw.build_additive_step(), random_step=_rw.normal(sigma_arr)
    )
    return SamplingAlgorithm(init=_rw.init, step=step_fn)


class TestPersistentSamplingSMCEndToEnd:
    def test_factory_returns_sampling_algorithm(self) -> None:
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            _logprior_fn,
            _loglikelihood_fn,
            inner_kernel=inner,
            mcmc_parameters={},
            n_schedule=_N_SCHEDULE,
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
            n_schedule=_N_SCHEDULE,
            num_mcmc_steps=5,
        )
        rng_key = jax.random.key(_SEED)
        initial_particles = jax.random.normal(rng_key, (_NUM_PARTICLES, _DIM))
        state = smc_alg.init(initial_particles)
        # PersistentSMCState.particles is a property returning the current iteration
        assert state.particles.shape == (_NUM_PARTICLES, _DIM)
        assert float(state.tempering_param) == 0.0

    def test_step_advances_iteration_and_tempering_param(self) -> None:
        """step_fn(rng_key, state, lmbda) — 3-arg, caller provides lmbda."""
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            _logprior_fn,
            _loglikelihood_fn,
            inner_kernel=inner,
            mcmc_parameters={},
            n_schedule=_N_SCHEDULE,
            num_mcmc_steps=5,
        )
        rng_key = jax.random.key(_SEED)
        init_key, step_key = jax.random.split(rng_key)
        initial_particles = jax.random.normal(init_key, (_NUM_PARTICLES, _DIM))
        state = smc_alg.init(initial_particles)

        # Caller provides lmbda (tempering parameter) — standard Persistent Sampling
        lmbda = jnp.array(0.3)
        state, info = jax.jit(smc_alg.step)(step_key, state, lmbda)

        assert (
            int(state.iteration) == 1
        ), f"iteration should be 1 after one step, got {state.iteration}"
        assert float(state.tempering_param) == pytest.approx(
            0.3
        ), f"tempering_param should be ~0.3 after one step, got {state.tempering_param}"

    def test_three_steps_shape_and_finite(self) -> None:
        """Run 3 SMC steps with increasing lmbda; verify shape and finiteness."""
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            _logprior_fn,
            _loglikelihood_fn,
            inner_kernel=inner,
            mcmc_parameters={},
            n_schedule=_N_SCHEDULE,
            num_mcmc_steps=5,
        )
        rng_key = jax.random.key(_SEED)
        init_key, smc_key = jax.random.split(rng_key)
        initial_particles = jax.random.normal(init_key, (_NUM_PARTICLES, _DIM))
        state = smc_alg.init(initial_particles)

        lambdas = [0.2, 0.5, 0.8]
        for lmbda_val in lambdas:
            smc_key, step_key = jax.random.split(smc_key)
            lmbda = jnp.array(lmbda_val)
            state, info = jax.jit(smc_alg.step)(step_key, state, lmbda)

        assert state.particles.shape == (_NUM_PARTICLES, _DIM), (
            f"Particle shape changed: expected ({_NUM_PARTICLES}, {_DIM}), "
            f"got {state.particles.shape}"
        )
        assert int(state.iteration) == 3
        assert jnp.all(
            jnp.isfinite(state.particles)
        ), "Particles contain non-finite values after 3 SMC steps"

    def test_persistent_particles_accumulated(self) -> None:
        """After k steps, persistent_particles has shape (n_schedule+1, N, dim)
        with the first (k+1) rows populated (row 0 = prior, rows 1..k = steps)."""
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        n_schedule = 5
        smc_alg = ENTRY.factory(
            _logprior_fn,
            _loglikelihood_fn,
            inner_kernel=inner,
            mcmc_parameters={},
            n_schedule=n_schedule,
            num_mcmc_steps=3,
        )
        rng_key = jax.random.key(_SEED)
        init_key, smc_key = jax.random.split(rng_key)
        initial_particles = jax.random.normal(init_key, (_NUM_PARTICLES, _DIM))
        state = smc_alg.init(initial_particles)

        # persistent_particles has shape (n_schedule + 1, num_particles, dim)
        assert state.persistent_particles.shape == (
            n_schedule + 1,
            _NUM_PARTICLES,
            _DIM,
        )

        # Take 2 steps
        for lmbda_val in [0.3, 0.7]:
            smc_key, step_key = jax.random.split(smc_key)
            state, _ = jax.jit(smc_alg.step)(step_key, state, jnp.array(lmbda_val))

        assert state.persistent_particles.shape == (
            n_schedule + 1,
            _NUM_PARTICLES,
            _DIM,
        )
        assert int(state.iteration) == 2


# ===========================================================================
# 5. Schema-validation negative test: missing required kwargs raises TypeError
# ===========================================================================


class TestPersistentSamplingFactoryNegative:
    def test_missing_inner_kernel_raises_type_error(self) -> None:
        """factory without inner_kernel raises TypeError."""
        with pytest.raises(TypeError):
            ENTRY.factory(_logprior_fn, _loglikelihood_fn, n_schedule=10)

    def test_missing_n_schedule_raises_type_error(self) -> None:
        """factory without n_schedule raises TypeError (keyword-only arg)."""
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        with pytest.raises(TypeError):
            ENTRY.factory(
                _logprior_fn,
                _loglikelihood_fn,
                inner_kernel=inner,
                mcmc_parameters={},
                # n_schedule intentionally omitted
            )

    def test_missing_mcmc_parameters_raises_type_error(self) -> None:
        """factory without mcmc_parameters raises TypeError (keyword-only arg)."""
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        with pytest.raises(TypeError):
            ENTRY.factory(
                _logprior_fn,
                _loglikelihood_fn,
                inner_kernel=inner,
                n_schedule=10,
                # mcmc_parameters intentionally omitted
            )
