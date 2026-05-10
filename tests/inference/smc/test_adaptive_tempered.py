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
"""Tests for the adaptive_tempered_smc SMC registry entry.

Covers:
1. ENTRY field correctness (name, family, default_inner_method,
   num_particles_default, HP space).
2. _COMPATIBLE_INNER excludes mclmc, adjusted_mclmc, adjusted_mclmc_dynamic.
3. HP space default values: target_ess uniform [0.3, 0.95],
   num_mcmc_steps int [1, 50].
4. End-to-end factory test with RWM inner kernel on synthetic 5-D Gaussian.
5. Schema-validation negative test: missing required kwargs raises TypeError.

Inner-kernel contract finding:
  blackjax SMC's from_mcmc.unshared_parameters_and_step_fn calls .shape on
  every value in mcmc_parameters, so mcmc_parameters must contain ONLY JAX
  arrays. Callable params (e.g. random_step for RWM) must be bound via
  functools.partial BEFORE passing as mcmc_step_fn to the SMC layer.
  The inner_kernel.step used by our factory must already have non-array
  params bound; inner_kernel.init is blackjax.mcmc.random_walk.init.
"""

import functools

import blackjax.mcmc.random_walk as _rw
import jax
import jax.numpy as jnp
import pytest
from blackjax.base import SamplingAlgorithm

from bjx_bench.inference.smc.adaptive_tempered import ENTRY

pytestmark = pytest.mark.fast

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


# ===========================================================================
# 1. ENTRY field correctness
# ===========================================================================


class TestAdaptiveTemperedSMCEntry:
    def test_name(self) -> None:
        assert ENTRY.name == "adaptive_tempered_smc"

    def test_family(self) -> None:
        assert ENTRY.family == "smc"

    def test_default_inner_method(self) -> None:
        assert ENTRY.default_inner_method == "rwm"

    def test_num_particles_default(self) -> None:
        assert ENTRY.num_particles_default == 1000

    def test_factory_is_callable(self) -> None:
        assert callable(ENTRY.factory)

    def test_step_kwargs_schema_empty(self) -> None:
        """Standard step(key, state) signature; no extra kwargs needed."""
        assert ENTRY.step_kwargs_schema == ()

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
        assert "target_ess" in names
        assert "num_mcmc_steps" in names


# ===========================================================================
# 2. _COMPATIBLE_INNER excludes mclmc family
# ===========================================================================


class TestAdaptiveTemperedCompatibleInner:
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


class TestAdaptiveTemperedHpSpace:
    def _hp_by_name(self, name: str):
        for hp in ENTRY.default_hp_space:
            if hp.name == name:
                return hp
        raise KeyError(f"HP '{name}' not found in ENTRY.default_hp_space")

    def test_target_ess_is_uniform(self) -> None:
        hp = self._hp_by_name("target_ess")
        assert hp.kind == "uniform"

    def test_target_ess_low(self) -> None:
        hp = self._hp_by_name("target_ess")
        assert hp.low == pytest.approx(0.3)

    def test_target_ess_high(self) -> None:
        hp = self._hp_by_name("target_ess")
        assert hp.high == pytest.approx(0.95)

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

    Returns a SamplingAlgorithm whose .step is a partial-bound raw kernel and
    whose .init is blackjax.mcmc.random_walk.init.
    """
    sigma_arr = jnp.full(dim, sigma)
    # Bind the non-array random_step at build time so mcmc_parameters stays array-only
    step_fn = functools.partial(
        _rw.build_additive_step(), random_step=_rw.normal(sigma_arr)
    )
    return SamplingAlgorithm(init=_rw.init, step=step_fn)


class TestAdaptiveTemperedSMCEndToEnd:
    def test_factory_returns_sampling_algorithm(self) -> None:
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            _logprior_fn,
            _loglikelihood_fn,
            inner_kernel=inner,
            mcmc_parameters={},
            target_ess=0.5,
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
            target_ess=0.5,
            num_mcmc_steps=5,
        )
        rng_key = jax.random.key(_SEED)
        initial_particles = jax.random.normal(rng_key, (_NUM_PARTICLES, _DIM))
        state = smc_alg.init(initial_particles)
        assert state.particles.shape == (_NUM_PARTICLES, _DIM)
        assert float(state.tempering_param) == 0.0

    def test_step_advances_tempering_param(self) -> None:
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            _logprior_fn,
            _loglikelihood_fn,
            inner_kernel=inner,
            mcmc_parameters={},
            target_ess=0.5,
            num_mcmc_steps=5,
        )
        rng_key = jax.random.key(_SEED)
        init_key, step_key = jax.random.split(rng_key)
        initial_particles = jax.random.normal(init_key, (_NUM_PARTICLES, _DIM))
        state = smc_alg.init(initial_particles)
        state, info = jax.jit(smc_alg.step)(step_key, state)
        assert (
            float(state.tempering_param) > 0.0
        ), f"tempering_param should be > 0 after one step, got {state.tempering_param}"

    def test_three_steps_shape_and_finite(self) -> None:
        """Run 3 SMC steps; verify particle shape and finiteness."""
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            _logprior_fn,
            _loglikelihood_fn,
            inner_kernel=inner,
            mcmc_parameters={},
            target_ess=0.5,
            num_mcmc_steps=5,
        )
        rng_key = jax.random.key(_SEED)
        init_key, smc_key = jax.random.split(rng_key)
        initial_particles = jax.random.normal(init_key, (_NUM_PARTICLES, _DIM))
        state = smc_alg.init(initial_particles)

        for _i in range(3):
            smc_key, step_key = jax.random.split(smc_key)
            state, info = jax.jit(smc_alg.step)(step_key, state)

        assert state.particles.shape == (_NUM_PARTICLES, _DIM), (
            f"Particle shape changed: expected ({_NUM_PARTICLES}, {_DIM}), "
            f"got {state.particles.shape}"
        )
        assert float(state.tempering_param) > 0.0
        assert jnp.all(
            jnp.isfinite(state.particles)
        ), "Particles contain non-finite values after 3 SMC steps"


# ===========================================================================
# 5. Schema-validation negative test: missing required kwargs raises TypeError
# ===========================================================================


class TestAdaptiveTemperedFactoryNegative:
    def test_missing_inner_kernel_raises_type_error(self) -> None:
        """factory(logprior, loglikelihood) without inner_kernel/mcmc_parameters raises."""
        with pytest.raises(TypeError):
            ENTRY.factory(_logprior_fn, _loglikelihood_fn)

    def test_missing_mcmc_parameters_raises_type_error(self) -> None:
        """factory without mcmc_parameters raises TypeError (keyword-only arg)."""
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        with pytest.raises(TypeError):
            ENTRY.factory(
                _logprior_fn,
                _loglikelihood_fn,
                inner_kernel=inner,
                # mcmc_parameters intentionally omitted
            )
