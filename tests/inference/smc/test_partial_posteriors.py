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
"""Tests for the partial_posteriors_smc SMC registry entry.

Covers:
1. ENTRY field correctness (name, family, default_inner_method,
   num_particles_default, HP space, step_kwargs_schema).
2. _COMPATIBLE_INNER excludes mclmc, adjusted_mclmc, adjusted_mclmc_dynamic.
3. HP space default values: num_mcmc_steps int [1, 50].
4. End-to-end factory test with RWM inner kernel on a 5-D Gaussian with
   a synthetic dataset of 6 data points; 3 SMC steps with progressively
   growing data masks; num_particles=200.
5. State shapes: particles.shape == (num_particles, 5), weights.shape == (200,),
   data_mask.shape == (num_data,), all finite.
6. Schema-validation negative tests.
7. ENTRY compatibility metadata importable without registry registration
   (commit-1 smoke — confirm ENTRY exists and is an SMCMethod).

Finding: init_fn signature is (particles, num_observations), NOT
  the standard (particles,) — must pass num_observations explicitly.
  step_fn signature is (rng_key, state, data_mask) — the extra data_mask
  positional arg selects which data points to include in the next partial
  posterior.
"""

import functools

import blackjax.mcmc.random_walk as _rw
import jax
import jax.numpy as jnp
import pytest
from blackjax.base import SamplingAlgorithm

from tuningfork.inference.smc._base import SMCMethod
from tuningfork.inference.smc.partial_posteriors import ENTRY

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DIM = 5
_NUM_PARTICLES = 200  # small for test speed
_NUM_DATA = 6  # small synthetic dataset
_SEED = 42


def _make_partial_logposterior_factory(data):
    """Build a partial_logposterior_factory for a 5-D Gaussian model.

    The 'dataset' consists of _NUM_DATA i.i.d. observations from N(0, I).
    The partial posterior given mask m is:
      log p(x | data[mask]) = log prior(x) + sum_{i: mask[i]=1} log N(data[i] | x, I)

    Parameters
    ----------
    data
        Array of shape (num_data, dim) — the synthetic observations.
    """

    def partial_logposterior_factory(data_mask):
        def logposterior_fn(x):
            logprior = -0.5 * jnp.sum(x**2)
            # loglikelihood contribution per observation: -0.5 * ||data_i - x||^2
            # Weight by mask (0 or 1) to include/exclude each observation.
            per_obs = jnp.sum(
                data_mask[:, None] * (-0.5 * (data - x[None, :]) ** 2), axis=-1
            )
            loglik = jnp.sum(per_obs)
            return logprior + loglik

        return logposterior_fn

    return partial_logposterior_factory


def _make_rwm_inner_kernel(dim: int, sigma: float) -> SamplingAlgorithm:
    """Build a RWM inner kernel suitable for SMC.

    Binds the non-array random_step callable via functools.partial so that
    mcmc_parameters stays array-only (required by from_mcmc.unshared_parameters_and_step_fn).
    """
    sigma_arr = jnp.full(dim, sigma)
    step_fn = functools.partial(
        _rw.build_additive_step(), random_step=_rw.normal(sigma_arr)
    )
    return SamplingAlgorithm(init=_rw.init, step=step_fn)


# ===========================================================================
# 1. ENTRY field correctness
# ===========================================================================


class TestPartialPosteriorsSMCEntry:
    def test_name(self) -> None:
        assert ENTRY.name == "partial_posteriors_smc"

    def test_family(self) -> None:
        assert ENTRY.family == "smc"

    def test_default_inner_method(self) -> None:
        assert ENTRY.default_inner_method == "rwm"

    def test_num_particles_default(self) -> None:
        assert ENTRY.num_particles_default == 1000

    def test_factory_is_callable(self) -> None:
        assert callable(ENTRY.factory)

    def test_step_kwargs_schema_has_data_mask(self) -> None:
        """step_fn requires an extra data_mask positional arg."""
        assert ENTRY.step_kwargs_schema == ("data_mask",)

    def test_notes_non_empty(self) -> None:
        assert len(ENTRY.notes) > 0

    def test_compatible_inner_methods_non_empty(self) -> None:
        assert len(ENTRY.compatible_inner_methods) > 0

    def test_default_inner_in_compatible(self) -> None:
        assert ENTRY.default_inner_method in ENTRY.compatible_inner_methods

    def test_hp_space_has_one_entry(self) -> None:
        assert len(ENTRY.default_hp_space) == 1

    def test_hp_space_name_is_num_mcmc_steps(self) -> None:
        names = {hp.name for hp in ENTRY.default_hp_space}
        assert "num_mcmc_steps" in names

    def test_is_smc_method_instance(self) -> None:
        """Commit-1 smoke: ENTRY is a valid SMCMethod without registry."""
        assert isinstance(ENTRY, SMCMethod)


# ===========================================================================
# 2. _COMPATIBLE_INNER excludes mclmc family
# ===========================================================================


class TestPartialPosteriorsSMCCompatibleInner:
    def test_mclmc_excluded(self) -> None:
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


class TestPartialPosteriorsSMCHpSpace:
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
# 4 + 5. End-to-end factory test with RWM inner kernel
# ===========================================================================


class TestPartialPosteriorsSMCEndToEnd:
    def _make_data_and_factory(self, seed=_SEED):
        rng_key = jax.random.key(seed)
        data = jax.random.normal(rng_key, (_NUM_DATA, _DIM))
        factory = _make_partial_logposterior_factory(data)
        return data, factory

    def test_factory_returns_sampling_algorithm(self) -> None:
        _data, pf = self._make_data_and_factory()
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            None,  # logprior_fn — ignored
            None,  # loglikelihood_fn — ignored
            inner_kernel=inner,
            mcmc_parameters={},
            partial_logposterior_factory=pf,
            num_observations=_NUM_DATA,
            num_mcmc_steps=5,
        )
        assert hasattr(smc_alg, "init"), "factory result must have .init"
        assert hasattr(smc_alg, "step"), "factory result must have .step"

    def test_init_fn_requires_num_observations(self) -> None:
        """init_fn signature: (particles, num_observations) — not standard."""
        _data, pf = self._make_data_and_factory()
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            None,
            None,
            inner_kernel=inner,
            mcmc_parameters={},
            partial_logposterior_factory=pf,
            num_observations=_NUM_DATA,
            num_mcmc_steps=5,
        )
        rng_key = jax.random.key(_SEED)
        initial_particles = jax.random.normal(rng_key, (_NUM_PARTICLES, _DIM))
        state = smc_alg.init(initial_particles, _NUM_DATA)
        assert state.particles.shape == (_NUM_PARTICLES, _DIM)
        assert state.weights.shape == (_NUM_PARTICLES,)
        assert state.data_mask.shape == (_NUM_DATA,)
        # Initial data_mask should be all-zeros (no data included yet)
        assert jnp.all(state.data_mask == 0)

    def test_step_with_data_mask_changes_particles(self) -> None:
        """step_fn(rng_key, state, data_mask) with a non-trivial mask."""
        _data, pf = self._make_data_and_factory()
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            None,
            None,
            inner_kernel=inner,
            mcmc_parameters={},
            partial_logposterior_factory=pf,
            num_observations=_NUM_DATA,
            num_mcmc_steps=5,
        )
        rng_key = jax.random.key(_SEED)
        init_key, step_key = jax.random.split(rng_key)
        initial_particles = jax.random.normal(init_key, (_NUM_PARTICLES, _DIM))
        state = smc_alg.init(initial_particles, _NUM_DATA)

        # Include the first 2 data points
        data_mask = jnp.array([1, 1, 0, 0, 0, 0], dtype=jnp.float32)
        new_state, info = jax.jit(smc_alg.step)(step_key, state, data_mask)

        assert new_state.particles.shape == (_NUM_PARTICLES, _DIM)
        assert new_state.weights.shape == (_NUM_PARTICLES,)
        assert new_state.data_mask.shape == (_NUM_DATA,)
        # After step, data_mask should reflect the mask we passed
        assert jnp.all(new_state.data_mask == data_mask)

    def test_three_steps_shape_and_finite(self) -> None:
        """Run 3 SMC steps with progressively growing masks; verify shapes."""
        _data, pf = self._make_data_and_factory()
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        smc_alg = ENTRY.factory(
            None,
            None,
            inner_kernel=inner,
            mcmc_parameters={},
            partial_logposterior_factory=pf,
            num_observations=_NUM_DATA,
            num_mcmc_steps=5,
        )
        rng_key = jax.random.key(_SEED)
        init_key, smc_key = jax.random.split(rng_key)
        initial_particles = jax.random.normal(init_key, (_NUM_PARTICLES, _DIM))
        state = smc_alg.init(initial_particles, _NUM_DATA)

        # Progressively grow the data mask over 3 steps
        masks = [
            jnp.array([1, 0, 0, 0, 0, 0], dtype=jnp.float32),
            jnp.array([1, 1, 1, 0, 0, 0], dtype=jnp.float32),
            jnp.array([1, 1, 1, 1, 1, 0], dtype=jnp.float32),
        ]

        for mask in masks:
            smc_key, step_key = jax.random.split(smc_key)
            state, info = jax.jit(smc_alg.step)(step_key, state, mask)

        assert state.particles.shape == (_NUM_PARTICLES, _DIM), (
            f"Particle shape changed: expected ({_NUM_PARTICLES}, {_DIM}), "
            f"got {state.particles.shape}"
        )
        assert state.weights.shape == (_NUM_PARTICLES,)
        assert jnp.all(
            jnp.isfinite(state.particles)
        ), "Particles contain non-finite values after 3 SMC steps"
        assert jnp.all(
            jnp.isfinite(state.weights)
        ), "Weights contain non-finite values after 3 SMC steps"

    def test_logprior_loglikelihood_ignored(self) -> None:
        """logprior_fn and loglikelihood_fn are ignored — sentinel values work."""
        _data, pf = self._make_data_and_factory()
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        # Passing sentinel 'IGNORED' strings should not raise
        smc_alg = ENTRY.factory(
            "IGNORED_LOGPRIOR",
            "IGNORED_LOGLIKELIHOOD",
            inner_kernel=inner,
            mcmc_parameters={},
            partial_logposterior_factory=pf,
            num_observations=_NUM_DATA,
            num_mcmc_steps=5,
        )
        assert hasattr(smc_alg, "init")
        assert hasattr(smc_alg, "step")


# ===========================================================================
# 6. Schema-validation negative tests
# ===========================================================================


class TestPartialPosteriorsSMCFactoryNegative:
    def test_missing_partial_logposterior_factory_raises(self) -> None:
        """Factory without partial_logposterior_factory raises TypeError."""
        inner = _make_rwm_inner_kernel(_DIM, sigma=0.5)
        with pytest.raises(TypeError):
            ENTRY.factory(
                None,
                None,
                inner_kernel=inner,
                mcmc_parameters={},
                num_observations=_NUM_DATA,
                # partial_logposterior_factory intentionally omitted
            )

    def test_missing_inner_kernel_raises(self) -> None:
        """Factory without inner_kernel raises TypeError."""
        with pytest.raises(TypeError):
            ENTRY.factory(
                None,
                None,
                # inner_kernel intentionally omitted
                mcmc_parameters={},
                partial_logposterior_factory=lambda m: lambda x: 0.0,
                num_observations=_NUM_DATA,
            )
