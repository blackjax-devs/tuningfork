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
"""Tests for bjx_bench.runner.smc.

Covers runner contract: shapes, JIT compatibility, termination conditions.
Intentionally minimal — does NOT audit MCMC accuracy.

Finding: blackjax TemperedSMCState uses field 'tempering_param'
(NOT 'lmbda'). The runner detects adaptive_tempered via
hasattr(state, 'tempering_param'). The while_loop history buffer is
pre-allocated at max_steps; the returned 'lmbda' key in the history dict
holds the tempering_param series.

Finding: partial_posteriors_smc step_fn(key, state, data_mask)
requires a data_mask argument. The runner's _run_smc_scan calls
smc_step_fn(subkey, state) so the caller MUST pre-bind data_mask via
functools.partial. For the step_through_data test we pre-bind a full
all-ones mask so each of the 10 scan steps sees all observations
(data_mask.sum() == 10 after each step).
"""

import functools

import blackjax
import blackjax.mcmc.random_walk as _rw
import blackjax.smc.partial_posteriors_path as _pp_path
import jax
import jax.numpy as jnp
import pytest
from blackjax.base import SamplingAlgorithm
from blackjax.smc import resampling as _resampling

from bjx_bench.runner.smc import init_particles_from_prior, run_smc

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

_DIM = 5
_NUM_PARTICLES = 200
_NUM_OBS = 10
_SEED = 42


def _logprior_fn(x):
    return -0.5 * jnp.sum(x**2)


def _loglikelihood_fn(x):
    return -0.5 * jnp.sum((x - 1.0) ** 2)


def _make_rwm_inner_kernel(dim: int, sigma: float = 0.5) -> SamplingAlgorithm:
    """RWM inner kernel suitable for SMC.

    Binds non-array random_step callable via functools.partial so that
    mcmc_parameters stays array-only (required by blackjax SMC layer).
    """
    sigma_arr = jnp.full(dim, sigma)
    step_fn = functools.partial(
        _rw.build_additive_step(), random_step=_rw.normal(sigma_arr)
    )
    return SamplingAlgorithm(init=_rw.init, step=step_fn)


def _make_adaptive_tempered_smc(dim: int = _DIM, num_particles: int = _NUM_PARTICLES):
    """Build adaptive_tempered_smc algorithm with RWM inner kernel."""
    inner = _make_rwm_inner_kernel(dim)
    smc_alg = blackjax.adaptive_tempered_smc(
        logprior_fn=_logprior_fn,
        loglikelihood_fn=_loglikelihood_fn,
        mcmc_step_fn=inner.step,
        mcmc_init_fn=inner.init,
        mcmc_parameters={},
        resampling_fn=_resampling.systematic,
        target_ess=0.5,
        num_mcmc_steps=5,
    )
    return smc_alg


def _make_partial_posteriors_smc(
    data: jax.Array,
    dim: int = _DIM,
    num_particles: int = _NUM_PARTICLES,
):
    """Build partial_posteriors_smc algorithm with RWM inner kernel.

    Parameters
    ----------
    data
        Shape (num_obs, dim) observations array.
    """
    inner = _make_rwm_inner_kernel(dim)

    def partial_logposterior_factory(data_mask):
        def logposterior_fn(x):
            logprior = -0.5 * jnp.sum(x**2)
            per_obs = jnp.sum(
                data_mask[:, None] * (-0.5 * (data - x[None, :]) ** 2), axis=-1
            )
            loglik = jnp.sum(per_obs)
            return logprior + loglik

        return logposterior_fn

    smc_alg = _pp_path.as_top_level_api(
        mcmc_step_fn=inner.step,
        mcmc_init_fn=inner.init,
        mcmc_parameters={},
        resampling_fn=_resampling.systematic,
        num_mcmc_steps=5,
        partial_logposterior_factory=partial_logposterior_factory,
    )
    return smc_alg


# ---------------------------------------------------------------------------
# 1. init_particles_from_prior shape
# ---------------------------------------------------------------------------


def test_init_particles_from_prior_shape():
    """N=200 particles from a 5-D Gaussian prior; assert shape (200, 5), no NaNs."""
    key = jax.random.key(_SEED)
    particles = init_particles_from_prior(
        key,
        prior_sample_fn=lambda k, n: jax.random.normal(k, (n, _DIM)),
        num_particles=_NUM_PARTICLES,
    )
    assert particles.shape == (
        _NUM_PARTICLES,
        _DIM,
    ), f"Expected shape ({_NUM_PARTICLES}, {_DIM}), got {particles.shape}"
    assert jnp.all(jnp.isfinite(particles)), "Particles contain non-finite values"


# ---------------------------------------------------------------------------
# 2. init_particles_from_prior JIT compatibility
# ---------------------------------------------------------------------------


def test_init_particles_jit_compatible():
    """Wrap init_particles_from_prior in jax.jit and assert it compiles + runs."""
    key = jax.random.key(_SEED)

    @jax.jit
    def jit_init(k):
        return init_particles_from_prior(
            k,
            prior_sample_fn=lambda rng, n: jax.random.normal(rng, (n, _DIM)),
            num_particles=_NUM_PARTICLES,
        )

    particles = jit_init(key)
    assert particles.shape == (_NUM_PARTICLES, _DIM)
    assert jnp.all(jnp.isfinite(particles))


# ---------------------------------------------------------------------------
# 3. run_smc adaptive_tempered reaches lambda=1
# ---------------------------------------------------------------------------


def test_run_smc_adaptive_tempered_reaches_lambda_one():
    """Smoke test: 5-D Gaussian prior+likelihood + RWM, num_particles=200.

    Asserts:
    - final_state.tempering_param >= 0.999 (target lambda_target=1.0)
    - particles are finite
    - history['lmbda'] is monotonically non-decreasing
    """
    key = jax.random.key(_SEED)
    init_key, run_key = jax.random.split(key)

    smc_alg = _make_adaptive_tempered_smc()
    initial_particles = jax.random.normal(init_key, (_NUM_PARTICLES, _DIM))
    init_state = smc_alg.init(initial_particles)

    final_state, history = run_smc(
        run_key,
        smc_init_state=init_state,
        smc_step_fn=smc_alg.step,
        max_steps=50,
        lambda_target=1.0,
    )

    # Termination: tempering_param should reach (or nearly reach) 1.0
    assert (
        float(final_state.tempering_param) >= 0.999
    ), f"Expected tempering_param >= 0.999, got {final_state.tempering_param}"

    # Particles should be finite
    assert jnp.all(
        jnp.isfinite(final_state.particles)
    ), "Particles contain non-finite values after adaptive_tempered_smc run"

    # History checks
    lmbda = history["lmbda"]
    assert lmbda.ndim == 1, f"Expected 1-D lmbda history, got shape {lmbda.shape}"
    assert len(lmbda) > 0, "Expected at least 1 SMC step in lmbda history"

    # Monotonically non-decreasing (tempering only moves forward)
    diffs = jnp.diff(lmbda)
    assert jnp.all(
        diffs >= -1e-6
    ), f"lmbda history is not monotonically non-decreasing: {lmbda}"


# ---------------------------------------------------------------------------
# 4. run_smc partial_posteriors steps through data
# ---------------------------------------------------------------------------


def test_run_smc_partial_posteriors_steps_through_data():
    """5-D Gaussian model, 10 IID observations, partial_posteriors_smc.

    We pre-bind the step function with a full all-ones data_mask so that
    every scan step includes all 10 observations.  After max_steps=10,
    data_mask.sum() should equal 10 (all observations included).
    """
    key = jax.random.key(_SEED)
    data_key, init_key, run_key = jax.random.split(key, 3)

    data = jax.random.normal(data_key, (_NUM_OBS, _DIM))
    smc_alg = _make_partial_posteriors_smc(data)

    initial_particles = jax.random.normal(init_key, (_NUM_PARTICLES, _DIM))
    init_state = smc_alg.init(initial_particles, _NUM_OBS)

    # Pre-bind data_mask=all-ones so the runner's smc_step_fn(key, state) works.
    full_mask = jnp.ones(_NUM_OBS, dtype=jnp.float32)
    step_fn_bound = functools.partial(smc_alg.step, data_mask=full_mask)

    final_state, history = run_smc(
        run_key,
        smc_init_state=init_state,
        smc_step_fn=step_fn_bound,
        max_steps=10,
    )

    # All 10 observations should be included in the final data_mask
    mask_sum = float(jnp.sum(final_state.data_mask))
    assert mask_sum == pytest.approx(
        10.0
    ), f"Expected data_mask.sum() == 10, got {mask_sum}"

    # Particles should be finite
    assert jnp.all(
        jnp.isfinite(final_state.particles)
    ), "Particles contain non-finite values after partial_posteriors_smc run"

    # History arrays are empty for non-tempering variant
    assert history["lmbda"].shape == (0,), (
        f"Expected empty lmbda history for partial_posteriors_smc, "
        f"got shape {history['lmbda'].shape}"
    )
