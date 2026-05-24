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
"""Smoke tests: gp_regression log_joint_fn + theta_init round-trip via
laplace_dhmc and laplace_dmhmc.

These tests verify that:
  1. ``gp_log_joint_fn`` is finite at the prior mean (phi=0, theta=0).
  2. ``gp_log_joint_fn`` has finite gradients at the same point.
  3. ``laplace_dhmc`` can complete a 5-step scan: no NaN, finite logdensity.
  4. ``laplace_dmhmc`` can complete a 5-step scan: no NaN, finite logdensity.

The tests do NOT check statistical quality — that is the statistician's
domain (Exp 4–5).  They only verify the plumbing: helpers exported from
``tuningfork.model.gp_regression`` round-trip cleanly through the samplers.

Markers
-------
The log_joint_fn / grad tests are ``fast`` (pure JAX evaluation, no chain).
The 5-step scan tests are ``slow`` (JAX compilation of the sampler).
"""

import jax
import jax.numpy as jnp
import pytest

from tuningfork.base_method.laplace_dhmc import ENTRY as DHMC_ENTRY
from tuningfork.base_method.laplace_dmhmc import ENTRY as DMHMC_ENTRY
from tuningfork.model.gp_regression import GP_THETA_INIT, N_OBS, gp_log_joint_fn

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SEED = 7
_N_STEPS = 5

# phi at the prior mean (all log-scales = 0 / -2 per prior)
_PHI_INIT = {
    "log_lengthscale": jnp.array(0.0),
    "log_kernel_scale": jnp.array(0.0),
    "log_noise_scale": jnp.array(-2.0),
}

# Diagonal IMM in phi-space (phi is 3-D: log_ls, log_ks, log_ns)
_PHI_DIM = 3
_INVERSE_MASS_MATRIX = jnp.ones(_PHI_DIM)

# Small step size to keep the Laplace L-BFGS well-conditioned at first init
_STEP_SIZE = 0.01


# ---------------------------------------------------------------------------
# 1. Fast structural checks on gp_log_joint_fn
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_log_joint_finite_at_prior_mean() -> None:
    """gp_log_joint_fn is finite at (f_raw=0, phi=prior mean)."""
    lp = gp_log_joint_fn(GP_THETA_INIT, _PHI_INIT)
    assert jnp.isfinite(lp), f"Expected finite log joint at prior mean, got {lp}"


@pytest.mark.fast
def test_log_joint_grad_finite_at_prior_mean() -> None:
    """Gradient of gp_log_joint_fn w.r.t. f_raw is finite at (f_raw=0, phi=prior mean)."""
    grad_fn = jax.grad(gp_log_joint_fn, argnums=0)
    grad = grad_fn(GP_THETA_INIT, _PHI_INIT)
    assert jnp.all(jnp.isfinite(grad)), "Gradient w.r.t. f_raw has non-finite values"


@pytest.mark.fast
def test_log_joint_grad_phi_finite_at_prior_mean() -> None:
    """Gradient of gp_log_joint_fn w.r.t. phi is finite at (f_raw=0, phi=prior mean)."""
    grad_fn = jax.grad(gp_log_joint_fn, argnums=1)
    grad_phi = grad_fn(GP_THETA_INIT, _PHI_INIT)
    for k, v in grad_phi.items():
        assert jnp.isfinite(v), f"Gradient w.r.t. phi[{k!r}] is not finite: {v}"


@pytest.mark.fast
def test_theta_init_shape() -> None:
    """GP_THETA_INIT has shape (N_OBS,) = (200,)."""
    assert GP_THETA_INIT.shape == (
        N_OBS,
    ), f"Expected GP_THETA_INIT.shape == ({N_OBS},), got {GP_THETA_INIT.shape}"


@pytest.mark.fast
def test_theta_init_zero() -> None:
    """GP_THETA_INIT is zeros (NCP prior mean)."""
    assert jnp.all(GP_THETA_INIT == 0.0), "GP_THETA_INIT should be all zeros"


# ---------------------------------------------------------------------------
# 2. Slow smoke tests: 5-step laplace_dhmc + laplace_dmhmc scans
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_laplace_dhmc_5step_smoke() -> None:
    """laplace_dhmc: 5-step scan on gp_regression completes without NaN."""
    algo = DHMC_ENTRY.factory(
        None,  # logdensity_fn — NOT USED by laplace family
        log_joint_fn=gp_log_joint_fn,
        theta_init=GP_THETA_INIT,
        step_size=_STEP_SIZE,
        inverse_mass_matrix=_INVERSE_MASS_MATRIX,
        maxiter=100,
    )
    state = algo.init(_PHI_INIT, jax.random.key(_SEED))

    assert jnp.isfinite(
        state.logdensity
    ), f"laplace_dhmc: init logdensity not finite: {state.logdensity}"
    assert jnp.all(
        jnp.isfinite(state.position["log_lengthscale"])
    ), "laplace_dhmc: init position has non-finite log_lengthscale"
    assert jnp.all(
        jnp.isfinite(state.theta_star)
    ), "laplace_dhmc: init theta_star has non-finite values"

    def one_step(carry, key):
        new_state, info = algo.step(key, carry)
        return new_state, info

    keys = jax.random.split(jax.random.key(_SEED + 1), _N_STEPS)
    final_state, infos = jax.lax.scan(one_step, state, keys)

    assert jnp.isfinite(
        final_state.logdensity
    ), f"laplace_dhmc: final logdensity not finite: {final_state.logdensity}"
    assert jnp.all(
        jnp.isfinite(final_state.theta_star)
    ), "laplace_dhmc: final theta_star has non-finite values"
    assert jnp.all(
        jnp.isfinite(infos.acceptance_rate)
    ), "laplace_dhmc: some acceptance_rates are not finite"


@pytest.mark.slow
def test_laplace_dmhmc_5step_smoke() -> None:
    """laplace_dmhmc: 5-step scan on gp_regression completes without NaN."""
    algo = DMHMC_ENTRY.factory(
        None,  # logdensity_fn — NOT USED by laplace family
        log_joint_fn=gp_log_joint_fn,
        theta_init=GP_THETA_INIT,
        step_size=_STEP_SIZE,
        inverse_mass_matrix=_INVERSE_MASS_MATRIX,
        maxiter=100,
    )
    state = algo.init(_PHI_INIT, jax.random.key(_SEED))

    assert jnp.isfinite(
        state.logdensity
    ), f"laplace_dmhmc: init logdensity not finite: {state.logdensity}"
    assert jnp.all(
        jnp.isfinite(state.theta_star)
    ), "laplace_dmhmc: init theta_star has non-finite values"

    def one_step(carry, key):
        new_state, info = algo.step(key, carry)
        return new_state, info

    keys = jax.random.split(jax.random.key(_SEED + 1), _N_STEPS)
    final_state, infos = jax.lax.scan(one_step, state, keys)

    assert jnp.isfinite(
        final_state.logdensity
    ), f"laplace_dmhmc: final logdensity not finite: {final_state.logdensity}"
    assert jnp.all(
        jnp.isfinite(final_state.theta_star)
    ), "laplace_dmhmc: final theta_star has non-finite values"
    assert jnp.all(
        jnp.isfinite(infos.acceptance_rate)
    ), "laplace_dmhmc: some acceptance_rates are not finite"
