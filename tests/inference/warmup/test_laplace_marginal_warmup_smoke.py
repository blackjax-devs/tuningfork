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
"""Smoke test for laplace-marginal warmup pathway (Decision 1 preflight).

This test verifies that the laplace-marginal warmup pathway is composable,
confirming Decision 1 (laplace_* variants are IN scope for the sweep).
The pathway: build laplace marginal logdensity → run window_adaptation on
phi-space → adapt IMM on marginal density → pass adapted params to laplace_hmc kernel.

Test procedure per Phase 1 specification:
1. Load eight_schools_ncp model.
2. Build laplace marginal logdensity via laplace_marginal_factory.
3. Run window_adaptation_diag_imm warmup on the marginal at n_warmup=500.
4. Assert no NaN/Inf in (step_size, IMM), shape == (dim(phi),).
5. Verify downstream laplace_hmc.step accepts the adapted params.

Phase 2a resolution (2026-05-18): the xfail that accompanied commit 042f630
has been removed. The adapter ``tuningfork.warmup._laplace_adapter`` routes
laplace_* base methods through ``blackjax.hmc`` for warmup so that
``blackjax.window_adaptation`` receives a proper algorithm object with
``.build_kernel`` and ``.init(position, logdensity_fn)``.
"""

import jax
import jax.numpy as jnp
import pytest
from blackjax.mcmc.laplace_marginal import laplace_marginal_factory

from tuningfork.base_method import BASE_METHODS
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.warmup import WARMUPS

pytestmark = pytest.mark.slow


def test_laplace_hmc_marginal_warmup_smoke():
    """Decision 1 smoke test: laplace-marginal warmup pathway composition.

    Verifies:
    1. Laplace marginal logdensity callable (C1).
    2. Window adaptation runs on phi-space without error (C2).
    3. Adapted (step_size, IMM) are finite and correctly shaped (C3).
    4. Downstream laplace_hmc kernel accepts adapted params (C4).
    """
    # C1: Load eight_schools_ncp and build laplace marginal logdensity
    eight_schools_ncp = MODELS["eight_schools_ncp"]
    key = jax.random.key(42)
    init_position, joint_logdensity_fn, _model_data = build_logdensity_fn(
        key, eight_schools_ncp
    )

    # Extract phi_init and theta_init from model structure.
    # eight_schools_ncp returns {phi, theta} where phi=(mu, tau), theta=(theta_raw,).
    phi_init = {"mu": init_position["mu"], "tau": init_position["tau"]}
    theta_init = {"theta_raw": init_position["theta_raw"]}

    # Build the log_joint callable (theta, phi) -> float used both by laplace
    # marginal construction and by the downstream laplace_hmc kernel.
    def log_joint_fn(theta, phi):
        return joint_logdensity_fn({"theta_raw": theta["theta_raw"], **phi})

    # Build laplace marginal: integrates out theta via L-BFGS.
    laplace = laplace_marginal_factory(log_joint_fn, theta_init)

    # Create marginal logdensity callable: phi → log p̂(phi | y)
    # Note: laplace() returns (lp, theta_star); extract the marginal log-prob.
    def marginal_logdensity_fn(phi):
        lp, _theta_star = laplace(phi)
        return lp

    # C2: Run window_adaptation_diag_imm on the marginal logdensity
    warmup = WARMUPS["window_adaptation_diag_imm"]
    laplace_hmc = BASE_METHODS["laplace_hmc"]

    # Note: for laplace_hmc, warmup runs on the marginal phi-space density.
    # The actual laplace_hmc kernel needs (log_joint_fn, theta_init) at step
    # time, but the warmup only sees the marginal density on phi.
    n_warmup = 500
    num_chains = 2

    states, adapted_params = warmup.runner(
        jax.random.key(0),
        phi_init,
        n_warmup,
        laplace_hmc,
        logdensity_fn=marginal_logdensity_fn,
        num_chains=num_chains,
    )

    # C3: Assert no NaN/Inf in adapted params
    step_size = adapted_params["step_size"]
    imm = adapted_params["inverse_mass_matrix"]

    assert jnp.all(jnp.isfinite(step_size)), f"step_size contains NaN/Inf: {step_size}"
    assert jnp.all(jnp.isfinite(imm)), f"IMM contains NaN/Inf: {imm}"

    # Check shape: phi is {mu, tau}, so dim(phi) = 2. IMM should be (num_chains, 2).
    expected_phi_dim = 2
    assert imm.shape == (
        num_chains,
        expected_phi_dim,
    ), f"IMM shape {imm.shape} != ({num_chains}, {expected_phi_dim})"

    # C4: Verify downstream laplace_hmc kernel accepts the adapted params.
    # Build the laplace_hmc kernel with the adapted (step_size, IMM).
    kernel = laplace_hmc.factory(
        None,  # logdensity_fn arg is NOT USED by laplace_hmc factory
        log_joint_fn=log_joint_fn,
        theta_init=theta_init,
        step_size=step_size[0],  # Use first chain's step_size
        inverse_mass_matrix=imm[0],  # Use first chain's IMM
        num_integration_steps=10,
    )

    # Initialize laplace_hmc state
    laplace_state = kernel.init(phi_init)

    # Run one kernel step to verify no runtime errors
    rng_key = jax.random.key(1)
    new_state, info = kernel.step(rng_key, laplace_state)

    # Basic sanity checks on the state
    assert jnp.all(
        jnp.isfinite(new_state.logdensity)
    ), f"Post-step logdensity is NaN/Inf: {new_state.logdensity}"
    assert jnp.all(
        jnp.isfinite(new_state.logdensity_grad["mu"])
    ), "Post-step gradient contains NaN/Inf"
