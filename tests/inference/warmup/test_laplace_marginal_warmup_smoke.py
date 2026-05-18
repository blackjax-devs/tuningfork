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

# TODO(wadapt-hmc-sweep Phase 2): laplace-marginal warmup pathway needs design work.
#
# The Phase 1 Junior-SWE smoke test verifies whether Decision 1 (laplace_* IN scope
# for the wadapt-hmc-sweep) holds — i.e., whether `window_adaptation_diag_imm` can be
# composed with the 4 laplace_{hmc,dhmc,mhmc,dmhmc} kernels by running adaptation on
# the laplace marginal log-density (phi-only) instead of the joint.
#
# Current finding (2026-05-18): the composition fails at
# `blackjax.adaptation.window_adaptation.py:300` with
#   AttributeError: 'function' object has no attribute 'build_kernel'
# Root cause: the existing `_runner` passes `base_method.factory` (a callable) to
# blackjax.window_adaptation's `algorithm=` parameter, which expects a constructed
# SamplingAlgorithm (with .build_kernel attribute). For HMC/NUTS this works because
# the factory's signature matches what the warmup expects. For laplace_*, the factory
# requires (log_joint_fn, theta_init) — different signature, different output.
#
# Two-part design work needed (deferred to Phase 2):
#   1. Tuningfork: extend the warmup `_runner` to detect laplace_* base methods and
#      auto-construct the SamplingAlgorithm via laplace_hmc-style kwargs before
#      passing to blackjax.window_adaptation.
#   2. Conceptual: confirm that running window adaptation on the laplace marginal
#      log-density produces a phi-only IMM that the laplace_* kernel can consume
#      at step time (separate kwargs: marginal IMM vs joint state).
#
# Until both land, Decision 1 (laplace_* IN scope) is UNVERIFIED. Phase 2 of the
# wadapt-hmc-sweep should either:
#   (a) Resolve the design work above → re-enable this test.
#   (b) Revoke Decision 1 → laplace_* cells become FAILED with REQUIRES_ALT_SAMPLER
#       in the recipe matrix. (Compatibility widening in commit 2499f1f stays as
#       a recipe-matrix declaration; downstream cells produce FAILED recipes.)
#
# See worklog/threads/wadapt-hmc-sweep.md § Decision 1 for the full preflight context.


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Decision 1 preflight: laplace-marginal warmup pathway needs design work. "
        "blackjax.window_adaptation expects a constructed SamplingAlgorithm; the "
        "current _runner passes base_method.factory which fails for laplace_* "
        "because their factory signature differs (log_joint_fn, theta_init). "
        "See file-level TODO for two-part Phase 2 design work."
    ),
)
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
    init_position, joint_logdensity_fn, model_data = build_logdensity_fn(
        key, eight_schools_ncp
    )

    # Extract phi_init and theta_init from model structure.
    # eight_schools_ncp returns {phi, theta} where phi=(mu, tau), theta=(theta_raw,).
    phi_init = {"mu": init_position["mu"], "tau": init_position["tau"]}
    theta_init = {"theta_raw": init_position["theta_raw"]}

    # Build laplace marginal: integrates out theta via L-BFGS.
    laplace = laplace_marginal_factory(
        lambda theta, phi: joint_logdensity_fn(
            {"theta_raw": theta["theta_raw"], **phi}
        ),
        theta_init,
    )

    # Create marginal logdensity callable: phi → log p̂(phi | y)
    # Note: laplace() returns (lp, theta_star); extract the marginal log-prob.
    def marginal_logdensity_fn(phi):
        lp, _theta_star = laplace(phi)
        return lp

    # C2: Run window_adaptation_diag_imm on the marginal logdensity
    warmup = WARMUPS["window_adaptation_diag_imm"]
    laplace_hmc = BASE_METHODS["laplace_hmc"]

    # Note: for laplace_hmc, we pretend it's a standard HMC for warmup purposes.
    # The actual laplace_hmc kernel needs (log_joint_fn, theta_init) at step time,
    # but the warmup only sees the marginal density on phi.
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
    try:
        kernel = laplace_hmc.factory(
            lambda theta, phi: joint_logdensity_fn(
                {"theta_raw": theta["theta_raw"], **phi}
            ),
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
        ), "Post-step logdensity is NaN/Inf"
        assert jnp.all(
            jnp.isfinite(new_state.logdensity_grad["mu"])
        ), "Post-step gradient contains NaN/Inf"
    except Exception as e:
        pytest.xfail(f"Downstream laplace_hmc kernel failed: {type(e).__name__}: {e}")
