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
"""Phase 2a E2E manual verify: laplace_hmc + window_adaptation_diag_imm + eight_schools_ncp.

Runs the full recipe-style pipeline at n_warmup=500, n_samples=1000 and asserts:
1. Adapted (step_size, IMM) are finite and in a plausible range.
2. IMM shape is (num_chains, dim_phi) = (2, 2) — NOT dim(joint).
3. After sampling with the adapted params: divergence rate < 5%.
4. ESS (lag-1 approximation) is plausible (> 10 for a 1000-step chain).

This is the E2E verification for the laplace warmup phase 2a definition of done.
The test takes ~60 s on CPU.
"""

import blackjax
import blackjax.util as bu
import jax
import jax.numpy as jnp
import pytest
from blackjax.mcmc.laplace_marginal import laplace_marginal_factory

from tuningfork.base_method import BASE_METHODS
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.warmup import WARMUPS

pytestmark = pytest.mark.slow


def test_laplace_hmc_e2e_warmup_and_sampling():
    """E2E verify: laplace_hmc + diag warmup + eight_schools_ncp.

    Full pipeline:
    1. Build laplace marginal logdensity on phi=(mu, tau).
    2. Run window_adaptation_diag_imm at n_warmup=500, num_chains=2.
    3. Assert IMM shape == (2, 2) [dim_phi=2, NOT dim_joint=10].
    4. Build laplace_hmc kernel with chain-0 adapted params.
    5. Run 1000 sampling steps.
    6. Assert divergence_rate < 5% and ESS > 10.
    """
    eight_schools_ncp = MODELS["eight_schools_ncp"]
    key = jax.random.key(42)
    init_position, joint_logdensity_fn, _model_data = build_logdensity_fn(
        key, eight_schools_ncp
    )

    phi_init = {"mu": init_position["mu"], "tau": init_position["tau"]}
    theta_init = {"theta_raw": init_position["theta_raw"]}

    def log_joint_fn(theta, phi):
        return joint_logdensity_fn({"theta_raw": theta["theta_raw"], **phi})

    laplace = laplace_marginal_factory(log_joint_fn, theta_init)

    def marginal_logdensity_fn(phi):
        lp, _theta_star = laplace(phi)
        return lp

    # --- Warmup ---
    warmup = WARMUPS["window_adaptation_diag_imm"]
    laplace_hmc_method = BASE_METHODS["laplace_hmc"]
    num_chains = 2

    states, adapted_params, *_ = warmup.runner(
        jax.random.key(0),
        phi_init,
        500,
        laplace_hmc_method,
        logdensity_fn=marginal_logdensity_fn,
        num_chains=num_chains,
    )

    step_size = adapted_params["step_size"]
    imm = adapted_params["inverse_mass_matrix"]

    # IMM dimensionality check: must be (num_chains, dim_phi=2)
    assert imm.shape == (num_chains, 2), (
        f"IMM shape {imm.shape} != ({num_chains}, 2). "
        "IMM must be phi-dimensional, not joint-dimensional."
    )
    assert jnp.all(jnp.isfinite(step_size)), f"step_size NaN/Inf: {step_size}"
    assert jnp.all(jnp.isfinite(imm)), f"IMM NaN/Inf: {imm}"
    assert jnp.all(step_size > 1e-5), f"step_size suspiciously small: {step_size}"
    assert jnp.all(step_size < 100.0), f"step_size suspiciously large: {step_size}"

    # --- Sampling with adapted params (chain 0) ---
    phi_start = jax.tree.map(lambda x: x[0], states).position

    final_state, history = bu.run_inference_algorithm(
        jax.random.key(1),
        blackjax.laplace_hmc(
            log_joint_fn,
            theta_init,
            float(step_size[0]),
            imm[0],
            num_integration_steps=10,
        ),
        num_steps=1000,
        initial_position=phi_start,
        transform=lambda state, info: (state.position, info),
    )

    positions, infos = history

    # Assert low divergence rate
    div_rate = float(jnp.mean(infos.is_divergent))
    assert div_rate < 0.05, (
        f"Divergence rate {div_rate:.4f} too high (> 5%). "
        "This may indicate the adapted step_size or IMM is poor."
    )

    # Basic ESS via lag-1 autocorrelation of mu
    mu_samples = positions["mu"]
    lag1_ac = float(jnp.corrcoef(mu_samples[:-1], mu_samples[1:])[0, 1])
    ess_approx = len(mu_samples) * (1 - lag1_ac) / (1 + lag1_ac)
    assert ess_approx > 10, (
        f"Approx ESS(mu)={ess_approx:.1f} < 10. "
        "Chain may not be mixing. lag-1 autocorr={lag1_ac:.3f}."
    )
