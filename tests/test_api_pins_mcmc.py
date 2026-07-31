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
"""Small tripwires for BlackJAX APIs used by generated sampling programs.

These pins deliberately cover emitted sampler construction, warmup helpers,
and the gradient-count evaluator.  They do not duplicate registry or wrapper
tests, nor do they freeze unrelated BlackJAX implementation details.
"""

import inspect

import blackjax
import jax
import jax.numpy as jnp
import pytest
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    ("api", "parameters"),
    [
        (
            "hmc",
            {
                "logdensity_fn",
                "step_size",
                "inverse_mass_matrix",
                "num_integration_steps",
            },
        ),
        ("nuts", {"logdensity_fn", "step_size", "inverse_mass_matrix"}),
        ("dynamic_hmc", {"logdensity_fn", "step_size", "inverse_mass_matrix"}),
        (
            "ghmc",
            {"logdensity_fn", "step_size", "momentum_inverse_scale", "alpha", "delta"},
        ),
        (
            "adjusted_mclmc",
            {
                "logdensity_fn",
                "step_size",
                "integration_steps_params",
                "inverse_mass_matrix",
            },
        ),
        (
            "adjusted_mclmc_dynamic",
            {
                "logdensity_fn",
                "step_size",
                "integration_steps_fn",
                "integration_steps_params",
                "inverse_mass_matrix",
            },
        ),
        (
            "rmhmc",
            {"logdensity_fn", "step_size", "mass_matrix", "num_integration_steps"},
        ),
    ],
)
def test_generated_sampler_factory_parameters(api, parameters):
    """Each listed emitted factory retains its material keyword arguments."""
    module_name = {
        "dynamic_hmc": "dynamic_hmc",
        "adjusted_mclmc_dynamic": "adjusted_mclmc_dynamic",
    }.get(api, api)
    module = __import__(f"blackjax.mcmc.{module_name}", fromlist=["as_top_level_api"])
    target = getattr(module, "as_top_level_api", None)
    if target is None:
        target = getattr(blackjax, api)
    missing = parameters - set(inspect.signature(target).parameters)
    assert not missing, f"blackjax.{api} lost emitted parameters: {sorted(missing)}"


def test_hmc_and_nuts_info_support_gradient_oracle_and_reference_stats():
    from blackjax.mcmc.hmc import HMCInfo
    from blackjax.mcmc.nuts import NUTSInfo

    for info_type in (HMCInfo, NUTSInfo):
        assert "num_integration_steps" in info_type._fields


def test_mclmc_contracts_used_by_generated_warmup_and_sampler():
    assert {"step_size", "L", "inverse_mass_matrix"} <= set(
        MCLMCAdaptationState._fields
    )

    def logdensity_fn(x):
        return -0.5 * jnp.sum(x**2)

    key = jax.random.key(0)
    state = blackjax.mclmc(logdensity_fn, L=1.0, step_size=0.1).init(jnp.zeros(3), key)
    result = blackjax.mclmc_find_L_and_step_size(
        blackjax.mclmc.build_kernel(), 5, state, key, logdensity_fn, True
    )
    assert len(result) == 3


def test_adjusted_mclmc_warmup_contracts():
    def logdensity_fn(x):
        return -0.5 * jnp.sum(x**2)

    state = blackjax.mcmc.adjusted_mclmc.init(jnp.zeros(3), logdensity_fn)
    result = blackjax.adjusted_mclmc_find_L_and_step_size(
        blackjax.mcmc.adjusted_mclmc.build_kernel(),
        logdensity_fn=logdensity_fn,
        num_steps=5,
        state=state,
        rng_key=jax.random.key(0),
        target=0.9,
    )
    assert len(result) == 3
    assert {"L", "step_size", "inverse_mass_matrix"} <= set(result[1]._fields)

    from blackjax.mcmc.adjusted_mclmc_dynamic import make_random_trajectory_length_fn

    assert int(make_random_trajectory_length_fn(True)(jax.random.key(0), 5.0)) >= 0


def test_laplace_factories_retain_state_constructor_parameters():
    for module_name in ("laplace_hmc", "laplace_dynamic_hmc"):
        module = __import__(
            f"blackjax.mcmc.{module_name}", fromlist=["as_top_level_api"]
        )
        params = set(inspect.signature(module.as_top_level_api).parameters)
        assert {
            "log_joint_fn",
            "theta_init",
            "step_size",
            "inverse_mass_matrix",
        } <= params
