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
"""Tripwires for BlackJAX warmups invoked by generated programs."""

import inspect

import blackjax
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.fast


def test_window_adaptation_contract_used_by_codegen_and_nuts_reference():
    signature = inspect.signature(blackjax.window_adaptation)
    assert {"initial_inverse_mass_matrix", "imm_shrinkage_to_previous"} <= set(
        signature.parameters
    )

    def logdensity_fn(x):
        return -0.5 * jnp.sum(x["x"] ** 2)

    for kernel in (blackjax.hmc, blackjax.nuts, blackjax.barker, blackjax.mala):
        adaptation = blackjax.window_adaptation(kernel, logdensity_fn)
        assert callable(adaptation.run)


def test_pathfinder_adaptation_contract():
    signature = inspect.signature(blackjax.pathfinder_adaptation)
    assert {"num_chains", "n_paths", "imm_estimator", "initial_step_size"} <= set(
        signature.parameters
    )

    from blackjax.adaptation.base import AdaptationResults

    assert {"state", "parameters"} <= set(AdaptationResults._fields)


def test_multipathfinder_contract_used_by_composed_warmup():
    from blackjax.vi.multipathfinder import psis_weights

    def logdensity_fn(x):
        return -0.5 * jnp.sum(x**2)

    state, _ = blackjax.multipathfinder(logdensity_fn).init(
        jax.random.key(0), jnp.zeros((2, 3)), num_samples=3
    )
    assert {"path_states", "samples", "logp", "logq"} <= set(state._fields)
    assert len(psis_weights(state)) == 2


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        ("meads_adaptation", {"logdensity_fn", "num_chains", "num_folds"}),
        ("chees_adaptation", {"logdensity_fn", "num_chains", "target_acceptance_rate"}),
    ],
)
def test_adaptation_factory_parameters(name, parameters):
    signature = inspect.signature(getattr(blackjax, name))
    assert parameters <= set(signature.parameters)


def test_adaptation_runs_return_emitter_parameters():
    key = jax.random.key(0)
    meads = blackjax.meads_adaptation(lambda x: -0.5 * jnp.sum(x**2), 2, num_folds=2)
    results, _ = meads.run(key, jnp.zeros((2, 3)), num_steps=2)
    assert {"step_size", "momentum_inverse_scale", "alpha", "delta"} <= set(
        results.parameters
    )

    import optax

    chees = blackjax.chees_adaptation(lambda x: -0.5 * jnp.sum(x**2), 2)
    results, _ = chees.run(key, jnp.zeros((2, 3)), 0.1, optax.adam(0.01), num_steps=2)
    assert {
        "step_size",
        "inverse_mass_matrix",
        "next_random_arg_fn",
        "integration_steps_fn",
    } <= set(results.parameters)


def test_pathfinder_state_and_lbfgs_helpers_are_emitter_dependencies():
    from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1

    assert callable(lbfgs_inverse_hessian_formula_1)
