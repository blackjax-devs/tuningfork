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
"""Fail-fast pins for the BlackJAX APIs emitted by tuningfork's SMC programs."""

import inspect

import pytest

pytestmark = pytest.mark.fast


def _assert_parameters(
    callable_obj, required: set[str], *, var_keyword: str | None = None
) -> None:
    parameters = inspect.signature(callable_obj).parameters
    missing = required - set(parameters)
    assert not missing, (
        f"{callable_obj!r} is missing generated-call parameters {sorted(missing)}; "
        f"current parameters: {list(parameters)}"
    )
    if var_keyword is not None:
        assert (
            parameters[var_keyword].kind is inspect.Parameter.VAR_KEYWORD
        ), f"{callable_obj!r} must retain **{var_keyword} for generated SMC options"


def test_adaptive_tempered_constructor_parameters() -> None:
    """The adaptive route emits these keyword arguments to BlackJAX."""
    from blackjax.smc.adaptive_tempered import as_top_level_api

    _assert_parameters(
        as_top_level_api,
        {
            "logprior_fn",
            "loglikelihood_fn",
            "mcmc_step_fn",
            "mcmc_init_fn",
            "mcmc_parameters",
            "resampling_fn",
            "target_ess",
            "num_mcmc_steps",
        },
        var_keyword="extra_parameters",
    )


def test_inner_kernel_tuning_constructor_parameters() -> None:
    """The HMC-tuning route emits these keyword arguments to BlackJAX."""
    from blackjax.smc.inner_kernel_tuning import as_top_level_api

    _assert_parameters(
        as_top_level_api,
        {
            "smc_algorithm",
            "logprior_fn",
            "loglikelihood_fn",
            "mcmc_step_fn",
            "mcmc_init_fn",
            "resampling_fn",
            "mcmc_parameter_update_fn",
            "initial_parameter_value",
            "num_mcmc_steps",
        },
        var_keyword="extra_parameters",
    )


def test_generated_smc_state_fields() -> None:
    """Generated programs read the tempered state fields directly."""
    from blackjax.smc.tempered import TemperedSMCState

    assert TemperedSMCState._fields == ("particles", "weights", "tempering_param")


def test_generated_smc_tuning_state_and_info_fields() -> None:
    """The tuning route unwraps its state and reads the update info field."""
    from blackjax.smc.base import SMCInfo
    from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride

    assert StateWithParameterOverride._fields == (
        "sampler_state",
        "parameter_override",
    )
    assert SMCInfo._fields == (
        "ancestors",
        "log_likelihood_increment",
        "update_info",
    )


def test_generated_smc_step_signatures() -> None:
    """Both emitted algorithms retain the standard ``step(key, state)`` shape."""
    from blackjax.smc.adaptive_tempered import as_top_level_api as adaptive
    from blackjax.smc.inner_kernel_tuning import as_top_level_api as tuning

    def _logdensity(_position):
        return 0.0

    def _inner_step(_key, _state, **_kwargs):
        return _state, None

    def _inner_init(_position):
        return None

    adaptive_algorithm = adaptive(
        logprior_fn=_logdensity,
        loglikelihood_fn=_logdensity,
        mcmc_step_fn=_inner_step,
        mcmc_init_fn=_inner_init,
        mcmc_parameters={},
        resampling_fn=lambda *_args: None,
        target_ess=0.5,
        num_mcmc_steps=1,
    )
    tuning_algorithm = tuning(
        smc_algorithm=lambda **_kwargs: adaptive_algorithm,
        logprior_fn=_logdensity,
        loglikelihood_fn=_logdensity,
        mcmc_step_fn=_inner_step,
        mcmc_init_fn=_inner_init,
        resampling_fn=lambda *_args: None,
        mcmc_parameter_update_fn=lambda *_args: {},
        initial_parameter_value={},
        num_mcmc_steps=1,
    )

    assert list(inspect.signature(adaptive_algorithm.step).parameters) == [
        "rng_key",
        "state",
    ]
    tuning_step_parameters = inspect.signature(tuning_algorithm.step).parameters
    assert list(tuning_step_parameters)[:2] == ["rng_key", "state"]
    assert (
        tuning_step_parameters["extra_step_parameters"].kind
        is inspect.Parameter.VAR_KEYWORD
    )

    for algorithm in (adaptive_algorithm, tuning_algorithm):
        assert list(inspect.signature(algorithm.init).parameters) == [
            "position",
            "rng_key",
        ]


def test_generated_smc_resampling_and_update_apis() -> None:
    """Pin helper APIs imported literally by generated SMC source."""
    from blackjax.smc import resampling
    from blackjax.smc.tuning.from_kernel_info import update_scale_from_acceptance_rate
    from blackjax.smc.tuning.from_particles import particles_as_rows

    assert callable(resampling.systematic)
    _assert_parameters(
        update_scale_from_acceptance_rate,
        {"scales", "acceptance_rates", "target_acceptance_rate"},
    )
    assert callable(particles_as_rows)
