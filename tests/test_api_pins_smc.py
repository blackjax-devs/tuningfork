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
"""Tripwire tests for BlackJAX SMC API shapes that bjx_bench relies on.

These are defensive: if BlackJAX upstream changes the return-tuple shape or
NamedTuple fields of any SMC algorithm we depend on, these tests fire with a
clear message pointing at the file in bjx-bench that needs an update.

Includes sections: 11 (adaptive_tempered_smc), 12 (partial_posteriors_smc +
inner_kernel_tuning).
"""

import inspect

import pytest

pytestmark = pytest.mark.fast


# ───────────────── 11. SMC family — adaptive_tempered_smc (P5.10) ─────────────────


def test_blackjax_adaptive_tempered_smc_factory_signature():
    """Tripwire: blackjax.smc.adaptive_tempered.as_top_level_api must have exactly
    {logprior_fn, loglikelihood_fn, mcmc_step_fn, mcmc_init_fn, mcmc_parameters,
    resampling_fn, target_ess, root_solver, num_mcmc_steps, extra_parameters}.

    Pinned at P5.10c: bjx_bench/inference/smc/adaptive_tempered.py calls
    blackjax.adaptive_tempered_smc(logprior_fn=..., loglikelihood_fn=...,
    mcmc_step_fn=..., mcmc_init_fn=..., mcmc_parameters=..., resampling_fn=...,
    target_ess=..., num_mcmc_steps=...).  If upstream renames or removes any of
    these parameters, the wrapper's factory calls fail silently.

    Note: extra_parameters is **extra_parameters (VAR_KEYWORD); it appears in
    inspect.signature but is not a named POSITIONAL_OR_KEYWORD param.
    """
    from blackjax.smc.adaptive_tempered import as_top_level_api

    sig = inspect.signature(as_top_level_api)
    expected_named = {
        "logprior_fn",
        "loglikelihood_fn",
        "mcmc_step_fn",
        "mcmc_init_fn",
        "mcmc_parameters",
        "resampling_fn",
        "target_ess",
        "root_solver",
        "num_mcmc_steps",
    }
    missing = expected_named - set(sig.parameters)
    assert not missing, (
        f"blackjax.smc.adaptive_tempered.as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/smc/adaptive_tempered.py if upstream API changed."
    )
    # Pin extra_parameters as VAR_KEYWORD (the **kwargs catch-all)
    assert "extra_parameters" in sig.parameters, (
        "blackjax.smc.adaptive_tempered.as_top_level_api lost the 'extra_parameters' "
        "VAR_KEYWORD parameter. Update bjx_bench/inference/smc/adaptive_tempered.py."
    )

    assert sig.parameters["extra_parameters"].kind == inspect.Parameter.VAR_KEYWORD, (
        "blackjax.smc.adaptive_tempered.as_top_level_api: 'extra_parameters' is no longer "
        "VAR_KEYWORD. Update bjx_bench/inference/smc/adaptive_tempered.py."
    )


def test_blackjax_smc_resampling_systematic_exists():
    """Tripwire: blackjax.smc.resampling.systematic must be callable.

    Pinned at P5.10c: bjx_bench/inference/smc/adaptive_tempered.py uses
    blackjax.smc.resampling.systematic as the default resampling function.
    If upstream renames or removes it, the wrapper silently falls back to
    importing a non-existent name at module load time (ImportError) or
    produces a confusing AttributeError at factory call time.
    """
    from blackjax.smc import resampling as _smc_resampling

    assert callable(_smc_resampling.systematic), (
        "blackjax.smc.resampling.systematic is not callable or does not exist. "
        "Update bjx_bench/inference/smc/adaptive_tempered.py: replace 'systematic' "
        "with the new resampling function name."
    )


# ───────────────── 12. SMC family — partial_posteriors_smc + inner_kernel_tuning (P5.10d) ─────────────────


def test_blackjax_partial_posteriors_path_as_top_level_api_signature():
    """Tripwire: blackjax.smc.partial_posteriors_path.as_top_level_api must have exactly
    {mcmc_step_fn, mcmc_init_fn, mcmc_parameters, resampling_fn, num_mcmc_steps,
    partial_logposterior_factory, update_strategy}.

    Pinned at P5.10d: bjx_bench/inference/smc/partial_posteriors.py calls
    blackjax.smc.partial_posteriors_path.as_top_level_api with keyword args
    mcmc_step_fn, mcmc_init_fn, mcmc_parameters, resampling_fn, num_mcmc_steps,
    partial_logposterior_factory.  If upstream renames or removes any of these,
    the wrapper's factory calls fail.

    Note: do NOT inspect blackjax.partial_posteriors_smc top-level — it collapses
    to *args/**kwargs via GenerateSamplingAPI.  Inspect the inner module directly.
    """
    import blackjax.smc.partial_posteriors_path as _pp_path

    sig = inspect.signature(_pp_path.as_top_level_api)
    expected_named = {
        "mcmc_step_fn",
        "mcmc_init_fn",
        "mcmc_parameters",
        "resampling_fn",
        "num_mcmc_steps",
        "partial_logposterior_factory",
        "update_strategy",
    }
    missing = expected_named - set(sig.parameters)
    assert not missing, (
        f"blackjax.smc.partial_posteriors_path.as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/smc/partial_posteriors.py if upstream API changed."
    )
    assert set(sig.parameters) == expected_named, (
        f"blackjax.smc.partial_posteriors_path.as_top_level_api has unexpected parameters. "
        f"Expected exactly {expected_named}, got {set(sig.parameters)}. "
        f"Update bjx_bench/inference/smc/partial_posteriors.py if upstream API changed."
    )


def test_blackjax_partial_posteriors_smc_state_fields():
    """Tripwire: PartialPosteriorsSMCState._fields must be ('particles', 'weights', 'data_mask').

    Pinned at P5.10d: bjx_bench/inference/smc/partial_posteriors.py documents
    this state shape; tests check state.particles.shape, state.weights.shape,
    state.data_mask.shape.  If upstream renames or adds/removes fields, our
    shape tests break silently (e.g., 'data_mask' renamed to 'mask' would make
    state.data_mask fail with AttributeError).
    """
    from blackjax.smc.partial_posteriors_path import PartialPosteriorsSMCState

    expected = ("particles", "weights", "data_mask")
    assert PartialPosteriorsSMCState._fields == expected, (
        f"BlackJAX PartialPosteriorsSMCState fields changed from {expected} to "
        f"{PartialPosteriorsSMCState._fields}. "
        f"Update bjx_bench/inference/smc/partial_posteriors.py notes and "
        f"tests/test_smc_method_partial_posteriors.py state-shape assertions."
    )


def test_blackjax_inner_kernel_tuning_as_top_level_api_signature():
    """Tripwire: blackjax.smc.inner_kernel_tuning.as_top_level_api must have exactly
    {smc_algorithm, logprior_fn, loglikelihood_fn, mcmc_step_fn, mcmc_init_fn,
    resampling_fn, mcmc_parameter_update_fn, initial_parameter_value,
    num_mcmc_steps, smc_returns_state_with_parameter_override, extra_parameters}.

    Pinned at P5.10d: bjx_bench/inference/smc/inner_kernel_tuning.py calls
    blackjax.smc.inner_kernel_tuning.as_top_level_api with all of these.
    If upstream renames or removes any, the wrapper's factory calls fail.

    Note: 'extra_parameters' is a VAR_KEYWORD (**kwargs catch-all) — pin its kind.
    """
    import blackjax.smc.inner_kernel_tuning as _ikt

    sig = inspect.signature(_ikt.as_top_level_api)
    expected_named = {
        "smc_algorithm",
        "logprior_fn",
        "loglikelihood_fn",
        "mcmc_step_fn",
        "mcmc_init_fn",
        "resampling_fn",
        "mcmc_parameter_update_fn",
        "initial_parameter_value",
        "num_mcmc_steps",
        "smc_returns_state_with_parameter_override",
    }
    missing = expected_named - set(sig.parameters)
    assert not missing, (
        f"blackjax.smc.inner_kernel_tuning.as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/smc/inner_kernel_tuning.py if upstream API changed."
    )
    # Pin extra_parameters as VAR_KEYWORD (the **kwargs catch-all)
    assert "extra_parameters" in sig.parameters, (
        "blackjax.smc.inner_kernel_tuning.as_top_level_api lost the 'extra_parameters' "
        "VAR_KEYWORD parameter. Update bjx_bench/inference/smc/inner_kernel_tuning.py."
    )
    assert sig.parameters["extra_parameters"].kind == inspect.Parameter.VAR_KEYWORD, (
        "blackjax.smc.inner_kernel_tuning.as_top_level_api: 'extra_parameters' is no longer "
        "VAR_KEYWORD. Update bjx_bench/inference/smc/inner_kernel_tuning.py."
    )


def test_blackjax_state_with_parameter_override_fields():
    """Tripwire: StateWithParameterOverride._fields must be ('sampler_state', 'parameter_override').

    Pinned at P5.10d: bjx_bench/inference/smc/inner_kernel_tuning.py documents
    that particles live at state.sampler_state.particles (not state.particles
    directly).  Tests access state.sampler_state.particles and
    state.parameter_override.  If upstream renames these fields, tests break.
    """
    from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride

    expected = ("sampler_state", "parameter_override")
    assert StateWithParameterOverride._fields == expected, (
        f"BlackJAX StateWithParameterOverride fields changed from {expected} to "
        f"{StateWithParameterOverride._fields}. "
        f"Update bjx_bench/inference/smc/inner_kernel_tuning.py notes and "
        f"tests/test_smc_method_inner_kernel_tuning.py state-access patterns."
    )


def test_blackjax_smc_registry_has_three_entries():
    """Tripwire: SMC_METHODS registry must contain all three registered entries
    after P5.10d (adaptive_tempered_smc, partial_posteriors_smc, inner_kernel_tuning).

    Pinned at P5.10d: verifies that the __init__.py registration is complete
    and that all three names are resolvable in the registry dict.
    """
    from bjx_bench.inference.smc import SMC_METHODS

    required = {
        "adaptive_tempered_smc",
        "partial_posteriors_smc",
        "inner_kernel_tuning",
    }
    missing = required - set(SMC_METHODS.keys())
    assert not missing, (
        f"SMC_METHODS registry is missing entries after P5.10d registration: {missing}. "
        f"Update bjx_bench/inference/smc/__init__.py to register all three SMC methods."
    )
