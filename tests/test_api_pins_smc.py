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


# ─── 13. Persistent sampling family (P5.11) ──────────────────────────────────


def test_blackjax_persistent_sampling_as_top_level_api_signature():
    """Tripwire: blackjax.smc.persistent_sampling.as_top_level_api must have exactly
    {logprior_fn, loglikelihood_fn, n_schedule, mcmc_step_fn, mcmc_init_fn,
    mcmc_parameters, resampling_fn, num_mcmc_steps, update_strategy}.

    Pinned at P5.11: bjx_bench/inference/smc/persistent_sampling.py calls
    blackjax.smc.persistent_sampling.as_top_level_api with keyword args
    logprior_fn, loglikelihood_fn, n_schedule, mcmc_step_fn, mcmc_init_fn,
    mcmc_parameters, resampling_fn, num_mcmc_steps.  If upstream renames or
    removes any of these, the wrapper's factory calls fail.

    Note: update_strategy has a default (update_and_take_last) but is a named
    POSITIONAL_OR_KEYWORD param — pin its presence.
    """
    from blackjax.smc.persistent_sampling import as_top_level_api

    sig = inspect.signature(as_top_level_api)
    expected_named = {
        "logprior_fn",
        "loglikelihood_fn",
        "n_schedule",
        "mcmc_step_fn",
        "mcmc_init_fn",
        "mcmc_parameters",
        "resampling_fn",
        "num_mcmc_steps",
        "update_strategy",
    }
    missing = expected_named - set(sig.parameters)
    assert not missing, (
        f"blackjax.smc.persistent_sampling.as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/smc/persistent_sampling.py if upstream API changed."
    )
    assert set(sig.parameters) == expected_named, (
        f"blackjax.smc.persistent_sampling.as_top_level_api has unexpected parameters. "
        f"Expected exactly {expected_named}, got {set(sig.parameters)}. "
        f"Update bjx_bench/inference/smc/persistent_sampling.py if upstream API changed."
    )


def test_blackjax_adaptive_persistent_sampling_as_top_level_api_signature():
    """Tripwire: blackjax.smc.adaptive_persistent_sampling.as_top_level_api must have
    exactly {logprior_fn, loglikelihood_fn, max_iterations, mcmc_step_fn, mcmc_init_fn,
    mcmc_parameters, resampling_fn, target_ess, num_mcmc_steps, update_strategy,
    root_solver}.

    Pinned at P5.11: bjx_bench/inference/smc/adaptive_persistent_sampling.py calls
    blackjax.smc.adaptive_persistent_sampling.as_top_level_api with keyword args
    logprior_fn, loglikelihood_fn, max_iterations, mcmc_step_fn, mcmc_init_fn,
    mcmc_parameters, resampling_fn, target_ess, num_mcmc_steps.  If upstream renames
    or removes any of these, the wrapper's factory calls fail.
    """
    from blackjax.smc.adaptive_persistent_sampling import as_top_level_api

    sig = inspect.signature(as_top_level_api)
    expected_named = {
        "logprior_fn",
        "loglikelihood_fn",
        "max_iterations",
        "mcmc_step_fn",
        "mcmc_init_fn",
        "mcmc_parameters",
        "resampling_fn",
        "target_ess",
        "num_mcmc_steps",
        "update_strategy",
        "root_solver",
    }
    missing = expected_named - set(sig.parameters)
    assert not missing, (
        f"blackjax.smc.adaptive_persistent_sampling.as_top_level_api is missing parameters: "
        f"{missing}. Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/smc/adaptive_persistent_sampling.py if upstream API changed."
    )
    assert set(sig.parameters) == expected_named, (
        f"blackjax.smc.adaptive_persistent_sampling.as_top_level_api has unexpected parameters. "
        f"Expected exactly {expected_named}, got {set(sig.parameters)}. "
        f"Update bjx_bench/inference/smc/adaptive_persistent_sampling.py if upstream API changed."
    )


def test_blackjax_adaptive_persistent_sampling_step_arity_is_two_arg():
    """Tripwire — META-004 candidate #8: step_fn of adaptive_persistent_sampling
    is 2-arg (rng_key, state), NOT 3-arg (rng_key, state, lmbda) as the docstring
    incorrectly states.

    Pinned at P5.11: bjx_bench/inference/smc/adaptive_persistent_sampling.py
    wraps the 2-arg step_fn as a standard bjx-bench step (step_kwargs_schema=()).
    If upstream silently changes step_fn to 3-arg, our wrapper would silently
    drop the lmbda arg, producing incorrect (delta=0) tempering.

    We pin BOTH aspects:
    1. The actual parameter count of step_fn is 2 (rng_key, state).
    2. The upstream docstring STILL says '(rng_key, state, lmbda)' — if that
       changes (docstring fix), the second assertion fires, reminding us to
       re-audit whether the actual arity also changed.

    When the docstring is eventually fixed upstream, remove or update the
    docstring mismatch assertion below.
    """
    import functools as _functools

    import blackjax.mcmc.random_walk as _rw
    import jax.numpy as _jnp
    from blackjax.base import SamplingAlgorithm as _SA
    from blackjax.smc import resampling as _resampling
    from blackjax.smc.adaptive_persistent_sampling import as_top_level_api

    # Build a minimal algorithm instance to inspect the step_fn arity
    def _logprior(x):
        return -0.5 * _jnp.sum(x**2)

    def _loglikelihood(x):
        return -0.5 * _jnp.sum((x - 1.0) ** 2)

    sigma_arr = _jnp.full(3, 0.5)
    _step = _functools.partial(
        _rw.build_additive_step(), random_step=_rw.normal(sigma_arr)
    )
    inner = _SA(init=_rw.init, step=_step)

    alg = as_top_level_api(
        logprior_fn=_logprior,
        loglikelihood_fn=_loglikelihood,
        max_iterations=5,
        mcmc_step_fn=inner.step,
        mcmc_init_fn=inner.init,
        mcmc_parameters={},
        resampling_fn=_resampling.systematic,
        target_ess=0.5,
        num_mcmc_steps=2,
    )

    step_sig = inspect.signature(alg.step)
    actual_params = list(step_sig.parameters)
    assert actual_params == ["rng_key", "state"], (
        f"adaptive_persistent_sampling step_fn arity changed from 2-arg to {actual_params}. "
        f"Update bjx_bench/inference/smc/adaptive_persistent_sampling.py: if now 3-arg, "
        f"set step_kwargs_schema=('lmbda',) and update the factory. "
        f"META-004 candidate #8 — check if docstring was also fixed upstream."
    )

    # Pin the docstring mismatch: upstream says '(rng_key, state, lmbda)' but step is 2-arg.
    # When upstream fixes the docstring, this assertion will fire — re-audit at that point.
    docstring = as_top_level_api.__doc__ or ""
    assert "lmbda" in docstring, (
        "blackjax.smc.adaptive_persistent_sampling.as_top_level_api docstring no longer "
        "mentions 'lmbda' in the step signature description. The upstream docstring/arity "
        "mismatch (META-004 #8) may have been fixed. Re-audit: confirm actual step arity "
        "is still 2-arg (rng_key, state) and update bjx_bench wrapper notes accordingly."
    )


def test_blackjax_persistent_smc_state_fields():
    """Tripwire: PersistentSMCState._fields must be
    ('persistent_particles', 'persistent_log_likelihoods', 'persistent_log_Z',
    'tempering_schedule', 'iteration').

    Pinned at P5.11: bjx_bench wrapper docs and tests access state.particles
    (property), state.tempering_param (property), state.iteration, and
    state.persistent_particles (direct field). If upstream renames or adds/removes
    fields, our access patterns break.
    """
    from blackjax.smc.persistent_sampling import PersistentSMCState

    expected = (
        "persistent_particles",
        "persistent_log_likelihoods",
        "persistent_log_Z",
        "tempering_schedule",
        "iteration",
    )
    assert PersistentSMCState._fields == expected, (
        f"BlackJAX PersistentSMCState fields changed from {expected} to "
        f"{PersistentSMCState._fields}. "
        f"Update bjx_bench/inference/smc/persistent_sampling.py notes and "
        f"tests/inference/smc/test_persistent_sampling.py state-access patterns."
    )


def test_blackjax_persistent_state_info_fields():
    """Tripwire: PersistentStateInfo._fields must be ('ancestors', 'update_info').

    Pinned at P5.11: tests access info.ancestors (resampling indices) and
    info.update_info (MCMC kernel info). If upstream renames these fields,
    our access patterns break silently.
    """
    from blackjax.smc.persistent_sampling import PersistentStateInfo

    expected = ("ancestors", "update_info")
    assert PersistentStateInfo._fields == expected, (
        f"BlackJAX PersistentStateInfo fields changed from {expected} to "
        f"{PersistentStateInfo._fields}. "
        f"Update bjx_bench/inference/smc/persistent_sampling.py notes and "
        f"tests/inference/smc/test_persistent_sampling.py info-access patterns."
    )


def test_smc_methods_registry_subset_after_p511():
    """Tripwire (META-011 subset check): SMC_METHODS must contain at minimum the five
    entries registered after P5.11 (adaptive_tempered_smc, partial_posteriors_smc,
    inner_kernel_tuning, persistent_sampling_smc, adaptive_persistent_sampling_smc).

    Uses subset check (not equality) per META-011: future additions to SMC_METHODS
    should not trigger this tripwire — only removals will.
    """
    from bjx_bench.inference.smc import SMC_METHODS

    required = {
        "adaptive_tempered_smc",
        "partial_posteriors_smc",
        "inner_kernel_tuning",
        "persistent_sampling_smc",
        "adaptive_persistent_sampling_smc",
    }
    missing = required - set(SMC_METHODS.keys())
    assert not missing, (
        f"SMC_METHODS registry is missing entries after P5.11 registration: {missing}. "
        f"Update bjx_bench/inference/smc/__init__.py to register all five SMC methods."
    )


# ───────── tempered_smc (P5.15.5) ────────────────────────────────────────────


def test_blackjax_tempered_smc_as_top_level_api_signature():
    """Tripwire: blackjax.smc.tempered.as_top_level_api must accept
    {logprior_fn, loglikelihood_fn, mcmc_step_fn, mcmc_init_fn, mcmc_parameters,
    resampling_fn, num_mcmc_steps, update_strategy, update_particles_fn}.

    Pinned at P5.15.5: bjx_bench/inference/smc/tempered.py calls
    _tempered.as_top_level_api(...) with these parameters.  Note that
    blackjax.tempered_smc (the top-level object) collapses to *args/**kwargs
    via GenerateSamplingAPI wrapping -- this tripwire inspects the inner module
    directly, same pattern as adaptive_tempered_smc tripwire (P5.10c).
    """
    import inspect

    from blackjax.smc.tempered import as_top_level_api

    sig = inspect.signature(as_top_level_api)
    expected = {
        "logprior_fn",
        "loglikelihood_fn",
        "mcmc_step_fn",
        "mcmc_init_fn",
        "mcmc_parameters",
        "resampling_fn",
        "num_mcmc_steps",
        "update_strategy",
        "update_particles_fn",
    }
    missing = expected - set(sig.parameters)
    assert not missing, (
        f"blackjax.smc.tempered.as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/smc/tempered.py if upstream API changed."
    )


def test_blackjax_tempered_smc_state_fields():
    """Tripwire: TemperedSMCState._fields must be ('particles', 'weights', 'tempering_param').

    Pinned at P5.15.5: bjx_bench/inference/smc/tempered.py documents that the
    state field is 'tempering_param' (NOT 'lmbda' as in persistent_sampling).
    If upstream renames this field, our wrapper's notes and tests/inference/smc/
    test_tempered.py state-access patterns break silently.

    Finding (P5.15.5): 'tempering_param' is the correct upstream spelling --
    distinct from 'lmbda' used in blackjax.smc.persistent_sampling.
    """
    from blackjax.smc.tempered import TemperedSMCState

    expected = ("particles", "weights", "tempering_param")
    assert TemperedSMCState._fields == expected, (
        f"BlackJAX TemperedSMCState fields changed from {expected} to "
        f"{TemperedSMCState._fields}. "
        f"Update bjx_bench/inference/smc/tempered.py notes and "
        f"tests/inference/smc/test_tempered.py state-access patterns. "
        f"NOTE: field name is 'tempering_param' (not 'lmbda')."
    )


def test_smc_methods_registry_subset_after_p5155():
    """Tripwire (META-011 subset check): SMC_METHODS must contain 'tempered_smc'
    after P5.15.5 registration.

    Subset check: future additions to SMC_METHODS do not break this test;
    only removal of 'tempered_smc' triggers it.
    """
    from bjx_bench.inference.smc import SMC_METHODS

    assert "tempered_smc" in SMC_METHODS, (
        f"SMC_METHODS is missing 'tempered_smc' after P5.15.5 registration. "
        f"Registered keys: {sorted(SMC_METHODS.keys())}. "
        f"Check bjx_bench/inference/smc/__init__.py imports."
    )
