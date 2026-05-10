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
"""Tripwire tests for BlackJAX MCMC API shapes that bjx_bench relies on.

These are defensive: if BlackJAX upstream changes the return-tuple shape or
NamedTuple fields of any MCMC kernel/adaptation we depend on, these tests fire
with a clear message pointing at the file in bjx-bench that needs an update.

Includes sections: 1 (MCLMC), 2 (MCLMCInfo), 3 (MCLMCAdaptationState),
4 (HMCInfo/NUTSInfo), 6 (GHMC), 7 (dynamic_hmc), 8 (adjusted_mclmc),
9 (elliptical_slice + mgrad_gaussian), 10 (irmh),
11 (standard-momentum 2x2: mhmc + dmhmc),
12 (Laplace-marginal 2x2: laplace_hmc + laplace_dhmc + laplace_mhmc + laplace_dmhmc).
"""

import blackjax
import jax
import jax.numpy as jnp
import pytest
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from blackjax.mcmc.mclmc import MCLMCInfo

pytestmark = pytest.mark.fast


# ───────────────── 1. mclmc_find_L_and_step_size return shape ─────────────────
def test_mclmc_find_L_and_step_size_returns_3_tuple():
    """Pinned in T2.6c: bjx_bench/calibration/tier_b.py:_run_warmup unpacks
    (state, params, _n_tuning_steps). If BlackJAX changes this to a 2-tuple
    (matching its docstring) or a different shape, the unpack fails opaquely.

    Caught at T2.6c commit 55792ac. Documented in WORKLOG META-004 watch.
    """

    def logdensity_fn(x):
        return -0.5 * jnp.sum(x["x"] ** 2)

    key = jax.random.key(0)
    # Build the kernel and state following tier_b.py:_run_warmup line 417-419
    default_kwargs = {"L": 1.0, "step_size": 0.1}
    kernel = blackjax.mclmc(logdensity_fn, **default_kwargs)
    init_state = kernel.init({"x": jnp.zeros(5)}, key)
    # Call mclmc_find_L_and_step_size with raw build_kernel (not the wrapper)
    mclmc_kernel = blackjax.mclmc.build_kernel()
    result = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel,
        num_steps=20,
        state=init_state,
        rng_key=key,
        logdensity_fn=logdensity_fn,
        diagonal_preconditioning=True,
    )
    assert len(result) == 3, (
        f"BlackJAX changed mclmc_find_L_and_step_size return arity from 3 to {len(result)}. "
        f"Update bjx_bench/calibration/tier_b.py:_run_warmup unpack accordingly."
    )


# ───────────────── 2. MCLMCInfo NamedTuple fields ─────────────────
def test_mclmc_info_fields():
    """Pinned in T2.4: bjx_bench/algorithms/mclmc.py grad_count_per_step is a
    constant 2 (NOT 2 × info.num_integration_steps), because MCLMCInfo lacks
    that field. If BlackJAX adds num_integration_steps to MCLMCInfo, our
    constant-2 formula becomes incorrect for any integrator beyond default.
    """
    expected = ("logdensity", "kinetic_change", "energy_change", "nonans")
    assert MCLMCInfo._fields == expected, (
        f"BlackJAX MCLMCInfo fields changed from {expected} to {MCLMCInfo._fields}. "
        f"Update bjx_bench/algorithms/mclmc.py:grad_count_per_step if num_integration_steps appeared."
    )


# ───────────────── 3. MCLMCAdaptationState fields (params dict) ─────────────────
def test_mclmc_adaptation_state_fields():
    """Pinned in T2.6c: bjx_bench/calibration/tier_b.py treats the warmup
    output as a dict-like with at least step_size and L. If BlackJAX renames
    or removes either, our trial-params merge silently misbehaves.
    """
    fields = MCLMCAdaptationState._fields
    for name in ("step_size", "L", "inverse_mass_matrix"):
        assert name in fields, (
            f"MCLMCAdaptationState lost field {name!r}. Current fields: {fields}. "
            f"Update bjx_bench/calibration/tier_b.py:_run_warmup."
        )


# ───────────────── 4. HMCInfo / NUTSInfo expose num_integration_steps ─────────────────
def test_hmc_nuts_info_have_num_integration_steps():
    """Pinned in T2.1+T2.2: grad_count_per_step for HMC and NUTS reads
    info.num_integration_steps. If that field disappears, our gradient-count
    accounting breaks silently (defaulting to 0?).
    """
    from blackjax.mcmc.hmc import HMCInfo
    from blackjax.mcmc.nuts import NUTSInfo

    assert "num_integration_steps" in HMCInfo._fields, (
        "blackjax.mcmc.hmc.HMCInfo lost num_integration_steps. "
        "Update bjx_bench/algorithms/{hmc,nuts}.py:grad_count_per_step."
    )
    assert "num_integration_steps" in NUTSInfo._fields, (
        "blackjax.mcmc.nuts.NUTSInfo lost num_integration_steps. "
        "Update bjx_bench/algorithms/{hmc,nuts}.py:grad_count_per_step."
    )


# ───────────────── 6. GHMC + MEADS (P5.5) ─────────────────


def test_blackjax_ghmc_factory_signature():
    """Tripwire: blackjax.ghmc must accept (logdensity_fn, step_size,
    momentum_inverse_scale, alpha, delta) as positional/keyword args.

    Pinned at P5.5: bjx_bench/inference/base_method/ghmc.py calls
    blackjax.ghmc(logdensity_fn, **trial_params) where trial_params includes
    step_size, momentum_inverse_scale, alpha, delta.  If upstream renames or
    removes any of these, factory calls in the BO loop fail silently.
    """
    import inspect

    from blackjax.mcmc.ghmc import as_top_level_api

    inner = inspect.signature(as_top_level_api)
    expected = {
        "logdensity_fn",
        "step_size",
        "momentum_inverse_scale",
        "alpha",
        "delta",
    }
    missing = expected - set(inner.parameters)
    assert not missing, (
        f"blackjax.mcmc.ghmc.as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(inner.parameters)}. "
        f"Update bjx_bench/inference/base_method/ghmc.py if upstream API changed."
    )


# ───────────────── 7. dynamic_hmc + CHEES (P5.6) ─────────────────


def test_blackjax_dynamic_hmc_factory_signature():
    """Tripwire: blackjax.dynamic_hmc inner API must accept (logdensity_fn,
    step_size, inverse_mass_matrix); also confirm dhmc alias.

    Pinned at P5.6: bjx_bench/inference/base_method/dynamic_hmc.py calls
    blackjax.dynamic_hmc(logdensity_fn, **trial_params) where trial_params
    includes step_size and inverse_mass_matrix (from CHEES warmup).  If
    upstream renames or removes any of these, factory calls in the BO loop
    fail silently.  Also confirms blackjax.dhmc is blackjax.dynamic_hmc.
    """
    import inspect

    from blackjax.mcmc.dynamic_hmc import as_top_level_api

    sig = inspect.signature(as_top_level_api)
    expected = {"logdensity_fn", "step_size", "inverse_mass_matrix"}
    missing = expected - set(sig.parameters)
    assert not missing, (
        f"blackjax.mcmc.dynamic_hmc.as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/base_method/dynamic_hmc.py if upstream API changed."
    )
    assert blackjax.dhmc is blackjax.dynamic_hmc, (
        "blackjax.dhmc alias broken: dhmc is not dynamic_hmc. "
        "Update bjx_bench/inference/base_method/dynamic_hmc.py alias note."
    )


def test_blackjax_chees_adaptation_signature():
    """Tripwire: blackjax.chees_adaptation must accept (logdensity_fn,
    num_chains, target_acceptance_rate) as parameters.

    Pinned at P5.6: bjx_bench/inference/warmup/chees.py calls
    blackjax.chees_adaptation(logdensity_fn, num_chains,
    target_acceptance_rate=..., max_leapfrog_steps=...).
    If upstream renames or removes any of these, the CHEES warmup fails.
    """
    import inspect

    sig = inspect.signature(blackjax.chees_adaptation)
    expected = {"logdensity_fn", "num_chains", "target_acceptance_rate"}
    missing = expected - set(sig.parameters)
    assert not missing, (
        f"blackjax.chees_adaptation is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/warmup/chees.py if upstream API changed."
    )


def test_blackjax_chees_adaptation_run_returns_2tuple():
    """Tripwire: chees_adaptation.run() must return a 2-tuple
    (AdaptationResults, AdaptationInfo).

    Pinned at P5.6: bjx_bench/inference/warmup/chees.py unpacks the result as
    (adaptation_results, _adaptation_info) = chees.run(...).
    If upstream changes to a 3-tuple or NamedTuple, the unpack fails.

    Note: unlike meads_adaptation.run(), chees_adaptation.run() requires
    step_size and optim as positional arguments.
    """
    import optax

    chees = blackjax.chees_adaptation(
        lambda x: -0.5 * jnp.sum(x**2),
        num_chains=4,
    )
    key = jax.random.key(0)
    optim = optax.adam(learning_rate=0.01)
    result = chees.run(key, jnp.zeros((4, 3)), 0.1, optim, num_steps=5)
    assert len(result) == 2, (
        f"chees_adaptation.run() should return a 2-tuple (AdaptationResults, AdaptationInfo), "
        f"got {len(result)}-tuple. Update bjx_bench/inference/warmup/chees.py."
    )
    adaptation_results, _adaptation_info = result
    assert hasattr(adaptation_results, "state"), (
        "AdaptationResults must have 'state' field; "
        "update bjx_bench/inference/warmup/chees.py."
    )
    assert hasattr(adaptation_results, "parameters"), (
        "AdaptationResults must have 'parameters' field; "
        "update bjx_bench/inference/warmup/chees.py."
    )
    for param_key in ("step_size", "inverse_mass_matrix"):
        assert param_key in adaptation_results.parameters, (
            f"chees_adaptation AdaptationResults.parameters lost key {param_key!r}. "
            f"Update bjx_bench/inference/warmup/chees.py."
        )
    # Callable params must also be present
    for callable_key in ("next_random_arg_fn", "integration_steps_fn"):
        assert callable_key in adaptation_results.parameters, (
            f"chees_adaptation AdaptationResults.parameters lost callable key {callable_key!r}. "
            f"Update bjx_bench/inference/warmup/chees.py."
        )


# ───────────────── 8. adjusted_mclmc + adjusted_mclmc_dynamic + adapter (P5.7) ─────────────────


def test_blackjax_adjusted_mclmc_factory_signature():
    """Tripwire: blackjax.mcmc.adjusted_mclmc.as_top_level_api must accept
    {logdensity_fn, step_size, integration_steps_params, inverse_mass_matrix}.

    Pinned at P5.7: bjx_bench/inference/base_method/adjusted_mclmc.py calls
    blackjax.adjusted_mclmc(logdensity_fn, step_size=...,
    integration_steps_params=(...,), inverse_mass_matrix=...).
    If upstream renames or removes any of these, factory calls in the BO loop
    fail silently.
    """
    import inspect

    from blackjax.mcmc.adjusted_mclmc import as_top_level_api

    sig = inspect.signature(as_top_level_api)
    expected = {
        "logdensity_fn",
        "step_size",
        "integration_steps_params",
        "inverse_mass_matrix",
    }
    missing = expected - set(sig.parameters)
    assert not missing, (
        f"blackjax.mcmc.adjusted_mclmc.as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/base_method/adjusted_mclmc.py if upstream API changed."
    )


def test_blackjax_adjusted_mclmc_dynamic_factory_signature():
    """Tripwire: blackjax.mcmc.adjusted_mclmc_dynamic.as_top_level_api must accept
    {logdensity_fn, step_size, integration_steps_fn, integration_steps_params,
    inverse_mass_matrix}.

    Pinned at P5.7: bjx_bench/inference/base_method/adjusted_mclmc_dynamic.py calls
    blackjax.adjusted_mclmc_dynamic(logdensity_fn, step_size=...,
    integration_steps_fn=..., integration_steps_params=(...,),
    inverse_mass_matrix=...).
    If upstream renames or removes any of these, factory calls fail silently.
    """
    import inspect

    from blackjax.mcmc.adjusted_mclmc_dynamic import as_top_level_api

    sig = inspect.signature(as_top_level_api)
    expected = {
        "logdensity_fn",
        "step_size",
        "integration_steps_fn",
        "integration_steps_params",
        "inverse_mass_matrix",
    }
    missing = expected - set(sig.parameters)
    assert not missing, (
        f"blackjax.mcmc.adjusted_mclmc_dynamic.as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/base_method/adjusted_mclmc_dynamic.py if upstream API changed."
    )


def test_blackjax_adjusted_mclmc_find_L_and_step_size_returns_3_tuple():
    """Tripwire: blackjax.adjusted_mclmc_find_L_and_step_size must return a 3-tuple
    (state, MCLMCAdaptationState, total_num_tuning_integrator_steps).

    Pinned at P5.7: bjx_bench/inference/warmup/adjusted_mclmc_tuning.py unpacks
    (s, adaptation_state, total_steps). If upstream changes to a 2-tuple (matching
    the vanilla mclmc docstring drift META-004), the unpack fails opaquely.

    META-004 instance #6: adjusted_mclmc_find_L_and_step_size docstring says
    'tuple containing the final state and final hyperparameters' but actually
    returns a 3-tuple including total_num_tuning_integrator_steps. Confirmed here.
    """

    def logdensity_fn(x):
        return -0.5 * jnp.sum(x**2)

    key = jax.random.key(0)
    init_state = blackjax.mcmc.adjusted_mclmc.init(jnp.zeros(10), logdensity_fn)
    mclmc_kernel = blackjax.mcmc.adjusted_mclmc.build_kernel()
    result = blackjax.adjusted_mclmc_find_L_and_step_size(
        mclmc_kernel,
        logdensity_fn=logdensity_fn,
        num_steps=100,
        state=init_state,
        rng_key=key,
        target=0.9,
    )
    assert len(result) == 3, (
        f"BlackJAX changed adjusted_mclmc_find_L_and_step_size return arity "
        f"from 3 to {len(result)}. "
        f"Update bjx_bench/inference/warmup/adjusted_mclmc_tuning.py unpack accordingly."
    )
    assert result[1]._fields == ("L", "step_size", "inverse_mass_matrix"), (
        f"MCLMCAdaptationState._fields changed from "
        f"('L', 'step_size', 'inverse_mass_matrix') to {result[1]._fields}. "
        f"Update bjx_bench/inference/warmup/adjusted_mclmc_tuning.py adapted_params dict."
    )


def test_blackjax_make_random_trajectory_length_fn_signature():
    """Tripwire: blackjax.mcmc.adjusted_mclmc_dynamic.make_random_trajectory_length_fn
    must accept 'random_trajectory_length' as parameter and return a callable
    (rng_arg, avg) -> int-castable scalar in a reasonable range.

    Pinned at P5.7: bjx_bench/inference/base_method/adjusted_mclmc_dynamic.py calls
    make_random_trajectory_length_fn(True) to get the integration_steps_fn.
    If upstream renames the parameter or changes the returned function's signature,
    the dynamic factory breaks.
    """
    import inspect

    from blackjax.mcmc.adjusted_mclmc_dynamic import make_random_trajectory_length_fn

    sig = inspect.signature(make_random_trajectory_length_fn)
    assert "random_trajectory_length" in sig.parameters, (
        f"make_random_trajectory_length_fn is missing 'random_trajectory_length' parameter. "
        f"Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/base_method/adjusted_mclmc_dynamic.py."
    )

    # Calling make_random_trajectory_length_fn(True) should return a callable
    # that takes (rng_arg, avg) and returns an int-castable scalar.
    steps_fn = make_random_trajectory_length_fn(True)
    assert callable(
        steps_fn
    ), "make_random_trajectory_length_fn(True) must return a callable."
    result = steps_fn(jax.random.key(0), 5.0)
    result_int = int(result)
    assert 0 <= result_int <= 100, (
        f"make_random_trajectory_length_fn(True)(key, 5.0) returned {result_int}, "
        f"expected an int-castable scalar in [0, ~10]. "
        f"Update bjx_bench/inference/base_method/adjusted_mclmc_dynamic.py."
    )


# ───────────────── 9. elliptical_slice + mgrad_gaussian (P5.8) ─────────────────


def test_blackjax_elliptical_slice_factory_signature():
    """Tripwire: blackjax.mcmc.elliptical_slice.as_top_level_api must accept
    {loglikelihood_fn, mean, cov}.

    Pinned at P5.8: bjx_bench/inference/base_method/elliptical_slice.py wraps
    blackjax.elliptical_slice(logdensity_fn, mean=prior_mean, cov=prior_cov)
    where the upstream positional arg is named 'loglikelihood_fn' (not 'logdensity_fn').
    If upstream renames this param, our wrapper silently diverges in naming convention
    and callers become confused about what the first arg should contain.
    """
    import inspect

    from blackjax.mcmc.elliptical_slice import as_top_level_api

    sig = inspect.signature(as_top_level_api)
    expected = {"loglikelihood_fn", "mean", "cov"}
    missing = expected - set(sig.parameters)
    assert not missing, (
        f"blackjax.mcmc.elliptical_slice.as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/base_method/elliptical_slice.py if upstream API changed. "
        f"CRITICAL: if 'loglikelihood_fn' was renamed, update the docstring warning in _factory "
        f"and the ENTRY notes — callers must supply a likelihood-ONLY function."
    )


def test_blackjax_mgrad_gaussian_factory_signature():
    """Tripwire: blackjax.mcmc.marginal_latent_gaussian.as_top_level_api must accept
    {logdensity_fn, covariance, mean, cov_svd, step_size}.

    Pinned at P5.8: bjx_bench/inference/base_method/mgrad_gaussian.py calls
    blackjax.mgrad_gaussian(logdensity_fn, covariance=prior_cov, mean=prior_mean,
    step_size=step_size).  If upstream renames or removes any of these, factory
    calls in the BO loop fail silently.
    """
    import inspect

    from blackjax.mcmc.marginal_latent_gaussian import as_top_level_api

    sig = inspect.signature(as_top_level_api)
    expected = {"logdensity_fn", "covariance", "mean", "cov_svd", "step_size"}
    missing = expected - set(sig.parameters)
    assert not missing, (
        f"blackjax.mcmc.marginal_latent_gaussian.as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/base_method/mgrad_gaussian.py if upstream API changed."
    )


def test_blackjax_ellip_slice_info_fields_and_marginal_info_fields():
    """Tripwire: pin EllipSliceInfo._fields and MarginalInfo._fields.

    Pinned at P5.8:
    - EllipSliceInfo._fields == ('momentum', 'theta', 'subiter'). The absence of
      'acceptance_rate' is intentional (slice sampler always accepts); if upstream
      adds it, our grad_count_per_step=0 and target_acceptance_rate=None decisions
      need revisiting.
    - MarginalInfo._fields == ('acceptance_rate', 'is_accepted', 'proposal'). Used
      in mgrad_gaussian's grad_count_per_step test via synthetic MarginalInfo.
      MarginalState._fields == ('position', 'logdensity', 'logdensity_grad', 'U_x',
      'U_grad_x') — the internal SVD representation; if it changes, tests that
      construct synthetic proposals break.
    """
    from blackjax.mcmc.elliptical_slice import EllipSliceInfo
    from blackjax.mcmc.marginal_latent_gaussian import MarginalInfo, MarginalState

    expected_ellip = ("momentum", "theta", "subiter")
    assert EllipSliceInfo._fields == expected_ellip, (
        f"BlackJAX EllipSliceInfo fields changed from {expected_ellip} to {EllipSliceInfo._fields}. "
        f"Update bjx_bench/inference/base_method/elliptical_slice.py notes and "
        f"tests/test_base_method_elliptical_slice.py accordingly. "
        f"If 'acceptance_rate' was ADDED, revisit target_acceptance_rate=None decision."
    )

    expected_marginal = ("acceptance_rate", "is_accepted", "proposal")
    assert MarginalInfo._fields == expected_marginal, (
        f"BlackJAX MarginalInfo fields changed from {expected_marginal} to {MarginalInfo._fields}. "
        f"Update bjx_bench/inference/base_method/mgrad_gaussian.py notes and "
        f"tests/test_base_method_mgrad_gaussian.py accordingly."
    )

    expected_marginal_state = (
        "position",
        "logdensity",
        "logdensity_grad",
        "U_x",
        "U_grad_x",
    )
    assert MarginalState._fields == expected_marginal_state, (
        f"BlackJAX MarginalState fields changed from {expected_marginal_state} to "
        f"{MarginalState._fields}. "
        f"Update tests/test_base_method_mgrad_gaussian.py synthetic MarginalState construction."
    )


# ───────────────── 10. irmh standalone (P5.9) ─────────────────


def test_blackjax_irmh_factory_signature():
    """Tripwire: blackjax.mcmc.random_walk.irmh_as_top_level_api must accept
    {logdensity_fn, proposal_distribution, proposal_logdensity_fn}.

    Pinned at P5.9: bjx_bench/inference/base_method/irmh.py calls
    blackjax.irmh(logdensity_fn, proposal_distribution=...,
    proposal_logdensity_fn=...).  If upstream renames or removes any of these,
    factory calls in the BO loop fail silently.
    """
    import inspect

    from blackjax.mcmc.random_walk import irmh_as_top_level_api

    sig = inspect.signature(irmh_as_top_level_api)
    expected = {"logdensity_fn", "proposal_distribution", "proposal_logdensity_fn"}
    missing = expected - set(sig.parameters)
    assert not missing, (
        f"blackjax.mcmc.random_walk.irmh_as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/base_method/irmh.py if upstream API changed."
    )
    assert set(sig.parameters) == expected, (
        f"blackjax.mcmc.random_walk.irmh_as_top_level_api has unexpected parameters. "
        f"Expected exactly {expected}, got {set(sig.parameters)}. "
        f"Update bjx_bench/inference/base_method/irmh.py if upstream API changed."
    )


def test_blackjax_irmh_alias_check():
    """Tripwire: blackjax.irmh top-level aliases must point at the correct inner functions.

    Pinned at P5.9: verifies that the top-level GenerateSamplingAPI wrapping
    for blackjax.irmh wires the right inner functions.  If upstream refactors
    random_walk.py (e.g. renames build_irmh or splits the module), this fires fast.

    Checks:
    - blackjax.irmh.differentiable is blackjax.mcmc.random_walk.irmh_as_top_level_api
    - blackjax.irmh.init is blackjax.mcmc.random_walk.init
    - blackjax.irmh.build_kernel is blackjax.mcmc.random_walk.build_irmh
    """
    from blackjax.mcmc.random_walk import build_irmh, init, irmh_as_top_level_api

    assert blackjax.irmh.differentiable is irmh_as_top_level_api, (
        "blackjax.irmh.differentiable is not blackjax.mcmc.random_walk.irmh_as_top_level_api. "
        "Upstream may have refactored the random_walk module. "
        "Update bjx_bench/inference/base_method/irmh.py and this tripwire."
    )
    assert blackjax.irmh.init is init, (
        "blackjax.irmh.init is not blackjax.mcmc.random_walk.init. "
        "Upstream may have introduced a separate IRMH init function. "
        "Update bjx_bench/inference/base_method/irmh.py and this tripwire."
    )
    assert blackjax.irmh.build_kernel is build_irmh, (
        "blackjax.irmh.build_kernel is not blackjax.mcmc.random_walk.build_irmh. "
        "Upstream may have renamed build_irmh. "
        "Update bjx_bench/inference/base_method/irmh.py and this tripwire."
    )


def test_blackjax_rwinfo_rwstate_fields():
    """Tripwire: pin RWInfo._fields and RWState._fields for IRMH and RWM.

    Pinned at P5.9: both IRMH and RWM share RWInfo and RWState.
    - RWInfo._fields == ('acceptance_rate', 'is_accepted', 'proposal')
    - RWState._fields == ('position', 'logdensity')

    If upstream refactors these NamedTuples (e.g. renames 'proposal' or adds
    fields), grad_count_per_step and the synthetic-info tests in
    test_base_method_irmh.py and test_base_method_mgrad_gaussian.py break.

    Note: MarginalInfo and MarginalState (also sharing this pattern) are
    already pinned in section 9 above.
    """
    from blackjax.mcmc.random_walk import RWInfo, RWState

    expected_rwinfo = ("acceptance_rate", "is_accepted", "proposal")
    assert RWInfo._fields == expected_rwinfo, (
        f"BlackJAX RWInfo fields changed from {expected_rwinfo} to {RWInfo._fields}. "
        f"Update tests/test_base_method_irmh.py synthetic RWInfo construction. "
        f"Also check test_base_method_mgrad_gaussian.py — MarginalInfo shares the same pattern."
    )

    expected_rwstate = ("position", "logdensity")
    assert RWState._fields == expected_rwstate, (
        f"BlackJAX RWState fields changed from {expected_rwstate} to {RWState._fields}. "
        f"Update tests/test_base_method_irmh.py synthetic RWState construction in grad_count tests."
    )


# ───────────── 11. Standard-momentum 2×2 (P5.13): mhmc + dmhmc ─────────────


def test_blackjax_mhmc_factory_signature():
    """Tripwire: blackjax.mhmc inner API must accept the same parameters as HMC.

    Pinned at P5.13: bjx_bench/inference/base_method/mhmc.py calls
    blackjax.mhmc(logdensity_fn, **trial_params) where trial_params includes
    step_size, inverse_mass_matrix, and num_integration_steps.  mhmc is a
    partial-applied HMC with multinomial_hmc_proposal; if upstream changes
    the as_top_level_api signature, factory calls in the BO loop fail silently.
    """
    import inspect

    from blackjax.mcmc.hmc import as_top_level_api

    sig = inspect.signature(as_top_level_api)
    expected = {
        "logdensity_fn",
        "step_size",
        "inverse_mass_matrix",
        "num_integration_steps",
    }
    missing = expected - set(sig.parameters)
    assert not missing, (
        f"blackjax.mcmc.hmc.as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/base_method/mhmc.py if upstream API changed."
    )


def test_blackjax_dmhmc_factory_signature():
    """Tripwire: blackjax.dmhmc inner API must accept the same parameters as dynamic_hmc.

    Pinned at P5.13: bjx_bench/inference/base_method/dmhmc.py calls
    blackjax.dmhmc(logdensity_fn, **trial_params) where trial_params includes
    step_size and inverse_mass_matrix (from CHEES warmup).  dmhmc is a
    partial-applied dynamic_hmc with multinomial_hmc_proposal; if upstream
    changes the as_top_level_api signature, factory calls in the BO loop fail.
    """
    import inspect

    from blackjax.mcmc.dynamic_hmc import as_top_level_api

    sig = inspect.signature(as_top_level_api)
    expected = {"logdensity_fn", "step_size", "inverse_mass_matrix"}
    missing = expected - set(sig.parameters)
    assert not missing, (
        f"blackjax.mcmc.dynamic_hmc.as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/base_method/dmhmc.py if upstream API changed."
    )


def test_blackjax_multinomial_hmc_alias():
    """Tripwire: blackjax.multinomial_hmc is blackjax.mhmc (backward-compat alias).

    Pinned at P5.13: documented at blackjax/__init__.py; if upstream renames the
    alias (e.g. removes multinomial_hmc or points it at a different factory),
    this fires immediately rather than silently at runtime.
    """
    assert blackjax.multinomial_hmc is blackjax.mhmc, (
        "blackjax.multinomial_hmc alias broken: multinomial_hmc is not mhmc. "
        "Update bjx_bench/inference/base_method/mhmc.py alias note and this tripwire."
    )


def test_blackjax_multinomial_hmc_proposal_exists():
    """Tripwire: blackjax.mcmc.hmc.multinomial_hmc_proposal exists and is callable.

    Pinned at P5.13: mhmc and dmhmc are partial-applied HMC/DynamicHMC with
    build_proposal=blackjax.mcmc.hmc.multinomial_hmc_proposal.  If upstream
    renames or removes this function, the wrappers silently fall back to the
    default proposal, changing the algorithm semantics.

    IMPORTANT: blackjax.multinomial_hmc is the SamplingAPI alias (NOT the
    proposal builder).  The proposal builder is at
    blackjax.mcmc.hmc.multinomial_hmc_proposal (module-level function).
    """
    assert hasattr(blackjax.mcmc.hmc, "multinomial_hmc_proposal"), (
        "blackjax.mcmc.hmc.multinomial_hmc_proposal does not exist. "
        "If upstream renamed or removed the multinomial proposal builder, "
        "update bjx_bench/inference/base_method/mhmc.py and dmhmc.py."
    )
    assert callable(blackjax.mcmc.hmc.multinomial_hmc_proposal), (
        "blackjax.mcmc.hmc.multinomial_hmc_proposal is not callable. "
        "Update bjx_bench/inference/base_method/mhmc.py and dmhmc.py."
    )


def test_base_methods_contains_mhmc_and_dmhmc():
    """Tripwire: BASE_METHODS registry must contain both 'mhmc' and 'dmhmc'.

    Pinned at P5.13 (META-011 dodge: subset check, not equality, so adding
    new algorithms doesn't break this test).  If either entry is accidentally
    dropped from __init__.py, this fires immediately.
    """
    from bjx_bench.inference.base_method import BASE_METHODS

    required = {"mhmc", "dmhmc"}
    missing = required - set(BASE_METHODS.keys())
    assert not missing, (
        f"BASE_METHODS is missing entries: {missing}. "
        f"Registered keys: {sorted(BASE_METHODS.keys())}. "
        f"Check bjx_bench/inference/base_method/__init__.py imports."
    )


# ───────── 12. Laplace-marginal 2×2 (P5.14b) ─────────────────────────────────


def test_blackjax_laplace_hmc_factory_signature():
    """Tripwire: blackjax.mcmc.laplace_hmc.as_top_level_api must accept
    {log_joint_fn, theta_init, step_size, inverse_mass_matrix,
    num_integration_steps, divergence_threshold, integrator, build_proposal}.

    Pinned at P5.14b: bjx_bench/inference/base_method/laplace_hmc.py calls
    blackjax.laplace_hmc(log_joint_fn, theta_init, step_size,
    inverse_mass_matrix, num_integration_steps, **optimizer_kwargs).
    If upstream renames or removes any of these, the wrapper fails silently.
    """
    import inspect

    from blackjax.mcmc.laplace_hmc import as_top_level_api

    sig = inspect.signature(as_top_level_api)
    expected = {
        "log_joint_fn",
        "theta_init",
        "step_size",
        "inverse_mass_matrix",
        "num_integration_steps",
        "divergence_threshold",
        "integrator",
        "build_proposal",
    }
    missing = expected - set(sig.parameters)
    assert not missing, (
        f"blackjax.mcmc.laplace_hmc.as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/base_method/laplace_hmc.py and "
        f"bjx_bench/inference/base_method/laplace_mhmc.py if upstream API changed."
    )


def test_blackjax_laplace_dhmc_factory_signature():
    """Tripwire: blackjax.mcmc.laplace_dynamic_hmc.as_top_level_api must accept
    {log_joint_fn, theta_init, step_size, inverse_mass_matrix,
    divergence_threshold, integrator, next_random_arg_fn, integration_steps_fn,
    integration_steps_params, build_proposal}.

    Pinned at P5.14b: bjx_bench/inference/base_method/laplace_dhmc.py calls
    blackjax.laplace_dhmc(log_joint_fn, theta_init, step_size,
    inverse_mass_matrix, **optimizer_kwargs).
    If upstream renames or removes any of these, the wrapper fails silently.
    """
    import inspect

    from blackjax.mcmc.laplace_dynamic_hmc import as_top_level_api

    sig = inspect.signature(as_top_level_api)
    expected = {
        "log_joint_fn",
        "theta_init",
        "step_size",
        "inverse_mass_matrix",
        "divergence_threshold",
        "integrator",
        "next_random_arg_fn",
        "integration_steps_fn",
        "build_proposal",
    }
    missing = expected - set(sig.parameters)
    assert not missing, (
        f"blackjax.mcmc.laplace_dynamic_hmc.as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update bjx_bench/inference/base_method/laplace_dhmc.py and "
        f"bjx_bench/inference/base_method/laplace_dmhmc.py if upstream API changed."
    )


def test_blackjax_laplace_hmc_state_fields():
    """Tripwire: LaplaceHMCState._fields must be
    ('position', 'logdensity', 'logdensity_grad', 'theta_star').

    Pinned at P5.14b: bjx_bench/inference/base_method/laplace_hmc.py and
    laplace_mhmc.py depend on the state carrying theta_star (the MAP latent
    at current phi, used for warm-starting L-BFGS).  If upstream renames or
    removes theta_star, warm-start and gradient accounting break silently.
    """
    from blackjax.mcmc.laplace_hmc import LaplaceHMCState

    expected = ("position", "logdensity", "logdensity_grad", "theta_star")
    assert LaplaceHMCState._fields == expected, (
        f"BlackJAX LaplaceHMCState fields changed from {expected} to "
        f"{LaplaceHMCState._fields}. "
        f"Update bjx_bench/inference/base_method/laplace_hmc.py, laplace_mhmc.py "
        f"and tests/inference/base_method/test_laplace_hmc.py."
    )


def test_blackjax_laplace_dynamic_hmc_state_fields():
    """Tripwire: LaplaceDynamicHMCState._fields must be
    ('position', 'logdensity', 'logdensity_grad', 'theta_star', 'random_generator_arg').

    Pinned at P5.14b: bjx_bench/inference/base_method/laplace_dhmc.py and
    laplace_dmhmc.py depend on the state carrying both theta_star (warm-start)
    and random_generator_arg (Halton step-count seed).  If either field is
    renamed or removed, the dynamic wrappers break at runtime.
    """
    from blackjax.mcmc.laplace_dynamic_hmc import LaplaceDynamicHMCState

    expected = (
        "position",
        "logdensity",
        "logdensity_grad",
        "theta_star",
        "random_generator_arg",
    )
    assert LaplaceDynamicHMCState._fields == expected, (
        f"BlackJAX LaplaceDynamicHMCState fields changed from {expected} to "
        f"{LaplaceDynamicHMCState._fields}. "
        f"Update bjx_bench/inference/base_method/laplace_dhmc.py, laplace_dmhmc.py "
        f"and tests/inference/base_method/test_laplace_dhmc.py."
    )


def test_blackjax_laplace_hmc_build_kernel_callable():
    """Tripwire: blackjax.mcmc.laplace_hmc.build_kernel must be callable.

    Pinned at P5.14b: documents that the composable-kernel layer exists.
    If upstream removes or renames build_kernel (e.g. following a module
    refactor), this fires immediately rather than at kernel construction time.
    """
    import blackjax.mcmc.laplace_hmc as lh

    assert callable(lh.build_kernel), (
        "blackjax.mcmc.laplace_hmc.build_kernel is not callable. "
        "Upstream may have removed or renamed the composable kernel layer. "
        "Update bjx_bench/inference/base_method/laplace_hmc.py and laplace_mhmc.py."
    )


def test_blackjax_laplace_dynamic_hmc_build_kernel_callable():
    """Tripwire: blackjax.mcmc.laplace_dynamic_hmc.build_kernel must be callable.

    Pinned at P5.14b: documents that the composable-kernel layer exists for
    the dynamic variant.  If upstream removes or renames build_kernel, this
    fires immediately rather than at kernel construction time.
    """
    import blackjax.mcmc.laplace_dynamic_hmc as ldh

    assert callable(ldh.build_kernel), (
        "blackjax.mcmc.laplace_dynamic_hmc.build_kernel is not callable. "
        "Upstream may have removed or renamed the composable kernel layer. "
        "Update bjx_bench/inference/base_method/laplace_dhmc.py and laplace_dmhmc.py."
    )


def test_base_methods_contains_laplace_marginal_2x2():
    """Tripwire: BASE_METHODS must contain all 4 Laplace-marginal entries.

    Pinned at P5.14b (META-011 dodge: subset check, not equality, so adding
    new algorithms doesn't break this test).  If any of the 4 entries is
    accidentally dropped from __init__.py, this fires immediately.
    """
    from bjx_bench.inference.base_method import BASE_METHODS

    required = {"laplace_hmc", "laplace_dhmc", "laplace_mhmc", "laplace_dmhmc"}
    missing = required - set(BASE_METHODS.keys())
    assert not missing, (
        f"BASE_METHODS is missing Laplace-marginal entries: {missing}. "
        f"Registered keys: {sorted(BASE_METHODS.keys())}. "
        f"Check bjx_bench/inference/base_method/__init__.py imports."
    )
