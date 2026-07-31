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
"""Tripwire tests for BlackJAX warmup/adaptation API shapes that tuningfork relies on.

These are defensive: if BlackJAX upstream changes the return-tuple shape or
function signatures of any adaptation we depend on, these tests fire with a
clear message pointing at the file in tuningfork that needs an update.

Includes sections: 5 (pathfinder / multipathfinder), and the unnumbered
window_adaptation section (for HMC/NUTS/Barker/MALA).
Also pins: _psis_weighted_mixture_covariance (used by multipathfinder_window_adaptation),
window_adaptation kwargs initial_inverse_mass_matrix + imm_shrinkage_to_previous.
"""

import blackjax
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.fast


# ───────────────── 5. pathfinder / multipathfinder callable + return shapes ─────────────────
def test_blackjax_pathfinder_callable():
    """Tripwire: if upstream pathfinder API changes its callable signature, fail fast.

    Pinned tripwire: tuningfork/warmup/pathfinder.py calls
    blackjax.pathfinder.approximate(rng_key, logdensity_fn, position, num_samples)
    and expects a 2-tuple (PathfinderState, PathfinderInfo) back.
    """
    assert callable(blackjax.pathfinder.approximate), (
        "blackjax.pathfinder.approximate must be callable; "
        "update tuningfork/warmup/pathfinder.py if API changed."
    )
    assert callable(blackjax.pathfinder.sample), (
        "blackjax.pathfinder.sample must be callable; "
        "update tuningfork/warmup/pathfinder.py if API changed."
    )

    def logdensity_fn(x):
        return -0.5 * jnp.sum(x**2)

    key = jax.random.key(0)
    result = blackjax.pathfinder.approximate(
        key, logdensity_fn, jnp.zeros(3), num_samples=5
    )
    assert len(result) == 2, (
        f"blackjax.pathfinder.approximate should return a 2-tuple (state, info), "
        f"got {len(result)}-tuple. Update tuningfork/warmup/pathfinder.py."
    )
    pf_state, _pf_info = result
    for field in ("elbo", "position", "grad_position", "alpha", "beta", "gamma"):
        assert hasattr(pf_state, field), (
            f"PathfinderState lost field {field!r}. "
            f"Update tuningfork/warmup/pathfinder.py."
        )


def test_blackjax_pathfinder_adaptation_signature():
    """Tripwire: blackjax.pathfinder_adaptation must accept num_chains + n_paths kwargs.

    Pinned tripwire: tuningfork/warmup/pathfinder.py and
    tuningfork/warmup/multipathfinder.py call
    blackjax.pathfinder_adaptation(algorithm, logdensity_fn, num_chains=..., n_paths=...).
    If upstream renames or removes these kwargs, fail fast here.
    """
    import inspect

    sig = inspect.signature(blackjax.pathfinder_adaptation)
    for kwarg in ("num_chains", "n_paths", "imm_estimator", "initial_step_size"):
        assert kwarg in sig.parameters, (
            f"blackjax.pathfinder_adaptation is missing parameter {kwarg!r}. "
            f"Current params: {list(sig.parameters)}. "
            f"Update tuningfork/warmup/pathfinder.py and multipathfinder.py."
        )

    # Verify AdaptationResults contract: .state + .parameters with step_size + IMM
    from blackjax.adaptation.base import AdaptationResults

    assert hasattr(AdaptationResults, "_fields"), (
        "AdaptationResults must be a NamedTuple with _fields. "
        "Update tuningfork/warmup/pathfinder.py and multipathfinder.py."
    )
    for field in ("state", "parameters"):
        assert field in AdaptationResults._fields, (
            f"AdaptationResults lost field {field!r}. "
            f"Update tuningfork/warmup/pathfinder.py and multipathfinder.py."
        )


def test_blackjax_multipathfinder_callable():
    """Tripwire: if upstream multipathfinder API changes, fail fast.

    Pinned tripwire: tuningfork/warmup/multipathfinder_window_adaptation.py calls
    blackjax.multipathfinder(logdensity_fn).init(key, positions, num_samples)
    and expects a 2-tuple (MultipathfinderState, PathfinderInfo) back.
    psis_weights(state) must return a 2-tuple (log_weights, pareto_k).
    """
    assert callable(blackjax.multipathfinder), (
        "blackjax.multipathfinder must be callable; "
        "update tuningfork/warmup/multipathfinder.py if API changed."
    )

    from blackjax.vi.multipathfinder import psis_weights

    def logdensity_fn(x):
        return -0.5 * jnp.sum(x**2)

    key = jax.random.key(0)
    mpf = blackjax.multipathfinder(logdensity_fn)
    result = mpf.init(key, jnp.zeros((2, 3)), num_samples=5)
    assert len(result) == 2, (
        f"multipathfinder.init should return a 2-tuple (state, info), "
        f"got {len(result)}-tuple. Update tuningfork/warmup/multipathfinder.py."
    )
    mpf_state, _info = result
    for field in ("path_states", "samples", "logp", "logq"):
        assert hasattr(mpf_state, field), (
            f"MultipathfinderState lost field {field!r}. "
            f"Update tuningfork/warmup/multipathfinder.py."
        )

    # psis_weights must return 2-tuple (log_weights, pareto_k)
    pw_result = psis_weights(mpf_state)
    assert len(pw_result) == 2, (
        f"psis_weights should return a 2-tuple (log_weights, pareto_k), "
        f"got {len(pw_result)}-tuple. Update tuningfork/warmup/multipathfinder.py."
    )


def test_blackjax_meads_adaptation_signature():
    """Tripwire: blackjax.meads_adaptation must accept (logdensity_fn,
    num_chains, num_folds) as parameters.

    Pinned tripwire: tuningfork/warmup/meads.py calls
    blackjax.meads_adaptation(logdensity_fn, num_chains, num_folds=num_folds, ...).
    If upstream renames or removes any of these, the MEADS warmup fails.
    """
    import inspect

    sig = inspect.signature(blackjax.meads_adaptation)
    expected = {"logdensity_fn", "num_chains", "num_folds"}
    missing = expected - set(sig.parameters)
    assert not missing, (
        f"blackjax.meads_adaptation is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update tuningfork/warmup/meads.py if upstream API changed."
    )


def test_blackjax_meads_adaptation_run_returns_2tuple():
    """Tripwire: meads_adaptation.run() must return a 2-tuple
    (AdaptationResults, AdaptationInfo).

    Pinned tripwire: tuningfork/warmup/meads.py unpacks the result as
    (adaptation_results, _adaptation_info) = meads.run(...).
    If upstream changes to a 3-tuple or NamedTuple, the unpack fails.
    """
    meads = blackjax.meads_adaptation(
        lambda x: -0.5 * jnp.sum(x**2),
        num_chains=4,
        num_folds=4,
    )
    key = jax.random.key(0)
    result = meads.run(key, jnp.zeros((4, 3)), num_steps=5)
    assert len(result) == 2, (
        f"meads_adaptation.run() should return a 2-tuple (AdaptationResults, AdaptationInfo), "
        f"got {len(result)}-tuple. Update tuningfork/warmup/meads.py."
    )
    adaptation_results, _adaptation_info = result
    assert hasattr(adaptation_results, "state"), (
        "AdaptationResults must have 'state' field; "
        "update tuningfork/warmup/meads.py."
    )
    assert hasattr(adaptation_results, "parameters"), (
        "AdaptationResults must have 'parameters' field; "
        "update tuningfork/warmup/meads.py."
    )
    for param_key in ("step_size", "momentum_inverse_scale", "alpha", "delta"):
        assert param_key in adaptation_results.parameters, (
            f"meads_adaptation AdaptationResults.parameters lost key {param_key!r}. "
            f"Update tuningfork/warmup/meads.py."
        )


# ───────────────── window_adaptation works for HMC/NUTS/Barker/MALA ─────────────────
def test_window_adaptation_constructs_for_supported_kernels():
    """Retained warmup and generated emitters construct every supported kernel.

    This is a construction-only tripwire for the window-adaptation contract;
    generated code exercises the resulting warmup path, while this test keeps
    the API check fast and independent of a sampling run.
    """

    def logdensity_fn(x):
        return -0.5 * jnp.sum(x["x"] ** 2)

    for kernel_factory in (blackjax.hmc, blackjax.nuts, blackjax.barker, blackjax.mala):
        try:
            adaptation = blackjax.window_adaptation(kernel_factory, logdensity_fn)
            assert hasattr(adaptation, "run"), (
                f"blackjax.window_adaptation(blackjax.{kernel_factory.__name__}) "
                "returned an object without .run; generated emitter contract changed."
            )
        except Exception as exc:
            raise AssertionError(
                f"blackjax.window_adaptation(blackjax.{kernel_factory.__name__}) "
                f"failed at construction: {type(exc).__name__}: {exc}"
            ) from exc


def test_blackjax_lbfgs_inverse_hessian_importable():
    """Tripwire: multipathfinder_window_adaptation uses lbfgs_inverse_hessian_formula_1
    from blackjax.optimizers.lbfgs for computing the PSIS-weighted mixture covariance.
    If upstream renames or removes it, fail fast here pointing at
    tuningfork/warmup/multipathfinder_window_adaptation.py.

    Note: we use our own local mixture covariance implementation (not the upstream
    _psis_weighted_mixture_covariance) because the upstream function assumes flat
    (n_paths, d) positions but PathfinderState.position stores pytree-structured form.
    """
    from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1

    assert callable(lbfgs_inverse_hessian_formula_1), (
        "blackjax.optimizers.lbfgs.lbfgs_inverse_hessian_formula_1 must be callable; "
        "update tuningfork/warmup/multipathfinder_window_adaptation.py "
        "if the upstream function was renamed or removed."
    )


def test_window_adaptation_accepts_initial_imm_kwarg():
    """Tripwire: window_adaptation must accept initial_inverse_mass_matrix kwarg.

    Pinned by tuningfork/warmup/multipathfinder_window_adaptation.py which passes
    initial_inverse_mass_matrix=<dense array> to seed the mass matrix.
    """
    import inspect

    sig = inspect.signature(blackjax.window_adaptation)
    assert "initial_inverse_mass_matrix" in sig.parameters, (
        "blackjax.window_adaptation is missing parameter 'initial_inverse_mass_matrix'. "
        "Update tuningfork/warmup/multipathfinder_window_adaptation.py."
    )


def test_window_adaptation_accepts_imm_shrinkage_kwarg():
    """Multipathfinder adaptation requires the shrinkage keyword."""
    import inspect

    sig = inspect.signature(blackjax.window_adaptation)
    assert (
        "imm_shrinkage_to_previous" in sig.parameters
    ), "blackjax.window_adaptation is missing parameter 'imm_shrinkage_to_previous'."


def test_mfvi_state_fields():
    """Tripwire: pin MFVIState._fields.

    tuningfork/base_method/meanfield_vi.py and
    tuningfork/warmup/meanfield_vi.py depend on MFVIState having
    fields ('mu', 'rho', 'opt_state').  If upstream renames or reorders
    these, the wrappers break silently.
    """
    from blackjax.vi.meanfield_vi import MFVIState

    assert MFVIState._fields == ("mu", "rho", "opt_state"), (
        f"MFVIState._fields changed: {MFVIState._fields}. "
        f"Update tuningfork/base_method/meanfield_vi.py and "
        f"tuningfork/warmup/meanfield_vi.py."
    )


def test_mfvi_info_fields():
    """Tripwire: pin MFVIInfo._fields.

    tuningfork wrappers expect MFVIInfo to have exactly ('elbo',).
    """
    from blackjax.vi.meanfield_vi import MFVIInfo

    assert MFVIInfo._fields == ("elbo",), (
        f"MFVIInfo._fields changed: {MFVIInfo._fields}. "
        f"Update tuningfork/base_method/meanfield_vi.py."
    )


def test_frvi_state_fields():
    """Tripwire: pin FRVIState._fields.

    tuningfork/base_method/fullrank_vi.py and
    tuningfork/warmup/fullrank_vi.py depend on FRVIState having
    fields ('mu', 'chol_params', 'opt_state').  If upstream renames or
    reorders these, the wrappers break silently.
    """
    from blackjax.vi.fullrank_vi import FRVIState

    assert FRVIState._fields == ("mu", "chol_params", "opt_state"), (
        f"FRVIState._fields changed: {FRVIState._fields}. "
        f"Update tuningfork/base_method/fullrank_vi.py and "
        f"tuningfork/warmup/fullrank_vi.py."
    )


def test_frvi_info_fields():
    """Tripwire: pin FRVIInfo._fields.

    tuningfork wrappers expect FRVIInfo to have exactly ('elbo',).
    """
    from blackjax.vi.fullrank_vi import FRVIInfo

    assert FRVIInfo._fields == ("elbo",), (
        f"FRVIInfo._fields changed: {FRVIInfo._fields}. "
        f"Update tuningfork/base_method/fullrank_vi.py."
    )


def test_meanfield_vi_as_top_level_api_signature():
    """Tripwire: pin meanfield_vi.as_top_level_api signature.

    The wrapper relies on as_top_level_api accepting
    (logdensity_fn, optimizer, num_samples, objective, stl_estimator).
    If upstream changes the signature, fail fast here.
    """
    import inspect

    import blackjax.vi.meanfield_vi as mf

    sig = inspect.signature(mf.as_top_level_api)
    expected_params = {
        "logdensity_fn",
        "optimizer",
        "num_samples",
        "objective",
        "stl_estimator",
    }
    actual_params = set(sig.parameters)
    missing = expected_params - actual_params
    assert not missing, (
        f"meanfield_vi.as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update tuningfork/base_method/meanfield_vi.py."
    )


def test_fullrank_vi_as_top_level_api_signature():
    """Tripwire: pin fullrank_vi.as_top_level_api signature.

    The wrapper relies on as_top_level_api accepting
    (logdensity_fn, optimizer, num_samples, objective, stl_estimator).
    If upstream changes the signature, fail fast here.
    """
    import inspect

    import blackjax.vi.fullrank_vi as fr

    sig = inspect.signature(fr.as_top_level_api)
    expected_params = {
        "logdensity_fn",
        "optimizer",
        "num_samples",
        "objective",
        "stl_estimator",
    }
    actual_params = set(sig.parameters)
    missing = expected_params - actual_params
    assert not missing, (
        f"fullrank_vi.as_top_level_api is missing parameters: {missing}. "
        f"Current params: {list(sig.parameters)}. "
        f"Update tuningfork/base_method/fullrank_vi.py."
    )
