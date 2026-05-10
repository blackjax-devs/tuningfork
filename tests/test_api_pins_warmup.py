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
"""Tripwire tests for BlackJAX warmup/adaptation API shapes that bjx_bench relies on.

These are defensive: if BlackJAX upstream changes the return-tuple shape or
function signatures of any adaptation we depend on, these tests fire with a
clear message pointing at the file in bjx-bench that needs an update.

Includes sections: 5 (pathfinder / multipathfinder), and the unnumbered
window_adaptation section (for HMC/NUTS/Barker/MALA).
"""

import blackjax
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.fast


# ───────────────── 5. pathfinder / multipathfinder callable + return shapes ─────────────────
def test_blackjax_pathfinder_callable():
    """P5.4 tripwire: if upstream pathfinder API changes its callable signature, fail fast.

    Pinned at P5.4: bjx_bench/inference/warmup/pathfinder.py calls
    blackjax.pathfinder.approximate(rng_key, logdensity_fn, position, num_samples)
    and expects a 2-tuple (PathfinderState, PathfinderInfo) back.
    """
    assert callable(blackjax.pathfinder.approximate), (
        "blackjax.pathfinder.approximate must be callable; "
        "update bjx_bench/inference/warmup/pathfinder.py if API changed."
    )
    assert callable(blackjax.pathfinder.sample), (
        "blackjax.pathfinder.sample must be callable; "
        "update bjx_bench/inference/warmup/pathfinder.py if API changed."
    )

    def logdensity_fn(x):
        return -0.5 * jnp.sum(x**2)

    key = jax.random.key(0)
    result = blackjax.pathfinder.approximate(
        key, logdensity_fn, jnp.zeros(3), num_samples=5
    )
    assert len(result) == 2, (
        f"blackjax.pathfinder.approximate should return a 2-tuple (state, info), "
        f"got {len(result)}-tuple. Update bjx_bench/inference/warmup/pathfinder.py."
    )
    pf_state, _pf_info = result
    for field in ("elbo", "position", "grad_position", "alpha", "beta", "gamma"):
        assert hasattr(pf_state, field), (
            f"PathfinderState lost field {field!r}. "
            f"Update bjx_bench/inference/warmup/pathfinder.py."
        )


def test_blackjax_multipathfinder_callable():
    """P5.4 tripwire: if upstream multipathfinder API changes, fail fast.

    Pinned at P5.4: bjx_bench/inference/warmup/multipathfinder.py calls
    blackjax.multipathfinder(logdensity_fn).init(key, positions, num_samples)
    and expects a 2-tuple (MultipathfinderState, PathfinderInfo) back.
    psis_weights(state) must return a 2-tuple (log_weights, pareto_k).
    """
    assert callable(blackjax.multipathfinder), (
        "blackjax.multipathfinder must be callable; "
        "update bjx_bench/inference/warmup/multipathfinder.py if API changed."
    )

    from blackjax.vi.multipathfinder import psis_weights

    def logdensity_fn(x):
        return -0.5 * jnp.sum(x**2)

    key = jax.random.key(0)
    mpf = blackjax.multipathfinder(logdensity_fn)
    result = mpf.init(key, jnp.zeros((2, 3)), num_samples=5)
    assert len(result) == 2, (
        f"multipathfinder.init should return a 2-tuple (state, info), "
        f"got {len(result)}-tuple. Update bjx_bench/inference/warmup/multipathfinder.py."
    )
    mpf_state, _info = result
    for field in ("path_states", "samples", "logp", "logq"):
        assert hasattr(mpf_state, field), (
            f"MultipathfinderState lost field {field!r}. "
            f"Update bjx_bench/inference/warmup/multipathfinder.py."
        )

    # psis_weights must return 2-tuple (log_weights, pareto_k)
    pw_result = psis_weights(mpf_state)
    assert len(pw_result) == 2, (
        f"psis_weights should return a 2-tuple (log_weights, pareto_k), "
        f"got {len(pw_result)}-tuple. Update bjx_bench/inference/warmup/multipathfinder.py."
    )


def test_blackjax_meads_adaptation_signature():
    """Tripwire: blackjax.meads_adaptation must accept (logdensity_fn,
    num_chains, num_folds) as parameters.

    Pinned at P5.5: bjx_bench/inference/warmup/meads.py calls
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
        f"Update bjx_bench/inference/warmup/meads.py if upstream API changed."
    )


def test_blackjax_meads_adaptation_run_returns_2tuple():
    """Tripwire: meads_adaptation.run() must return a 2-tuple
    (AdaptationResults, AdaptationInfo).

    Pinned at P5.5: bjx_bench/inference/warmup/meads.py unpacks the result as
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
        f"got {len(result)}-tuple. Update bjx_bench/inference/warmup/meads.py."
    )
    adaptation_results, _adaptation_info = result
    assert hasattr(adaptation_results, "state"), (
        "AdaptationResults must have 'state' field; "
        "update bjx_bench/inference/warmup/meads.py."
    )
    assert hasattr(adaptation_results, "parameters"), (
        "AdaptationResults must have 'parameters' field; "
        "update bjx_bench/inference/warmup/meads.py."
    )
    for param_key in ("step_size", "momentum_inverse_scale", "alpha", "delta"):
        assert param_key in adaptation_results.parameters, (
            f"meads_adaptation AdaptationResults.parameters lost key {param_key!r}. "
            f"Update bjx_bench/inference/warmup/meads.py."
        )


# ───────────────── window_adaptation works for HMC/NUTS/Barker/MALA ─────────────────
def test_window_adaptation_constructs_for_supported_kernels():
    """Pinned in T2.6b: bjx_bench/calibration/tier_b.py:_run_warmup uses
    blackjax.window_adaptation for kernels with needs_mass_matrix=True (and
    structurally for MALA in T2.6c, even though MALA's needs_mass_matrix=False
    — sanity check). Verifies the construction path stays valid.
    """

    def logdensity_fn(x):
        return -0.5 * jnp.sum(x["x"] ** 2)

    for kernel_factory in (blackjax.hmc, blackjax.nuts, blackjax.barker, blackjax.mala):
        # window_adaptation construction (no .run) — fast smoke
        try:
            wa = blackjax.window_adaptation(kernel_factory, logdensity_fn)
            assert hasattr(wa, "run"), (
                f"blackjax.window_adaptation(blackjax.{kernel_factory.__name__}) "
                f"returned object lacking .run; surface changed."
            )
        except Exception as exc:
            raise AssertionError(
                f"blackjax.window_adaptation(blackjax.{kernel_factory.__name__}) "
                f"failed at construction: {type(exc).__name__}: {exc}"
            ) from exc
