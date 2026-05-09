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
"""MCLMC warmup via ``blackjax.mclmc_find_L_and_step_size``.

This warmup is *only* compatible with the ``mclmc`` algorithm.  It runs
the MCLMC-specific adaptation routine which finds good values of ``L``
(trajectory length) and ``step_size`` together with a diagonal inverse
mass matrix via preconditioning.

The 3-tuple returned by ``blackjax.mclmc_find_L_and_step_size`` is::

    (IntegratorState, MCLMCAdaptationState, total_tuning_steps: int)

where ``MCLMCAdaptationState._fields = ('L', 'step_size', 'inverse_mass_matrix')``.
This was pinned in Phase 2 at ``tests/test_blackjax_api_pins.py``.

The third value ``total_tuning_steps`` is threaded into the returned
``adapted_params`` dict under the key ``"_total_tuning_steps"`` so
``Recipe.calibration_budget`` (Phase 3.2) can capture the actual gradient
spend during warmup.

Runner signature (uniform across all warmups)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, **kwargs) -> (state, adapted_params)
"""

from typing import Any

import blackjax
import jax

from bjx_bench.inference.warmup._base import Warmup

__all__ = ["ENTRY"]


def _runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,
    base_method: Any,  # BaseMethod; not imported to avoid circular dep at module level
    *,
    logdensity_fn: Any,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run ``blackjax.mclmc_find_L_and_step_size`` and return ``(state, adapted_params)``.

    Parameters
    ----------
    rng_key
        JAX random key; internally split into ``init_key`` (for MCLMC state
        initialisation) and ``warmup_key`` (for the adaptation run).
    init_position
        Initial unconstrained parameter dict.
    n_warmup
        Number of adaptation steps passed as ``num_steps``.
    base_method
        ``BaseMethod`` entry for MCLMC (carries ``factory``,
        ``default_hp_space``).
    logdensity_fn
        BlackJAX-compatible log-density function.
    **kwargs
        Ignored; present for interface uniformity.

    Returns
    -------
    state
        Post-adaptation ``IntegratorState``.
    adapted_params
        Dict with keys::

            "L"                     : float — adapted trajectory length
            "step_size"             : float — adapted step size
            "inverse_mass_matrix"   : Array — diagonal preconditioner
            "_total_tuning_steps"   : int   — total gradient evals in adaptation
                                              (for Recipe.calibration_budget)
    """
    from bjx_bench.calibration.tier_b import default_params_for

    init_key, warmup_key = jax.random.split(rng_key, 2)

    # Build kernel with default params to get an initial MCLMC state.
    # mclmc.init requires an rng_key to generate the initial unit-vector
    # momentum — all other kernels don't need it.
    defaults = default_params_for(base_method)
    kernel = base_method.factory(logdensity_fn, **defaults)
    init_state = kernel.init(init_position, init_key)

    # mclmc_find_L_and_step_size takes the raw build_kernel function output,
    # not the SamplingAlgorithm wrapper.  Use the module-level build_kernel.
    mclmc_kernel = blackjax.mclmc.build_kernel()
    state, adaptation_state, total_tuning_steps = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel,
        num_steps=n_warmup,
        state=init_state,
        rng_key=warmup_key,
        logdensity_fn=logdensity_fn,
        diagonal_preconditioning=True,
    )
    # MCLMCAdaptationState._fields = ('L', 'step_size', 'inverse_mass_matrix')
    adapted: dict[str, Any] = {
        "L": float(adaptation_state.L),
        "step_size": float(adaptation_state.step_size),
        "inverse_mass_matrix": adaptation_state.inverse_mass_matrix,
        # P3.2 will fold this into Recipe.calibration_budget
        "_total_tuning_steps": int(total_tuning_steps),
    }
    return state, adapted


ENTRY = Warmup(
    name="mclmc_tuning",
    runner=_runner,
    compatible_methods=("mclmc",),
    notes=(
        "MCLMC-specific adaptation via blackjax.mclmc_find_L_and_step_size. "
        "Finds L, step_size, and a diagonal inverse_mass_matrix jointly. "
        "Returns _total_tuning_steps for calibration_budget accounting (P3.2). "
        "Not compatible with any other kernel (HMC/NUTS use window_adaptation)."
    ),
)
