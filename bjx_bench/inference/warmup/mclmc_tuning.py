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

Runner signature (multi-chain contract, P5.0c)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, num_chains: int = 4, **kwargs)
    -> (states, adapted_params)

Where:

- ``rng_key`` is a single key; split internally into ``num_chains`` keys.
- ``init_position`` is a single pytree (one chain's worth); replicated
  across chains internally via ``_maybe_replicate`` unless the caller
  pre-batches it (leading dim == ``num_chains``).
- ``states`` is a batched pytree with leading dim ``num_chains``.
- ``adapted_params`` contains per-chain values::

      "L"                     : (num_chains,) — adapted trajectory lengths
      "step_size"             : (num_chains,) — adapted step sizes
      "inverse_mass_matrix"   : (num_chains, d) — diagonal preconditioners
      "_total_tuning_steps"   : int — total gradient evals in adaptation
                                      (summed across chains; for
                                      Recipe.calibration_budget)

Note on vmap and ``blackjax.mclmc_find_L_and_step_size``: the tuning
function is vmapped over (init_key, warmup_key, init_position).  The
``_total_tuning_steps`` scalar is averaged across chains (all chains run
the same ``num_steps``) and stored as a Python int.
"""

from typing import Any

import blackjax
import jax
import jax.numpy as jnp

from bjx_bench.inference.warmup._base import Warmup, _maybe_replicate

__all__ = ["ENTRY"]


def _runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,
    base_method: Any,  # BaseMethod; not imported to avoid circular dep at module level
    *,
    logdensity_fn: Any,
    num_chains: int = 4,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run ``blackjax.mclmc_find_L_and_step_size`` over ``num_chains`` chains.

    Parameters
    ----------
    rng_key
        JAX random key; split internally into ``2 * num_chains`` subkeys
        (``num_chains`` init keys + ``num_chains`` warmup keys).
    init_position
        Initial unconstrained parameter dict.  A SINGLE pytree (one chain's
        worth).  Replicated across ``num_chains`` unless pre-batched.
    n_warmup
        Number of adaptation steps passed as ``num_steps``.
    base_method
        ``BaseMethod`` entry for MCLMC (carries ``factory``,
        ``default_hp_space``).
    logdensity_fn
        BlackJAX-compatible log-density function.
    num_chains
        Number of independent chains to run in parallel via ``jax.vmap``.
        Default ``4``, matching Stan/NumPyro convention.  Pass
        ``num_chains=1`` explicitly for BO trials.
    **kwargs
        Ignored; present for interface uniformity.

    Returns
    -------
    states
        Post-adaptation ``IntegratorState``, batched over ``num_chains``.
        ``states.position`` has shape ``(num_chains, d)``.
    adapted_params
        Dict with keys::

            "L"                     : (num_chains,) — adapted trajectory lengths
            "step_size"             : (num_chains,) — adapted step sizes
            "inverse_mass_matrix"   : (num_chains, d) — diagonal preconditioners
            "_total_tuning_steps"   : int — total gradient evals across all
                                            chains (for Recipe.calibration_budget)
    """
    from bjx_bench.calibration.tier_b import default_params_for

    # Split rng_key: num_chains init_keys + num_chains warmup_keys
    all_keys = jax.random.split(rng_key, 2 * num_chains)
    init_keys = all_keys[:num_chains]
    warmup_keys = all_keys[num_chains:]

    # Replicate init_position across chains.  Pass-through if pre-batched.
    init_positions = _maybe_replicate(init_position, num_chains)

    # Build default params for creating initial MCLMC states.
    defaults = default_params_for(base_method)
    kernel = base_method.factory(logdensity_fn, **defaults)

    # mclmc.init requires an rng_key to generate the initial unit-vector
    # momentum — vmap over (init_key, init_position).
    @jax.vmap
    def init_one(k: jax.Array, x0: Any) -> Any:
        return kernel.init(x0, k)

    init_states = init_one(init_keys, init_positions)

    # mclmc_find_L_and_step_size takes the raw build_kernel function output,
    # not the SamplingAlgorithm wrapper.  Use the module-level build_kernel.
    mclmc_kernel = blackjax.mclmc.build_kernel()

    # vmap the tuning over (warmup_key, init_state).
    # NOTE: mclmc_find_L_and_step_size is NOT designed for vmap out-of-the-box;
    # we vmap the Python wrapper that calls it.  This works because JAX traces
    # through the function as long as no Python-level branching depends on
    # traced values.  _total_tuning_steps is a static int (same across chains),
    # so we collect it from chain 0 after vmapping.
    @jax.vmap
    def tune_one(k: jax.Array, state: Any) -> tuple[Any, Any, Any]:
        s, adaptation_state, total_steps = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel,
            num_steps=n_warmup,
            state=state,
            rng_key=k,
            logdensity_fn=logdensity_fn,
            diagonal_preconditioning=True,
        )
        return s, adaptation_state, total_steps

    states, adaptation_states, total_tuning_steps_per_chain = tune_one(
        warmup_keys, init_states
    )

    # total_tuning_steps is the same for all chains (same num_steps).
    # Take the value from chain 0 and convert to Python int.
    total_tuning_steps = int(jnp.asarray(total_tuning_steps_per_chain)[0])

    # MCLMCAdaptationState._fields = ('L', 'step_size', 'inverse_mass_matrix')
    adapted: dict[str, Any] = {
        "L": adaptation_states.L,  # shape (num_chains,)
        "step_size": adaptation_states.step_size,  # shape (num_chains,)
        "inverse_mass_matrix": adaptation_states.inverse_mass_matrix,  # (num_chains, d)
        # P3.2 will fold this into Recipe.calibration_budget
        "_total_tuning_steps": total_tuning_steps,
    }
    return states, adapted


ENTRY = Warmup(
    name="mclmc_tuning",
    runner=_runner,
    compatible_methods=("mclmc",),
    notes=(
        "MCLMC-specific adaptation via blackjax.mclmc_find_L_and_step_size. "
        "Finds L, step_size, and a diagonal inverse_mass_matrix jointly. "
        "Returns _total_tuning_steps for calibration_budget accounting (P3.2). "
        "Not compatible with any other kernel (HMC/NUTS use window_adaptation).  "
        "P5.0c: multi-chain by default (num_chains=4 via jax.vmap); per-chain "
        "L/step_size/IMM returned as (num_chains,)/(num_chains,)/(num_chains, d) "
        "arrays; _total_tuning_steps is summed across chains."
    ),
)
