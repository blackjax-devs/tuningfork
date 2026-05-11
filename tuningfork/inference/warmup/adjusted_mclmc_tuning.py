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
"""Adjusted-MCLMC warmup via ``blackjax.adjusted_mclmc_find_L_and_step_size``.

This warmup is compatible with both ``adjusted_mclmc`` and
``adjusted_mclmc_dynamic``.  It runs the adjusted-MCLMC-specific adaptation
routine (which finds good values of ``L``, ``step_size``, and a diagonal
inverse mass matrix via preconditioning) using the **static** adjusted-MCLMC
kernel ``blackjax.mcmc.adjusted_mclmc.build_kernel()``.

Using the static kernel for tuning (regardless of whether the sampling step
uses the dynamic variant) matches the upstream adjusted-MCLMC test convention:
the adapter is integrator-agnostic, and the adapted ``(L, step_size, IMM)``
values are fully compatible with both static and dynamic sampling.

The 3-tuple returned by ``blackjax.adjusted_mclmc_find_L_and_step_size`` is::

    (IntegratorState, MCLMCAdaptationState, total_num_tuning_integrator_steps)

where ``MCLMCAdaptationState._fields = ('L', 'step_size', 'inverse_mass_matrix')``.
This mirrors the vanilla ``mclmc_find_L_and_step_size``  A.1 pattern.

The third value ``total_tuning_steps`` is threaded into the returned
``adapted_params`` dict under the key ``"_total_tuning_steps"`` so
``Recipe.calibration_budget`` can capture the actual gradient
spend during warmup.

Runner signature (multi-chain contract)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, num_chains: int = 4, target: float = 0.9, **kwargs)
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

Note: unlike vanilla mclmc, ``blackjax.mcmc.adjusted_mclmc.init`` does NOT
require an rng_key — it takes only ``(position, logdensity_fn)``.

Note on ``target``: ``adjusted_mclmc_find_L_and_step_size`` requires an
explicit ``target`` acceptance rate (no default).  The canonical value is
``0.9`` (per upstream adjusted-MCLMC tests).
"""

from typing import Any

import blackjax
import jax
import jax.numpy as jnp

from tuningfork.inference.warmup._base import Warmup, _maybe_replicate

__all__ = ["ENTRY"]


def _runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,
    base_method: Any,  # BaseMethod; not imported to avoid circular dep at module level
    *,
    logdensity_fn: Any,
    num_chains: int = 4,
    target: float = 0.9,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run ``blackjax.adjusted_mclmc_find_L_and_step_size`` over ``num_chains`` chains.

    Parameters
    ----------
    rng_key
        JAX random key; split internally into ``num_chains`` warmup keys.
        (No separate init keys — adjusted_mclmc.init does not require rng_key.)
    init_position
        Initial unconstrained parameter dict.  A SINGLE pytree (one chain's
        worth).  Replicated across ``num_chains`` unless pre-batched.
    n_warmup
        Number of adaptation steps passed as ``num_steps``.
    base_method
        ``BaseMethod`` entry (``adjusted_mclmc`` or ``adjusted_mclmc_dynamic``).
    logdensity_fn
        BlackJAX-compatible log-density function.
    num_chains
        Number of independent chains to run in parallel via ``jax.vmap``.
        Default ``4``.
    target
        Target acceptance rate for the adjusted-MCLMC adaptation.
        Default ``0.9`` (canonical adjusted-MCLMC value from upstream tests).
    **kwargs
        Ignored; present for interface uniformity.

    Returns
    -------
    states
        Post-adaptation ``HMCState``, batched over ``num_chains``.
        ``states.position`` has shape ``(num_chains, d)``.
    adapted_params
        Dict with keys::

            "L"                     : (num_chains,) — adapted trajectory lengths
            "step_size"             : (num_chains,) — adapted step sizes
            "inverse_mass_matrix"   : (num_chains, d) — diagonal preconditioners
            "_total_tuning_steps"   : int — gradient evals in adaptation
    """
    # Split rng_key: num_chains warmup keys only.
    # adjusted_mclmc.init does not require an rng_key (no random momentum init).
    warmup_keys = jax.random.split(rng_key, num_chains)

    # Replicate init_position across chains.  Pass-through if pre-batched.
    init_positions = _maybe_replicate(init_position, num_chains)

    # Init states: adjusted_mclmc.init(position, logdensity_fn) — no rng_key.
    @jax.vmap
    def init_one(x0: Any) -> Any:
        return blackjax.mcmc.adjusted_mclmc.init(x0, logdensity_fn)

    init_states = init_one(init_positions)

    # adjusted_mclmc_find_L_and_step_size takes the raw build_kernel output,
    # not the SamplingAlgorithm wrapper.  Use the static adjusted_mclmc kernel
    # for adaptation (adapter is integrator-agnostic; works for both static and
    # dynamic downstream samplers).
    adj_mclmc_kernel = blackjax.mcmc.adjusted_mclmc.build_kernel()

    @jax.vmap
    def tune_one(k: jax.Array, state: Any) -> tuple[Any, Any, Any]:
        s, adaptation_state, total_steps = blackjax.adjusted_mclmc_find_L_and_step_size(
            adj_mclmc_kernel,
            logdensity_fn=logdensity_fn,
            num_steps=n_warmup,
            state=state,
            rng_key=k,
            target=target,
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
        "_total_tuning_steps": total_tuning_steps,
    }
    return states, adapted


ENTRY = Warmup(
    name="adjusted_mclmc_tuning",
    runner=_runner,
    compatible_methods=("adjusted_mclmc", "adjusted_mclmc_dynamic"),
    notes=(
        "Adjusted-MCLMC-specific adaptation via blackjax.adjusted_mclmc_find_L_and_step_size. "
        "Finds L, step_size, and a diagonal inverse_mass_matrix jointly. "
        "Uses static adjusted_mclmc kernel for tuning (adapter is integrator-agnostic). "
        "Canonical target acceptance rate: 0.9 (per upstream adjusted-MCLMC tests). "
        "Returns _total_tuning_steps for calibration_budget accounting. "
        "Compatible with both adjusted_mclmc (static) and adjusted_mclmc_dynamic (random N). "
        "multi-chain by default (num_chains=4 via jax.vmap); per-chain "
        "L/step_size/IMM returned as (num_chains,)/(num_chains,)/(num_chains, d) "
        "arrays; _total_tuning_steps is a Python int (chain-0 value, same across chains)."
    ),
)
