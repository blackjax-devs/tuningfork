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
"""CHEES (Change in Estimator of Step Size) warmup, wrapping
``blackjax.chees_adaptation``.

CHEES (Hoffman et al. 2022) is a **dynamic-HMC-specific** adaptation routine
that simultaneously tunes ``step_size``, ``inverse_mass_matrix``, and the
trajectory-length distribution (via ``integration_steps_fn``,
``next_random_arg_fn``, and ``integration_steps_params``).  Like MEADS, CHEES
is **multi-chain by construction**: all chains run jointly in a single
adaptation call — chains are inputs, not loop iterations.

CHEES adapts both step_size AND the trajectory-length distribution.  Like
MEADS but for dynamic-HMC instead of GHMC; both are multi-chain-by-construction
adaptation procedures.

Compatibility
-------------
CHEES is paired exclusively with ``dynamic_hmc``
(``compatible_methods=("dynamic_hmc",)``).  Passing any other base method
raises ``ValueError``.

Upstream API note
-----------------
``blackjax.chees_adaptation.run`` has a different signature from
``meads_adaptation.run``.  It requires ``step_size`` and ``optim`` (an optax
optimizer) as additional positional arguments::

    chees.run(rng_key, positions, step_size, optim, num_steps=n_warmup)

This wrapper hard-codes an ``optax.adam(learning_rate=0.01)`` optimizer for the
trajectory-length parameters (the standard CHEES default) and accepts
``step_size`` as a pass-through kwarg (default ``0.5``).

Adapted parameters returned by upstream
----------------------------------------
``chees_adaptation.run`` returns a 2-tuple ``(AdaptationResults, AdaptationInfo)``
where ``AdaptationResults.parameters`` is a dict with keys:

- ``step_size`` — scalar (shared across chains)
- ``inverse_mass_matrix`` — shape ``(d,)`` (shared across chains)
- ``next_random_arg_fn`` — Python callable (shared; not array-broadcastable)
- ``integration_steps_fn`` — Python callable (shared; not array-broadcastable)
- ``integration_steps_params`` — shape ``(1,)`` array

Runner signature (multi-chain contract)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, num_chains: int = 4,
            step_size: float = 0.5, **kwargs)
    -> (states, adapted_params)

Where:

- ``rng_key`` is passed directly to ``chees_adaptation.run``.
- ``init_position`` is a single pytree (one chain's worth); replicated to
  ``(num_chains, d)`` via ``_maybe_replicate`` unless pre-batched.
- ``states`` is a batched ``DynamicHMCState`` pytree with leading dim ``num_chains``.
- ``adapted_params`` contains:

  ========================= =============================================
  Key                       Shape / Type
  ========================= =============================================
  ``step_size``             ``(num_chains,)`` — scalar broadcast
  ``inverse_mass_matrix``   ``(num_chains, d)`` — broadcast from ``(d,)``
  ``next_random_arg_fn``    Python callable (passed through; shared)
  ``integration_steps_fn``  Python callable (passed through; shared)
  ``integration_steps_params`` ``(1,)`` array (passed through; shared)
  ``_chees_target_acceptance_rate`` float — value used
  ``_chees_max_leapfrog_steps``     int — cap used
  ========================= =============================================

References
----------
- Hoffman, M. D., Radul, A., & Sountsov, P. (2022). An adaptive-MCMC scheme
  for setting trajectory lengths in Hamiltonian Monte Carlo. In *AISTATS 2022*.
"""

from typing import Any

import blackjax
import jax
import jax.numpy as jnp
import optax

from bjx_bench.inference.warmup._base import Warmup, _maybe_replicate

__all__ = ["ENTRY"]

# Default CHEES hyperparameters (matching upstream defaults)
_DEFAULT_CHEES_TARGET_ACCEPTANCE_RATE: float = 0.651
_DEFAULT_CHEES_MAX_LEAPFROG_STEPS: int = 1000
_DEFAULT_CHEES_OPTIM_LR: float = 0.01
_DEFAULT_CHEES_STEP_SIZE: float = 0.5


def _runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,
    base_method: Any,  # BaseMethod; not imported to avoid circular dep at module level
    *,
    logdensity_fn: Any,
    num_chains: int = 4,
    step_size: float = _DEFAULT_CHEES_STEP_SIZE,
    target_acceptance_rate: float = _DEFAULT_CHEES_TARGET_ACCEPTANCE_RATE,
    max_leapfrog_steps: int = _DEFAULT_CHEES_MAX_LEAPFROG_STEPS,
    optim_learning_rate: float = _DEFAULT_CHEES_OPTIM_LR,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run ``blackjax.chees_adaptation`` over ``num_chains`` chains jointly.

    Unlike ``stan_window`` which vmaps per-chain window adaptation independently,
    CHEES runs a **single** multi-chain adaptation call.  All chains participate
    together to adapt step size and trajectory length.

    Parameters
    ----------
    rng_key
        JAX random key for the adaptation run.  Passed directly to
        ``chees_adaptation.run``; split internally by CHEES.
    init_position
        Initial unconstrained parameter dict/array (one chain's worth).
        Replicated across ``num_chains`` via ``_maybe_replicate`` unless the
        caller pre-batches with a leading dim of ``num_chains``.
    n_warmup
        Number of adaptation steps.
    base_method
        ``BaseMethod`` entry (must be ``dynamic_hmc``; verified by compatibility
        guard).
    logdensity_fn
        BlackJAX-compatible log-density function.
    num_chains
        Number of chains for the joint CHEES adaptation.  Default ``4``.
    step_size
        Initial step size passed to CHEES.  CHEES will adapt it.  Default
        ``0.5`` (a reasonable starting point for most posteriors).
    target_acceptance_rate
        CHEES target acceptance rate.  Default ``0.651`` (upstream default).
    max_leapfrog_steps
        Maximum leapfrog steps per trajectory.  Default ``1000`` (upstream
        default).
    optim_learning_rate
        Learning rate for the optax Adam optimizer used to adapt trajectory
        length parameters.  Default ``0.01`` (standard CHEES default).
    **kwargs
        Additional keyword arguments (ignored; kept for API uniformity).

    Returns
    -------
    states
        Post-warmup ``DynamicHMCState`` pytree with leading dim ``num_chains``.
        ``states.position`` has shape ``(num_chains, d)``.
    adapted_params
        Dict with keys:

        - ``"step_size"``: shape ``(num_chains,)`` (scalar broadcast from
          CHEES shared estimate).
        - ``"inverse_mass_matrix"``: shape ``(num_chains, d)`` (broadcast from
          ``(d,)`` CHEES estimate).
        - ``"next_random_arg_fn"``: Python callable (CHEES-adapted; shared
          across chains; not broadcastable — passed through as-is).
        - ``"integration_steps_fn"``: Python callable (CHEES-adapted; shared
          across chains; not broadcastable — passed through as-is).
        - ``"integration_steps_params"``: shape ``(1,)`` array (CHEES-adapted
          trajectory length param; shared across chains; passed through as-is).
        - ``"_chees_target_acceptance_rate"``: Python float — value used.
        - ``"_chees_max_leapfrog_steps"``: Python int — cap used.

    Raises
    ------
    ValueError
        If ``base_method.name != "dynamic_hmc"`` (incompatibility guard).

    Notes
    -----
    CHEES upstream signature is different from MEADS: ``chees_adaptation.run``
    requires ``step_size`` and ``optim`` (optax optimizer) as positional
    arguments after ``positions``.  This wrapper creates an ``optax.adam``
    optimizer internally.  The callable params (``next_random_arg_fn``,
    ``integration_steps_fn``) are Python functions returned by CHEES and are
    passed through to the downstream kernel factory unchanged.
    """
    # Build CHEES adaptation object
    chees = blackjax.chees_adaptation(
        logdensity_fn,
        num_chains,
        target_acceptance_rate=target_acceptance_rate,
        max_leapfrog_steps=max_leapfrog_steps,
    )

    # Replicate init_position to (num_chains, *leaf.shape); pass-through if pre-batched.
    init_positions = _maybe_replicate(init_position, num_chains)

    # Build optax optimizer for trajectory-length adaptation (standard CHEES default).
    optim = optax.adam(learning_rate=optim_learning_rate)

    # Run CHEES: single call handles all num_chains chains jointly.
    # Returns (AdaptationResults, AdaptationInfo).
    # AdaptationResults.state: DynamicHMCState with leading dim num_chains.
    # AdaptationResults.parameters: dict with:
    #   step_size (scalar), inverse_mass_matrix (d,),
    #   next_random_arg_fn (callable), integration_steps_fn (callable),
    #   integration_steps_params (1,).
    (adaptation_results, _adaptation_info) = chees.run(
        rng_key, init_positions, step_size, optim, num_steps=n_warmup
    )

    states = adaptation_results.state
    raw_params = adaptation_results.parameters

    # Extract adapted scalar params and broadcast to (num_chains,) / (num_chains, d).
    # CHEES returns a single shared estimate:
    #   step_size → scalar ()
    #   inverse_mass_matrix → shape (d,)
    # Callable params (next_random_arg_fn, integration_steps_fn) cannot be
    # broadcast — they are passed through as Python functions (shared).
    step_size_scalar = jnp.asarray(raw_params["step_size"])
    raw_imm = raw_params["inverse_mass_matrix"]

    # Flatten the inverse_mass_matrix pytree to a single (d,) array.
    # For plain arrays this is a no-op.
    imm_leaves = jax.tree.leaves(raw_imm)
    if len(imm_leaves) == 1:
        imm_flat = imm_leaves[0]  # shape (d,)
    else:
        imm_flat = jnp.concatenate(
            [leaf.reshape(-1) for leaf in imm_leaves]
        )  # shape (d,)

    adapted_params: dict[str, Any] = {
        # Numeric params: broadcast to (num_chains,) / (num_chains, d)
        "step_size": jnp.broadcast_to(step_size_scalar, (num_chains,)),
        "inverse_mass_matrix": jnp.broadcast_to(
            imm_flat[None, :], (num_chains, imm_flat.shape[0])
        ),
        # Callable / non-broadcastable params: pass through as-is (shared across chains)
        "next_random_arg_fn": raw_params["next_random_arg_fn"],
        "integration_steps_fn": raw_params["integration_steps_fn"],
        "integration_steps_params": raw_params["integration_steps_params"],
        # Sidecar metadata
        "_chees_target_acceptance_rate": float(target_acceptance_rate),
        "_chees_max_leapfrog_steps": int(max_leapfrog_steps),
    }

    return states, adapted_params


ENTRY = Warmup(
    name="chees",
    runner=_runner,
    compatible_methods=("dynamic_hmc",),
    notes=(
        "CHEES (Change in Estimator of Step Size) adaptation for dynamic-HMC. "
        "Adapts both step_size AND the trajectory-length distribution (integration_steps_fn, "
        "next_random_arg_fn, integration_steps_params).  Like MEADS but for dynamic-HMC "
        "instead of GHMC; both are multi-chain-by-construction adaptation procedures. "
        "Upstream API note: chees_adaptation.run() requires step_size and an optax optimizer "
        "as positional args (unlike meads_adaptation.run); this wrapper provides optax.adam "
        "internally.  Callable params (next_random_arg_fn, integration_steps_fn) are Python "
        "functions returned by CHEES and passed through unchanged — they cannot be broadcast "
        "to (num_chains,) shape.  numeric params (step_size, inverse_mass_matrix) are "
        "broadcast from the shared CHEES estimate to (num_chains,) / (num_chains, d). "
        "dynamic_hmc-specific; not compatible with HMC, NUTS, GHMC, or any other kernel. "
        "multi-chain by default (num_chains=4); target_acceptance_rate=0.651 (CHEES default)."
    ),
)
