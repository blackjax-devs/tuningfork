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
"""MEADS (Maximum-Eigenvalue Adapted Dual-Averaging Step-size) warmup,
wrapping ``blackjax.meads_adaptation``.

MEADS is a **GHMC-specific** adaptation routine that simultaneously tunes
``step_size``, ``momentum_inverse_scale`` (the inverse mass matrix diagonal),
``alpha`` (momentum persistence), and ``delta`` (slice-sampling parameter).
Unlike ``stan_window`` which vmaps per-chain window adaptation, **MEADS runs a
single multi-chain adaptation internally**: chains are cross-validated across
``num_folds`` folds to estimate the maximum eigenvalue of the target
covariance, which drives the step-size schedule.  Chains are *inputs*, not
loop iterations.

Compatibility
-------------
MEADS is paired exclusively with GHMC (``compatible_methods=("ghmc",)``).
Passing any other base method raises ``ValueError``.

Multi-chain constraint
----------------------
``blackjax.meads_adaptation`` requires ``num_chains ≥ num_folds`` (default
``num_folds=4``).  The runner enforces this and raises ``ValueError`` if
violated.  Recommended: use the default ``num_chains=4`` so that
``num_chains == num_folds``, or any multiple thereof.

Runner signature (multi-chain contract, P5.0c)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, num_chains: int = 4, num_folds: int = 4,
            step_size_multiplier: float = 0.5,
            damping_slowdown: float = 1.0, **kwargs)
    -> (states, adapted_params)

Where:

- ``rng_key`` is a single key passed directly to ``meads_adaptation.run``.
- ``init_position`` is a single pytree (one chain's worth); replicated to
  ``(num_chains, d)`` via ``_maybe_replicate`` unless pre-batched.
- ``states`` is a batched ``GHMCState`` pytree with leading dim ``num_chains``.
- ``adapted_params`` contains the MEADS-adapted keys broadcast to
  ``num_chains`` shape:

  ========================= ===========================================
  Key                       Shape
  ========================= ===========================================
  ``step_size``             ``(num_chains,)`` — scalar broadcast
  ``momentum_inverse_scale``  ``(num_chains, d)`` — broadcast from ``(d,)``
  ``alpha``                 ``(num_chains,)`` — scalar broadcast
  ``delta``                 ``(num_chains,)`` — scalar broadcast
  ``_meads_num_folds``      ``int`` — number of folds used
  ========================= ===========================================

Upstream note
-------------
``blackjax.meads_adaptation`` returns a single (not per-chain) set of adapted
parameters: ``step_size`` is a scalar, ``momentum_inverse_scale`` has shape
``(d,)``, and ``alpha``/``delta`` are scalars.  This wrapper broadcasts all
scalar parameters to ``(num_chains,)`` and tiles the mass matrix to
``(num_chains, d)`` so downstream callers see the standard multi-chain shape.
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
    num_folds: int = 4,
    step_size_multiplier: float = 0.5,
    damping_slowdown: float = 1.0,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run ``blackjax.meads_adaptation`` over ``num_chains`` chains jointly.

    Unlike ``stan_window`` which vmaps per-chain window adaptation independently,
    MEADS runs a **single** multi-chain adaptation call.  All chains participate
    together to cross-validate across folds.

    Parameters
    ----------
    rng_key
        JAX random key for the adaptation run.  Passed directly to
        ``meads_adaptation.run``; split internally by MEADS.
    init_position
        Initial unconstrained parameter dict/array (one chain's worth).
        Replicated across ``num_chains`` via ``_maybe_replicate`` unless the
        caller pre-batches with a leading dim of ``num_chains``.
    n_warmup
        Number of adaptation steps.
    base_method
        ``BaseMethod`` entry (must be ``ghmc``; verified by compatibility guard).
    logdensity_fn
        BlackJAX-compatible log-density function.
    num_chains
        Number of chains for the joint MEADS adaptation.  Must be ≥
        ``num_folds``.  Default ``4`` matches ``num_folds`` default.
    num_folds
        Number of folds for cross-validation inside MEADS.  Must be ≤
        ``num_chains``.  Default ``4`` (MEADS upstream default).
    step_size_multiplier
        Multiplier applied to the estimated maximum eigenvalue to set the
        initial step size.  See upstream ``blackjax.meads_adaptation`` docs.
    damping_slowdown
        Slows down the damping adaptation.  See upstream docs.
    **kwargs
        Additional keyword arguments (ignored; kept for API uniformity).

    Returns
    -------
    states
        Post-warmup ``GHMCState`` pytree with leading dim ``num_chains``.
        ``states.position`` has shape ``(num_chains, d)``.
    adapted_params
        Dict with keys:
        - ``"step_size"``: shape ``(num_chains,)`` (scalar broadcast).
        - ``"momentum_inverse_scale"``: shape ``(num_chains, d)`` (tiled).
        - ``"alpha"``: shape ``(num_chains,)`` (scalar broadcast).
        - ``"delta"``: shape ``(num_chains,)`` (scalar broadcast).
        - ``"_meads_num_folds"``: Python int (number of folds used).

    Raises
    ------
    ValueError
        If ``num_chains < num_folds``.
    ValueError
        If ``base_method.name != "ghmc"`` (incompatibility guard).
    """
    if num_chains < num_folds:
        raise ValueError(
            f"meads warmup: num_chains ({num_chains}) must be ≥ num_folds "
            f"({num_folds}).  MEADS requires at least one chain per fold for "
            f"cross-validation.  Either increase num_chains or decrease num_folds."
        )

    # Build MEADS adaptation object
    meads = blackjax.meads_adaptation(
        logdensity_fn,
        num_chains,
        num_folds=num_folds,
        step_size_multiplier=step_size_multiplier,
        damping_slowdown=damping_slowdown,
    )

    # Replicate init_position to (num_chains, *leaf.shape); pass-through if pre-batched.
    # _maybe_replicate preserves the pytree structure (dict, array, etc.) — MEADS .run()
    # accepts any ArrayLikeTree with a leading num_chains dimension, matching the
    # logdensity_fn signature.
    init_positions = _maybe_replicate(init_position, num_chains)

    # Run MEADS: single call handles all num_chains chains jointly.
    # Returns (AdaptationResults, AdaptationInfo).
    # AdaptationResults.state: GHMCState with leading dim num_chains.
    # AdaptationResults.parameters: dict with scalar step_size, (d,) momentum_inverse_scale,
    #                                scalar alpha, scalar delta.
    (adaptation_results, _adaptation_info) = meads.run(
        rng_key, init_positions, num_steps=n_warmup
    )

    states = adaptation_results.state
    raw_params = adaptation_results.parameters

    # Extract raw (shared) adapted parameters from MEADS.
    # MEADS returns a single shared estimate:
    #   step_size, alpha, delta → scalars ()
    #   momentum_inverse_scale  → pytree matching position structure, e.g.
    #                              {"x": array(d,)} or plain array(d,).
    # Broadcast all to (num_chains,) / (num_chains, d) to satisfy the
    # multi-chain contract.
    step_size_scalar = jnp.asarray(raw_params["step_size"])
    alpha_scalar = jnp.asarray(raw_params["alpha"])
    delta_scalar = jnp.asarray(raw_params["delta"])
    raw_imm = raw_params["momentum_inverse_scale"]

    # Flatten the momentum_inverse_scale pytree to a single (d,) array.
    # For plain arrays this is a no-op; for dict-based positions it
    # concatenates all leaves along the last axis.
    imm_leaves = jax.tree.leaves(raw_imm)
    if len(imm_leaves) == 1:
        imm_flat = imm_leaves[0]  # shape (d,)
    else:
        imm_flat = jnp.concatenate(
            [leaf.reshape(-1) for leaf in imm_leaves]
        )  # shape (d,)

    adapted_params: dict[str, Any] = {
        "step_size": jnp.broadcast_to(step_size_scalar, (num_chains,)),
        "momentum_inverse_scale": jnp.broadcast_to(
            imm_flat[None, :], (num_chains, imm_flat.shape[0])
        ),
        "alpha": jnp.broadcast_to(alpha_scalar, (num_chains,)),
        "delta": jnp.broadcast_to(delta_scalar, (num_chains,)),
        "_meads_num_folds": num_folds,
    }

    return states, adapted_params


ENTRY = Warmup(
    name="meads",
    runner=_runner,
    compatible_methods=("ghmc",),
    notes=(
        "MEADS (Maximum-Eigenvalue Adapted Dual-Averaging Step-size) warmup for GHMC. "
        "Unlike stan_window which vmaps per-chain adaptation, MEADS runs a single "
        "multi-chain adaptation that cross-validates across num_folds folds; chains "
        "are inputs, not loop iterations.  Requires num_chains ≥ num_folds (default 4). "
        "Adapts step_size, momentum_inverse_scale, alpha, and delta jointly. "
        "GHMC-specific; not compatible with HMC, NUTS, or any other kernel. "
        "P5.5: multi-chain by default (num_chains=4); adapted_params broadcast from "
        "MEADS scalar output to (num_chains,) shape for contract uniformity."
    ),
)
