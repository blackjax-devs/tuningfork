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
"""Single-path Pathfinder warmup: init-and-IMM provider via variational surrogate.

This warmup is a thin shim around ``blackjax.pathfinder_adaptation`` with
``num_chains`` chains and a single L-BFGS path (``n_paths=1`` equivalent via
``effective_n_paths == 1`` dispatch in upstream).

**Breaking change from pre-PR-B contract**: ``inverse_mass_matrix`` is now
dense ``(num_chains, d, d)`` (broadcast from the shared ``(d, d)`` L-BFGS
inverse Hessian) instead of the old diagonal ``(num_chains, d)`` form
(which returned the ``alpha`` field = diagonal of the L-BFGS approximation).
This matches the upstream blackjax PR #919 uniform dense-IMM contract.

Runner signature (multi-chain contract)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, step_size_default=1.0,
            num_chains=4, **kwargs)
    -> (states, adapted_params)

Where:

- ``rng_key`` is a single key.
- ``init_position`` is a single pytree (one chain's worth).
- ``states`` is a batched pytree with leading dim ``num_chains``.
- ``adapted_params`` contains:

  =============================================  ====================  ======================================
  Key                                            Shape                 Notes
  =============================================  ====================  ======================================
  ``step_size``                                  ``(num_chains,)``     Per-chain adapted step size from DA
  ``inverse_mass_matrix``                        ``(num_chains, d, d)``  Dense IMM broadcast per chain
  =============================================  ====================  ======================================

Sidecar keys (underscore prefix) are metadata for ``calibration_budget``
and are NOT forwarded to the base-method kernel as hyperparameters.

Compatible with: ``nuts``, ``hmc``, ``mala``, ``rwm``, ``barker``.
NOT compatible with ``mclmc`` (different geometry — microcanonical momentum,
no Gaussian inverse mass matrix in the HMC sense).
"""

from typing import Any

import blackjax
import jax
import jax.numpy as jnp

from tuningfork.warmup._base import Warmup

__all__ = ["ENTRY"]

# Algorithms compatible with HMC-style inverse_mass_matrix.
_COMPATIBLE = ("nuts", "hmc", "mala", "rwm", "barker")


def _runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,
    base_method: Any,  # BaseMethod; not imported to avoid circular dep
    *,
    logdensity_fn: Any,
    step_size_default: float = 1.0,
    num_chains: int = 4,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run single-path Pathfinder via blackjax.pathfinder_adaptation.

    Thin shim around ``blackjax.pathfinder_adaptation(num_chains=num_chains,
    n_paths=1)`` (single L-BFGS path broadcast across all chains).

    Parameters
    ----------
    rng_key
        JAX random key.
    init_position
        Initial unconstrained parameter pytree (one chain's worth).
    n_warmup
        Number of dual-averaging adaptation steps per chain.
    base_method
        ``BaseMethod`` entry.  Used for compatibility check and extra HP defaults.
    logdensity_fn
        BlackJAX-compatible log-density function.
    step_size_default
        Initial step size for dual-averaging.  Default ``1.0``.
    num_chains
        Number of independent chains.  Default ``4``.
    **kwargs
        Additional keyword arguments forwarded to ``pathfinder_adaptation``
        (e.g. ``maxiter``, ``maxcor`` for the L-BFGS optimiser).

    Returns
    -------
    states
        Post-adaptation BlackJAX kernel states, batched over ``num_chains``.
        ``states.position`` has shape ``(num_chains, d)`` or
        ``(num_chains, ...)`` for dict-based positions.
    adapted_params
        Dict with:

        - ``"step_size"``: ``(num_chains,)`` per-chain adapted step size.
        - ``"inverse_mass_matrix"``: ``(num_chains, d, d)`` dense IMM,
          broadcast from the shared ``(d, d)`` L-BFGS inverse Hessian.

    Raises
    ------
    ValueError
        If ``base_method.name`` is not in the compatible set.
    """
    if base_method.name not in _COMPATIBLE:
        raise ValueError(
            f"pathfinder warmup is not compatible with base_method "
            f"{base_method.name!r}; compatible: {_COMPATIBLE}"
        )

    # Build extra kwargs for the algorithm (e.g. num_integration_steps for HMC).
    from tuningfork.base_method import default_value_for_space

    extra_kwargs: dict[str, Any] = dict(kwargs)
    for space in base_method.default_hp_space:
        if space.name not in ("step_size", "inverse_mass_matrix"):
            if space.name not in extra_kwargs:
                extra_kwargs[space.name] = default_value_for_space(space)

    # n_paths=1 forces PATH B (multichain single-path) in pathfinder_adaptation:
    # upstream computes effective_n_paths = n_paths if n_paths is not None else num_chains,
    # so n_paths=None with num_chains=4 would give effective_n_paths=4 → PATH C
    # (multipathfinder), which calls _psis_weighted_mixture_covariance and fails
    # on dict-position pytrees.  n_paths=1 always forces PATH A (num_chains=1)
    # or PATH B (num_chains>1) — single L-BFGS fit, broadcast to all chains.
    adaptation = blackjax.pathfinder_adaptation(
        base_method.factory,
        logdensity_fn,
        num_chains=num_chains,
        n_paths=1,  # explicit: effective_n_paths=1 → PATH A/B, never PATH C
        initial_step_size=step_size_default,
        **extra_kwargs,
    )

    results, _info = adaptation.run(rng_key, init_position, num_steps=n_warmup)
    # results.state:
    #   num_chains=1 → PATH A → single-chain state (NO leading batch dim)
    #   num_chains>1 → PATH B → vmapped state (leading dim = num_chains)
    # results.parameters["step_size"]: scalar (PATH A) or (num_chains,) (PATH B)
    # results.parameters["inverse_mass_matrix"]: (d, d) shared

    state = results.state
    if num_chains == 1:
        # PATH A returns an unbatched single-chain state; add the leading batch
        # dim of 1 so the multi-chain contract is uniformly satisfied.
        state = jax.tree.map(lambda x: x[None], state)

    shared_imm = results.parameters["inverse_mass_matrix"]  # (d, d)
    # Broadcast shared IMM to (num_chains, d, d) to match per-chain contract.
    imm_per_chain = jnp.broadcast_to(
        shared_imm[None, :, :], (num_chains,) + shared_imm.shape
    )

    # Normalise step_size to always be (num_chains,).
    step_size_raw = results.parameters["step_size"]
    step_size = (
        jnp.full((num_chains,), step_size_raw)
        if jnp.asarray(step_size_raw).ndim == 0
        else jnp.asarray(step_size_raw)
    )

    adapted_params: dict[str, Any] = {
        "step_size": step_size,  # (num_chains,)
        "inverse_mass_matrix": imm_per_chain,  # (num_chains, d, d)
    }

    return state, adapted_params


ENTRY = Warmup(
    name="pathfinder",
    runner=_runner,
    compatible_methods=_COMPATIBLE,
    notes=(
        "Single-path Pathfinder warmup (thin shim around blackjax.pathfinder_adaptation "
        "with num_chains and n_paths=None — single L-BFGS path). "
        "Adapts step size via dual-averaging over n_warmup steps. "
        "Returns per-chain adapted step_size (num_chains,) and dense (num_chains, d, d) "
        "IMM broadcast from the shared L-BFGS inverse Hessian. "
        "NOTE: IMM is now dense (num_chains, d, d); old diagonal (num_chains, d) "
        "contract changed in PR B (warmup-collapse-pathfinder-shims). "
        "Compatible: nuts, hmc, mala, rwm, barker. "
        "NOT compatible with mclmc (microcanonical geometry)."
    ),
)
