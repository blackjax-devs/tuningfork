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

This warmup runs one independent Pathfinder L-BFGS optimisation per chain
(vmapped over ``(key, init_position)`` pairs), draws one initial position from
the per-chain variational surrogate, and derives a diagonal inverse mass matrix
from the per-chain surrogate covariance (the ``alpha`` field of the
``PathfinderState``).

**No step_size adaptation** is performed.  A scalar default of ``1.0`` is
returned for every chain.  The downstream sampler should rely on its own
dual-averaging adaptation (e.g. NUTS's window adaptation) or Bayesian
optimisation to tune the step size; this warmup only provides a better
initialisation than a flat prior sample.

Runner signature (multi-chain contract)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, step_size_default=1.0,
            num_chains: int = 4, **kwargs)
    -> (states, adapted_params)

Where:

- ``rng_key`` is a single key; split internally into ``num_chains`` keys.
- ``init_position`` is a single pytree (one chain's worth); replicated
  across chains internally via ``_maybe_replicate`` unless the caller
  pre-batches it (leading dim == ``num_chains``).
- ``states`` is a batched pytree with leading dim ``num_chains``;
  ``states.position`` has shape ``(num_chains, d)``.
- ``adapted_params`` contains:

  =============================================  =================  ====================================
  Key                                            Shape              Notes
  =============================================  =================  ====================================
  ``step_size``                                  ``(num_chains,)``  Constant ``step_size_default`` per chain
  ``inverse_mass_matrix``                        ``(num_chains, d)``  Diagonal of per-chain L-BFGS inv-Hessian (``alpha`` field)
  ``_pathfinder_logZ_estimate``                  ``(num_chains,)``  Per-chain ELBO (sidecar — routes to calibration_budget, not base_method_params)
  =============================================  =================  ====================================

Sidecar keys (underscore prefix) are metadata for ``calibration_budget``
and are NOT forwarded to the base-method kernel as hyperparameters.

Compatible with: ``nuts``, ``hmc``, ``mala``, ``rwm``, ``barker``.
NOT compatible with ``mclmc`` (different geometry — microcanonical momentum,
no Gaussian inverse mass matrix in the HMC sense).

Upstream API (BlackJAX >= 0.9.x):

- ``blackjax.pathfinder.approximate(rng_key, logdensity_fn, initial_position,
  num_samples=200, ...)``
  → ``(PathfinderState, PathfinderInfo)``
- ``PathfinderState._fields``: ``('elbo', 'position', 'grad_position',
  'alpha', 'beta', 'gamma')``
  where ``alpha`` is shape ``(d,)`` — the diagonal of the L-BFGS
  inverse Hessian approximation.
- Positions inside ``PathfinderState`` retain the original pytree structure
  (e.g. a dict of arrays).  We use ``jax.flatten_util.ravel_pytree`` to
  convert to/from a flat ``(d,)`` array for ``bfgs_sample``.

Positions are drawn via the L-BFGS surrogate (``bfgs_sample``) to produce
geometrically informed starting points that are typically much closer to
the posterior than a flat prior draw.
"""

from typing import Any

import blackjax
import jax
import jax.numpy as jnp
from blackjax.vi.pathfinder import bfgs_sample
from jax.flatten_util import ravel_pytree

from bjx_bench.inference.warmup._base import Warmup, _maybe_replicate

__all__ = ["ENTRY"]

# Algorithms that accept an inverse_mass_matrix and are therefore compatible
# with Pathfinder's init-and-IMM output.  mclmc is excluded: its geometry
# is microcanonical and its inverse_mass_matrix is a diagonal preconditioner
# for Euclidean distance, not an HMC-style covariance.
_COMPATIBLE = ("nuts", "hmc", "mala", "rwm", "barker")


def _runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,  # noqa: ARG001 — accepted for interface uniformity; not used
    base_method: Any,  # BaseMethod; not imported to avoid circular dep
    *,
    logdensity_fn: Any,
    step_size_default: float = 1.0,
    num_chains: int = 4,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run single-path Pathfinder independently per chain via ``jax.vmap``.

    For each chain, one Pathfinder L-BFGS run is performed from the chain's
    initial position.  A single init position is drawn from the per-chain
    variational surrogate.  The diagonal of the per-chain L-BFGS inverse
    Hessian approximation (``PathfinderState.alpha``) is returned as the
    ``inverse_mass_matrix``.

    Parameters
    ----------
    rng_key
        JAX random key.  Split internally into ``num_chains`` independent
        per-chain keys, each used for one Pathfinder run and one draw.
    init_position
        Initial unconstrained parameter pytree (one chain's worth).  The
        runner replicates it to ``(num_chains, ...)`` unless the caller
        pre-batches it (leading dim == ``num_chains``).
    n_warmup
        Accepted for interface uniformity; not used.  Pathfinder's
        optimisation budget is controlled by ``**kwargs`` (e.g. ``maxiter``).
    base_method
        ``BaseMethod`` entry.  Used for kernel-state initialisation after
        the Pathfinder run.
    logdensity_fn
        BlackJAX-compatible log-density function.
    step_size_default
        Constant step size assigned to every chain.  Default ``1.0``.
        The downstream sampler should adapt this via dual-averaging.
    num_chains
        Number of independent chains to initialise.  Default ``4``.
        The returned ``states`` have leading dim ``num_chains``.
    **kwargs
        Forwarded to ``blackjax.pathfinder.approximate`` (e.g. ``maxiter``,
        ``maxcor``, ``num_samples``).

    Returns
    -------
    states
        Post-Pathfinder kernel states, batched over ``num_chains``.
        ``states.position`` has shape ``(num_chains, d)`` (or ``(num_chains, ...)``
        for dict/pytree positions).
        The position of each chain is a single draw from the per-chain
        variational surrogate.
    adapted_params
        Dict with:

        - ``"step_size"``: ``(num_chains,)`` array, constant ``step_size_default``.
        - ``"inverse_mass_matrix"``: ``(num_chains, d)``, per-chain ``alpha``
          (diagonal of L-BFGS inverse Hessian).
        - ``"_pathfinder_logZ_estimate"``: ``(num_chains,)``, per-chain ELBO
          (sidecar metadata; underscore prefix marks it as non-HP).

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

    # Build the unravel function from a SINGLE-chain position.
    # If init_position is pre-batched (leading dim == num_chains), extract
    # one chain's worth so that ravel_pytree gives the right (d,) unravel fn.
    _leaves = jax.tree.leaves(init_position)
    _is_prebatched = bool(
        _leaves and _leaves[0].shape and _leaves[0].shape[0] == num_chains
    )
    if _is_prebatched:
        _single_pos = jax.tree.map(lambda x: x[0], init_position)
    else:
        _single_pos = init_position
    _dummy_flat, unravel_fn = ravel_pytree(_single_pos)
    d = int(_dummy_flat.shape[0])

    # Split key into per-chain Pathfinder keys and per-chain sampling keys.
    pf_key, sample_key = jax.random.split(rng_key)
    chain_pf_keys = jax.random.split(pf_key, num_chains)
    chain_sample_keys = jax.random.split(sample_key, num_chains)

    # Replicate init_position across chains.  Pass-through if pre-batched.
    init_positions = _maybe_replicate(init_position, num_chains)

    # Pull Pathfinder-specific kwargs; remainder is forwarded to .approximate().
    num_samples = kwargs.pop("num_samples", 200)
    pf_kwargs: dict[str, Any] = kwargs

    @jax.vmap
    def run_one_pathfinder(
        pf_k: jax.Array, sample_k: jax.Array, x0: Any
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Run Pathfinder once; return (flat_init_pos, alpha, elbo)."""
        pf_state, _info = blackjax.pathfinder.approximate(
            pf_k, logdensity_fn, x0, num_samples=num_samples, **pf_kwargs
        )
        # Ravel the pytree position to a flat (d,) array for bfgs_sample.
        flat_pos, _ = ravel_pytree(pf_state.position)
        flat_grad, _ = ravel_pytree(pf_state.grad_position)
        # Draw one init position from the variational surrogate.
        # bfgs_sample returns (samples: (1, d), logq: (1,)).
        drawn_flat, _logq = bfgs_sample(
            sample_k,
            1,
            flat_pos,
            flat_grad,
            pf_state.alpha,
            pf_state.beta,
            pf_state.gamma,
        )
        # drawn_flat: (1, d) → take first sample.
        init_pos_flat = drawn_flat[0]  # (d,)
        return init_pos_flat, pf_state.alpha, pf_state.elbo

    flat_init_positions, alpha_per_chain, elbo_per_chain = run_one_pathfinder(
        chain_pf_keys, chain_sample_keys, init_positions
    )
    # flat_init_positions: (num_chains, d)
    # alpha_per_chain:     (num_chains, d)
    # elbo_per_chain:      (num_chains,)

    # Convert flat (num_chains, d) positions back to the original pytree
    # structure: apply unravel_fn independently for each chain.
    init_positions_pytree = jax.vmap(unravel_fn)(flat_init_positions)
    # init_positions_pytree: pytree with leading dim num_chains

    # Build the kernel with neutral HPs for state initialisation.
    # The actual IMM comes from adapted_params; we inject an identity here
    # just to satisfy the kernel constructor.
    init_defaults: dict[str, Any] = {}
    if base_method.needs_mass_matrix:
        init_defaults["inverse_mass_matrix"] = jnp.ones(d)

    from bjx_bench.calibration.tier_b import default_value_for_space

    for space in base_method.default_hp_space:
        if space.name not in ("step_size", "inverse_mass_matrix"):
            if space.name not in init_defaults:
                init_defaults[space.name] = default_value_for_space(space)
    init_defaults.setdefault("step_size", step_size_default)

    kernel = base_method.factory(logdensity_fn, **init_defaults)

    @jax.vmap
    def init_one(pos: Any) -> Any:
        return kernel.init(pos)

    states = init_one(init_positions_pytree)

    adapted_params: dict[str, Any] = {
        "step_size": jnp.full((num_chains,), step_size_default),
        "inverse_mass_matrix": alpha_per_chain,  # (num_chains, d)
        "_pathfinder_logZ_estimate": elbo_per_chain,  # (num_chains,)
    }

    return states, adapted_params


ENTRY = Warmup(
    name="pathfinder",
    runner=_runner,
    compatible_methods=_COMPATIBLE,
    notes=(
        "Single-path Pathfinder warmup: runs one independent L-BFGS "
        "Pathfinder optimisation per chain via jax.vmap.  Draws one init "
        "position from each chain's variational surrogate and returns the "
        "per-chain L-BFGS inverse-Hessian diagonal (alpha) as the "
        "inverse_mass_matrix.  No step_size adaptation: returns a constant "
        "scalar default (1.0) per chain.  Sidecar: "
        "_pathfinder_logZ_estimate (per-chain ELBO).  "
        "Compatible: nuts, hmc, mala, rwm, barker.  "
        "NOT compatible with mclmc (microcanonical geometry)."
    ),
)
