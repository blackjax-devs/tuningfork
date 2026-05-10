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
"""Multi-path Pathfinder warmup: init-and-IMM provider via PSIS importance resampling.

This warmup runs one ``blackjax.multipathfinder`` (multi-path Pathfinder) fit
that starts from ``n_paths`` independent initial positions.  It then draws
``num_chains`` init positions from the mixture via Pareto-Smoothed Importance
Sampling (PSIS) resampling, and derives a single diagonal inverse mass matrix
from the post-PSIS empirical covariance (replicated identically to each chain).

The PSIS diagnostic (Pareto-k) is returned as a sidecar key for diagnostics.
A Pareto-k < 0.5 indicates reliable IS; k > 0.7 may indicate unreliable
importance weights — downstream calibration code may use this to warn or
fall back to ``pathfinder`` (single-path).

Runner signature (multi-chain contract, P5.0c)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, n_paths=None, num_samples_per_path=200,
            n_imm_samples=2000, step_size_default=1.0,
            num_chains: int = 4, **kwargs)
    -> (states, adapted_params)

Where:

- ``rng_key`` is a single key; used for multi-path Pathfinder + PSIS resampling.
- ``init_position`` is a single pytree (one chain's worth); replicated to
  ``(n_paths, ...)`` for the multi-path fit unless pre-batched
  (leading dim == ``n_paths``).
- ``states`` is a batched pytree with leading dim ``num_chains``;
  ``states.position`` has shape ``(num_chains, d)`` or ``(num_chains, ...)``
  for dict-based positions.
- ``adapted_params`` contains:

  ==================================  =================  ==================================
  Key                                 Shape              Notes
  ==================================  =================  ==================================
  ``step_size``                       ``(num_chains,)``  Constant ``step_size_default`` per chain
  ``inverse_mass_matrix``             ``(num_chains, d)``  Post-PSIS empirical variance, same value per chain
  ``_multipathfinder_psis_pareto_k``  scalar             PSIS Pareto-k diagnostic
  ==================================  =================  ==================================

Sidecar keys (underscore prefix) are metadata for ``calibration_budget``
and are NOT forwarded to the base-method kernel as hyperparameters.

Compatible with: ``nuts``, ``hmc``, ``mala``, ``rwm``, ``barker``.
NOT compatible with ``mclmc`` (different geometry — microcanonical momentum,
no Gaussian inverse mass matrix in the HMC sense).

Upstream API (BlackJAX >= 0.9.x):

- ``blackjax.multipathfinder(logdensity_fn)`` returns a ``VIAlgorithm``
  with ``init``, ``step``, ``sample`` methods.
- ``mpf.init(rng_key, initial_positions, num_samples=200)``
  → ``(MultipathfinderState, PathfinderInfo)``
  where ``MultipathfinderState._fields = ('path_states', 'samples', 'logp', 'logq')``.
  ``samples`` has shape ``(n_paths, num_samples, ...)`` (pytree-structured).
- ``psis_weights(state)`` → ``(log_weights, pareto_k)``
  where ``log_weights`` has shape ``(n_paths * num_samples,)``.

The IMM (inverse mass matrix) is the post-PSIS empirical variance:
we draw ``n_imm_samples`` positions from the PSIS mixture and compute
``jnp.var(flat_draws, axis=0)``, where ``flat_draws`` has shape
``(n_imm_samples, d)`` and ``d`` is the total number of parameters.
The same ``(d,)`` vector is replicated to each chain so all chains share
the same preconditioner.
"""

from typing import Any

import blackjax
import jax
import jax.numpy as jnp
from blackjax.vi.multipathfinder import psis_weights
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
    n_paths: int | None = None,
    num_samples_per_path: int = 200,
    n_imm_samples: int = 2000,
    step_size_default: float = 1.0,
    num_chains: int = 4,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run multi-path Pathfinder and draw init positions via PSIS resampling.

    A single multi-path Pathfinder fit is performed from ``n_paths`` independent
    starting positions.  ``num_chains`` init positions are then drawn from the
    importance-resampled mixture.  The post-PSIS empirical covariance diagonal
    is returned as the ``inverse_mass_matrix`` (same value replicated to each
    chain).

    Parameters
    ----------
    rng_key
        JAX random key.  Used for the multi-path Pathfinder fit and for
        PSIS resampling.
    init_position
        Initial unconstrained parameter pytree (one chain's worth).  The
        runner replicates it to ``(n_paths, ...)`` for the multi-path fit
        unless the caller pre-batches it (leading dim == ``n_paths``).
    n_warmup
        Accepted for interface uniformity; not used.  Pathfinder's
        optimisation budget is controlled by ``num_samples_per_path``
        and ``**kwargs``.
    base_method
        ``BaseMethod`` entry.  Used for kernel-state initialisation.
    logdensity_fn
        BlackJAX-compatible log-density function.
    n_paths
        Number of independent L-BFGS paths for the multi-path Pathfinder
        run.  Defaults to ``num_chains`` (one path per chain).
    num_samples_per_path
        Number of samples drawn per path to estimate ELBO / PSIS weights.
        Default ``200``.
    n_imm_samples
        Number of PSIS-resampled draws used to estimate the post-PSIS
        empirical covariance for the ``inverse_mass_matrix``.  Default ``2000``.
    step_size_default
        Constant step size assigned to every chain.  Default ``1.0``.
    num_chains
        Number of independent chains to initialise.  Default ``4``.
    **kwargs
        Forwarded to the multi-path Pathfinder ``init`` call (e.g.
        ``maxiter``, ``maxcor``).

    Returns
    -------
    states
        Post-Pathfinder kernel states, batched over ``num_chains``.
        ``states.position`` has shape ``(num_chains, d)`` or
        ``(num_chains, ...)`` for dict-based positions.
        Each chain's position is drawn from the PSIS-resampled mixture.
    adapted_params
        Dict with:

        - ``"step_size"``: ``(num_chains,)`` constant ``step_size_default``.
        - ``"inverse_mass_matrix"``: ``(num_chains, d)``, post-PSIS empirical
          variance replicated across chains.
        - ``"_multipathfinder_psis_pareto_k"``: scalar PSIS Pareto-k diagnostic.

    Raises
    ------
    ValueError
        If ``base_method.name`` is not in the compatible set.
    """
    if base_method.name not in _COMPATIBLE:
        raise ValueError(
            f"multipathfinder warmup is not compatible with base_method "
            f"{base_method.name!r}; compatible: {_COMPATIBLE}"
        )

    if n_paths is None:
        n_paths = num_chains

    # Build the unravel function from a SINGLE-chain position.
    # If init_position is pre-batched (leading dim == n_paths), extract one
    # chain's worth so that ravel_pytree gives the right (d,) unravel fn.
    _leaves = jax.tree.leaves(init_position)
    _is_prebatched = bool(
        _leaves and _leaves[0].shape and _leaves[0].shape[0] == n_paths
    )
    if _is_prebatched:
        _single_pos = jax.tree.map(lambda x: x[0], init_position)
    else:
        _single_pos = init_position
    _dummy_flat, unravel_fn = ravel_pytree(_single_pos)
    d = int(_dummy_flat.shape[0])

    # Split key: Pathfinder, chain resampling, IMM estimation.
    pf_key, resample_key, imm_key = jax.random.split(rng_key, 3)

    # Replicate init_position to (n_paths, ...) for the multi-path fit.
    # _maybe_replicate passes through if already pre-batched (leading dim == n_paths).
    init_positions = _maybe_replicate(init_position, n_paths)

    # Run multi-path Pathfinder.
    # mpf.init returns (MultipathfinderState, PathfinderInfo).
    mpf = blackjax.multipathfinder(logdensity_fn)
    mpf_state, _pf_info = mpf.init(
        pf_key, init_positions, num_samples=num_samples_per_path, **kwargs
    )
    # mpf_state.samples: (n_paths, num_samples_per_path, ...) — pytree structured.

    # Compute PSIS importance weights.
    log_weights, pareto_k = psis_weights(mpf_state)
    # log_weights: (n_paths * num_samples_per_path,)
    # pareto_k: scalar

    # Flatten sample pool to (n_paths * num_samples_per_path, ...).
    samples_flat = jax.tree.map(
        lambda x: x.reshape(-1, *x.shape[2:]), mpf_state.samples
    )
    # samples_flat is still a pytree; leaves have shape (n_paths*num_samples, ...).

    total_pool = log_weights.shape[0]
    probs = jnp.exp(log_weights)

    # Draw num_chains init positions via PSIS resampling.
    init_indices = jax.random.choice(
        resample_key, total_pool, shape=(num_chains,), replace=True, p=probs
    )
    init_from_psis = jax.tree.map(lambda x: x[init_indices], samples_flat)
    # init_from_psis: pytree with leading dim num_chains.

    # Estimate post-PSIS empirical covariance diagonal for the IMM.
    # Draw a large sample from the PSIS mixture and compute per-parameter variance.
    imm_indices = jax.random.choice(
        imm_key, total_pool, shape=(n_imm_samples,), replace=True, p=probs
    )
    imm_samples_pytree = jax.tree.map(lambda x: x[imm_indices], samples_flat)
    # Flatten each sample to (d,) for variance estimation.
    flat_imm_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(imm_samples_pytree)
    # flat_imm_samples: (n_imm_samples, d)

    # Empirical variance diagonal; clamp to avoid near-zero entries.
    imm_diag = jnp.var(flat_imm_samples, axis=0)  # (d,)
    imm_diag = jnp.clip(imm_diag, min=1e-6)

    # Replicate the shared IMM to each chain: (num_chains, d).
    imm_per_chain = jnp.broadcast_to(imm_diag[None, :], (num_chains, d))

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

    states = init_one(init_from_psis)

    adapted_params: dict[str, Any] = {
        "step_size": jnp.full((num_chains,), step_size_default),
        "inverse_mass_matrix": imm_per_chain,  # (num_chains, d)
        "_multipathfinder_psis_pareto_k": pareto_k,  # scalar
    }

    return states, adapted_params


ENTRY = Warmup(
    name="multipathfinder",
    runner=_runner,
    compatible_methods=_COMPATIBLE,
    notes=(
        "Multi-path Pathfinder warmup (P5.4): runs one multi-path Pathfinder "
        "fit from n_paths independent starting positions (default: n_paths == "
        "num_chains).  Draws num_chains init positions from the PSIS "
        "importance-resampled mixture.  Returns the post-PSIS empirical "
        "covariance diagonal as the inverse_mass_matrix (same value replicated "
        "to each chain).  No step_size adaptation: returns a constant scalar "
        "default (1.0) per chain.  Sidecar: "
        "_multipathfinder_psis_pareto_k (PSIS Pareto-k diagnostic). "
        "Compatible: nuts, hmc, mala, rwm, barker.  "
        "NOT compatible with mclmc (microcanonical geometry)."
    ),
)
