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

Thin shim around ``blackjax.pathfinder_adaptation`` dispatching to the
multi-path path (PATH C: ``effective_n_paths >= 2``).  The upstream call
handles multipathfinder + PSIS importance resampling + PSIS-weighted L-BFGS
mixture-covariance IMM (analytic, law of total variance) + per-chain
dual-averaging in a single jit-friendly run.

**IMM contract**: dense ``(num_chains, d, d)`` (broadcast from the shared
``(d, d)`` upstream IMM).  Pre-PR-B (warmup-collapse-pathfinder-shims) the
contract was diagonal ``(num_chains, d)``; no committed recipe used the
diagonal form (post-PR #31 stoch_vol switched to ``window_adaptation_diag_imm``).

Runner signature (multi-chain contract)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, n_paths=None, num_samples_per_path=200,
            step_size_default=1.0, num_chains=4, **kwargs)
    -> (states, adapted_params)

Where:

- ``rng_key`` is a single key; used for pathfinder fit + PSIS resampling + DA.
- ``init_position`` is a single pytree (one chain's worth).  Forwarded
  to ``blackjax.pathfinder_adaptation.run`` which handles replication
  internally.
- ``states`` is a batched pytree with leading dim ``num_chains``.
- ``adapted_params`` contains:

  ==================================  ======================  ====================================
  Key                                 Shape                   Notes
  ==================================  ======================  ====================================
  ``step_size``                       ``(num_chains,)``       Per-chain adapted step size from DA
  ``inverse_mass_matrix``             ``(num_chains, d, d)``  Shared dense IMM broadcast per chain
  ``_multipathfinder_psis_pareto_k``  scalar                  PSIS Pareto-k diagnostic
  ==================================  ======================  ====================================

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
    n_paths: int | None = None,
    num_samples_per_path: int = 200,
    step_size_default: float = 1.0,
    num_chains: int = 4,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run multi-path Pathfinder + PSIS resampling + dual-averaging via upstream.

    Multi-path Pathfinder fit from ``n_paths`` starting points → PSIS
    importance-resampled init positions → shared dense ``(d, d)`` IMM via
    PSIS-weighted L-BFGS mixture covariance → per-chain dual-averaging.
    All steps delegated to ``blackjax.pathfinder_adaptation(num_chains=N,
    n_paths=K, imm_estimator="lbfgs_psis_mixture")``.

    Parameters
    ----------
    rng_key
        JAX random key.
    init_position
        Initial unconstrained parameter pytree (one chain's worth).
        Forwarded to ``blackjax.pathfinder_adaptation.run``.
    n_warmup
        Number of dual-averaging adaptation steps per chain.
    base_method
        ``BaseMethod`` entry.  Used for compatibility check and the algorithm
        passed to ``pathfinder_adaptation`` (``base_method.factory``).
    logdensity_fn
        BlackJAX-compatible log-density function.
    n_paths
        Number of independent L-BFGS paths for the multi-path Pathfinder
        run.  Defaults to ``num_chains`` (one path per chain).
    num_samples_per_path
        Number of samples drawn per path to estimate ELBO / PSIS weights.
        Default ``200``.
    step_size_default
        Initial step size for dual-averaging.  Default ``1.0``.
    num_chains
        Number of independent chains to initialise.  Default ``4``.
    **kwargs
        Additional keyword arguments forwarded to ``pathfinder_adaptation``
        (e.g. ``num_integration_steps`` for HMC) after default-HP injection
        from ``base_method.default_hp_space``.

    Returns
    -------
    states
        Post-adaptation BlackJAX kernel states, batched over ``num_chains``.
    adapted_params
        Dict with:

        - ``"step_size"``: ``(num_chains,)`` per-chain adapted step size.
        - ``"inverse_mass_matrix"``: ``(num_chains, d, d)`` dense IMM
          (broadcast from the shared upstream ``(d, d)``).
        - ``"_multipathfinder_psis_pareto_k"``: scalar PSIS Pareto-k.

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

    # Inject default HPs for base_method that aren't step_size or IMM.
    # Mirrors the pattern used by other warmup wrappers in this module.
    from tuningfork.calibration.tune import default_value_for_space

    extra_kwargs: dict[str, Any] = dict(kwargs)
    for space in base_method.default_hp_space:
        if space.name not in ("step_size", "inverse_mass_matrix"):
            if space.name not in extra_kwargs:
                extra_kwargs[space.name] = default_value_for_space(space)

    # Delegate the full multipathfinder + PSIS + DA pipeline to upstream.
    warmup = blackjax.pathfinder_adaptation(
        base_method.factory,
        logdensity_fn,
        num_chains=num_chains,
        n_paths=n_paths,
        num_samples_per_path=num_samples_per_path,
        initial_step_size=step_size_default,
        imm_estimator="lbfgs_psis_mixture",
        **extra_kwargs,
    )
    results, _info = warmup.run(rng_key, init_position, num_steps=n_warmup)

    # Adapt upstream's return shape to tuningfork's ENTRY contract.
    # ``inverse_mass_matrix``: upstream returns shared ``(d, d)``; broadcast
    # to ``(num_chains, d, d)`` to match the per-chain-IMM contract this
    # ENTRY exposes to downstream consumers.
    # ``step_size``: upstream returns ``(num_chains,)`` for the multi-chain
    # paths (PATH B / PATH C with ``num_chains >= 2``).  For the single-chain
    # case (PATH A or PATH C with ``num_chains == 1``) upstream returns a
    # scalar ``()`` + unbatched state; the tuningfork ENTRY contract requires
    # leading dim ``num_chains`` even when ``num_chains == 1``, so we add a
    # broadcast (``atleast_1d`` for the scalar, ``x[None]`` over the state
    # pytree).
    # Sidecar: upstream emits ``_pathfinder_psis_pareto_k``; this ENTRY
    # historically used ``_multipathfinder_psis_pareto_k``.  Forward under
    # the historical name to avoid breaking downstream readers.
    step_size = results.parameters["step_size"]
    imm_dense = results.parameters["inverse_mass_matrix"]  # (d, d)
    state = results.state

    if num_chains == 1:
        step_size = jnp.atleast_1d(step_size)  # () -> (1,)
        state = jax.tree.map(lambda x: x[None], state)

    imm_per_chain = jnp.broadcast_to(
        imm_dense[None, :, :], (num_chains,) + imm_dense.shape
    )
    pareto_k = results.parameters.get("_pathfinder_psis_pareto_k")

    adapted_params: dict[str, Any] = {
        "step_size": step_size,
        "inverse_mass_matrix": imm_per_chain,
        "_multipathfinder_psis_pareto_k": pareto_k,
    }

    return state, adapted_params


ENTRY = Warmup(
    name="multipathfinder",
    runner=_runner,
    compatible_methods=_COMPATIBLE,
    notes=(
        "Multi-path Pathfinder warmup: thin shim over "
        "blackjax.pathfinder_adaptation(num_chains, n_paths>=2, "
        "imm_estimator='lbfgs_psis_mixture'). "
        "Multi-path Pathfinder fit from n_paths independent starting positions "
        "(default n_paths == num_chains). PSIS-resamples num_chains init positions. "
        "Derives shared dense (d, d) IMM via the PSIS-weighted L-BFGS mixture "
        "covariance (law of total variance). Per-chain dual-averaging step-size "
        "adaptation for n_warmup steps. "
        "Returns per-chain step_size (num_chains,) and dense (num_chains, d, d) "
        "IMM broadcast from the shared estimate. "
        "Sidecar: _multipathfinder_psis_pareto_k (PSIS Pareto-k diagnostic). "
        "IMM shape is dense (num_chains, d, d); pre-PR-B contract was diagonal "
        "(num_chains, d). Compatible: nuts, hmc, mala, rwm, barker. "
        "NOT compatible with mclmc (microcanonical geometry)."
    ),
)
