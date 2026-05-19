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
multi-path path (PATH C: ``effective_n_paths >= 2``).

NOTE: The upstream ``blackjax.pathfinder_adaptation`` multi-path dispatch
has a bug in ``_psis_weighted_mixture_covariance`` — it assumes
``path_states.position`` is a flat ``(n_paths, d)`` array, but
``PathfinderState.position`` stores the pytree-structured (unravelled) form.
This causes a ``ValueError`` for dict-position models.  We work around this
by computing the mixture covariance ourselves via ``_psis_mixture_covariance_flat``
which flattens positions with ``ravel_pytree`` before the einsum.

**Breaking change from pre-PR-B contract**: ``inverse_mass_matrix`` is now
dense ``(num_chains, d, d)`` (broadcast from the shared ``(d, d)`` upstream
IMM) instead of the old diagonal ``(num_chains, d)`` form.  No committed
recipe uses the old diagonal contract (post-PR #31 the stoch_vol groundtruth
switched to ``window_adaptation_diag_imm``).

Runner signature (multi-chain contract)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, n_paths=None, num_samples_per_path=200,
            step_size_default=1.0, num_chains=4, **kwargs)
    -> (states, adapted_params)

Where:

- ``rng_key`` is a single key; used for pathfinder fit + PSIS resampling + DA.
- ``init_position`` is a single pytree (one chain's worth).
- ``states`` is a batched pytree with leading dim ``num_chains``.
- ``adapted_params`` contains:

  ==================================  ====================  ====================================
  Key                                 Shape                 Notes
  ==================================  ====================  ====================================
  ``step_size``                       ``(num_chains,)``     Per-chain adapted step size from DA
  ``inverse_mass_matrix``             ``(num_chains, d, d)``  Dense IMM broadcast per chain
  ``_multipathfinder_psis_pareto_k``  scalar                PSIS Pareto-k diagnostic
  ==================================  ====================  ====================================

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
from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1
from blackjax.vi.multipathfinder import psis_weights
from jax.flatten_util import ravel_pytree

from tuningfork.warmup._base import Warmup, _maybe_replicate

__all__ = ["ENTRY"]

# Algorithms compatible with HMC-style inverse_mass_matrix.
_COMPATIBLE = ("nuts", "hmc", "mala", "rwm", "barker")


def _psis_mixture_covariance_flat(
    path_states: Any,
    log_weights: jax.Array,
    num_samples_per_path: int,
) -> jax.Array:
    """Compute PSIS-weighted mixture covariance using flattened per-path positions.

    Workaround for the upstream ``_psis_weighted_mixture_covariance`` bug:
    ``PathfinderState.position`` stores pytree-structured (unravelled) positions
    but the upstream function assumes flat ``(n_paths, d)`` arrays.

    See also: tuningfork/warmup/multipathfinder_window_adaptation.py for the
    same workaround with documentation.

    Returns
    -------
    jax.Array
        Dense ``(d, d)`` PSIS-weighted mixture covariance.
    """
    n_paths = log_weights.shape[0] // num_samples_per_path
    log_weights_per_path = log_weights.reshape(n_paths, num_samples_per_path)
    log_path_weights = jax.scipy.special.logsumexp(log_weights_per_path, axis=1)
    log_path_weights_norm = log_path_weights - jax.scipy.special.logsumexp(
        log_path_weights
    )
    w = jnp.exp(log_path_weights_norm)  # (n_paths,)
    # Flatten pytree positions to (n_paths, d).
    mu_per_path = jax.vmap(lambda x: ravel_pytree(x)[0])(
        path_states.position
    )  # (n_paths, d)
    sigmas = jax.vmap(lbfgs_inverse_hessian_formula_1)(
        path_states.alpha, path_states.beta, path_states.gamma
    )  # (n_paths, d, d)
    mu_mix = jnp.einsum("i,id->d", w, mu_per_path)  # (d,)
    sigma_within = jnp.einsum("i,ijk->jk", w, sigmas)  # (d, d)
    delta = mu_per_path - mu_mix[None, :]  # (n_paths, d)
    sigma_between = jnp.einsum("i,ij,ik->jk", w, delta, delta)  # (d, d)
    return sigma_within + sigma_between  # (d, d)


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
    """Run multi-path Pathfinder + PSIS resampling + dual-averaging.

    Multi-path Pathfinder fit from ``n_paths`` starting points → PSIS
    importance-resampled init positions → shared dense ``(d, d)`` IMM via
    PSIS-weighted L-BFGS mixture covariance → per-chain dual-averaging.

    NOTE: Uses bespoke multi-path implementation rather than delegating to
    ``blackjax.pathfinder_adaptation(n_paths>1)`` because the upstream
    ``_psis_weighted_mixture_covariance`` function fails for dict-position
    models (sees pytree positions, not flat arrays as it assumes).

    Parameters
    ----------
    rng_key
        JAX random key.
    init_position
        Initial unconstrained parameter pytree (one chain's worth).
    n_warmup
        Number of dual-averaging adaptation steps per chain.
    base_method
        ``BaseMethod`` entry.  Used for compatibility check and DA.
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
        Additional keyword arguments forwarded to the algorithm (e.g.
        ``num_integration_steps`` for HMC).

    Returns
    -------
    states
        Post-adaptation BlackJAX kernel states, batched over ``num_chains``.
    adapted_params
        Dict with:

        - ``"step_size"``: ``(num_chains,)`` per-chain adapted step size.
        - ``"inverse_mass_matrix"``: ``(num_chains, d, d)`` dense IMM.
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

    # --- Step 1: run multipathfinder ----------------------------------------
    pf_key, resample_key, adapt_key = jax.random.split(rng_key, 3)

    init_positions = _maybe_replicate(init_position, n_paths)
    mpf = blackjax.multipathfinder(logdensity_fn)
    mpf_state, _pf_info = mpf.init(
        pf_key, init_positions, num_samples=num_samples_per_path
    )

    # --- Step 2: derive dense (d, d) IMM via PSIS-weighted L-BFGS mixture ----
    log_weights, pareto_k = psis_weights(mpf_state)
    imm_dense = _psis_mixture_covariance_flat(
        mpf_state.path_states, log_weights, num_samples_per_path
    )  # (d, d)

    # --- Step 3: PSIS-resample num_chains init positions ----------------------
    total_pool = log_weights.shape[0]
    probs = jnp.exp(log_weights)
    samples_flat = jax.tree.map(
        lambda x: x.reshape(-1, *x.shape[2:]), mpf_state.samples
    )
    init_indices = jax.random.choice(
        resample_key, total_pool, shape=(num_chains,), replace=True, p=probs
    )
    init_from_psis = jax.tree.map(lambda x: x[init_indices], samples_flat)

    # --- Step 4: per-chain pathfinder_adaptation with shared IMM (PATH B) -----
    # Use n_paths=1 so pathfinder_adaptation enters PATH B (single-path
    # broadcast) and doesn't call _psis_weighted_mixture_covariance.
    # We override the init positions externally via the per-chain DA vmap below.
    from tuningfork.calibration.tune import default_value_for_space

    extra_kwargs: dict[str, Any] = dict(kwargs)
    for space in base_method.default_hp_space:
        if space.name not in ("step_size", "inverse_mass_matrix"):
            if space.name not in extra_kwargs:
                extra_kwargs[space.name] = default_value_for_space(space)

    # Use the PathfinderAdaptationState machinery via window_adaptation with
    # the pre-computed IMM as a fixed mass matrix — but use window_adaptation
    # is cleaner. Actually: use pathfinder_adaptation PATH B (n_paths=1) and
    # just do per-chain DA via vmap over the PSIS-resampled inits.
    # Simpler: use blackjax.pathfinder_adaptation(num_chains=1, n_paths=1) per
    # chain, vmap over PSIS-resampled inits, use the pre-computed imm_dense.

    # Actually the cleanest is: use adaptation.base() directly.
    from blackjax.adaptation.pathfinder_adaptation import base as pf_adapt_base

    adapt_init, adapt_init_from_imm, adapt_update, adapt_final = pf_adapt_base(
        target_acceptance_rate=0.80
    )
    mcmc_kernel = base_method.factory.build_kernel()

    def one_step(carry, k):
        state, adaptation_state = carry
        new_state, info = mcmc_kernel(
            k,
            state,
            logdensity_fn,
            adaptation_state.step_size,
            adaptation_state.inverse_mass_matrix,
            **extra_kwargs,
        )
        new_adapt = adapt_update(
            adaptation_state, new_state.position, info.acceptance_rate
        )
        return (new_state, new_adapt), None

    @jax.vmap
    def run_one_chain(pos: Any, chain_key: jax.Array) -> tuple[Any, Any]:
        init_state = base_method.factory.init(pos, logdensity_fn)
        init_adapt = adapt_init_from_imm(imm_dense, step_size_default)
        step_keys = jax.random.split(chain_key, n_warmup)
        (last_state, last_adapt), _ = jax.lax.scan(
            one_step, (init_state, init_adapt), step_keys
        )
        step_size, _ = adapt_final(last_adapt)
        return last_state, step_size

    chain_keys = jax.random.split(adapt_key, num_chains)
    states, step_sizes = run_one_chain(init_from_psis, chain_keys)
    # states: batched over num_chains
    # step_sizes: (num_chains,) adapted

    # Broadcast shared dense IMM to (num_chains, d, d).
    imm_per_chain = jnp.broadcast_to(
        imm_dense[None, :, :], (num_chains,) + imm_dense.shape
    )

    adapted_params: dict[str, Any] = {
        "step_size": step_sizes,  # (num_chains,)
        "inverse_mass_matrix": imm_per_chain,  # (num_chains, d, d)
        "_multipathfinder_psis_pareto_k": pareto_k,  # scalar
    }

    return states, adapted_params


ENTRY = Warmup(
    name="multipathfinder",
    runner=_runner,
    compatible_methods=_COMPATIBLE,
    notes=(
        "Multi-path Pathfinder warmup: runs one multi-path Pathfinder fit "
        "from n_paths independent starting positions (default: n_paths == num_chains). "
        "PSIS-resamples num_chains init positions. "
        "Derives shared dense (d, d) IMM via PSIS-weighted L-BFGS mixture covariance "
        "(law of total variance). Runs per-chain dual-averaging for n_warmup steps. "
        "Returns per-chain adapted step_size (num_chains,) and dense (num_chains, d, d) "
        "IMM broadcast from the shared estimate. "
        "Sidecar: _multipathfinder_psis_pareto_k (PSIS Pareto-k diagnostic). "
        "NOTE: IMM is now dense (num_chains, d, d); old diagonal (num_chains, d) "
        "contract changed in PR B (warmup-collapse-pathfinder-shims). "
        "Step size is now adapted (DA over n_warmup steps) instead of constant default. "
        "Compatible: nuts, hmc, mala, rwm, barker. "
        "NOT compatible with mclmc (microcanonical geometry)."
    ),
)
