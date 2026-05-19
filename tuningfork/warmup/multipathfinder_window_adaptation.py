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
"""Paper-canonical composed warmup: multipathfinder init + window adaptation.

Zhang et al. 2022 (JMLR § 4, Birthday GP case study) recommend
multi-path Pathfinder as an **init-strategy preceding adaptive HMC**
(= window adaptation), not as a standalone adaptation.  This warmup
implements that composition exactly:

1. Run ``blackjax.multipathfinder`` to get a mixture of L-BFGS Laplace
   approximations from ``n_paths`` independent paths.
2. Derive a shared dense ``(d, d)`` inverse mass matrix via the
   PSIS-weighted L-BFGS mixture covariance (law of total variance —
   the same estimator used by ``blackjax.pathfinder_adaptation`` with
   ``imm_estimator="lbfgs_psis_mixture"``).
3. PSIS-resample ``num_chains`` init positions from the mixture.
4. Run ``blackjax.window_adaptation`` with:
   - ``is_mass_matrix_diagonal=False`` (dense IMM throughout),
   - ``initial_inverse_mass_matrix=imm_dense`` (seeds window 1 with
     the multipathfinder-derived covariance),
   - ``imm_shrinkage_to_previous=20.0`` (medium pseudo-count — keeps
     the multipathfinder IMM influential across windows, per the
     docstring scale-points: 0 = Stan default, 5 = light, 20 = medium,
     50 = heavy persistence).
5. Use the BEST PSIS draw (highest weight) as the warmup init position;
   vmap the per-chain DA over the ``num_chains`` PSIS-resampled inits.

The ``imm_shrinkage_to_previous=20.0`` default encodes the intent that the
multipathfinder IMM seeds window 1 *and* stays influential through subsequent
windows — otherwise the P0 seed (``initial_inverse_mass_matrix``) is only
effective for the first window under Stan's Welford-reset design.

Runner signature (multi-chain contract)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, n_paths=None, num_samples_per_path=200,
            imm_shrinkage_to_previous=20.0, target_acceptance=0.80,
            step_size_default=1.0, num_chains=4, **kwargs)
    -> (states, adapted_params)

Where:

- ``rng_key`` is a single key; used for multipathfinder fit + PSIS resampling
  + window adaptation.
- ``init_position`` is a single pytree (one chain's worth); replicated to
  ``(n_paths, ...)`` for the multi-path fit.
- ``states`` is a batched pytree with leading dim ``num_chains``.
- ``adapted_params`` contains:

  ==================================  ====================  =====================================
  Key                                 Shape                 Notes
  ==================================  ====================  =====================================
  ``step_size``                       ``(num_chains,)``     Per-chain adapted step size from DA
  ``inverse_mass_matrix``             ``(num_chains, d, d)``  Shared dense IMM broadcast per chain
  ``_multipathfinder_psis_pareto_k``  scalar                PSIS Pareto-k diagnostic
  ==================================  ====================  =====================================

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
from blackjax.adaptation.pathfinder_adaptation import _psis_weighted_mixture_covariance
from blackjax.vi.multipathfinder import psis_weights

from tuningfork.warmup._base import Warmup, _maybe_replicate
from tuningfork.warmup._laplace_adapter import resolve_warmup_algorithm

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
    imm_shrinkage_to_previous: float = 20.0,
    target_acceptance: float = 0.80,
    step_size_default: float = 1.0,
    num_chains: int = 4,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run multipathfinder then window_adaptation (paper-canonical composition).

    Implements the Zhang et al. 2022 recommendation: multi-path Pathfinder
    as an init-strategy preceding adaptive HMC.  The multipathfinder IMM
    seeds ``window_adaptation``'s initial mass matrix and medium pseudo-count
    shrinkage keeps it influential across windows.

    Parameters
    ----------
    rng_key
        JAX random key.  Used for the multipathfinder fit, PSIS resampling,
        and window adaptation.
    init_position
        Initial unconstrained parameter pytree (one chain's worth).  The
        runner replicates it to ``(n_paths, ...)`` for the multi-path fit.
    n_warmup
        Number of window-adaptation steps.
    base_method
        ``BaseMethod`` entry.  Used for the warmup algorithm selection and
        extra HP injection.
    logdensity_fn
        BlackJAX-compatible log-density function.
    n_paths
        Number of independent L-BFGS paths for the multi-path Pathfinder
        run.  Defaults to ``num_chains`` (one path per chain).
    num_samples_per_path
        Number of samples drawn per path to estimate ELBO / PSIS weights.
        Default ``200``.
    imm_shrinkage_to_previous
        Pseudo-count controlling shrinkage of the per-window IMM toward the
        previous window's IMM inside ``window_adaptation``.  ``0.0`` = Stan
        default (no shrinkage, Welford-reset resets fully).  ``20.0``
        (medium) keeps the multipathfinder-derived IMM influential across
        windows.  ``50.0`` = heavy persistence.  Default ``20.0``.
    target_acceptance
        Dual-averaging target acceptance rate for window adaptation.
        Default ``0.80``.
    step_size_default
        Initial step size for dual-averaging in window adaptation.
        Default ``1.0``.
    num_chains
        Number of independent chains to initialise.  Default ``4``.
    **kwargs
        Additional keyword arguments forwarded to ``window_adaptation``
        (e.g. ``num_integration_steps`` for HMC).

    Returns
    -------
    states
        Post-warmup BlackJAX kernel states, batched over ``num_chains``.
        ``states.position`` has shape ``(num_chains, d)`` or
        ``(num_chains, ...)`` for dict-based positions.
        Each chain's position is drawn from the PSIS-resampled mixture.
    adapted_params
        Dict with:

        - ``"step_size"``: ``(num_chains,)`` per-chain adapted step size.
        - ``"inverse_mass_matrix"``: ``(num_chains, d, d)``, shared dense
          IMM broadcast to each chain.
        - ``"_multipathfinder_psis_pareto_k"``: scalar PSIS Pareto-k.

    Raises
    ------
    ValueError
        If ``base_method.name`` is not in the compatible set.
    """
    if base_method.name not in _COMPATIBLE:
        raise ValueError(
            f"multipathfinder_window_adaptation warmup is not compatible with "
            f"base_method {base_method.name!r}; compatible: {_COMPATIBLE}"
        )

    if n_paths is None:
        n_paths = num_chains

    # --- Step 1: run multipathfinder ----------------------------------------
    pf_key, resample_key, adapt_key = jax.random.split(rng_key, 3)

    # Replicate init_position to (n_paths, ...) for the multi-path fit.
    init_positions = _maybe_replicate(init_position, n_paths)

    mpf = blackjax.multipathfinder(logdensity_fn)
    mpf_state, _pf_info = mpf.init(
        pf_key, init_positions, num_samples=num_samples_per_path
    )

    # --- Step 2: derive dense (d, d) IMM via PSIS-weighted L-BFGS mixture ----
    # Delegated to upstream blackjax (which handles pytree positions correctly
    # post-PR #922).  This is the same estimator that
    # ``blackjax.pathfinder_adaptation(imm_estimator="lbfgs_psis_mixture")``
    # uses internally — analytic law-of-total-variance over the per-path
    # Laplace approximations.
    log_weights, pareto_k = psis_weights(mpf_state)
    imm_dense = _psis_weighted_mixture_covariance(mpf_state, log_weights)  # (d, d)

    # --- Step 3: PSIS-resample num_chains init positions ----------------------
    total_pool = log_weights.shape[0]
    probs = jnp.exp(log_weights)

    # Flatten sample pool for indexing.
    samples_flat = jax.tree.map(
        lambda x: x.reshape(-1, *x.shape[2:]), mpf_state.samples
    )

    init_indices = jax.random.choice(
        resample_key, total_pool, shape=(num_chains,), replace=True, p=probs
    )
    init_from_psis = jax.tree.map(lambda x: x[init_indices], samples_flat)
    # init_from_psis: pytree with leading dim num_chains

    # --- Step 4: window adaptation seeded with the multipathfinder IMM -------
    from tuningfork.calibration.tune import default_value_for_space

    # Build extra kwargs for window_adaptation (inject HP defaults that the
    # warmup kernel needs but are not step_size/IMM).
    extra_kwargs: dict[str, Any] = dict(kwargs)
    for space in base_method.default_hp_space:
        if space.name not in ("step_size", "inverse_mass_matrix"):
            if space.name not in extra_kwargs:
                extra_kwargs[space.name] = default_value_for_space(space)

    # For laplace_* base methods, substitute blackjax.hmc as the warmup
    # algorithm (see window_adaptation_diag_imm.py for rationale).
    warmup_algorithm, warmup_kwargs = resolve_warmup_algorithm(
        base_method, extra_kwargs
    )

    warmup = blackjax.window_adaptation(
        warmup_algorithm,
        logdensity_fn,
        is_mass_matrix_diagonal=False,
        initial_inverse_mass_matrix=imm_dense,
        imm_shrinkage_to_previous=imm_shrinkage_to_previous,
        target_acceptance_rate=target_acceptance,
        initial_step_size=step_size_default,
        **warmup_kwargs,
    )

    # --- Step 5: vmap window adaptation over num_chains PSIS-resampled inits -
    chain_keys = jax.random.split(adapt_key, num_chains)

    @jax.vmap
    def run_one(k: jax.Array, x0: Any) -> tuple[Any, Any]:
        (state, params), _info = warmup.run(k, x0, n_warmup)
        return state, params

    states, per_chain_params = run_one(chain_keys, init_from_psis)
    # per_chain_params["step_size"]: (num_chains,)
    # per_chain_params["inverse_mass_matrix"]: (num_chains, d, d) — each chain
    #   gets the same seed IMM from window_adaptation (with medium shrinkage
    #   toward it), but vmap allows per-chain DA to diverge slightly.

    adapted_params: dict[str, Any] = {
        "step_size": per_chain_params["step_size"],  # (num_chains,)
        "inverse_mass_matrix": per_chain_params[
            "inverse_mass_matrix"
        ],  # (num_chains, d, d)
        "_multipathfinder_psis_pareto_k": pareto_k,  # scalar
    }

    return states, adapted_params


ENTRY = Warmup(
    name="multipathfinder_window_adaptation",
    runner=_runner,
    compatible_methods=_COMPATIBLE,
    notes=(
        "Paper-canonical composed warmup (Zhang et al. 2022 § 4): "
        "multi-path Pathfinder as init-strategy preceding window adaptation. "
        "Runs multipathfinder (n_paths=num_chains by default) to derive a shared "
        "dense (d, d) IMM via the PSIS-weighted L-BFGS mixture covariance. "
        "PSIS-resamples num_chains init positions. Passes the dense IMM as "
        "initial_inverse_mass_matrix to window_adaptation, with "
        "imm_shrinkage_to_previous=20.0 (medium persistence) so the "
        "multipathfinder IMM seed remains influential across windows. "
        "Returns per-chain adapted step_size (num_chains,), dense IMM "
        "(num_chains, d, d), and PSIS Pareto-k sidecar. "
        "Compatible: nuts, hmc, mala, rwm, barker. "
        "NOT compatible with mclmc (microcanonical geometry)."
    ),
)
