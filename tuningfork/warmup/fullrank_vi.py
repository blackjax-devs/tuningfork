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
"""Full-rank VI warmup: init-and-IMM provider via variational surrogate.

This warmup runs a single full-rank VI optimisation (shared across all
chains), then:

1. Draws ``num_chains`` initial positions from the fitted variational
   distribution ``N(mu, L @ L.T)`` where ``L`` is the Cholesky factor
   recovered from ``chol_params`` — each chain gets an independent draw.
2. Returns the **dense inverse mass matrix** (IMM) as
   ``L @ L.T`` — the covariance of the fitted variational distribution,
   broadcast identically across all chains (the VI fit is shared; only init
   positions differ).

No step-size adaptation is performed.  A scalar default of ``1.0`` is
returned for every chain.  The downstream sampler should rely on its own
dual-averaging adaptation (e.g. NUTS window adaptation) or Bayesian
optimisation to tune the step size; this warmup only provides a better
initialisation and dense IMM than a flat prior sample.

Runner signature (multi-chain contract)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, step_size_default=1.0,
            num_chains: int = 4,
            num_optimization_steps: int = 20_000,
            optimizer=optax.adam(1e-2),
            num_samples_per_step: int = 5,
            **kwargs)
    -> (states, adapted_params)

Where:

- ``rng_key`` is a single key; split internally.
- ``init_position`` is a single pytree (one chain's worth); replicated
  across chains internally via ``_maybe_replicate``.
- ``states`` is a batched pytree with leading dim ``num_chains``;
  ``states.position`` has shape ``(num_chains, d)``.
- ``adapted_params`` contains:

  =============================================  =================  ==========================================
  Key                                            Shape              Notes
  =============================================  =================  ==========================================
  ``step_size``                                  ``(num_chains,)``  Constant ``step_size_default`` per chain
  ``inverse_mass_matrix``                        ``(num_chains, d, d)``  Dense covariance ``L @ L.T``
  ``_frvi_elbo``                                 scalar             Final ELBO from the VI fit (sidecar)
  =============================================  =================  ==========================================

Sidecar keys (underscore prefix) are metadata for ``calibration_budget``
and are NOT forwarded to the base-method kernel as hyperparameters.

Compatible with: ``nuts``, ``hmc``, ``mala``, ``rwm``, ``barker``.
NOT compatible with ``mclmc`` (different geometry — microcanonical momentum,
no Gaussian inverse mass matrix in the HMC sense).

**Applicability**: recommended for ``d <= 30``.  For higher-dimensional
problems, ``meanfield_vi`` warmup is preferred.

Upstream API (BlackJAX >= 0.9.x):

- ``blackjax.vi.fullrank_vi.init(position, optimizer)``
  → ``FRVIState(mu, chol_params, opt_state)``
- ``blackjax.vi.fullrank_vi.step(rng_key, state, logdensity_fn, optimizer, num_samples)``
  → ``(FRVIState, FRVIInfo(elbo=...))``
- ``blackjax.vi.fullrank_vi.sample(rng_key, state, num_samples)``
  → PyTree with leading dim ``num_samples``
- ``FRVIState._fields``: ``('mu', 'chol_params', 'opt_state')``
  where ``chol_params`` is a flattened vector of length ``d*(d+1)/2``
  containing log-diagonal entries followed by lower-triangular off-diagonal
  entries (row-major).  The Cholesky factor ``L`` has
  ``L[i, i] = exp(chol_params[i])`` and ``L[i, j] = chol_params[d + ...]``
  for ``i > j``.  The covariance is ``L @ L.T``.
"""

from typing import Any

import blackjax.vi.fullrank_vi as fr
import jax
import jax.numpy as jnp

from tuningfork.base_method._base import HyperparamSpace
from tuningfork.warmup._base import Warmup
from tuningfork.warmup._vi_warmup_runner import _vi_warmup_runner

__all__ = ["ENTRY"]

# Algorithms that accept an inverse_mass_matrix and are therefore compatible
# with full-rank VI's init-and-IMM output.  mclmc is excluded: its geometry
# is microcanonical and its inverse_mass_matrix is a diagonal preconditioner
# for Euclidean distance, not an HMC-style covariance.
_COMPATIBLE = ("nuts", "hmc", "mala", "rwm", "barker")

# Production default for num_optimization_steps.
_DEFAULT_N_OPT_STEPS = 20_000


def _unflatten_cholesky(chol_params: jax.Array, dim: int) -> jax.Array:
    """Reconstruct the Cholesky factor from a flattened parameter vector.

    Mirrors ``blackjax.vi.fullrank_vi._unflatten_cholesky`` (private) to
    avoid importing a private helper.  Diagonal entries are in log-space:
    ``L[i, i] = exp(chol_params[i])``.

    Parameters
    ----------
    chol_params
        Flattened Cholesky parameters, shape ``(d*(d+1)//2,)``.
    dim
        Dimensionality ``d``.

    Returns
    -------
    jax.Array
        Lower-triangular Cholesky factor ``L``, shape ``(d, d)``.
    """
    tril = jnp.zeros((dim, dim))
    tril = tril.at[jnp.tril_indices(dim, k=-1)].set(chol_params[dim:])
    diag = jnp.exp(chol_params[:dim])
    return tril + jnp.diag(diag)


def _frvi_imm_extractor(final_vi_state: fr.FRVIState, d: int) -> tuple[Any, tuple]:
    """Extract dense IMM from full-rank VI state via Cholesky reconstruction.

    Returns ``(dense_cov, broadcast_shape)`` where
    ``dense_cov = L @ L.T`` has shape ``(d, d)`` and
    ``broadcast_shape = (d, d)``.
    """
    chol_factor = _unflatten_cholesky(final_vi_state.chol_params, d)
    dense_cov = chol_factor @ chol_factor.T  # (d, d)
    return dense_cov, (d, d)


def _runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,  # step_size dual-averaging budget (0 = skip, use step_size_default)
    base_method: Any,  # BaseMethod; not imported to avoid circular dep
    *,
    logdensity_fn: Any,
    step_size_default: float = 1.0,
    num_chains: int = 4,
    num_optimization_steps: int = _DEFAULT_N_OPT_STEPS,
    optimizer: Any = None,
    num_samples_per_step: int = 5,
    target_acceptance_rate: float = 0.8,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run full-rank VI once (shared fit); draw ``num_chains`` init positions.

    The VI optimisation is performed once and the resulting variational
    distribution is shared across all chains.  ``num_chains`` independent
    initial positions are drawn from the fitted distribution.  The dense
    covariance ``L @ L.T`` (recovered from ``chol_params``) is returned as
    the ``inverse_mass_matrix``, broadcast identically across all chains.

    Parameters
    ----------
    rng_key
        JAX random key.  Split internally into a key for the VI loop and
        a key for drawing ``num_chains`` initial positions.
    init_position
        Initial unconstrained parameter pytree (one chain's worth).  The
        runner uses this as the starting point for VI optimisation, and
        replicates it to ``(num_chains, ...)`` for state initialisation
        unless the caller pre-batches it (leading dim == ``num_chains``).
    n_warmup
        Accepted for interface uniformity; not used.  The VI optimisation
        budget is controlled by ``num_optimization_steps``.
    base_method
        ``BaseMethod`` entry.  Used for kernel-state initialisation after
        the VI fit.
    logdensity_fn
        BlackJAX-compatible log-density function.
    step_size_default
        Constant step size assigned to every chain.  Default ``1.0``.
        The downstream sampler should adapt this via dual-averaging.
    num_chains
        Number of independent chains to initialise.  Default ``4``.
        The returned ``states`` have leading dim ``num_chains``.
    num_optimization_steps
        Number of Adam optimisation steps for the VI loop.
        Default ``20_000`` (production); use ``5_000`` in tests.
    optimizer
        Optax ``GradientTransformation``.  Defaults to ``optax.adam(1e-2)``.
    num_samples_per_step
        Number of Monte Carlo samples per VI gradient step.  Default ``5``
        (upstream default).
    **kwargs
        Accepted for interface uniformity; ignored.

    Returns
    -------
    states
        Post-VI kernel states, batched over ``num_chains``.
        ``states.position`` has shape ``(num_chains, d)`` (or
        ``(num_chains, ...)`` for dict/pytree positions).  Each chain's
        position is an independent draw from the fitted variational
        distribution.
    adapted_params
        Dict with:

        - ``"step_size"``: ``(num_chains,)`` array, constant ``step_size_default``.
        - ``"inverse_mass_matrix"``: ``(num_chains, d, d)``, per-chain copy of
          the dense covariance ``L @ L.T`` (identical across chains — the VI
          fit is shared).
        - ``"_frvi_elbo"``: scalar, final ELBO from the VI optimisation
          (sidecar metadata; underscore prefix marks it as non-HP).

    Raises
    ------
    ValueError
        If ``base_method.name`` is not in the compatible set.
    """
    if base_method.name not in _COMPATIBLE:
        raise ValueError(
            f"fullrank_vi warmup is not compatible with base_method "
            f"{base_method.name!r}; compatible: {_COMPATIBLE}"
        )

    return _vi_warmup_runner(
        rng_key,
        init_position,
        n_warmup,
        base_method,
        vi_module=fr,
        imm_extractor_fn=_frvi_imm_extractor,
        elbo_sidecar_key="_frvi_elbo",
        default_n_opt_steps=_DEFAULT_N_OPT_STEPS,
        logdensity_fn=logdensity_fn,
        step_size_default=step_size_default,
        num_chains=num_chains,
        num_optimization_steps=num_optimization_steps,
        optimizer=optimizer,
        num_samples_per_step=num_samples_per_step,
        target_acceptance_rate=target_acceptance_rate,
        **kwargs,
    )


ENTRY = Warmup(
    name="fullrank_vi",
    runner=_runner,
    compatible_methods=_COMPATIBLE,
    default_hp_space=(
        HyperparamSpace("num_optimization_steps", "int", low=1_000, high=50_000),
    ),
    notes=(
        "Full-rank VI warmup: runs a single full-rank VI "
        "optimisation (shared across all chains) via jax.lax.scan over "
        "num_optimization_steps Adam steps.  Draws num_chains independent "
        "initial positions from the fitted variational distribution N(mu, L@L.T) "
        "and returns the dense covariance L@L.T as the inverse_mass_matrix "
        "(shared across all chains — the fit is shared, only init positions "
        "differ).  No step_size adaptation: returns a constant scalar "
        "default (1.0) per chain.  "
        "Sidecar: _frvi_elbo (final ELBO scalar).  "
        "Compatible: nuts, hmc, mala, rwm, barker.  "
        "NOT compatible with mclmc (microcanonical geometry).  "
        "Recommended ONLY for d <= 30: the Cholesky parameterisation has "
        "O(d^2) parameters which become expensive at high dimension. "
        "Use meanfield_vi warmup for d > 30.  "
        "Production default: num_optimization_steps=20_000, "
        "optimizer=optax.adam(1e-2).  Use 5_000 in tests."
    ),
)
