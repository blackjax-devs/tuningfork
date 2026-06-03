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
import optax

from tuningfork.warmup._base import Warmup

__all__ = ["ENTRY"]

# Algorithms that accept an inverse_mass_matrix and are therefore compatible
# with full-rank VI's init-and-IMM output.  mclmc is excluded: its geometry
# is microcanonical and its inverse_mass_matrix is a diagonal preconditioner
# for Euclidean distance, not an HMC-style covariance.
_COMPATIBLE = ("nuts", "hmc", "mala", "rwm", "barker")

# Default optimizer — overridable via runner kwarg.
_adam_default = optax.adam(1e-2)


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


def _runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,  # step_size dual-averaging budget (0 = skip, use step_size_default)
    base_method: Any,  # BaseMethod; not imported to avoid circular dep
    *,
    logdensity_fn: Any,
    step_size_default: float = 1.0,
    num_chains: int = 4,
    num_optimization_steps: int = 20_000,
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

    if optimizer is None:
        optimizer = _adam_default

    # Build the unravel function from a SINGLE-chain position.
    _leaves = jax.tree.leaves(init_position)
    _is_prebatched = bool(
        _leaves and _leaves[0].shape and _leaves[0].shape[0] == num_chains
    )
    if _is_prebatched:
        _single_pos = jax.tree.map(lambda x: x[0], init_position)
    else:
        _single_pos = init_position
    _dummy_flat, unravel_fn = jax.flatten_util.ravel_pytree(_single_pos)
    d = int(_dummy_flat.shape[0])

    # Split key: one for the VI loop, one for drawing init positions.
    vi_key, sample_key = jax.random.split(rng_key)

    # --- Run VI optimisation (single fit, shared across all chains) ---
    vi_init = fr.init(_single_pos, optimizer)

    def one_step(carry: fr.FRVIState, step_key: jax.Array):
        new_state, info = fr.step(
            step_key, carry, logdensity_fn, optimizer, num_samples_per_step
        )
        return new_state, info

    vi_keys = jax.random.split(vi_key, num_optimization_steps)
    final_vi_state, vi_infos = jax.lax.scan(one_step, vi_init, vi_keys)

    # Final ELBO (scalar) — last step's ELBO value.
    final_elbo = vi_infos.elbo[-1]

    # --- Extract dense IMM from the fitted variational distribution ---
    # chol_params → L (lower triangular Cholesky factor)
    # covariance = L @ L.T  (positive definite, shape (d, d))
    chol_factor = _unflatten_cholesky(final_vi_state.chol_params, d)
    dense_cov = chol_factor @ chol_factor.T  # (d, d)

    # --- Draw num_chains initial positions from the fitted distribution ---
    chain_sample_keys = jax.random.split(sample_key, num_chains)

    @jax.vmap
    def draw_one(key: jax.Array) -> jax.Array:
        """Draw one position from the fitted variational distribution."""
        samples = fr.sample(key, final_vi_state, num_samples=1)
        # samples is a pytree with leading dim 1; take the first draw.
        pos = jax.tree.map(lambda x: x[0], samples)
        flat_pos, _ = jax.flatten_util.ravel_pytree(pos)
        return flat_pos  # (d,)

    flat_init_positions = draw_one(chain_sample_keys)  # (num_chains, d)

    # Convert flat (num_chains, d) positions back to the original pytree.
    init_positions_pytree = jax.vmap(unravel_fn)(flat_init_positions)

    # --- Build extra kwargs for the downstream kernel ---
    from tuningfork.calibration.tune import default_value_for_space

    _extra_kwargs: dict[str, Any] = {}
    if base_method.needs_mass_matrix:
        _extra_kwargs["inverse_mass_matrix"] = dense_cov  # VI dense IMM
    for space in base_method.default_hp_space:
        if (
            space.name not in ("step_size", "inverse_mass_matrix")
            and space.name not in _extra_kwargs
        ):
            _extra_kwargs[space.name] = default_value_for_space(space)

    # --- Step_size adaptation via incremental dual averaging (VI IMM frozen) ---
    # n_warmup > 0: run n_warmup steps of Nesterov DA from chain-0's VI position.
    # The VI dense IMM is frozen throughout; only step_size is adapted.
    # n_warmup == 0: skip adaptation, use step_size_default.
    if n_warmup > 0:
        from blackjax.adaptation.step_size import dual_averaging_adaptation as _da_adapt

        _da_target = (
            float(target_acceptance_rate) if target_acceptance_rate is not None else 0.8
        )
        _da_init_fn, _da_update_fn, _da_final_fn = _da_adapt(target=_da_target)
        _da_s0 = _da_init_fn(float(step_size_default))

        _sa_kernel_0 = base_method.factory(
            logdensity_fn, step_size=float(step_size_default), **_extra_kwargs
        )
        _sa_init_state = _sa_kernel_0.init(
            jax.tree.map(lambda x: x[0], init_positions_pytree)
        )

        def _sa_one_step(carry: tuple, step_key: jax.Array) -> tuple:
            mcmc_state, da_state = carry
            current_ss = jnp.exp(da_state.log_step_size)
            new_mcmc_state, mcmc_info = base_method.factory(
                logdensity_fn, step_size=current_ss, **_extra_kwargs
            ).step(step_key, mcmc_state)
            _accept = jnp.asarray(
                getattr(
                    mcmc_info,
                    "acceptance_rate",
                    getattr(mcmc_info, "is_accepted", jnp.asarray(0.5)),
                )
            )
            new_da_state = _da_update_fn(da_state, jnp.mean(_accept))
            return (new_mcmc_state, new_da_state), None

        _sa_key = jax.random.fold_in(rng_key, 999)
        _sa_keys = jax.random.split(_sa_key, n_warmup)
        (_, _sa_final_da), _ = jax.lax.scan(
            _sa_one_step, (_sa_init_state, _da_s0), _sa_keys
        )
        _adapted_step_size = float(jnp.exp(_sa_final_da.log_step_size_avg))
    else:
        _adapted_step_size = float(step_size_default)

    # --- Build kernel states for each chain at the adapted step_size ---
    kernel = base_method.factory(
        logdensity_fn, step_size=_adapted_step_size, **_extra_kwargs
    )

    @jax.vmap
    def init_one(pos: Any) -> Any:
        return kernel.init(pos)

    states = init_one(init_positions_pytree)

    # Broadcast the shared dense IMM across all chains: (num_chains, d, d).
    imm_per_chain = jnp.broadcast_to(dense_cov[None, :, :], (num_chains, d, d))

    adapted_params: dict[str, Any] = {
        "step_size": jnp.full((num_chains,), _adapted_step_size),
        "inverse_mass_matrix": imm_per_chain,  # (num_chains, d, d)
        "_frvi_elbo": final_elbo,  # scalar sidecar
    }

    return states, adapted_params


ENTRY = Warmup(
    name="fullrank_vi",
    runner=_runner,
    compatible_methods=_COMPATIBLE,
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
