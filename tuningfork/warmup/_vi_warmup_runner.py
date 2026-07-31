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
"""Shared runner body for mean-field and full-rank VI warmup variants.

The meanfield_vi and fullrank_vi warmup runners share ~120 lines of identical
logic: pre-batch detection, key split, vi.init + one_step scan, ELBO read,
draw loop, unravel, extra-kwargs build, dual-averaging step-size adaptation,
kernel-state init, and adapted_params assembly.

The ONLY difference is the IMM extraction:
- **meanfield_vi**: ``diag_imm = exp(2 * rho_flat)``, shape ``(d,)``;
  broadcast to ``(num_chains, d)``.
- **fullrank_vi**: Cholesky extraction ``_unflatten_cholesky(chol_params, d)``
  → ``dense_cov = chol @ chol.T``, shape ``(d, d)``; broadcast to
  ``(num_chains, d, d)``.

This module exposes ``_vi_warmup_runner`` which accepts an
``imm_extractor_fn`` to inject that difference, plus ``elbo_sidecar_key``
(``"_mfvi_elbo"`` vs ``"_frvi_elbo"``) and ``default_n_opt_steps`` as
variant-specific constants.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

__all__ = ["_vi_warmup_runner"]


def _vi_warmup_runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,
    base_method: Any,
    *,
    vi_module: Any,
    imm_extractor_fn: Callable[[Any, int], tuple[Any, Any]],
    elbo_sidecar_key: str,
    default_n_opt_steps: int,
    logdensity_fn: Any,
    step_size_default: float = 1.0,
    num_chains: int = 4,
    num_optimization_steps: int | None = None,
    optimizer: Any = None,
    num_samples_per_step: int = 5,
    target_acceptance_rate: float = 0.8,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Shared runner body for VI warmup variants.

    Runs a single VI optimisation (shared across all chains), draws
    ``num_chains`` initial positions from the fitted distribution, extracts
    the IMM via ``imm_extractor_fn``, optionally adapts the step size via
    Nesterov dual averaging, and initialises kernel states.

    Parameters
    ----------
    rng_key
        JAX random key.  Split internally into a key for the VI loop and
        a key for drawing ``num_chains`` initial positions.
    init_position
        Initial unconstrained parameter pytree (one chain's worth).
    n_warmup
        When > 0, runs ``n_warmup`` Nesterov DA steps to adapt the step size
        with the VI IMM frozen.  When 0, returns ``step_size_default``.
    base_method
        ``BaseMethod`` entry used for kernel-state initialisation.
    vi_module
        BlackJAX VI module (e.g. ``blackjax.vi.meanfield_vi`` or
        ``blackjax.vi.fullrank_vi``).  Must expose ``init``, ``step``, and
        ``sample`` functions.
    imm_extractor_fn
        ``(final_vi_state, d: int) -> (imm_single_chain, broadcast_shape)``
        where ``imm_single_chain`` is the per-chain IMM (shape ``(d,)`` for
        diagonal or ``(d, d)`` for dense) and ``broadcast_shape`` is the
        tuple to broadcast it to ``(num_chains, *broadcast_shape)``.
    elbo_sidecar_key
        Key for the ELBO sidecar in ``adapted_params`` (e.g.
        ``"_mfvi_elbo"`` or ``"_frvi_elbo"``).
    default_n_opt_steps
        Default for ``num_optimization_steps`` when the caller does not
        supply it.
    logdensity_fn
        BlackJAX-compatible log-density function.
    step_size_default
        Constant step size assigned to every chain.  Default ``1.0``.
    num_chains
        Number of independent chains to initialise.  Default ``4``.
    num_optimization_steps
        Number of Adam steps for the VI loop.  Falls back to
        ``default_n_opt_steps`` when ``None``.
    optimizer
        Optax ``GradientTransformation``.  Falls back to
        ``optax.adam(1e-2)`` when ``None``.
    num_samples_per_step
        Monte Carlo samples per VI gradient step.  Default ``5``.
    target_acceptance_rate
        Dual-averaging target when ``n_warmup > 0``.  Default ``0.8``.
    **kwargs
        Accepted for interface uniformity; ignored.

    Returns
    -------
    states
        Post-VI kernel states, batched over ``num_chains``.
    adapted_params
        Dict with ``"step_size"``, ``"inverse_mass_matrix"``, and the
        ``elbo_sidecar_key`` entry.
    """
    import optax

    if num_optimization_steps is None:
        num_optimization_steps = default_n_opt_steps

    if optimizer is None:
        optimizer = optax.adam(1e-2)

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
    vi_init = vi_module.init(_single_pos, optimizer)

    def one_step(carry: Any, step_key: jax.Array) -> tuple[Any, Any]:
        new_state, info = vi_module.step(
            step_key, carry, logdensity_fn, optimizer, num_samples_per_step
        )
        return new_state, info

    vi_keys = jax.random.split(vi_key, num_optimization_steps)
    final_vi_state, vi_infos = jax.lax.scan(one_step, vi_init, vi_keys)

    # Final ELBO (scalar) — last step's ELBO value.
    final_elbo = vi_infos.elbo[-1]

    # --- Extract IMM from the fitted variational distribution ---
    # Delegated to imm_extractor_fn — the ONLY variant-specific logic.
    imm_single_chain, broadcast_shape = imm_extractor_fn(final_vi_state, d)

    # --- Draw num_chains initial positions from the fitted distribution ---
    chain_sample_keys = jax.random.split(sample_key, num_chains)

    @jax.vmap
    def draw_one(key: jax.Array) -> jax.Array:
        """Draw one position from the fitted variational distribution."""
        samples = vi_module.sample(key, final_vi_state, num_samples=1)
        pos = jax.tree.map(lambda x: x[0], samples)
        flat_pos, _ = jax.flatten_util.ravel_pytree(pos)
        return flat_pos  # (d,)

    flat_init_positions = draw_one(chain_sample_keys)  # (num_chains, d)

    # Convert flat (num_chains, d) positions back to the original pytree.
    init_positions_pytree = jax.vmap(unravel_fn)(flat_init_positions)

    # --- Build extra kwargs for the downstream kernel ---
    from tuningfork.base_method import default_value_for_space

    _extra_kwargs: dict[str, Any] = {}
    if base_method.needs_mass_matrix:
        _extra_kwargs["inverse_mass_matrix"] = imm_single_chain
    for space in base_method.default_hp_space:
        if (
            space.name not in ("step_size", "inverse_mass_matrix")
            and space.name not in _extra_kwargs
        ):
            _extra_kwargs[space.name] = default_value_for_space(space)

    # --- Step_size adaptation via incremental dual averaging (VI IMM frozen) ---
    # n_warmup > 0: run n_warmup steps of Nesterov DA from chain-0's VI position.
    # The VI IMM is frozen throughout; only step_size is adapted.
    # n_warmup == 0: skip adaptation, use step_size_default.
    if n_warmup > 0:
        from blackjax.adaptation.step_size import dual_averaging_adaptation as _da_adapt

        # Defensive: target_acceptance_rate may be None if not set in warmup_params.
        _da_target = (
            float(target_acceptance_rate) if target_acceptance_rate is not None else 0.8
        )
        _da_init_fn, _da_update_fn, _da_final_fn = _da_adapt(target=_da_target)
        _da_s0 = _da_init_fn(float(step_size_default))

        # Init from chain-0's VI-drawn position
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

    # Broadcast the shared IMM across all chains.
    imm_per_chain = jnp.broadcast_to(
        jnp.expand_dims(imm_single_chain, axis=0),
        (num_chains,) + broadcast_shape,
    )

    adapted_params: dict[str, Any] = {
        "step_size": jnp.full((num_chains,), _adapted_step_size),
        "inverse_mass_matrix": imm_per_chain,
        elbo_sidecar_key: final_elbo,
    }

    return states, adapted_params
