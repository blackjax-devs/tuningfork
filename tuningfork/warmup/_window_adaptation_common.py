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
"""Shared runner body for all three window-adaptation warmup variants.

The diag, dense, and low-rank variants differ in ONLY ONE line — the call that
constructs the ``warmup`` object (``blackjax.window_adaptation`` with
``is_mass_matrix_diagonal=True/False``, or
``blackjax.window_adaptation_low_rank``).  Everything else is identical.

This module exposes ``_window_adaptation_body`` which receives a
``warmup_builder_fn`` callable to inject that one-line difference and executes
the shared 20-line runner body.
"""

from collections.abc import Callable
from typing import Any

import jax

from tuningfork.warmup._base import _maybe_replicate
from tuningfork.warmup._laplace_adapter import resolve_warmup_algorithm

__all__ = ["_window_adaptation_body"]


def _window_adaptation_body(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,
    base_method: Any,
    *,
    logdensity_fn: Any,
    target_acceptance_rate: float | None = None,
    num_chains: int = 4,
    warmup_builder_fn: Callable[..., Any],
    **kwargs: Any,
) -> tuple[Any, dict[str, Any], Any]:
    """Shared runner body for window-adaptation variants.

    Parameters
    ----------
    rng_key
        JAX random key for the adaptation run.  Split internally into
        ``num_chains`` independent per-chain keys.
    init_position
        Initial unconstrained parameter pytree (one chain's worth).  The runner
        replicates it across ``num_chains`` unless the caller pre-batches it
        (leading dim == ``num_chains``).
    n_warmup
        Number of adaptation steps.
    base_method
        ``BaseMethod`` entry (carries ``factory``, ``default_hp_space``,
        ``target_acceptance_rate``).
    logdensity_fn
        BlackJAX-compatible log-density function.
    target_acceptance_rate
        Override for the dual-averaging target.  Falls back to
        ``base_method.target_acceptance_rate``, then ``0.80``.
    num_chains
        Number of independent chains to run in parallel via ``jax.vmap``.
        Default ``4``. Pass ``num_chains=1`` for isolated adaptation checks.
    warmup_builder_fn
        Callable ``(warmup_algorithm, logdensity_fn, target_acceptance_rate,
        **warmup_kwargs) -> warmup`` that constructs the BlackJAX warmup object.
        This is the ONLY line that differs across diag / dense / low_rank.
    **kwargs
        Additional keyword arguments forwarded to ``warmup_builder_fn``
        (e.g. ``num_integration_steps`` for HMC, ``max_rank`` for low-rank).

    Returns
    -------
    states
        Post-warmup BlackJAX kernel states, batched over ``num_chains``.
    adapted_params
        Dict with at least ``"step_size"`` and ``"inverse_mass_matrix"``.
    kernel_info
        Per-step kernel info (NUTSInfo / HMCInfo) for wge accounting.
    """
    from tuningfork.base_method import default_value_for_space

    target = target_acceptance_rate or base_method.target_acceptance_rate or 0.80

    # Build extra kwargs: inject default values for any HP that is NOT step_size
    # or inverse_mass_matrix (those come from the adaptation itself).
    extra_kwargs: dict[str, Any] = dict(kwargs)  # caller-supplied overrides first
    for space in base_method.default_hp_space:
        if space.name not in ("step_size", "inverse_mass_matrix"):
            if space.name not in extra_kwargs:
                extra_kwargs[space.name] = default_value_for_space(space)

    # For laplace_* base methods, substitute blackjax.hmc as the warmup
    # algorithm so that window_adaptation receives a proper algorithm object
    # with .build_kernel and .init(position, logdensity_fn).
    warmup_algorithm, warmup_kwargs = resolve_warmup_algorithm(
        base_method, extra_kwargs
    )

    # The ONE line that differs across variants — injected via warmup_builder_fn.
    warmup = warmup_builder_fn(
        warmup_algorithm,
        logdensity_fn,
        target_acceptance_rate=target,
        **warmup_kwargs,
    )

    # Split the key for num_chains independent runs.
    chain_keys = jax.random.split(rng_key, num_chains)

    # Replicate init_position across chains.  Pass-through if pre-batched.
    init_positions = _maybe_replicate(init_position, num_chains)

    # vmap the warmup.run over (key, init_position).
    # Return adapt_info.info (per-step kernel info) as a third value so callers
    # can CUMSUM num_integration_steps for exact wge.
    @jax.vmap
    def run_one(k: jax.Array, x0: Any) -> tuple[Any, Any, Any]:
        (state, params), adapt_info = warmup.run(k, x0, n_warmup)
        return state, params, adapt_info.info

    states, adapted_params, kernel_info = run_one(chain_keys, init_positions)
    return states, dict(adapted_params), kernel_info
