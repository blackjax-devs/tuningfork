"""Stan-style window-adaptation warmup, wrapping ``blackjax.window_adaptation``.

This warmup runs dual-averaging step-size adaptation together with diagonal
mass-matrix estimation, matching the Stan HMC/NUTS default.  It is
compatible with any BlackJAX kernel that accepts an ``inverse_mass_matrix``
keyword argument (HMC, NUTS, Barker, MALA — verified in Phase 2 tripwire
tests in ``tests/test_blackjax_api_pins.py``).

Runner signature (uniform across all warmups)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, target_acceptance_rate=None, **kwargs)
    -> (state, adapted_params)

The ``adapted_params`` dict always contains at least ``"step_size"``
and ``"inverse_mass_matrix"`` on successful adaptation.  If the
``base_method`` has a BO-tunable HP that is NOT step_size or
inverse_mass_matrix (e.g. ``num_integration_steps`` for HMC), the
default value for that HP is injected into the ``window_adaptation``
call so the warmup kernel can construct itself; BO trials later override
those HPs via trial_params.
"""

from __future__ import annotations

from typing import Any

import blackjax
import jax

from bjx_bench.inference.warmup._base import Warmup

__all__ = ["ENTRY"]


def _runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,
    base_method: Any,  # BaseMethod; not imported to avoid circular dep at module level
    *,
    logdensity_fn: Any,
    target_acceptance_rate: float | None = None,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run blackjax.window_adaptation and return ``(state, adapted_params)``.

    Parameters
    ----------
    rng_key
        JAX random key for the adaptation run.
    init_position
        Initial unconstrained parameter dict (from the model's prior sample).
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
    **kwargs
        Additional keyword arguments forwarded to ``window_adaptation``
        (e.g. ``num_integration_steps`` for HMC — the warmup kernel needs
        it to build its leapfrog integrator, even though BO will tune it
        later).

    Returns
    -------
    state
        Post-warmup BlackJAX kernel state.
    adapted_params
        Dict with at least ``"step_size"`` and ``"inverse_mass_matrix"``.
    """
    from bjx_bench.calibration.tier_b import default_value_for_space

    target = target_acceptance_rate or base_method.target_acceptance_rate or 0.80

    # Build extra kwargs for the warmup call: inject default values for any
    # HP that the kernel needs during warmup but is NOT step_size or
    # inverse_mass_matrix (those come from the adaptation itself).
    extra_kwargs: dict[str, Any] = dict(kwargs)  # caller-supplied overrides first
    for space in base_method.default_hp_space:
        if space.name not in ("step_size", "inverse_mass_matrix"):
            if space.name not in extra_kwargs:
                extra_kwargs[space.name] = default_value_for_space(space)

    warmup = blackjax.window_adaptation(
        base_method.factory,
        logdensity_fn,
        target_acceptance_rate=target,
        **extra_kwargs,
    )
    (state, adapted_params), _info = warmup.run(rng_key, init_position, n_warmup)
    return state, dict(adapted_params)


ENTRY = Warmup(
    name="stan_window",
    runner=_runner,
    compatible_methods=("hmc", "nuts", "barker", "mala"),
    notes=(
        "Standard Stan window adaptation: dual-averaging step_size + diagonal "
        "mass matrix.  Compatible with hmc, nuts, barker, mala (all kernels "
        "that accept inverse_mass_matrix).  Verified in Phase 2 tripwire tests."
    ),
)
