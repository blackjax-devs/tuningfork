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
"""Named registry for SMC inner-kernel parameter update functions (W6, Phase 8B.1).

The ``mcmc_parameter_update_fn`` required by
``blackjax.smc.inner_kernel_tuning`` is a callable — not JSON-serialisable.
This module provides a string-key registry so that SMCRecipe can store the
update strategy as a serialisable ``(strategy_name, kwargs)`` pair and resolve
it back to a callable at build time.

Public API
----------
``build_parameter_update_fn(strategy_name, **kwargs)``
    Resolve a strategy name to a ``(key, smc_state, smc_info) -> dict[str, Array]``
    callable ready to pass as ``mcmc_parameter_update_fn``.
``PARAMETER_UPDATE_STRATEGIES``
    Dict mapping strategy name → factory description (for introspection).

Registered strategies
---------------------
``"none"``
    No update — parameters are held fixed across all SMC steps (plain
    ``adaptive_tempered_smc`` without inner-kernel tuning).  Returns
    ``{}``, leaving ``parameter_override`` unchanged.

``"step_size_from_acceptance_rate"``
    Updates step_size (scalar) using acceptance rates from the MCMC
    mutation phase.  Uses ``blackjax.smc.tuning.from_kernel_info.
    update_scale_from_acceptance_rate``.  Extra kwarg: ``target_acceptance``
    (float, default 0.65 for HMC, 0.234 for RWM).

``"imm_from_particles"``
    Updates diagonal inverse-mass-matrix from particle-cloud variance.
    Uses ``blackjax.smc.tuning.from_particles.inverse_mass_matrix_from_particles``.
    No extra kwargs.

``"step_size_and_imm_from_particles"``
    Combined: step_size from acceptance rate + diagonal IMM from particle
    variance.  The statistically recommended default for HMC inner kernel
    (Phase 8B.1, statistician verdict).  Extra kwarg: ``target_acceptance``
    (float, default 0.65 for HMC).
"""

from typing import Any

from blackjax.smc.tuning.from_kernel_info import update_scale_from_acceptance_rate
from blackjax.smc.tuning.from_particles import inverse_mass_matrix_from_particles

__all__ = ["PARAMETER_UPDATE_STRATEGIES", "build_parameter_update_fn"]


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _make_none_update_fn() -> Any:
    """No-op: holds all parameters fixed."""

    def _update(rng_key: Any, smc_state: Any, smc_info: Any) -> dict:
        return {}

    return _update


def _get_particles(smc_state: Any) -> Any:
    """Extract particles from an SMC state (handles inner_kernel_tuning wrapping)."""
    try:
        return smc_state.sampler_state.particles  # inner_kernel_tuning
    except AttributeError:
        return smc_state.particles  # plain adaptive_tempered_smc


def _get_parameter_override(smc_state: Any) -> dict:
    """Extract the current parameter_override dict from an SMC state."""
    try:
        return dict(smc_state.parameter_override)  # StateWithParameterOverride
    except AttributeError:
        return {}


def _make_step_size_from_acceptance_fn(target_acceptance: float = 0.65) -> Any:
    """Step-size adaptation from per-particle acceptance rates.

    CRITICAL: must return a dict with the SAME keys as the current
    parameter_override (blackjax inner_kernel_tuning replaces the full
    parameter_override dict on each step — partial returns cause pytree
    structure mismatches in jax.lax.scan).
    """

    def _update(rng_key: Any, smc_state: Any, smc_info: Any) -> dict:
        current_params = _get_parameter_override(smc_state)
        # smc_info.update_info carries MCMC step info (HMCInfo / RWMInfo).
        try:
            acceptance_rates = smc_info.update_info.acceptance_rate
            current_ss = current_params["step_size"]
            new_ss = update_scale_from_acceptance_rate(
                current_ss, acceptance_rates, target_acceptance_rate=target_acceptance
            )
            return {**current_params, "step_size": new_ss}
        except (AttributeError, KeyError):
            # Kernel doesn't expose acceptance_rate or step_size not in params.
            # Return current params unchanged to preserve pytree structure.
            return current_params

    return _update


def _make_imm_from_particles_fn() -> Any:
    """Diagonal IMM from particle cloud variance.

    CRITICAL: must return all current parameter keys (not just imm) to preserve
    the pytree structure expected by blackjax inner_kernel_tuning.
    """

    def _update(rng_key: Any, smc_state: Any, smc_info: Any) -> dict:
        current_params = _get_parameter_override(smc_state)
        try:
            particles = _get_particles(smc_state)
            new_imm = inverse_mass_matrix_from_particles(particles)
            return {**current_params, "inverse_mass_matrix": new_imm}
        except AttributeError:
            return current_params

    return _update


def _make_step_size_and_imm_fn(target_acceptance: float = 0.65) -> Any:
    """Combined: step_size from acceptance + diagonal IMM from particles.

    Recommended default for HMC inner kernel (Phase 8B.1). Always returns
    a dict with all current parameter keys to preserve the pytree structure
    required by blackjax inner_kernel_tuning's scan loop.
    """

    def _update(rng_key: Any, smc_state: Any, smc_info: Any) -> dict:
        current_params = _get_parameter_override(smc_state)
        # MUST start from current_params and preserve ALL keys — inner_kernel_tuning
        # REPLACES parameter_override with the returned dict; any missing key
        # causes a jax.lax.scan pytree structure mismatch.
        result = dict(current_params)
        # Update step_size if acceptance_rate is available.
        try:  # noqa: SIM105
            acceptance_rates = smc_info.update_info.acceptance_rate
            if "step_size" in result:
                result["step_size"] = update_scale_from_acceptance_rate(
                    result["step_size"],
                    acceptance_rates,
                    target_acceptance_rate=target_acceptance,
                )
        except Exception:  # noqa: BLE001 — catch any JAX/Python exception
            pass  # preserve current step_size unchanged
        # Update IMM from particle cloud variance.
        try:  # noqa: SIM105
            particles = _get_particles(smc_state)
            result["inverse_mass_matrix"] = inverse_mass_matrix_from_particles(
                particles
            )
        except Exception:  # noqa: BLE001
            pass  # preserve current IMM unchanged
        return result

    return _update


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PARAMETER_UPDATE_STRATEGIES: dict[str, str] = {
    "none": "No update — holds all parameters fixed (plain adaptive_tempered_smc).",
    "step_size_from_acceptance_rate": (
        "Step-size adaptation from per-particle acceptance rates. "
        "Extra kwarg: target_acceptance (float, default 0.65 for HMC)."
    ),
    "imm_from_particles": (
        "Diagonal IMM from particle-cloud variance "
        "(Buchholz et al. 2019 §3.1). No extra kwargs."
    ),
    "step_size_and_imm_from_particles": (
        "Combined: step_size from acceptance rate + diagonal IMM from particle "
        "variance. Recommended for HMC inner kernel (Phase 8B.1). "
        "Extra kwarg: target_acceptance (float, default 0.65 for HMC)."
    ),
}

_FACTORIES: dict = {
    "none": _make_none_update_fn,
    "step_size_from_acceptance_rate": _make_step_size_from_acceptance_fn,
    "imm_from_particles": _make_imm_from_particles_fn,
    "step_size_and_imm_from_particles": _make_step_size_and_imm_fn,
}


def build_parameter_update_fn(
    strategy_name: str,
    **kwargs: Any,
) -> Any:
    """Resolve a strategy name to a ``(key, smc_state, smc_info) -> dict`` callable.

    Parameters
    ----------
    strategy_name
        One of the keys in ``PARAMETER_UPDATE_STRATEGIES``.
    **kwargs
        Strategy-specific keyword arguments.  For
        ``step_size_from_acceptance_rate`` and
        ``step_size_and_imm_from_particles``: ``target_acceptance`` (float).

    Returns
    -------
    Callable
        A ``(rng_key, smc_state, smc_info) -> dict[str, Array]`` callable
        suitable for ``mcmc_parameter_update_fn`` in
        ``blackjax.smc.inner_kernel_tuning``.

    Raises
    ------
    KeyError
        If ``strategy_name`` is not registered.
    """
    if strategy_name not in _FACTORIES:
        raise KeyError(
            f"Unknown parameter_update_strategy {strategy_name!r}. "
            f"Valid: {sorted(_FACTORIES.keys())}"
        )
    return _FACTORIES[strategy_name](**kwargs)
