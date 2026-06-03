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
    variance.  Extra kwarg: ``target_acceptance`` (float, default 0.65 for HMC).

Per-cell classification guide (statistician-confirmed, Phase 8B.1)
-------------------------------------------------------------------
The choice of update strategy depends on the posterior geometry of each SMC
cell.  Use this table when wiring new cells:

**step_size_from_acceptance_rate (step_size-only)**
    - Appropriate for: near-Gaussian or smoothly-sharpening posteriors (GLMs,
      logistic regression, well-conditioned likelihoods).
    - Rationale: the particle-cloud covariance is well-behaved throughout
      tempering; IMM-from-particles adds lag (computed from the *current*
      particle cloud which lags the target) and can overshoot in the final
      temperature jump.  Step-size adaptation is sufficient.
    - Examples: logistic_synthetic, gmm_25 (moot — 1 temp step), any GLM.

**step_size_and_imm_from_particles (step_size + IMM)**
    - Appropriate for: strongly non-Gaussian or scale-varying posteriors
      (funnels, heavy tails, hierarchical models with scale uncertainty).
    - Rationale: the diagonal IMM tracks per-dimension curvature changes
      across temperatures; beneficial when the posterior geometry shifts
      substantially from prior to posterior (funnel compression, tail shrinkage).
    - Examples: neals_funnel (certified PASS z=1.579 with this strategy).
    - Warning: do NOT use for near-Gaussian cells where IMM lag causes overshoot.

**none**
    - Appropriate for: plain adaptive_tempered_smc (no inner_kernel_tuning),
      or cells where all kernel params are pre-bound via functools.partial.
    - Examples: adaptive_tempered_smc + RWM baseline cells.
"""

from typing import Any

from blackjax.smc.tuning.from_kernel_info import update_scale_from_acceptance_rate

__all__ = ["PARAMETER_UPDATE_STRATEGIES", "build_parameter_update_fn"]


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _make_none_update_fn(
    initial_parameter_value: dict[str, Any] | None = None, **kwargs: Any
) -> Any:
    """No-op: returns initial_parameter_value unchanged each step.

    For inner_kernel_tuning, we must return the full parameter dict (all keys)
    — returning {} would drop all params from parameter_override.
    """
    _ip = dict(initial_parameter_value) if initial_parameter_value else {}

    def _update(rng_key: Any, smc_state: Any, smc_info: Any) -> dict:
        return _ip

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


def _make_step_size_from_acceptance_fn(
    initial_parameter_value: dict[str, Any] | None = None,
    target_acceptance: float = 0.65,
    **kwargs: Any,
) -> Any:
    """Step-size adaptation from per-particle acceptance rates.

    Uses ``initial_parameter_value["step_size"]`` as the reference scale for
    ``update_scale_from_acceptance_rate``.  NOTE: blackjax inner_kernel_tuning
    passes the UNDERLYING ``TemperedSMCState`` (not ``StateWithParameterOverride``)
    to this function, so the CURRENT step_size is not accessible from ``smc_state``.
    We use the initial scale as a fixed reference and adapt relative to it.
    Must return ALL keys from ``initial_parameter_value`` to preserve pytree structure.
    """
    _ip = dict(initial_parameter_value) if initial_parameter_value else {}
    _init_ss = _ip.get("step_size", None)

    def _update(rng_key: Any, smc_state: Any, smc_info: Any) -> dict:
        result = dict(_ip)  # start with full initial dict (preserves all keys)
        if _init_ss is not None:
            try:
                acceptance_rates = smc_info.update_info.acceptance_rate
                result["step_size"] = update_scale_from_acceptance_rate(
                    _init_ss, acceptance_rates, target_acceptance_rate=target_acceptance
                )
            except Exception:  # noqa: BLE001
                pass  # leave result["step_size"] as initial value
        return result

    return _update


def _make_imm_from_particles_fn(
    initial_parameter_value: dict[str, Any] | None = None, **kwargs: Any
) -> Any:
    """Diagonal IMM from particle cloud variance.

    Must return ALL keys from ``initial_parameter_value`` (inner_kernel_tuning
    replaces the full parameter_override dict; missing keys cause scan errors).

    The IMM is returned as a diagonal 1D array broadcast to match the shape of
    ``initial_parameter_value["inverse_mass_matrix"]``.  For example, if the
    initial IMM has shape ``(N, d)`` (per-particle), the update returns
    ``(N, d)`` with the same diagonal variance vector tiled N times.
    This keeps the JAX scan pytree shape constraint satisfied.
    """
    _ip = dict(initial_parameter_value) if initial_parameter_value else {}
    _init_imm = _ip.get("inverse_mass_matrix", None)

    def _update(rng_key: Any, smc_state: Any, smc_info: Any) -> dict:
        result = dict(_ip)  # start with full initial dict
        if _init_imm is None:
            return result
        try:
            import jax.numpy as _jnp  # noqa: PLC0415
            from blackjax.smc.tuning.from_particles import (  # noqa: PLC0415
                particles_as_rows,
            )

            particles = _get_particles(smc_state)
            # Compute per-dim variance across all particles → diagonal IMM.
            particles_flat = particles_as_rows(particles)  # (N, d)
            var = _jnp.var(particles_flat, axis=0)  # (d,) diagonal
            # Clip to prevent degenerate near-zero variances.
            var = _jnp.maximum(var, 1e-6)
            # Broadcast to match initial IMM shape (e.g. (N, d) or (1, d)).
            target_shape = _jnp.asarray(_init_imm).shape
            if len(target_shape) == 1:
                new_imm = var  # (d,)
            else:
                n_particles = target_shape[0]
                new_imm = _jnp.tile(var[None, :], (n_particles, 1))  # (N, d)
            result["inverse_mass_matrix"] = new_imm
        except Exception:  # noqa: BLE001
            pass  # leave IMM as initial value
        return result

    return _update


def _make_step_size_and_imm_fn(
    initial_parameter_value: dict[str, Any] | None = None,
    target_acceptance: float = 0.65,
    **kwargs: Any,
) -> Any:
    """Combined: step_size from acceptance rate + diagonal IMM from particles.

    Recommended default for HMC inner kernel (Phase 8B.1). Uses
    ``initial_parameter_value`` as the structural template so that ALL keys
    are preserved (required by blackjax inner_kernel_tuning's scan pytree check).

    Since blackjax passes ``TemperedSMCState`` (not ``StateWithParameterOverride``)
    to this function, the current step_size is not accessible — we use
    ``initial_parameter_value["step_size"]`` as the reference scale.
    """
    _ip = dict(initial_parameter_value) if initial_parameter_value else {}
    _init_ss = _ip.get("step_size", None)

    def _update(rng_key: Any, smc_state: Any, smc_info: Any) -> dict:
        result = dict(_ip)  # start with full initial dict (preserves all keys)
        # Update step_size using acceptance_rate from current MCMC mutation.
        if _init_ss is not None:
            try:
                acceptance_rates = smc_info.update_info.acceptance_rate
                result["step_size"] = update_scale_from_acceptance_rate(
                    _init_ss, acceptance_rates, target_acceptance_rate=target_acceptance
                )
            except Exception:  # noqa: BLE001
                pass  # leave result["step_size"] as initial value
        # Update IMM from particle cloud variance (shape must match initial IMM).
        if _ip.get("inverse_mass_matrix") is not None:
            try:
                import jax.numpy as _jnp  # noqa: PLC0415
                from blackjax.smc.tuning.from_particles import (  # noqa: PLC0415
                    particles_as_rows,
                )

                particles = _get_particles(smc_state)
                particles_flat = particles_as_rows(particles)  # (N, d)
                var = _jnp.var(particles_flat, axis=0)  # (d,) diagonal variances
                var = _jnp.maximum(var, 1e-6)  # clip near-zero variances
                # Broadcast to match initial IMM shape: (N, d) or (1, d) or (d,).
                target_shape = _jnp.asarray(_ip["inverse_mass_matrix"]).shape
                if len(target_shape) == 1:
                    result["inverse_mass_matrix"] = var  # (d,)
                else:
                    n_p = target_shape[0]
                    result["inverse_mass_matrix"] = _jnp.tile(var[None, :], (n_p, 1))
            except Exception:  # noqa: BLE001
                pass  # leave IMM as initial value
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
    *,
    initial_parameter_value: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Resolve a strategy name to a ``(key, smc_state, smc_info) -> dict`` callable.

    The returned callable is passed as ``mcmc_parameter_update_fn`` to
    ``blackjax.smc.inner_kernel_tuning``.  Blackjax calls it with
    ``(rng_key, inner_smc_state, smc_info)`` where ``inner_smc_state`` is the
    UNDERLYING SMC state (e.g. ``TemperedSMCState``), NOT the
    ``StateWithParameterOverride`` wrapper.  This means the current
    ``parameter_override`` dict (with step_size + IMM) is NOT accessible via
    ``inner_smc_state`` — it must be captured in the closure via
    ``initial_parameter_value``.

    The returned callable MUST return a dict with exactly the same keys as
    ``initial_parameter_value`` (blackjax inner_kernel_tuning replaces the
    full ``parameter_override`` dict on each step; partial returns cause
    ``jax.lax.scan`` pytree structure mismatches).

    Parameters
    ----------
    strategy_name
        One of the keys in ``PARAMETER_UPDATE_STRATEGIES``.
    initial_parameter_value
        The initial ``mcmc_parameters`` dict (e.g.
        ``{"step_size": jnp.ones(N)*0.1, "inverse_mass_matrix": jnp.ones((N, d))}``).
        Required for strategies that update step_size (captured in closure as
        the reference scale).  ``None`` falls back to IMM-only update.
    **kwargs
        Strategy-specific keyword arguments.  For step_size strategies:
        ``target_acceptance`` (float, default 0.65 for HMC).

    Returns
    -------
    Callable
        A ``(rng_key, smc_state, smc_info) -> dict[str, Array]`` callable.

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
    return _FACTORIES[strategy_name](
        initial_parameter_value=initial_parameter_value, **kwargs
    )
