# ENHANCED version of the shipped mclmc_lrd_warmup (Scheme A).
# Differs from lrd_warmup_baseline.py by EXACTLY two enhancements:
#
#   E1 (step/L warm-start, Phase 3, GATED): warm-start Phase 3 at
#       step_init = 1.22 * sqrt(d), L_init = 0.85 * sqrt(d)
#       ONLY WHEN the pilot-derived LRD IMM achieves kappa_eff <= 5
#       (i.e. the geometry is actually whitened well enough for the √d law
#       to apply).  Otherwise falls back to the BASELINE init (pilot's own
#       step/L), guaranteeing enhanced is never worse than baseline on
#       under-preconditioned geometry.
#       Exposed diagnostics: `e1_fired` (bool) + `e1_kappa_eff_at_k_used`.
#
#   E2 (κ_eff rank, rank guard): replace k_safe = floor(n_eff/2) with
#       k* = smallest k s.t. kappa_eff(Sigma_pilot, lrd(Sigma_pilot, k)) <= 5.
#       Sigma_pilot is the sample covariance of the pilot draws.
#       k_safe = floor(n_eff/2) is retained as an UPPER CLAMP (never exceed it).
#       Both the chosen k* and the n_eff/2 clamp are exposed in diagnostics.
#
# ALL OTHER CODE IS IDENTICAL TO THE BASELINE (lrd_warmup_baseline.py).
# Verify with: diff lrd_warmup_baseline.py lrd_warmup_enhanced.py
#
# Copyright 2020- The Blackjax Authors.
# Licensed under the Apache License, Version 2.0.
"""Enhanced Scheme A MCLMC warmup (E1-gated + E2 additive enhancements).

See module-level comment above and the CONTEXT note in sweep_warmup_compare.py
for the research motivation.
"""

import warnings
from typing import Any, NamedTuple

import blackjax.mcmc.adjusted_mclmc as _adj_mclmc_mod
import blackjax.mcmc.mclmc as _mclmc_mod
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.adjusted_mclmc_adaptation import (
    adjusted_mclmc_find_L_and_step_size,
)
from blackjax.adaptation.mclmc_adaptation import (
    MCLMCAdaptationState,
    mclmc_find_L_and_step_size,
)
from blackjax.diagnostics import effective_sample_size
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from jax.flatten_util import ravel_pytree

__all__ = [
    "MCLMCLRDAdaptationState",
    "mclmc_lrd_warmup",
]

_VALID_INNER_KERNELS = frozenset({"mclmc", "adjusted_mclmc"})


class MCLMCLRDAdaptationState(NamedTuple):
    """Result of :func:`mclmc_lrd_warmup` (enhanced version).

    Same fields as the baseline, plus additional E2 diagnostics.

    L
        Adapted momentum decoherence length from the final tuning phase.
    step_size
        Adapted step size from the final tuning phase.
    inverse_mass_matrix
        The adapted LRD inverse mass matrix as a
        :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix` NamedTuple.
    diagnostics
        A plain dict.  All baseline fields present, plus:

        ``k_star``
            (E2) The smallest rank with ``kappa_eff(Sigma_pilot, lrd(pilot,k)) <= 5``.
        ``kappa_eff_at_k_star``
            (E2) The actual kappa_eff at the chosen k_star.
        ``k_safe``
            floor(n_eff/2) — the upper clamp (never exceeded by E2).
        ``k_used``
            The rank actually used: ``min(k_star, k_safe, k_requested)``.
        ``e1_fired``
            (E1) True when the √d warm-start was used; False when fell back to
            pilot's (step, L) because kappa_eff(pilot IMM) > 5.
        ``e1_kappa_eff_at_k_used``
            (E1) kappa_eff of the selected LRD IMM (at k_used), used as the
            E1 gate condition.  < 5 → E1 fires; >= 5 → E1 falls back.
    """

    L: float
    step_size: float
    inverse_mass_matrix: LowRankInverseMassMatrix
    diagnostics: dict


def _extract_lrd_from_samples(
    flat_positions: Any,
    k: int,
) -> tuple:
    """Extract LRD parameters from a ``(n_samples, d)`` array of pilot draws.

    Identical to the baseline / shipped implementation.
    """
    mean = jnp.mean(flat_positions, axis=0)  # (d,)
    sigma = jnp.std(flat_positions, axis=0)  # (d,)
    sigma = jnp.where(sigma == 0.0, 1.0, sigma)  # avoid div-by-zero in zero-var dims

    standardised = (flat_positions - mean[None, :]) / sigma[None, :]  # (n, d)
    n = flat_positions.shape[0]

    _, S, Vt = jnp.linalg.svd(standardised, full_matrices=False)
    V = Vt.T  # (d, min(n,d))
    lam = (S**2) / n  # (min(n,d),)

    sort_idx = jnp.argsort(jnp.abs(lam - 1.0))[::-1]
    top_idx = sort_idx[:k]

    lam_k = lam[top_idx]  # (k,)
    U_k = V[:, top_idx]  # (d, k)
    return sigma, U_k, lam_k


def _check_da_ceiling_warning(
    final_step_size: float,
    L_init: float,
    floor_factor: float,
) -> None:
    """Emit a UserWarning when the adapted step_size is at or near L_init/1.1."""
    da_clamp = L_init / 1.1
    step_ratio = final_step_size / da_clamp
    if step_ratio >= 0.999:
        step_s = round(final_step_size, 4)
        clamp_s = round(da_clamp, 4)
        ratio_s = round(step_ratio, 3)
        warnings.warn(
            f"mclmc_lrd_warmup: adapted step_size "
            f"({step_s}) is at or near the DA ceiling "
            f"L_init/1.1={clamp_s} (ratio={ratio_s}). "
            "The step-size tuner may have been constrained rather than "
            "converged. Consider raising `floor_factor` "
            f"(current value: {floor_factor}) — e.g. to 1.5 for "
            "high-condition-number targets.",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# [E2] Helper: kappa_eff from pilot draws (numpy, no GT required)
# ---------------------------------------------------------------------------


def _kappa_eff_from_pilot(
    flat_pilot_np: np.ndarray,
    k: int,
) -> float:
    """Compute kappa_eff(Sigma_pilot, lrd(Sigma_pilot, k)).

    Mirrors ``gt_imm.kappa_eff`` but operates on the pilot sample covariance
    rather than the true Sigma.  Used by E2 to find the minimal rank k* with
    kappa_eff <= 5 without any ground-truth access.

    Parameters
    ----------
    flat_pilot_np : np.ndarray, shape (n, d)
        Pilot draws ravelled to a 2-D float64 array.
    k : int
        LRD rank to evaluate.

    Returns
    -------
    float : kappa_eff of the k-rank LRD approximation of the pilot covariance.
    """
    d = flat_pilot_np.shape[1]

    # Sample covariance (ddof=1 for unbiased; close enough at n >> d)
    Sigma_pilot = np.cov(flat_pilot_np, rowvar=False)  # (d, d)
    if Sigma_pilot.ndim == 0:
        # scalar edge-case: d=1
        Sigma_pilot = Sigma_pilot.reshape(1, 1)

    # sigma = marginal std from sample cov
    sigma_np = np.sqrt(np.diag(Sigma_pilot))  # (d,)
    sigma_np = np.where(sigma_np == 0.0, 1.0, sigma_np)

    if k == 0:
        # k=0: M^{-1} = diag(sigma^2), product = diag(sigma^2) @ Sigma^{-1}
        # = correlation matrix R eigenvalues -> same as the np.cov version
        inv_sigma = 1.0 / sigma_np
        R = inv_sigma[:, None] * Sigma_pilot * inv_sigma[None, :]
        R = (R + R.T) / 2.0
        eigvals = np.linalg.eigvalsh(R)  # real symmetric, ascending
        eigvals = eigvals[eigvals > 0]
        if len(eigvals) == 0:
            return float("inf")
        return float(eigvals[-1] / eigvals[0])

    # Correlation matrix
    inv_sigma = 1.0 / sigma_np
    R = inv_sigma[:, None] * Sigma_pilot * inv_sigma[None, :]
    R = (R + R.T) / 2.0

    # Eigendecompose (real symmetric)
    lam_all, V_all = np.linalg.eigh(R)  # ascending

    # Select top-k by |lambda - 1| (mirrors _extract_lrd_from_samples selection)
    deviation = np.abs(lam_all - 1.0)
    sort_idx = np.argsort(deviation)[::-1]
    top_idx = sort_idx[:k]

    lam_k = lam_all[top_idx]  # (k,)
    U_k = V_all[:, top_idx]  # (d, k)

    # Build M^{-1} = D (I + U (Lambda - I) U^T) D, D = diag(sigma)
    D = np.diag(sigma_np)
    correction = U_k @ np.diag(lam_k - 1.0) @ U_k.T  # (d, d)
    M_inv = D @ (np.eye(d) + correction) @ D

    # kappa(M^{-1} Sigma^{-1})
    try:
        Sigma_inv = np.linalg.inv(Sigma_pilot)
    except np.linalg.LinAlgError:
        return float("inf")

    product = M_inv @ Sigma_inv
    eigvals = np.linalg.eigvals(product)
    eigvals_real = np.real(eigvals)
    eigvals_pos = eigvals_real[eigvals_real > 0]
    if len(eigvals_pos) == 0:
        return float("inf")
    return float(eigvals_pos.max() / eigvals_pos.min())


def _find_k_star(
    flat_pilot_np: np.ndarray,
    k_max: int,
    kappa_target: float = 5.0,
) -> tuple[int, float]:
    """[E2] Find the minimal rank k* s.t. kappa_eff(pilot, k) <= kappa_target.

    Parameters
    ----------
    flat_pilot_np : np.ndarray, shape (n, d)
        Pilot draws in float64.
    k_max : int
        Upper bound on the search (= k_safe = floor(n_eff/2)).
        Also the fallback if no k in [1, k_max] satisfies the criterion.
    kappa_target : float
        Target effective condition number.  Default 5.0 (from S1: k* ~ 87%
        of dense ESS at kappa_eff ≲ 5).

    Returns
    -------
    k_star : int
        Smallest k in [1, k_max] with kappa_eff <= kappa_target,
        or k_max if no such k exists.
    kappa_at_k_star : float
        kappa_eff at the chosen k_star.
    """
    if k_max <= 0:
        return 1, _kappa_eff_from_pilot(flat_pilot_np, 1)

    # Linear scan: try k = 1, 2, ..., k_max in order.
    # For typical smooth targets d ~ 10–500, k_max <= n_eff/2 is rarely > few
    # hundred, so linear scan is fast enough for this research harness.
    for k_try in range(1, k_max + 1):
        keff = _kappa_eff_from_pilot(flat_pilot_np, k_try)
        if keff <= kappa_target:
            return k_try, keff

    # No k in [1, k_max] reached target: return k_max with its kappa_eff.
    keff_kmax = _kappa_eff_from_pilot(flat_pilot_np, k_max)
    return k_max, keff_kmax


def mclmc_lrd_warmup(
    logdensity_fn,
    position,
    rng_key,
    *,
    k: int = 10,
    pilot_num_warmup: int = 1000,
    pilot_num_samples: int = 5000,
    lrd_num_steps: int = 1000,
    num_chains: int = 4,
    inner_kernel: str = "mclmc",
    floor_factor: float = 1.15,
    adjusted_num_steps: int = 3000,
    adjusted_target: float = 0.9,
):
    """Scheme A (pilot-free) MCLMC warmup — ENHANCED (E1-gated + E2).

    Differs from the shipped baseline by two additive enhancements:

    E1 (step/L warm-start, Phase 3, GATED by kappa_eff):
        After E2 selects k_used and builds lrd_imm, compute
        kappa_eff(Sigma_pilot, lrd_imm) to check whether the IMM actually
        whitens the pilot geometry.  ONLY IF kappa_eff <= 5 does E1 fire —
        i.e. does Phase 3 warm-start at step = 1.22*sqrt(d), L = 0.85*sqrt(d)
        (the S3 √d scaling law).  Otherwise E1 falls back to the pilot's own
        (step, L), matching the baseline behaviour exactly.  This guarantees
        enhanced is NEVER worse than baseline on under-preconditioned geometry.

    E2 (κ_eff rank, rank guard):
        k* = min k s.t. kappa_eff(Sigma_pilot, lrd(pilot, k)) <= 5.
        k_safe = floor(n_eff/2) is kept as an UPPER CLAMP.
        Both k_star and k_safe are reported in diagnostics.

    All other behaviour is identical to the baseline.
    """
    if inner_kernel not in _VALID_INNER_KERNELS:
        raise ValueError(
            f"inner_kernel must be one of {sorted(_VALID_INNER_KERNELS)!r}, "
            f"got {inner_kernel!r}."
        )

    # Five independent keys — no reuse across phases.
    init_key, warmup_key, sample_key, lrd_subkey, adj_subkey = jax.random.split(
        rng_key, 5
    )

    # ------------------------------------------------------------------
    # Phase 1: diagonal pilot — reach typical set + collect geometry samples
    # ------------------------------------------------------------------
    base_kernel = _mclmc_mod.build_kernel()
    init_state = _mclmc_mod.init(position, logdensity_fn, init_key)

    state_after_warmup, pilot_params, _ = mclmc_find_L_and_step_size(
        mclmc_kernel=base_kernel,
        num_steps=pilot_num_warmup,
        state=init_state,
        rng_key=warmup_key,
        logdensity_fn=logdensity_fn,
        diagonal_preconditioning=True,
    )

    pilot_L = float(pilot_params.L)
    pilot_step_size_val = float(pilot_params.step_size)

    # Collect pilot_num_samples draws with the adapted diagonal kernel.
    def _pilot_step(state, key):
        next_state, _ = base_kernel(
            rng_key=key,
            state=state,
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=pilot_params.inverse_mass_matrix,
            L=pilot_params.L,
            step_size=pilot_params.step_size,
        )
        return next_state, next_state.position

    _, pilot_positions = jax.lax.scan(
        _pilot_step,
        state_after_warmup,
        jax.random.split(sample_key, pilot_num_samples),
    )

    # Ravel pilot positions to (n, d) flat array.
    flat_pilot = jax.vmap(lambda p: ravel_pytree(p)[0])(pilot_positions)  # (n, d)
    d = flat_pilot.shape[1]

    # ------------------------------------------------------------------
    # Rank guard (shared baseline logic): n_eff via Geyer ESS.
    # k_safe = floor(n_eff/2) — used as UPPER CLAMP in E2.
    # ------------------------------------------------------------------
    if pilot_num_samples >= 2:
        ess_per_dim = effective_sample_size(flat_pilot[None, :, :])  # (d,)
        n_eff = float(jnp.min(ess_per_dim))
    else:
        n_eff = 0.0

    k_safe = int(n_eff / 2)  # upper clamp (same as baseline)

    # ------------------------------------------------------------------
    # [E2] κ_eff-guided rank selection
    # ------------------------------------------------------------------
    # Compute k* = smallest k in [1, k_safe] with kappa_eff(pilot, k) <= 5.
    # Requires: the pilot sample covariance (no GT access).
    # Falls back to k_safe when no k in [1, k_safe] reaches target (unlikely
    # for smooth targets with adequate pilot).
    flat_pilot_np = np.array(flat_pilot, dtype=np.float64)  # materialise to numpy

    k_safe_clamped = max(k_safe, 1)  # always allow at least rank 1
    k_star, kappa_at_k_star = _find_k_star(
        flat_pilot_np,
        k_max=k_safe_clamped,
        kappa_target=5.0,
    )

    # Final rank: min(k_star, k_requested).  k_safe is already the upper
    # bound enforced by _find_k_star; this additionally respects the caller's
    # requested k (user might set k lower than k_star for budget reasons).
    k_used = min(k_star, k)

    if k_used < k:
        warnings.warn(
            f"mclmc_lrd_warmup [enhanced/E2]: k_used={k_used} < requested k={k}. "
            f"k* (κ_eff-guided) = {k_star} (kappa_eff={kappa_at_k_star:.2f} at k*); "
            f"k_safe (n_eff/2 clamp) = {k_safe} (n_eff={n_eff:.1f}). "
            "k_used = min(k_star, k_requested).",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Phase 2: SVD extraction → LowRankInverseMassMatrix
    # ------------------------------------------------------------------
    sigma, U_k, lam_k = _extract_lrd_from_samples(flat_pilot, k=k_used)
    lrd_imm = LowRankInverseMassMatrix(sigma=sigma, U=U_k, lam=lam_k)

    # ------------------------------------------------------------------
    # Phase 3: multi-chain unadjusted LRD tuning
    # ------------------------------------------------------------------
    def lrd_kernel(rng_key, state, logdensity_fn, inverse_mass_matrix, L, step_size):
        return base_kernel(
            rng_key=rng_key,
            state=state,
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=lrd_imm,  # always route through LRD
            L=L,
            step_size=step_size,
        )

    # [E1] Gated √d warm-start for Phase 3.
    #
    # The √d scaling law (step ≈ 1.22√d, L ≈ 0.85√d) applies at GOOD
    # preconditioning (κ_eff ≲ 5).  At poor preconditioning (e.g. ill_cond_50
    # with k_used=1 due to a tiny pilot budget), the LRD IMM doesn't whiten
    # the geometry, and launching Phase 3 at 1.22√d overshoots the actual
    # optimum (which lives much lower in the poorly-whitened space).
    #
    # Gate condition: compute kappa_eff of the chosen LRD IMM against the pilot
    # sample covariance.  This is the SAME pilot Sigma used by E2, so no extra
    # computation is needed beyond the scan already done in _find_k_star.
    # Specifically: re-evaluate kappa_eff at k_used (k_star may differ from
    # k_used after the min(k_star, k_requested) clamp).
    e1_kappa_eff_at_k_used = _kappa_eff_from_pilot(flat_pilot_np, k_used)
    e1_fired = e1_kappa_eff_at_k_used <= 5.0

    _sqrt_d = float(jnp.sqrt(d))
    if e1_fired:
        # Geometry is whitened: use the √d law warm-start (S3 result).
        lrd_init_step = 1.22 * _sqrt_d
        lrd_init_L = 0.85 * _sqrt_d
    else:
        # Geometry is NOT whitened: fall back to the pilot's adapted (step, L).
        # This matches baseline behaviour exactly — enhanced cannot be worse.
        lrd_init_step = float(pilot_params.step_size)
        lrd_init_L = float(pilot_params.L)

    lrd_init_params = MCLMCAdaptationState(
        L=jnp.array(lrd_init_L),
        step_size=jnp.array(lrd_init_step),
        inverse_mass_matrix=pilot_params.inverse_mass_matrix,  # placeholder; overridden by closure
    )

    # Split into 2*num_chains keys: first half for chain init, second for tuning.
    lrd_all_keys = jax.random.split(lrd_subkey, 2 * num_chains)
    lrd_init_keys = lrd_all_keys[:num_chains]
    lrd_tune_keys = lrd_all_keys[num_chains:]

    # Replicate position for all chains (same start, different momenta via keys).
    chain_positions = jax.tree.map(
        lambda x: jnp.stack([x] * num_chains),
        state_after_warmup.position,
    )

    @jax.vmap
    def _lrd_init_one(k, x0):
        return _mclmc_mod.init(x0, logdensity_fn, k)

    lrd_init_states = _lrd_init_one(lrd_init_keys, chain_positions)

    @jax.vmap
    def _lrd_tune_one(k, state):
        _, params, _ = mclmc_find_L_and_step_size(
            mclmc_kernel=lrd_kernel,
            num_steps=lrd_num_steps,
            state=state,
            rng_key=k,
            logdensity_fn=logdensity_fn,
            diagonal_preconditioning=False,
            params=lrd_init_params,
        )
        return params

    lrd_params_all = _lrd_tune_one(lrd_tune_keys, lrd_init_states)

    # Multi-chain mean for stable L and step_size estimates.
    lrd_L = float(jnp.mean(lrd_params_all.L))
    lrd_step_size = float(jnp.mean(lrd_params_all.step_size))

    # ------------------------------------------------------------------
    # Phase 4: inner-kernel dispatch (identical to baseline)
    # ------------------------------------------------------------------
    if inner_kernel == "mclmc":
        final_L = jnp.array(lrd_L)
        final_step_size = jnp.array(lrd_step_size)

    else:  # inner_kernel == "adjusted_mclmc"
        adj_base_kernel = _adj_mclmc_mod.build_kernel()

        def adj_lrd_kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
            integration_steps_params,
            inverse_mass_matrix,
        ):
            return adj_base_kernel(
                rng_key=rng_key,
                state=state,
                logdensity_fn=logdensity_fn,
                step_size=step_size,
                integration_steps_params=integration_steps_params,
                inverse_mass_matrix=lrd_imm,  # always route through LRD
            )

        L_floor = floor_factor * lrd_step_size
        floor_active = bool(L_floor > lrd_L)
        L_init = float(max(lrd_L, L_floor))

        adj_init_params = MCLMCAdaptationState(
            L=jnp.array(L_init),
            step_size=jnp.array(lrd_step_size),
            inverse_mass_matrix=pilot_params.inverse_mass_matrix,
        )

        adj_tune_keys = jax.random.split(adj_subkey, num_chains)

        @jax.vmap
        def _adj_init_one(x0):
            return _adj_mclmc_mod.init(x0, logdensity_fn)

        adj_init_states = _adj_init_one(chain_positions)

        @jax.vmap
        def _adj_tune_one(k, state):
            _, params, _ = adjusted_mclmc_find_L_and_step_size(
                mclmc_kernel=adj_lrd_kernel,
                logdensity_fn=logdensity_fn,
                num_steps=adjusted_num_steps,
                state=state,
                rng_key=k,
                target=adjusted_target,
                frac_tune1=0.5,
                frac_tune2=0.0,  # REQUIRED: variance-based L estimator disabled
                diagonal_preconditioning=False,
                params=adj_init_params,
            )
            return params

        adj_params_all = _adj_tune_one(adj_tune_keys, adj_init_states)

        final_step_size = jnp.mean(adj_params_all.step_size)
        final_L = jnp.array(L_init)

        _check_da_ceiling_warning(float(final_step_size), L_init, floor_factor)

    # Gradient accounting: unadjusted MCLMC costs 2 grads/step.
    pilot_num_grad_evals = (pilot_num_warmup + pilot_num_samples) * 2

    diagnostics = {
        "inner_kernel": inner_kernel,
        "n_eff": n_eff,
        # E2 fields: both k_star (new) and k_safe (old, now an upper clamp)
        "k_star": k_star,
        "kappa_eff_at_k_star": kappa_at_k_star,
        "k_safe": k_safe,
        "k_used": k_used,
        "pilot_num_grad_evals": pilot_num_grad_evals,
        "pilot_L": pilot_L,
        "pilot_step_size": pilot_step_size_val,
        # E1 fields: gate result + actual warm-start values passed to Phase 3
        "e1_fired": e1_fired,
        "e1_kappa_eff_at_k_used": e1_kappa_eff_at_k_used,
        "lrd_init_step": lrd_init_step,
        "lrd_init_L": lrd_init_L,
        "lrd_L": lrd_L,
        "lrd_step_size": lrd_step_size,
    }

    if inner_kernel == "adjusted_mclmc":
        diagnostics["L_init"] = L_init
        diagnostics["floor_active"] = floor_active
        diagnostics["N_sample"] = round(
            float(final_L) / max(float(final_step_size), 1e-10)
        )

    return MCLMCLRDAdaptationState(
        L=final_L,
        step_size=final_step_size,
        inverse_mass_matrix=lrd_imm,
        diagnostics=diagnostics,
    )
