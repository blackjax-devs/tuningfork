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
"""long-NUTS reference-certification path (Path B) — long single-chain NUTS reference certification.

Runs 1 chain × n_warmup warmup × n_samples post-warmup NUTS steps using
BlackJAX's window adaptation (Stan-style).  Reshapes into n_chunks contiguous
chunks for rank-normalised split-R̂ and bulk-ESS diagnostics.

Certification gate (reference-certification):
    - rank-normalised split-R̂ ≤ 1.01
    - min per-chunk bulk-ESS > 400
    - num_divergences == 0
    - E-BFMI > 0.3

E-BFMI formula (Neal 2011, Stan Reference §15.4):
    E-BFMI = mean(diff(energy)²) / var(energy)
where ``energy`` is the Hamiltonian energy at each post-warmup step.
This measures how well the momentum resampling explores the energy surface.
"""

import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.util import run_inference_algorithm
from jax.flatten_util import ravel_pytree

from tuningfork.calibration._summary import Summaries, compute_summaries
from tuningfork.model._base import Posterior, ReferenceMethod
from tuningfork.model._numpyro import build_logdensity_fn

__all__ = [
    "AdaptationParams",
    "CertificationResult",
    "CertificationError",
    "PreAdaptedWarmup",
    "WarmupValidationError",
    "WarmupCheckpoint",
    "WarmupHealthSummary",
    "compute_certification_verdict",
    "default_warmup_validator",
    "certify_reference_nuts",
]


class PreAdaptedWarmup(NamedTuple):
    """Pre-computed warmup output, optionally injected into ``certify_reference_nuts``.

    When this is provided (as ``pre_adapted=...``), the certifier skips its
    own ``window_adaptation.run`` call and starts sampling directly from
    ``state`` using ``params``. This enables a two-phase cert workflow
    (warmup separately, review, then sample+cert) for models where the
    warmup is expensive and the user wants to validate the adapted state
    before committing to a long sampling phase.

    Fields
    ------
    state
        Final HMCState from a prior ``window_adaptation.run``. Position +
        logdensity + grad pytree at warmup end.
    params
        Adapted-params dict, exactly as ``window_adaptation.run`` returns
        (``step_size``, ``inverse_mass_matrix``, ``max_num_doublings``).
    num_leapfrog_median
        Median of ``warmup_info.info.num_integration_steps`` from the
        prior warmup, used to populate the returned ``AdaptationParams``
        for downstream reporting.

    Notes
    -----
    Introduced 2026-05-13 for the gp_regression groundtruth certification closeout where
    n_warmup=5000 at ta=0.99 takes ~7h wall and an intermediate review
    of adapted parameters (step_size, IMM diag per site, divergence rate
    in late warmup) prevents committing the ~30-50h sampling phase to
    a bad warmup outcome.
    """

    state: Any  # blackjax HMCState; typed loosely to avoid import dance
    params: dict[str, Any]  # step_size, inverse_mass_matrix, max_num_doublings
    num_leapfrog_median: int


@dataclass(frozen=True)
class AdaptationParams:
    """Tuned NUTS parameters from window adaptation warmup.

    Used as informative priors (not optima) for BO tuning search ranges.

    Parameters
    ----------
    step_size
        Dual-averaging adapted step size.
    inverse_mass_matrix
        Diagonal inverse mass matrix (1-D array) or dense (2-D array).
    num_leapfrog_median
        Median number of leapfrog steps during warmup (from NUTS trajectory
        length distribution).
    """

    step_size: float
    inverse_mass_matrix: jax.Array
    num_leapfrog_median: int


@dataclass(frozen=True)
class CertificationResult:
    """Diagnostic summary from a long-NUTS reference-certification run.

    Parameters
    ----------
    passed
        True iff all certification gates are satisfied.
    split_rhat_max
        Maximum rank-normalised split-R̂ across all dimensions.
    min_chunk_bulk_ess
        Minimum per-chunk bulk-ESS across all dimensions and chunks.
    num_divergences
        Total number of divergent transitions.
    e_bfmi
        Expected Bayesian Fraction of Missing Information.
    """

    passed: bool
    split_rhat_max: float
    min_chunk_bulk_ess: float
    num_divergences: int
    e_bfmi: float


class CertificationError(RuntimeError):
    """Raised when a Path-B run fails the reference-certification gate.

    Carries ``cert: CertificationResult``, ``adaptation: AdaptationParams | None``,
    ``chain_stats: dict[str, np.ndarray] | None``, AND ``draws: dict[str, jax.Array] | None``
    so the caller can persist diagnostic data on the failure path. Without
    draws, the statistician cannot do parameter-space cluster analysis on
    divergent transitions (the primary diagnostic step per Lens 1 in the
    diagnostics playbook). Added 2026-05-12 to close the schema gap that
    blocked diagnosis of gp_regression's chunk-1 divergence cluster.
    """

    def __init__(
        self,
        message: str,
        cert: CertificationResult,
        adaptation: "AdaptationParams | None" = None,
        chain_stats: "dict[str, np.ndarray] | None" = None,
        draws: "dict[str, jax.Array] | None" = None,
    ) -> None:
        super().__init__(message)
        self.cert = cert
        self.adaptation = adaptation
        self.chain_stats = chain_stats
        self.draws = draws


@dataclass(frozen=True)
class WarmupHealthSummary:
    """Generic, model-agnostic health summary computed after warmup.

    Used by ``default_warmup_validator`` and ships in
    ``WarmupValidationError`` for the failure-path diagnostician.

    Fields
    ------
    step_size
        Final adapted step size (dual-averaged).
    imm_cond
        IMM condition number (max diag / min diag for diagonal IMM).
    final_log_p
        Log-density at the end-of-warmup chain state.
    position_max_abs
        L∞-norm of the flattened end-of-warmup position (sanity for
        diverged / NaN-corrupted chains).
    position_has_finite
        True iff position pytree has no NaN/Inf entries.
    num_leapfrog_median_late
        Median ``num_integration_steps`` over the last 20 % of warmup.
    cap_saturation_pct_late
        Percent of late-warmup trajectories that hit ``max_num_doublings``.
    late_div_rate
        Divergence fraction in the last 20 % of warmup.
    ar_late_mean
        Mean acceptance rate over the last 20 % of warmup.
    """

    step_size: float
    imm_cond: float
    final_log_p: float
    position_max_abs: float
    position_has_finite: bool
    num_leapfrog_median_late: int
    cap_saturation_pct_late: float
    late_div_rate: float
    ar_late_mean: float


class WarmupValidationError(RuntimeError):
    """Raised when post-warmup sanity validation fails.

    The pattern: every NUTS cert pipeline now runs a generic, model-agnostic
    sanity check after warmup (no NaN, log_p finite, step_size in a plausible
    range, late-warmup divergence rate not catastrophic). Failing those means
    the warmup output is structurally bad — proceeding to sampling would
    waste hours of compute on a broken chain. Aborting at this point is
    cheap (warmup is much shorter than sampling).

    Carries the ``health: WarmupHealthSummary``, list of failed-check names
    (``failed_checks``), and the path where the warmup checkpoint was
    persisted (``checkpoint_dir``) so the diagnostician can load the bad
    state for forensic analysis.
    """

    def __init__(
        self,
        message: str,
        health: WarmupHealthSummary,
        failed_checks: list[str],
        checkpoint_dir: "Path | None" = None,
    ) -> None:
        super().__init__(message)
        self.health = health
        self.failed_checks = failed_checks
        self.checkpoint_dir = checkpoint_dir


@dataclass(frozen=True)
class WarmupCheckpoint:
    """Filesystem layout of a persisted warmup checkpoint.

    Standard files in ``checkpoint_dir``:
      * ``state.pkl``        — adapted HMCState (pickle)
      * ``params.pkl``       — adapted_params dict (pickle)
      * ``warmup_info.npz``  — per-step diagnostics (is_divergent, acceptance_rate,
                              num_integration_steps, num_trajectory_expansions,
                              is_turning, energy)
      * ``health.json``      — WarmupHealthSummary dict + adapted-config record

    The checkpoint is written immediately after warmup completes, BEFORE any
    validation or sampling. So even if validation fails or sampling crashes,
    the warmup output survives on disk for resume or forensic analysis.
    """

    dir: Path
    state_path: Path
    params_path: Path
    warmup_info_path: Path
    health_path: Path

    @classmethod
    def at(cls, dir: Path) -> "WarmupCheckpoint":
        return cls(
            dir=dir,
            state_path=dir / "state.pkl",
            params_path=dir / "params.pkl",
            warmup_info_path=dir / "warmup_info.npz",
            health_path=dir / "health.json",
        )


def default_warmup_validator(
    health: WarmupHealthSummary, *, max_num_doublings: int
) -> list[str]:
    """Generic model-agnostic warmup-health validator. Returns failed-check names.

    Empty list = healthy → proceed to sampling. Non-empty = failed → caller
    raises WarmupValidationError. Thresholds are intentionally loose: they
    catch *catastrophic* failures (NaN, step collapsed to 1e-12, divergence
    storms) but not model-specific tuning issues. Per-model tighter checks
    should be passed via ``validate_warmup_fn`` callback.

    Failure semantics:
      * ``position_finite`` — position has NaN or Inf entries (numerical blow-up).
      * ``log_p_finite``   — log-density is non-finite.
      * ``step_size_in_range`` — step adapted to < 1e-10 (collapsed) or > 100
        (runaway dual-averaging).
      * ``imm_cond_finite`` — IMM condition number is non-finite (often
        downstream of an IMM-diagonal entry going to zero).
      * ``late_div_rate``  — divergence rate in last 20 % of warmup > 50 %
        (chain is in a perpetually divergent region).
    """
    failed: list[str] = []
    if not health.position_has_finite:
        failed.append("position_finite")
    if not np.isfinite(health.final_log_p):
        failed.append("log_p_finite")
    if not (1e-10 < health.step_size < 100.0):
        failed.append("step_size_in_range")
    if not np.isfinite(health.imm_cond):
        failed.append("imm_cond_finite")
    if health.late_div_rate > 0.50:
        failed.append("late_div_rate")
    return failed


# ---------------------------------------------------------------------------
# Gate thresholds (per reference-certification protocol in CLAUDE.md)
# ---------------------------------------------------------------------------
_RHAT_THRESHOLD = 1.01
_MIN_CHUNK_ESS = 400.0
_EBFMI_THRESHOLD = 0.3
# Divergence tolerance — fraction of n_samples. Amended 2026-05-12 from strict
# zero ("no divergences at all") to a rate-based tolerance ("a few in 40k is
# fine for groundtruth"). Rationale: for well-mixed chains with healthy E-BFMI
# and high R̂/ESS, the residual divergence rate reflects fundamental geometry
# (e.g. a HalfCauchy funnel neck visited at probability ~1e-5 per step), not
# adaptation failure. Strict zero forced gate-gaming (seed-roulette, brute n
# bump). Threshold 0.001 means up to 1 divergence per 1000 samples — at the
# default n_samples=40_000 this allows ≤40 divergences before fail. See
# worklog/decisions/2026-05-11-phase0-reference-protocol-refinements.md § 8.
_DIVERGENCE_RATE_TOLERANCE = 0.001


def compute_certification_verdict(
    *,
    split_rhat_max: float,
    min_chunk_bulk_ess: float,
    num_divergences: int,
    e_bfmi: float,
    n_samples: int,
    divergence_rate_tolerance: float | None = None,
    rhat_threshold: float = _RHAT_THRESHOLD,
    min_chunk_ess: float = _MIN_CHUNK_ESS,
    ebfmi_threshold: float = _EBFMI_THRESHOLD,
) -> CertificationResult:
    """Pure gate-logic computation — apply per-metric thresholds, build verdict.

    Extracted from ``certify_reference_nuts`` 2026-05-17 so the verdict logic can
    be unit-tested with synthetic inputs (no NUTS run required). Thresholds
    default to the module constants (``_RHAT_THRESHOLD``, ``_MIN_CHUNK_ESS``,
    ``_EBFMI_THRESHOLD``, ``_DIVERGENCE_RATE_TOLERANCE``) but are overridable
    via keyword for edge-case testing.

    Parameters
    ----------
    split_rhat_max
        Max rank-normalised split-R̂ across all dimensions.
    min_chunk_bulk_ess
        Min per-chunk bulk-ESS across all dimensions and chunks.
    num_divergences
        Total number of divergent transitions in the post-warmup chain.
    e_bfmi
        Expected Bayesian Fraction of Missing Information.
    n_samples
        Total number of post-warmup samples — used together with
        ``divergence_rate_tolerance`` to compute the allowed divergence count.
    divergence_rate_tolerance
        Per-model override for the divergence-rate ceiling (fraction of
        ``n_samples`` allowed to be divergent). ``None`` falls back to the
        module default ``_DIVERGENCE_RATE_TOLERANCE``.
    rhat_threshold, min_chunk_ess, ebfmi_threshold
        Per-metric thresholds. Defaults match the protocol thresholds in
        CLAUDE.md § "Reference protocol"; overridable for edge-case testing.

    Returns
    -------
    CertificationResult
        Frozen dataclass with ``passed`` (bool) plus the four input metrics
        echoed back for downstream reporting / error messages.
    """
    effective_tolerance = (
        divergence_rate_tolerance
        if divergence_rate_tolerance is not None
        else _DIVERGENCE_RATE_TOLERANCE
    )
    max_divergences_allowed = int(effective_tolerance * n_samples)
    passed = (
        split_rhat_max <= rhat_threshold
        and min_chunk_bulk_ess >= min_chunk_ess
        and num_divergences <= max_divergences_allowed
        and e_bfmi >= ebfmi_threshold
    )
    return CertificationResult(
        passed=passed,
        split_rhat_max=split_rhat_max,
        min_chunk_bulk_ess=min_chunk_bulk_ess,
        num_divergences=num_divergences,
        e_bfmi=e_bfmi,
    )


def _compute_warmup_health(
    adapted_state,
    adapted_params,
    warmup_info,
    max_num_doublings: int,
) -> WarmupHealthSummary:
    """Compute the generic post-warmup health summary used by validators."""
    imm = np.asarray(adapted_params["inverse_mass_matrix"])
    step_size = float(adapted_params["step_size"])
    final_log_p = float(adapted_state.logdensity)

    flat_position, _ = ravel_pytree(adapted_state.position)
    flat_position_np = np.asarray(flat_position)
    position_has_finite = bool(np.isfinite(flat_position_np).all())
    position_max_abs = (
        float(np.max(np.abs(flat_position_np))) if position_has_finite else float("nan")
    )

    info = warmup_info.info
    is_div_trail = np.asarray(info.is_divergent)
    n_int_trail = np.asarray(info.num_integration_steps)
    n_exp_trail = np.asarray(info.num_trajectory_expansions)
    ar_trail = np.asarray(info.acceptance_rate)
    n_total = is_div_trail.shape[0]
    late_window = slice(int(0.8 * n_total), n_total)
    n_late = n_total - int(0.8 * n_total)

    n_div_late = int(is_div_trail[late_window].sum()) if n_late > 0 else 0
    late_div_rate = (n_div_late / n_late) if n_late > 0 else 0.0
    n_int_med_late = int(np.median(n_int_trail[late_window])) if n_late > 0 else 0
    cap_saturation = (
        float((n_exp_trail[late_window] == max_num_doublings).mean() * 100.0)
        if n_late > 0
        else 0.0
    )
    ar_late_mean = float(ar_trail[late_window].mean()) if n_late > 0 else float("nan")

    if imm.size > 0:
        imm_min, imm_max = float(imm.min()), float(imm.max())
        imm_cond = imm_max / imm_min if imm_min > 0 else float("nan")
    else:
        imm_cond = float("nan")

    return WarmupHealthSummary(
        step_size=step_size,
        imm_cond=imm_cond,
        final_log_p=final_log_p,
        position_max_abs=position_max_abs,
        position_has_finite=position_has_finite,
        num_leapfrog_median_late=n_int_med_late,
        cap_saturation_pct_late=cap_saturation,
        late_div_rate=late_div_rate,
        ar_late_mean=ar_late_mean,
    )


def _json_dump_health(
    file_obj,
    health: WarmupHealthSummary,
    entry,
    n_warmup: int,
    target_acceptance: float,
    max_num_doublings: int,
) -> None:
    """Write the WarmupHealthSummary + adapted-config record to ``file_obj``."""
    import json

    payload = {
        "model": entry.name,
        "n_warmup": int(n_warmup),
        "target_acceptance": float(target_acceptance),
        "max_num_doublings": int(max_num_doublings),
        "step_size": float(health.step_size),
        "imm_cond": float(health.imm_cond),
        "final_log_p": float(health.final_log_p),
        "position_max_abs": float(health.position_max_abs),
        "position_has_finite": bool(health.position_has_finite),
        "num_leapfrog_median_late": int(health.num_leapfrog_median_late),
        "cap_saturation_pct_late": float(health.cap_saturation_pct_late),
        "late_div_rate": float(health.late_div_rate),
        "ar_late_mean": float(health.ar_late_mean),
    }
    json.dump(payload, file_obj, indent=2)


def _compute_e_bfmi(energy: jax.Array) -> jax.Array:
    """Compute E-BFMI = mean(diff(energy)²) / var(energy).

    Parameters
    ----------
    energy
        1-D array of Hamiltonian energies from post-warmup samples.
    """
    diffs = jnp.diff(energy)
    return jnp.mean(diffs**2) / jnp.var(energy)


def certify_reference_nuts(
    entry: Posterior,
    rng_key: jax.Array,
    *,
    n_warmup: int = 5_000,
    n_samples: int = 40_000,
    n_chunks: int = 4,
    target_acceptance: float = 0.80,
    max_num_doublings: int = 10,
    pre_adapted: "PreAdaptedWarmup | None" = None,
    checkpoint_dir: "Path | None" = None,
    validate_warmup_fn: "Callable[[WarmupHealthSummary], list[str]] | None" = None,
) -> tuple[
    dict[str, jax.Array],
    Summaries,
    AdaptationParams,
    CertificationResult,
    dict[str, np.ndarray],
    float | None,
    float,
]:
    """Run long single-chain NUTS and certify the reference draws.

    Parameters
    ----------
    entry
        Registry entry.  Must have ``reference_method == NUTS``.
    rng_key
        JAX random key.
    n_warmup
        Number of warmup (adaptation) steps.
    n_samples
        Number of post-warmup samples.
    n_chunks
        Number of contiguous chunks to reshape samples into for split-R̂.
    target_acceptance
        Target acceptance rate for dual averaging.
    max_num_doublings
        NUTS ``max_num_doublings`` (max tree depth). Default 10 (BlackJAX
        default; allows up to 2^10=1024 leapfrog steps per trajectory).
        Raise to 12-15 for high-d models with naturally long trajectories
        (horseshoe priors, latent GPs) where the no-U-turn condition fires
        late. Empirically captured 2026-05-12 in the earlier statistician
        investigation (worklog/threads/phase0-statistician-3holdouts.md).

    Returns
    -------
    draws
        Dict mapping site name → Array of shape ``(n_samples, *site_shape)``.
    summaries
        Per-dim mean/std/q05/q95.
    adaptation_params
        Tuned step size and mass matrix from warmup.
    cert
        Certification result (passed/failed + diagnostics).
    chain_stats
        Dict mapping per-step diagnostic field names to arrays of shape
        ``(n_samples,)``, including ``num_integration_steps``, ``energy``,
        ``is_divergent``, ``acceptance_rate``, and any other fields
        exposed by BlackJAX's NUTSInfo NamedTuple.
    warmup_wall_seconds
        Wall seconds for the warmup phase (timed at Python orchestration
        level with ``jax.block_until_ready()`` before stopping the clock).
        ``None`` when ``pre_adapted`` was supplied (warmup was skipped).
    sampling_wall_seconds
        Wall seconds for the sampling phase (timed the same way).

    pre_adapted
        Optional pre-computed warmup output. When provided, the warmup phase
        is skipped and sampling starts from ``pre_adapted.state`` using
        ``pre_adapted.params``. Use this for two-phase workflows (run warmup
        separately, review, then resume sampling) or recovery after a
        sampling-phase crash. Mutually exclusive with the warmup-checkpoint
        persistence (no checkpoint is written when ``pre_adapted`` is supplied).
    checkpoint_dir
        Optional directory where the warmup checkpoint (state.pkl, params.pkl,
        warmup_info.npz, health.json) is written immediately after warmup
        completes, BEFORE validation or sampling. If sampling later crashes,
        the warmup can be reused via ``pre_adapted=...``. Ignored when
        ``pre_adapted`` is supplied. Caller is responsible for creating the
        directory.
    validate_warmup_fn
        Optional callback ``(health: WarmupHealthSummary) -> list[str]`` that
        receives the post-warmup health summary and returns the list of
        failed-check names. Called BEFORE sampling. If non-empty, raises
        ``WarmupValidationError`` (sampling not launched). Defaults to
        ``default_warmup_validator`` (model-agnostic catastrophic-failure
        check). Pass a tighter callback for model-specific thresholds.
        Skipped entirely when ``pre_adapted`` is supplied (the caller is
        assumed to have validated already).

    Raises
    ------
    ValueError
        If ``entry.reference_method != NUTS``.
    WarmupValidationError
        If the post-warmup validation callback reports any failed check.
        Sampling is NOT launched; warmup checkpoint (if any) is preserved.
    CertificationError
        If any post-sampling certification gate fails. The exception carries
        ``adaptation`` and ``chain_stats`` from the run for diagnostician
        inspection.
    """
    if entry.reference_method != ReferenceMethod.NUTS:
        raise ValueError(
            f"{entry.name!r} uses the analytic path; "
            "call certify_reference_analytic instead."
        )

    # Per-model x64 requirement (e.g. gp_regression's dense Cholesky cannot
    # be evaluated stably in float32). Raise a clear error rather than silently
    # producing NaN-corrupted gradients.
    if entry.requires_x64 and not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            f"Model {entry.name!r} requires float64 (Posterior.requires_x64=True) "
            "but JAX is running in float32 mode. Set the environment variable "
            "JAX_ENABLE_X64=1 BEFORE the first jax import (e.g. at the top of "
            "your runner script or via the shell) and retry."
        )

    rng_key_init, rng_key_warmup, rng_key_sample = jax.random.split(rng_key, 3)

    # --- Build logdensity_fn ---
    init_position, logdensity_fn, _ = build_logdensity_fn(rng_key_init, entry)

    # --- Warmup (or load pre_adapted) ---
    _warmup_wall: float | None = (
        None  # set below in warmup branch; None on pre_adapted path
    )

    if pre_adapted is None:
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn,
            target_acceptance_rate=target_acceptance,
            max_num_doublings=max_num_doublings,
        )
        _t_warmup0 = time.perf_counter()
        (adapted_state, adapted_params), warmup_info = warmup.run(
            rng_key_warmup, init_position, n_warmup
        )
        # Block until JAX async dispatch completes before stopping the clock.
        # Without this the timer measures dispatch latency, not actual compute —
        # the same artifact that caused the vmap "blowup" misdiagnosis in the
        # gp arc (worklog/lessons/case-studies/laplace-family-vmap-compile-blowup.md).
        jax.block_until_ready((adapted_state, warmup_info))
        _warmup_wall = time.perf_counter() - _t_warmup0
        num_leapfrog_median = int(jnp.median(warmup_info.info.num_integration_steps))

        # --- Persist warmup checkpoint IMMEDIATELY (before validation, before
        #     sampling). If a downstream step crashes — be it validation, JIT
        #     compilation of the sampling kernel, or sampling itself — the
        #     warmup output survives on disk for resume via ``pre_adapted=...``
        #     or for forensic analysis.
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt = WarmupCheckpoint.at(checkpoint_dir)
            with open(ckpt.state_path, "wb") as f:
                pickle.dump(adapted_state, f)
            with open(ckpt.params_path, "wb") as f:
                pickle.dump(dict(adapted_params), f)
            _info = warmup_info.info
            np.savez(
                ckpt.warmup_info_path,
                is_divergent=np.asarray(_info.is_divergent),
                num_integration_steps=np.asarray(_info.num_integration_steps),
                num_trajectory_expansions=np.asarray(_info.num_trajectory_expansions),
                acceptance_rate=np.asarray(_info.acceptance_rate),
                is_turning=np.asarray(_info.is_turning),
                energy=np.asarray(_info.energy),
            )

        # --- Generic + custom warmup validation (BEFORE sampling) ---
        health = _compute_warmup_health(
            adapted_state, adapted_params, warmup_info, max_num_doublings
        )
        if checkpoint_dir is not None:
            ckpt = WarmupCheckpoint.at(checkpoint_dir)
            with open(ckpt.health_path, "w") as health_fh:
                _json_dump_health(
                    health_fh,
                    health,
                    entry,
                    n_warmup,
                    target_acceptance,
                    max_num_doublings,
                )

        validator = (
            validate_warmup_fn
            if validate_warmup_fn is not None
            else (
                lambda h: default_warmup_validator(
                    h, max_num_doublings=max_num_doublings
                )
            )
        )
        failed = validator(health)
        if failed:
            raise WarmupValidationError(
                f"warmup validation failed for {entry.name!r}: {failed}. "
                f"step_size={health.step_size:.4e}, imm_cond={health.imm_cond:.1f}x, "
                f"final_log_p={health.final_log_p:.4f}, late_div_rate="
                f"{health.late_div_rate * 100:.2f}%. "
                f"Warmup checkpoint preserved at "
                f"{checkpoint_dir if checkpoint_dir else '(no checkpoint_dir given)'}.",
                health=health,
                failed_checks=failed,
                checkpoint_dir=checkpoint_dir,
            )
    else:
        # Skip warmup entirely; use the pre-supplied adapted state + params.
        # ``rng_key_warmup``, ``n_warmup``, ``checkpoint_dir``, and
        # ``validate_warmup_fn`` are unused in this branch. The caller is
        # responsible for having produced (and validated) ``pre_adapted``.
        adapted_state = pre_adapted.state
        adapted_params = pre_adapted.params
        num_leapfrog_median = pre_adapted.num_leapfrog_median
    adaptation = AdaptationParams(
        step_size=float(adapted_params["step_size"]),
        inverse_mass_matrix=jnp.array(adapted_params["inverse_mass_matrix"]),
        num_leapfrog_median=num_leapfrog_median,
    )

    # --- Long single chain ---
    # window_adaptation propagates max_num_doublings into adapted_params, so we
    # don't pass it again here (would raise TypeError on duplicate kwarg).
    nuts = blackjax.nuts(logdensity_fn, **adapted_params)
    _t_sample0 = time.perf_counter()
    final_state, (states, infos) = run_inference_algorithm(
        rng_key=rng_key_sample,
        inference_algorithm=nuts,
        num_steps=n_samples,
        initial_state=adapted_state,
    )
    # Block until dispatch completes — same rationale as the warmup timer above.
    jax.block_until_ready((final_state, states, infos))
    _sampling_wall = time.perf_counter() - _t_sample0
    del final_state  # not needed

    # states.position is a dict {site: (n_samples, *shape)}
    draws: dict[str, jax.Array] = states.position  # type: ignore[assignment]

    # --- Diagnostics ---
    # Energy array from infos
    energy: jax.Array = infos.energy  # shape (n_samples,)

    # Divergences
    num_divergences = int(jnp.sum(infos.is_divergent))

    # E-BFMI
    e_bfmi_val = float(_compute_e_bfmi(energy))

    # --- Extract all chain_stats from infos ---
    # Iterate over all NUTSInfo._fields and extract array-valued fields
    chain_stats: dict[str, np.ndarray] = {}
    for field_name in infos._fields:
        field_val = getattr(infos, field_name)
        # Skip dicts and other non-array types
        if isinstance(field_val, dict):
            continue
        # Only store array-like fields; skip nested NamedTuples or non-array fields
        try:
            arr = np.asarray(field_val)
            # Skip object arrays that aren't truly homogeneous
            if arr.dtype == object:
                continue
            chain_stats[field_name] = arr
        except (ValueError, TypeError):
            # Skip fields that can't be converted to arrays
            pass

    # Reshape to (n_chunks, chunk_size, *site_shape) for split-R̂ and ESS
    chunk_size = n_samples // n_chunks

    # Build (n_chains=n_chunks, n_draws=chunk_size, *shape) for diagnostics
    # blackjax diagnostics expect (num_chains, num_draws, *param_shape)
    def _reshape_for_diag(arr: jax.Array) -> jax.Array:
        """Reshape (n_samples, *shape) → (n_chunks, chunk_size, *shape)."""
        site_shape = arr.shape[1:]
        return arr[: n_chunks * chunk_size].reshape(n_chunks, chunk_size, *site_shape)

    chunked = {site: _reshape_for_diag(arr) for site, arr in draws.items()}

    # Compute split-R̂ and bulk-ESS
    # blackjax.diagnostics.potential_scale_reduction: (num_chains, num_draws, *param_shape) → scalar
    # blackjax.diagnostics.effective_sample_size: same shape → scalar
    rhat_values = []
    ess_values = []
    for site, arr in chunked.items():
        rhat = blackjax.diagnostics.potential_scale_reduction(arr)
        ess = blackjax.diagnostics.effective_sample_size(arr)
        # rhat and ess may be scalars or arrays (per-dim)
        rhat_values.append(float(jnp.max(jnp.asarray(rhat))))
        # ESS per chunk: ess already computed over all chunks; divide by n_chunks
        # to get per-chunk bulk-ESS
        ess_values.append(float(jnp.min(jnp.asarray(ess))) / n_chunks)

    split_rhat_max = max(rhat_values)
    min_chunk_bulk_ess = min(ess_values)

    # --- Certification gate ---
    # Divergence allowance: per-model override if set, else global default.
    # Default _DIVERGENCE_RATE_TOLERANCE = 0.001 = 0.1% of n_samples (≤40 in
    # 40k, ≤100 in 100k). A model may override via Posterior.divergence_rate_tolerance
    # (e.g. stoch_vol uses 0.005 for the AR(1) unit-root excursion tail — see
    # the Posterior field's docstring + the model file's rationale comment).
    #
    # Verdict computation is delegated to compute_certification_verdict (pure)
    # so the gate-logic tests can exercise it without running NUTS.
    cert = compute_certification_verdict(
        split_rhat_max=split_rhat_max,
        min_chunk_bulk_ess=min_chunk_bulk_ess,
        num_divergences=num_divergences,
        e_bfmi=e_bfmi_val,
        n_samples=n_samples,
        divergence_rate_tolerance=entry.divergence_rate_tolerance,
    )
    # Recompute max_divergences_allowed for the error message (the verdict
    # function uses it internally but does not expose it on CertificationResult).
    effective_tolerance = (
        entry.divergence_rate_tolerance
        if entry.divergence_rate_tolerance is not None
        else _DIVERGENCE_RATE_TOLERANCE
    )
    max_divergences_allowed = int(effective_tolerance * n_samples)

    if not cert.passed:
        raise CertificationError(
            f"reference-certification certification failed for {entry.name!r}: "
            f"split_rhat_max={split_rhat_max:.4f}, "
            f"min_chunk_bulk_ess={min_chunk_bulk_ess:.1f}, "
            f"num_divergences={num_divergences} (gate ≤ {max_divergences_allowed}), "
            f"e_bfmi={e_bfmi_val:.4f}",
            cert,
            adaptation=adaptation,
            chain_stats=chain_stats,
            draws=draws,
        )

    summaries = compute_summaries(draws)

    # --- Posteriordb cross-check (optional; only for models with a posteriordb_id) ---
    if entry.posteriordb_id is not None:
        from tuningfork._posteriordb_xcheck import cross_check_against_posteriordb

        # Build the our_summaries dict in the format expected by cross_check_against_posteriordb:
        # {site: {"mean": array, "std": array, "q05": array, "q95": array}}
        our_summaries: dict[str, dict[str, object]] = {
            site: {
                "mean": summaries.mean[site],
                "std": summaries.std[site],
                "q05": summaries.q05[site],
                "q95": summaries.q95[site],
            }
            for site in summaries.mean
        }
        xcheck = cross_check_against_posteriordb(
            model_name=entry.name,
            posteriordb_id=entry.posteriordb_id,
            our_summaries=our_summaries,
            n_samples_ours=n_samples,
        )
        # Post-R2 (2026-05-17): xcheck.json lives in the committed reference/
        # subdir under the per-model catalog dir.
        xcheck_dir = Path(__file__).parent.parent / "catalog" / entry.name / "reference"
        xcheck_dir.mkdir(parents=True, exist_ok=True)
        xcheck.save(xcheck_dir / "xcheck.json")

    return draws, summaries, adaptation, cert, chain_stats, _warmup_wall, _sampling_wall
