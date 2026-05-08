"""Tier-A NUTS path (Path B) — long single-chain NUTS reference certification.

Runs 1 chain × n_warmup warmup × n_samples post-warmup NUTS steps using
BlackJAX's window adaptation (Stan-style).  Reshapes into n_chunks contiguous
chunks for rank-normalised split-R̂ and bulk-ESS diagnostics.

Certification gate (Tier-A):
    - rank-normalised split-R̂ ≤ 1.01
    - min per-chunk bulk-ESS > 400
    - num_divergences == 0
    - E-BFMI > 0.3

E-BFMI formula (Neal 2011, Stan Reference §15.4):
    E-BFMI = mean(diff(energy)²) / var(energy)
where ``energy`` is the Hamiltonian energy at each post-warmup step.
This measures how well the momentum resampling explores the energy surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
from blackjax.util import run_inference_algorithm

from bjx_bench.calibration._summary import Summaries, compute_summaries
from bjx_bench.model._base import Posterior, ReferenceMethod
from bjx_bench.model._numpyro import build_logdensity_fn

__all__ = [
    "AdaptationParams",
    "CertificationResult",
    "CertificationError",
    "certify_reference_nuts",
]


@dataclass(frozen=True)
class AdaptationParams:
    """Tuned NUTS parameters from window adaptation warmup.

    Used as informative priors (not optima) for Tier-B search ranges.

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
    """Diagnostic summary from a Tier-A NUTS run.

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
    """Raised when a Path-B run fails the Tier-A gate.

    Carries ``cert: CertificationResult`` so the caller can log the failure
    and decide to re-run with more samples or a different seed.
    """

    def __init__(self, message: str, cert: CertificationResult) -> None:
        super().__init__(message)
        self.cert = cert


# ---------------------------------------------------------------------------
# Gate thresholds (per Tier-A protocol in CLAUDE.md)
# ---------------------------------------------------------------------------
_RHAT_THRESHOLD = 1.01
_MIN_CHUNK_ESS = 400.0
_EBFMI_THRESHOLD = 0.3


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
    n_samples: int = 100_000,
    n_chunks: int = 10,
    target_acceptance: float = 0.80,
) -> tuple[
    dict[str, jax.Array],
    Summaries,
    AdaptationParams,
    CertificationResult,
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

    Raises
    ------
    ValueError
        If ``entry.reference_method != NUTS``.
    CertificationError
        If any certification gate fails.
    """
    if entry.reference_method != ReferenceMethod.NUTS:
        raise ValueError(
            f"{entry.name!r} uses the analytic path; "
            "call certify_reference_analytic instead."
        )

    rng_key_init, rng_key_warmup, rng_key_sample = jax.random.split(rng_key, 3)

    # --- Build logdensity_fn ---
    init_position, logdensity_fn, _ = build_logdensity_fn(rng_key_init, entry)

    # --- Window adaptation (warmup) ---
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        target_acceptance_rate=target_acceptance,
    )
    (adapted_state, adapted_params), warmup_info = warmup.run(
        rng_key_warmup, init_position, n_warmup
    )
    adaptation = AdaptationParams(
        step_size=float(adapted_params["step_size"]),
        inverse_mass_matrix=jnp.array(adapted_params["inverse_mass_matrix"]),
        num_leapfrog_median=int(jnp.median(warmup_info.info.num_integration_steps)),
    )

    # --- Long single chain ---
    nuts = blackjax.nuts(logdensity_fn, **adapted_params)
    final_state, (states, infos) = run_inference_algorithm(
        rng_key=rng_key_sample,
        inference_algorithm=nuts,
        num_steps=n_samples,
        initial_state=adapted_state,
    )
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
    passed = (
        split_rhat_max <= _RHAT_THRESHOLD
        and min_chunk_bulk_ess >= _MIN_CHUNK_ESS
        and num_divergences == 0
        and e_bfmi_val >= _EBFMI_THRESHOLD
    )

    cert = CertificationResult(
        passed=passed,
        split_rhat_max=split_rhat_max,
        min_chunk_bulk_ess=min_chunk_bulk_ess,
        num_divergences=num_divergences,
        e_bfmi=e_bfmi_val,
    )

    if not passed:
        raise CertificationError(
            f"Tier-A certification failed for {entry.name!r}: "
            f"split_rhat_max={split_rhat_max:.4f}, "
            f"min_chunk_bulk_ess={min_chunk_bulk_ess:.1f}, "
            f"num_divergences={num_divergences}, "
            f"e_bfmi={e_bfmi_val:.4f}",
            cert,
        )

    summaries = compute_summaries(draws)

    # --- Posteriordb cross-check (optional; only for models with a posteriordb_id) ---
    if entry.posteriordb_id is not None:
        from bjx_bench.reference._posteriordb_xcheck import (
            cross_check_against_posteriordb,
        )

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
        xcheck_dir = Path(__file__).parent.parent / "reference" / "posteriordb_xcheck"
        xcheck_dir.mkdir(parents=True, exist_ok=True)
        xcheck.save(xcheck_dir / f"{entry.name}.json")

    return draws, summaries, adaptation, cert
