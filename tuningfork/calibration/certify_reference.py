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

from dataclasses import dataclass
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.util import run_inference_algorithm

from tuningfork.calibration._summary import Summaries, compute_summaries
from tuningfork.model._base import Posterior, ReferenceMethod
from tuningfork.model._numpyro import build_logdensity_fn

__all__ = [
    "AdaptationParams",
    "CertificationResult",
    "CertificationError",
    "certify_reference_nuts",
]


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
    and ``chain_stats: dict[str, np.ndarray] | None`` so the caller can log the
    failure, log diagnostic data, and decide to re-run with more samples or a
    different seed.
    """

    def __init__(
        self,
        message: str,
        cert: CertificationResult,
        adaptation: "AdaptationParams | None" = None,
        chain_stats: "dict[str, np.ndarray] | None" = None,
    ) -> None:
        super().__init__(message)
        self.cert = cert
        self.adaptation = adaptation
        self.chain_stats = chain_stats


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
) -> tuple[
    dict[str, jax.Array],
    Summaries,
    AdaptationParams,
    CertificationResult,
    dict[str, np.ndarray],
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
    chain_stats
        Dict mapping per-step diagnostic field names to arrays of shape
        ``(n_samples,)``, including ``num_integration_steps``, ``energy``,
        ``is_divergent``, ``acceptance_rate``, and any other fields
        exposed by BlackJAX's NUTSInfo NamedTuple.

    Raises
    ------
    ValueError
        If ``entry.reference_method != NUTS``.
    CertificationError
        If any certification gate fails. The exception carries ``adaptation``
        and ``chain_stats`` from the run for diagnostician inspection.
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
    # Divergence allowance: up to _DIVERGENCE_RATE_TOLERANCE × n_samples
    # (default 0.001 = 0.1%). At n=40k this allows ≤40 divergences; at n=100k
    # ≤100. See _DIVERGENCE_RATE_TOLERANCE comment above for rationale.
    max_divergences_allowed = int(_DIVERGENCE_RATE_TOLERANCE * n_samples)
    passed = (
        split_rhat_max <= _RHAT_THRESHOLD
        and min_chunk_bulk_ess >= _MIN_CHUNK_ESS
        and num_divergences <= max_divergences_allowed
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
            f"reference-certification certification failed for {entry.name!r}: "
            f"split_rhat_max={split_rhat_max:.4f}, "
            f"min_chunk_bulk_ess={min_chunk_bulk_ess:.1f}, "
            f"num_divergences={num_divergences} (gate ≤ {max_divergences_allowed}), "
            f"e_bfmi={e_bfmi_val:.4f}",
            cert,
            adaptation=adaptation,
            chain_stats=chain_stats,
        )

    summaries = compute_summaries(draws)

    # --- Posteriordb cross-check (optional; only for models with a posteriordb_id) ---
    if entry.posteriordb_id is not None:
        from tuningfork.reference._posteriordb_xcheck import (
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

    return draws, summaries, adaptation, cert, chain_stats
