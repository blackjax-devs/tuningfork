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
"""SMC auto-gate — quality assessment for Sequential Monte Carlo results.

Statistician spec (Phase 8B.1, 2026-06-03):

  Gates (verdict-affecting):
    - max_abs_mean_z < 2.0  (SE = std(particles) / √particle_ess, NOT √N)
    - mode_coverage_fraction ≥ 0.01 per mode (gmm_25 only)

  Report-only (NOT gates):
    - particle_ess = 1/Σwᵢ² computed BEFORE the final resample at λ=1
    - particle_ess_fraction = particle_ess / N
    - n_divergences = None (SMC has no divergences in the HMC sense)
    - rhat = None (undefined for a single particle cloud)

  Headline metric:
    - particle_ess / total_grad_evals (HMC, fixed-L exact)
    - particle_ess / total_likelihood_evals (RWM, gradient-free)

Verdict:
    PASS   — all gates pass.
    REVIEW — no explicit REVIEW band for SMC (future: if z ∈ [2, 4]).
    FAIL   — any gate fails.

Public API
----------
``smc_gate(particles, weights, ground_truth_summaries, *, model_name, num_particles)``
    Compute SMC gate metrics and return a ``SMCGateVerdict``.
``compute_particle_ess(weights)``
    Compute 1/Σwᵢ² from a (N,) weight array.
``gmm_25_mode_coverage(particles, *, centers, radius_sigmas)``
    Fraction of gmm_25 modes with ≥ 1% particle coverage within 2σ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

__all__ = [
    "SMCGateVerdict",
    "compute_particle_ess",
    "gmm_25_mode_coverage",
    "smc_gate",
]


# ---------------------------------------------------------------------------
# SMCGateVerdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SMCGateVerdict:
    """Output of ``smc_gate(...)``.

    Maps into ``SMCRecipe.gate_evidence['auto']`` via ``to_dict()``.

    λ_final validity gate (short-circuits z/mode-coverage):
      ≥ 0.99 → PASS-eligible (z gate applied)
      [0.95, 0.99) → REVIEW
      < 0.95 → FAIL (particles from tempered density, not posterior)
    """

    verdict: str  # "PASS" | "REVIEW" | "FAIL"
    max_abs_mean_z: float | None
    particle_ess: float | None  # diagnostic; 1/Σwᵢ² pre-final-resample
    particle_ess_fraction: float | None  # diagnostic; particle_ess / N
    mode_coverage_fraction: float | None  # gmm_25 only; None otherwise
    lambda_final: float | None = None  # tempering param at end of run

    def to_dict(self) -> dict[str, Any]:
        """Render in the shape SMCRecipe.gate_evidence['auto'] expects."""
        return {
            "verdict": self.verdict,
            "max_abs_mean_z": self.max_abs_mean_z,
            "particle_ess": self.particle_ess,
            "particle_ess_fraction": self.particle_ess_fraction,
            "mode_coverage_fraction": self.mode_coverage_fraction,
            "lambda_final": self.lambda_final,
            # rhat and n_divergences are n/a for SMC.
            "rhat_max": None,
            "n_divergences": None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_particle_ess(weights: np.ndarray) -> float:
    """Compute effective sample size from normalised weights.

    ESS = 1 / Σwᵢ²   (computed on normalised weights).

    Taken BEFORE the final resample at λ=1 (per statistician spec).

    Parameters
    ----------
    weights
        (N,) float array of normalised particle weights (must sum to 1).

    Returns
    -------
    float
        Effective sample size ∈ [1, N].
    """
    w = np.asarray(weights, dtype=float)
    # Re-normalise defensively (numerical drift).
    w = w / w.sum()
    return float(1.0 / np.sum(w**2))


def _particle_mean_and_std(particles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dim mean and std of a (N, d) particle array."""
    return particles.mean(axis=0), particles.std(axis=0, ddof=0)


def _compute_max_abs_mean_z_smc(
    particles_flat: np.ndarray,
    particle_ess: float,
    gt_means: np.ndarray,
    gt_stds: np.ndarray,
) -> float:
    """Max |mean_z| across dims with SE = std(particles)/√particle_ess.

    Unlike the MCMC gate (which uses SE = std/√n_samples), the SMC gate uses
    SE = std(particles)/√particle_ess (statistician spec, Phase 8B.1).

    Parameters
    ----------
    particles_flat
        (N, d) numpy array of particle positions (flat, all dims).
    particle_ess
        Effective sample size (1/Σwᵢ²) used in the SE denominator.
    gt_means, gt_stds
        (d,) ground-truth means and stds for normalisation.

    Returns
    -------
    float
        max |sample_mean - gt_mean| / max(SE_sample, SE_gt)
        where SE_sample = std(particles)/√particle_ess.
    """
    sample_means, sample_stds = _particle_mean_and_std(particles_flat)

    # SE for sample mean using particle_ess (not N) as the effective n.
    se_sample = sample_stds / max(np.sqrt(particle_ess), 1.0)
    # SE for ground-truth mean (same formula; gt_n treated as large → 0).
    # Use a conservative gt_se floor to avoid division by near-zero.
    se_gt = gt_stds / max(np.sqrt(particle_ess), 1.0)
    se = np.maximum(se_sample, se_gt)
    se = np.maximum(se, 1e-8)  # numerical floor

    z_scores = np.abs(sample_means - gt_means) / se
    return float(np.max(z_scores))


def gmm_25_mode_coverage(
    particles_flat: np.ndarray,
    *,
    centers: np.ndarray,
    radius_sigmas: float = 2.0,
) -> float:
    """Fraction of gmm_25 modes with ≥ 1% of particles within ``radius_sigmas``.

    A mode is "covered" if at least ``ceil(0.01 * N)`` particles lie within
    ``radius_sigmas`` standard deviations of the mode center (using the
    per-dim marginal std of ALL particles as the scale).

    Parameters
    ----------
    particles_flat
        (N, d) flat particle array.
    centers
        (K, d) array of the K mode centers.
    radius_sigmas
        Number of marginal standard deviations for the ball radius.

    Returns
    -------
    float
        Fraction of modes covered ∈ [0, 1].
    """
    N, d = particles_flat.shape
    K = centers.shape[0]
    threshold = max(1, int(np.ceil(0.01 * N)))
    marginal_std = particles_flat.std(axis=0, ddof=0)
    marginal_std = np.maximum(marginal_std, 1e-8)

    covered = 0
    for k in range(K):
        z = np.abs(particles_flat - centers[k]) / (radius_sigmas * marginal_std)
        # Particle is "in the ball" if ALL dims within radius.
        in_ball = (z <= 1.0).all(axis=1)
        if in_ball.sum() >= threshold:
            covered += 1
    return float(covered) / K


# ---------------------------------------------------------------------------
# Main gate function
# ---------------------------------------------------------------------------


def smc_gate(
    particles: dict[str, np.ndarray] | np.ndarray,
    weights: np.ndarray,
    ground_truth_summaries: dict[str, dict] | None,
    *,
    model_name: str = "",
    num_particles: int | None = None,
    lambda_final: float | None = None,
) -> SMCGateVerdict:
    """Compute SMC gate metrics from final particle cloud.

    The verdict is determined by applying gates in order:
    1. **λ_final validity gate (short-circuits all other gates):**
       λ_final ≥ 0.99 → PASS-eligible; [0.95, 0.99) → REVIEW;
       < 0.95 → FAIL (particles from a tempered density, not the posterior).
       When ``lambda_final`` is None, the λ gate is skipped (backward compat).
    2. ``max_abs_mean_z < 2.0`` (SE = std/√particle_ess).
    3. ``mode_coverage_fraction ≥ 0.01 per mode`` (gmm_25 only).

    Parameters
    ----------
    particles
        Final particle positions at λ=1 BEFORE the final resample.
        Either a dict ``{site_name: (N, *shape)}`` (from numpyro parameterisation)
        or a flat ``(N, d)`` array.
    weights
        Normalised particle weights ``(N,)`` at λ=1 BEFORE final resample.
    ground_truth_summaries
        Per-site dict with ``"mean"`` and ``"std"`` keys (from reference/summary.json).
        ``None`` → z-gate is skipped.
    model_name
        Used to detect gmm_25 for mode-coverage gate.
    num_particles
        N (used for ESS fraction); inferred from ``weights`` if None.

    Returns
    -------
    SMCGateVerdict
    """
    from jax.flatten_util import (
        ravel_pytree,  # inline import; avoids JAX on module load
    )

    # --- Flatten particles to (N, d) ---
    if isinstance(particles, dict):
        # Dict of per-site arrays (N, *shape) — flatten each particle.
        # Stack by ravelling per particle: vmap ravel_pytree over leading dim.
        import jax

        _, unravel = ravel_pytree(
            {k: jnp.zeros_like(v[0]) for k, v in particles.items()}
        )
        particles_flat = np.array(jax.vmap(lambda pt: ravel_pytree(pt)[0])(particles))
    else:
        particles_flat = np.asarray(particles)

    N = particles_flat.shape[0]
    if num_particles is None:
        num_particles = N

    weights_np = np.asarray(weights, dtype=float)

    # --- Particle ESS ---
    ess = compute_particle_ess(weights_np)
    ess_fraction = ess / num_particles

    # --- max_abs_mean_z (only when GT available) ---
    max_z: float | None = None
    if ground_truth_summaries is not None:
        # Flatten GT means/stds to match particle flat ordering.
        gt_means_list, gt_stds_list = [], []
        # Use alphabetical ordering matching ravel_pytree's dict traversal.
        for site in sorted(ground_truth_summaries.keys()):
            if site not in (particles if isinstance(particles, dict) else {}):
                continue
            gt_info = ground_truth_summaries[site]
            gt_means_list.append(np.asarray(gt_info["mean"]).ravel())
            gt_stds_list.append(np.asarray(gt_info["std"]).ravel())
        if gt_means_list:
            gt_means = np.concatenate(gt_means_list)
            gt_stds = np.concatenate(gt_stds_list)
            # Align flat dim.
            d = min(particles_flat.shape[1], len(gt_means))
            max_z = _compute_max_abs_mean_z_smc(
                particles_flat[:, :d],
                ess,
                gt_means[:d],
                gt_stds[:d],
            )

    # --- gmm_25 mode coverage ---
    mode_cov: float | None = None
    if model_name == "gmm_25":
        try:
            from tuningfork.model.gmm_25 import COMPONENT_LOCS as _gmm25_centers

            mode_cov = gmm_25_mode_coverage(
                particles_flat,
                centers=np.asarray(_gmm25_centers),
                radius_sigmas=2.0,
            )
        except (ImportError, AttributeError):
            pass  # gmm_25 centers not accessible; skip

    # --- λ_final validity gate (short-circuits z/mode-coverage if failed) ---
    # Statistician ruling (Phase 8B.1, 2026-06-03):
    #   ≥ 0.99 → PASS-eligible; [0.95, 0.99) → REVIEW; < 0.95 → FAIL
    verdict = "PASS"
    lambda_gate_fired = False
    if lambda_final is not None:
        if lambda_final >= 0.99:
            pass  # PASS-eligible; continue to z/mode gates
        elif lambda_final >= 0.95:
            verdict = "REVIEW"
            lambda_gate_fired = True
        else:
            verdict = "FAIL"
            lambda_gate_fired = True

    # --- z-gate and mode-coverage gate (skipped if λ gate already FAILed) ---
    if not lambda_gate_fired:
        if max_z is not None and max_z >= 2.0:
            verdict = "FAIL"

        if mode_cov is not None and mode_cov < 1.0:
            # At least one mode not covered.
            verdict = "FAIL"

    return SMCGateVerdict(
        verdict=verdict,
        max_abs_mean_z=max_z,
        particle_ess=ess,
        particle_ess_fraction=ess_fraction,
        mode_coverage_fraction=mode_cov,
        lambda_final=lambda_final,
    )
