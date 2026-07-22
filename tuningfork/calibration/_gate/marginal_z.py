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
"""Shared calibrated marginal-z test — single source of truth for both GT gates.

This module holds the three calibration decisions from PR #245 (commit 700cfac)
that apply to ALL per-dimension z-tests comparing a benchmark/generated run
against a ground-truth reference:

  1. **Pooled SE denominator**::

         se_denom = max(sqrt(se_a² + se_b²), _SE_FLOOR)

     Prior formula: ``max(se_a, se_b)``, which inflates z by up to √2 at equal
     SE and by less at unequal SE (always ≥ √(1/2) × pooled).

  2. **Dimension-aware Bonferroni threshold**::

         ν       = 2·(n_chains − 1)
         z_crit  = t.ppf(1 − α/(2·D_total), ν)

     Prior: fixed z < 4.0 (benchmark gate) or fixed z < 3.0 (verify gate),
     both insensitive to model dimensionality — generous for high-D, strict
     for low-D models.

  3. **Materiality co-primary gate** (strict > on boundary)::

         HARD FAIL  iff  z_d > z_crit  AND  |Δμ_d|/std_b_d > _TAU_SCI
         REVIEW     iff  z_d > z_crit  AND  |Δμ_d|/std_b_d ≤ _TAU_SCI
         PASS       otherwise

     Prior: z > threshold → FAIL regardless of effect size |Δμ| / σ.

Routing
-------
Both GT-correctness gates route through this module so the formula lives in
exactly one place:

- ``tuningfork.groundtruth._verify._check_coherence`` — the ``--verify`` CLI
  coherence path (imports ``_SE_FLOOR``, ``_TAU_SCI``, ``bonferroni_z_crit``).
- ``tuningfork.calibration._gate.gt_compare._compute_gt_compare`` — the recipe
  benchmark / auto-gate path (imports the same constants + ``_DEFAULT_NU``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as scipy_stats

__all__ = [
    "_DEFAULT_NU",
    "_SE_FLOOR",
    "_TAU_SCI",
    "bonferroni_z_crit",
    "marginal_z_verdict",
]

# ---------------------------------------------------------------------------
# Module-level constants: single source of truth across both gate paths.
# ---------------------------------------------------------------------------

# SE floor: prevents division-by-zero on scalar / near-zero-SE sites.
# Under the dual gate this only affects REVIEW labelling, not hard-fail.
_SE_FLOOR: float = 1e-8

# Materiality threshold: |Δμ| / std_ref must strictly exceed this to be a
# hard FAIL.  Boundary (mat == _TAU_SCI) is REVIEW, not FAIL (strict >).
# Mirrors the W1 gate sibling in calibration/_gate/w1_realm.py.
_TAU_SCI: float = 0.05

# Default ν (degrees of freedom) for the benchmark gate path, where mc_samples
# arrives as a single chunk (n_chunks=1, so arr.shape[0] == 1).
#
# JUDGMENT CALL FLAG — statistician validation requested:
# When n_chunks=1 the "chains" dimension of mc_samples collapses to 1, giving
# ν = 2·(1 − 1) = 0 (degenerate: z_crit → ∞, trivially passing everything).
# To avoid a degenerate gate, _compute_gt_compare falls back to _DEFAULT_NU = 9
# when mc_n_chains == 1.
#
# Rationale for the value 9:
#   The standard multichain GT protocol (PR #228+) uses 10 chains × 10 000
#   draws.  The between-chain SE for that GT has ν = 10 − 1 = 9 df, which is
#   the bottleneck uncertainty in the pooled SE denominator (the benchmark run
#   has ESS >> 9 in well-behaved dims).  Using ν = 9 is therefore conservative
#   (small ν → fat t-tails → higher z_crit → harder to hard-fail).
#
# Note: models using the legacy single-chain GT (summary.json, n_samples=40000)
# do NOT have a between-chain SE; their GT SE is std/sqrt(40000), approximately
# equivalent to ∞ df.  In those cases ν = 9 is still conservative — the true
# ν is the benchmark run's per-dim ESS, which is typically 500-2000 >> 9 — but
# the benchmark SE usually dominates, making z_crit insensitive to ν.
_DEFAULT_NU: int = 9


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def bonferroni_z_crit(D_total: int, nu: int, alpha: float = 0.05) -> float:
    """Bonferroni-corrected t critical value for D_total independent tests.

    Parameters
    ----------
    D_total
        Number of scalar dimensions being tested (Bonferroni denominator).
        Returns ``inf`` when ``D_total <= 0`` (degenerate: no dims to test).
    nu
        Degrees of freedom for the t distribution.  Must be >= 1.
    alpha
        Family-wise error rate (default 0.05).

    Returns
    -------
    float
        z_crit = t.ppf(1 - alpha/(2*D_total), nu).
    """
    if D_total <= 0:
        return float("inf")
    return float(scipy_stats.t.ppf(1.0 - alpha / (2.0 * D_total), nu))


def marginal_z_verdict(
    mean_a: np.ndarray,
    std_a: np.ndarray,
    se_a: np.ndarray,
    mean_b: np.ndarray,
    std_b: np.ndarray,
    se_b: np.ndarray,
    D_total: int,
    n_chains: int,
    alpha: float = 0.05,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    """Calibrated per-site marginal-z verdict (all three PR #245 decisions).

    Applies pooled SE denominator, Bonferroni-corrected t threshold, and
    materiality co-primary gate to a single site's per-dimension arrays.
    The caller must provide ``D_total`` counting scalar dims across ALL sites
    that share the same z_crit (Bonferroni denominator is global).

    Parameters
    ----------
    mean_a, std_a, se_a
        Per-dim arrays for distribution A (generated / benchmark run).
        ``std_a`` is accepted for API completeness but not used internally
        (materiality is computed relative to the reference, ``std_b``).
    mean_b, std_b, se_b
        Per-dim arrays for distribution B (committed / ground truth).
        ``std_b`` serves as the materiality reference: mat_d = |Δμ_d| / std_b_d.
    D_total
        Total scalar dims across ALL sites (Bonferroni denominator).
        Caller must aggregate this from all sites before calling.
    n_chains
        Number of Markov chains for the generating distribution.
        ν = 2·(n_chains − 1).  Must be > 1 to produce a finite z_crit
        (n_chains == 1 gives ν = 0 → z_crit = ∞).  For the benchmark path
        where n_chains == 1 (n_chunks=1), callers should supply a defensible
        effective n_chains or use ``_DEFAULT_NU`` directly via
        ``bonferroni_z_crit``.
    alpha
        Family-wise error rate (default 0.05).

    Returns
    -------
    all_pass : bool
        True when no dimension in this site is a hard fail.
    per_dim_verdicts : list of dict
        One dict per dimension with keys ``z``, ``mat``, ``verdict``
        (``"PASS"``, ``"REVIEW"``, or ``"FAIL"``).
    meta : dict
        Keys: ``z_crit``, ``nu``, ``D_total``, ``max_z``, ``n_review``,
        ``n_fail``.
    """
    mean_a = np.asarray(mean_a, dtype=float).ravel()
    se_a = np.asarray(se_a, dtype=float).ravel()
    mean_b = np.asarray(mean_b, dtype=float).ravel()
    std_b = np.asarray(std_b, dtype=float).ravel()
    se_b = np.asarray(se_b, dtype=float).ravel()

    # Decision 1: pooled SE denominator, floored.
    se_denom = np.maximum(np.sqrt(se_a**2 + se_b**2), _SE_FLOOR)
    z_vals = np.abs(mean_a - mean_b) / se_denom

    # Decision 3 (part 1): materiality in units of std_b.
    std_b_safe = np.where(std_b == 0.0, 1.0, std_b)
    mat_vals = np.abs(mean_a - mean_b) / std_b_safe

    # Decision 2: Bonferroni-corrected t threshold.
    nu = 2 * (n_chains - 1)
    z_crit = bonferroni_z_crit(D_total, nu, alpha)

    # Decision 3 (part 2): per-dim verdict — strict > on materiality boundary.
    over_z = z_vals > z_crit
    over_mat = mat_vals > _TAU_SCI

    n_fail = int(np.sum(over_z & over_mat))
    n_review = int(np.sum(over_z & ~over_mat))
    max_z = float(np.max(z_vals)) if len(z_vals) > 0 else 0.0

    per_dim_verdicts: list[dict[str, Any]] = [
        {
            "z": float(z),
            "mat": float(m),
            "verdict": (
                "FAIL"
                if (z > z_crit and m > _TAU_SCI)
                else "REVIEW" if (z > z_crit) else "PASS"
            ),
        }
        for z, m in zip(z_vals, mat_vals)
    ]

    all_pass = n_fail == 0
    meta: dict[str, Any] = {
        "z_crit": z_crit,
        "nu": nu,
        "D_total": D_total,
        "max_z": max_z,
        "n_review": n_review,
        "n_fail": n_fail,
    }

    return all_pass, per_dim_verdicts, meta
