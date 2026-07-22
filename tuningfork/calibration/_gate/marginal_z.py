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

  2. **Dimension-aware Bonferroni threshold** — two regimes, selected by ``n_chains``::

     **Verify (coherence) regime** — ``n_chains`` is a positive integer::

         ν       = 2·(n_chains − 1)
         z_crit  = t.ppf(1 − α/(2·D_total), ν)    [bonferroni_z_crit]

     Used when SE is a *between-chain* SE (finite small-sample df).
     ``tuningfork.groundtruth._verify._check_coherence`` uses this form.

     **Benchmark (correctness) regime** — ``n_chains=None``::

         z_crit  = Φ⁻¹(1 − α/(2·D_total))          [bonferroni_z_crit_normal]

     Used when SE is per-dim-ESS-based (``std/√ESS``).  ESS is typically in
     the thousands → large Welch–Satterthwaite pooled df → normal limit.
     At D=503 (stoch_vol), normal z_crit ≈ 3.892; t-df6 gives ≈ 9.09
     (too lenient — insensitive to the observed max-order-statistic floor ≈ 3.53).

     Prior: fixed z < 4.0 (benchmark gate) or fixed z < 3.0 (verify gate),
     both insensitive to model dimensionality.

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
  Uses the t-df form (between-chain SE, finite n_chains).
- ``tuningfork.calibration._gate.gt_compare._compute_gt_compare`` — the recipe
  benchmark / auto-gate path (imports ``bonferroni_z_crit_normal``).
  Uses the normal-Bonferroni form (per-dim-ESS SE, large df).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as scipy_stats

__all__ = [
    "_DEFAULT_NU",
    "_SE_FLOOR",
    "_TAU_SCI",
    "_TAU_SCI_BENCHMARK",
    "bonferroni_z_crit",
    "bonferroni_z_crit_normal",
    "marginal_z_verdict",
]

# ---------------------------------------------------------------------------
# Module-level constants: single source of truth across both gate paths.
# ---------------------------------------------------------------------------

# SE floor: prevents division-by-zero on scalar / near-zero-SE sites.
# Under the dual gate this only affects REVIEW labelling, not hard-fail.
_SE_FLOOR: float = 1e-8

# Materiality threshold — VERIFY (coherence) regime (groundtruth/_verify.py).
# |Δμ| / std_ref must strictly exceed this to be a hard FAIL.
# Boundary (mat == _TAU_SCI) is REVIEW, not FAIL (strict >).
# Mirrors the W1 gate sibling in calibration/_gate/w1_realm.py.
_TAU_SCI: float = 0.05

# Materiality threshold — BENCHMARK (correctness) regime (_gate/gt_compare.py).
# Looser than _TAU_SCI because GT-correctness is an inherently noisier measure:
# the recipe's typical per-seed worst-marginal biases cluster at 0.042–0.049σ
# (5/6 seeds) with the outlier seed-18 at 0.085σ.  _TAU_SCI=0.05 bisects the
# cluster itself, causing false hard-FAILs on MC-noise excursions that collapse
# to ~0.02–0.04σ on +chains/+warmup/+samples.  _TAU_SCI_BENCHMARK=0.15 sits
# clearly above the cluster (43% margin on the worst seed, seed-18 at 0.085σ)
# while still flagging genuine ≥0.15σ biases.  The coherence threshold (0.05)
# is tighter because _verify.py compares two high-quality runs where MC noise
# is negligible; the benchmark compares a short nightly run against GT.
_TAU_SCI_BENCHMARK: float = 0.15

# Default ν for the t-df Bonferroni form — VERIFY (coherence) regime ONLY.
#
# NOT used by the benchmark gate, which uses ``bonferroni_z_crit_normal``
# (large df from ESS-based SE → normal limit; see PR #245 decision doc).
#
# Value = 9: conservative fallback when n_chains is not provided to a t-df
# caller (corresponds to ~10-chain GT; ν = 2·(10−1) = 18, halved as a
# one-sided conservative estimate).  Callers with known n_chains should
# pass 2·(n_chains−1) explicitly via ``bonferroni_z_crit``.
_DEFAULT_NU: int = 9


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def bonferroni_z_crit(D_total: int, nu: int, alpha: float = 0.05) -> float:
    """Bonferroni-corrected t critical value for D_total independent tests.

    **Verify (coherence) regime**: use when SE is a between-chain SE (finite
    n_chains, small df).  For the benchmark / correctness regime where SE is
    per-dim-ESS-based (large df → normal limit), use
    ``bonferroni_z_crit_normal`` instead.

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


def bonferroni_z_crit_normal(D_total: int, alpha: float = 0.05) -> float:
    """Bonferroni-corrected normal critical value for D_total independent tests.

    **Benchmark (correctness) regime**: use when SE is per-dim-ESS-based
    (``std/√ESS`` with ESS typically in the thousands → large
    Welch–Satterthwaite pooled df → normal limit).  For the verify /
    coherence regime with finite between-chain SE, use ``bonferroni_z_crit``.

    Spot values (alpha=0.05):
      D=503  (stoch_vol):       z_crit ≈ 3.892
      D=26   (german_credit):   z_crit ≈ 3.102
      D=1500 (high-D):          z_crit ≈ 4.149
      D=1    (single-dim):      z_crit ≈ 1.960

    Parameters
    ----------
    D_total
        Number of scalar dimensions being tested (Bonferroni denominator).
        Returns ``inf`` when ``D_total <= 0`` (degenerate: no dims to test).
    alpha
        Family-wise error rate (default 0.05).

    Returns
    -------
    float
        z_crit = Phi^{-1}(1 - alpha/(2*D_total)).
    """
    if D_total <= 0:
        return float("inf")
    return float(scipy_stats.norm.ppf(1.0 - alpha / (2.0 * D_total)))


def marginal_z_verdict(
    mean_a: np.ndarray,
    std_a: np.ndarray,
    se_a: np.ndarray,
    mean_b: np.ndarray,
    std_b: np.ndarray,
    se_b: np.ndarray,
    D_total: int,
    n_chains: int | None = None,
    alpha: float = 0.05,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    """Calibrated per-site marginal-z verdict (all three PR #245 decisions).

    Applies pooled SE denominator, Bonferroni-corrected threshold, and
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
        Controls which z_crit regime is used:

        ``None`` (default) — **benchmark / correctness regime**.  SE is
        per-dim-ESS-based (high df) → normal-Bonferroni z_crit via
        ``bonferroni_z_crit_normal``.  ``meta["nu"]`` is ``None``.

        ``int`` — **verify / coherence regime**.  SE is between-chain SE
        (finite df).  ν = 2·(n_chains − 1); z_crit via ``bonferroni_z_crit``.
        Must be > 1 to produce a finite z_crit.
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
        Keys: ``z_crit``, ``nu`` (``None`` when normal-approx), ``D_total``,
        ``max_z``, ``n_review``, ``n_fail``.
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

    # Decision 2: Bonferroni threshold, regime-selected by n_chains.
    if n_chains is None:
        # Benchmark regime: ESS-based SE, large df → normal-Bonferroni.
        nu: int | None = None
        z_crit = bonferroni_z_crit_normal(D_total, alpha)
    else:
        # Verify regime: between-chain SE, finite df.
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
