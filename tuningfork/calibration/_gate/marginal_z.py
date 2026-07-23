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
"""Shared calibrated-z constants and Bonferroni helpers for GT gate paths.

This module provides the **shared pieces** used by both GT-comparison gates
in tuningfork:

- ``_SE_FLOOR``, ``_TAU_SCI``, ``_TAU_SCI_BENCHMARK`` — materiality constants.
- ``bonferroni_z_crit``, ``bonferroni_z_crit_normal`` — regime-specific
  Bonferroni critical values.

The per-dimension verdict logic is intentionally **inlined** per gate so each
gate can apply its own SE formula, materiality constant, and df regime without
coupling:

- ``tuningfork.groundtruth._verify._check_coherence`` — the ``--verify`` CLI
  coherence path (imports ``_SE_FLOOR``, ``_TAU_SCI``, ``bonferroni_z_crit``).
  Uses the t-df form (between-chain SE, finite n_chains) and TAU_SCI=0.05.
- ``tuningfork.calibration._gate.gt_compare._compute_gt_compare`` — the recipe
  benchmark / auto-gate path (imports ``_TAU_SCI_BENCHMARK``,
  ``bonferroni_z_crit_normal``).  Uses the normal-Bonferroni form (per-dim-ESS
  SE, large df) and TAU_SCI_BENCHMARK=0.15.

Full helper-unification (routing both gates through a shared verdict function)
is a deferred follow-on PR.

Calibration decisions from PR #245 (commit 700cfac)
----------------------------------------------------

  1. **Pooled SE denominator** (inlined per gate)::

         se_denom = max(sqrt(se_a² + se_b²), _SE_FLOOR)

     Prior formula: ``max(se_a, se_b)``, which inflates z by up to √2 at equal
     SE and by less at unequal SE (always ≥ √(1/2) × pooled).

  2. **Dimension-aware Bonferroni threshold** — two regimes::

     **Verify (coherence) regime** — ``bonferroni_z_crit`` (t-df form)::

         ν       = 2·(n_chains − 1)
         z_crit  = t.ppf(1 − α/(2·D_total), ν)

     Used when SE is a *between-chain* SE (finite small-sample df).

     **Benchmark (correctness) regime** — ``bonferroni_z_crit_normal``::

         z_crit  = Φ⁻¹(1 − α/(2·D_total))

     Used when SE is per-dim-ESS-based (``std/√ESS``).  ESS is typically in
     the thousands → large Welch–Satterthwaite pooled df → normal limit.
     At D=503 (stoch_vol), normal z_crit ≈ 3.892.

     Empirical calibration: E[max 503 |N(0,1)|] = 3.243 ± 0.001 (500k-rep MC),
     so a null 503-dim recipe sits on a ≈3.24 max-|z| floor — z_crit=3.892
     sits only 0.65 above the null floor.  The old fixed z<4.0 gate was
     0.76 above the null floor (marginally tighter, but insensitive to D).

     Prior: fixed z < 4.0 (benchmark gate) or fixed z < 3.0 (verify gate),
     both insensitive to model dimensionality.

  3. **Materiality co-primary gate** (strict > on boundary, inlined per gate)::

         HARD FAIL  iff  z_d > z_crit  AND  |Δμ_d|/std_b_d > TAU_SCI[_REGIME]
         REVIEW     iff  z_d > z_crit  AND  |Δμ_d|/std_b_d ≤ TAU_SCI[_REGIME]
         PASS       otherwise

     TAU_SCI=0.05 for the coherence gate; TAU_SCI_BENCHMARK=0.15 for the
     correctness gate (see constant docstrings for calibration rationale).
     Prior: z > threshold → FAIL regardless of effect size |Δμ| / σ.
"""

from __future__ import annotations

from scipy import stats as scipy_stats

__all__ = [
    "_SE_FLOOR",
    "_TAU_SCI",
    "_TAU_SCI_BENCHMARK",
    "bonferroni_z_crit",
    "bonferroni_z_crit_normal",
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
