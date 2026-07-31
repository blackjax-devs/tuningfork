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
"""Statistician auto-gate — automated quality assessment of MCMC samples.

``auto_gate`` computes MCMC quality metrics from post-warmup chain output and
renders a 3-band verdict (PASS / REVIEW / FAIL).  The verdict maps directly into
``Recipe.gate_evidence["auto"]`` via ``AutoGateVerdict.to_dict()``.

Verdict semantics (NON-BLOCKING — the Statistician agent can override):
  PASS   — all evaluated thresholds within the ``pass`` band; cell can
            auto-commit at LOW.
  REVIEW — at least one metric in the ``review`` band; none in FAIL;
            agent inspection requested.
  FAIL   — at least one metric in FAIL (worst-of-all aggregation); MED
            workflow needed.

Metrics evaluated:
  - ``rhat_max``       : max rank-normalised split-R̂ across all params/dims.
  - ``min_bulk_ess``   : min bulk-ESS across all params/dims.
  - ``n_divergences``  : total divergent transitions (rate-tolerant threshold;
                         see ``DEFAULT_THRESHOLDS["n_divergences"]``).
  - ``n_nonfinite_proposals`` : false entries in MCLMC ``info.nonans``, kept
                                separate from divergences. Zero contributes
                                PASS; a positive count provisionally contributes
                                REVIEW until a calibrated boundary exists.
  - ``max_abs_mean_z`` : max |sample_mean - gt_mean| / max(SE_sample, SE_gt)
                         across all params/dims; only when ground truth is
                         available. At ensemble scale (min_bulk_ess > 6400),
                         z-driven FAILs are demoted to REVIEW (advisory realm;
                         see issue #223). Small-realm (ess ≤ 6400): NHST verdict
                         unchanged.

Per-model threshold overrides are applied via ``Posterior.tags``; see
``resolve_thresholds`` for the recognised tag → relaxation mapping.

Implementation note
-------------------
This module is the **public facade**.  The computation is split into stage
modules under ``tuningfork.calibration._gate``:

  - ``_gate.constants`` — thresholds, ESS ceiling, verdict-rank tables.
  - ``_gate.bands``     — ``sidak_t_pass``, ``resolve_thresholds``, classify/
                          margin helpers, VI-mode threshold override.
  - ``_gate.layout``    — ``_samples_to_multichain`` (single→multichain).
  - ``_gate.mixing``    — ``_compute_mixing_stats`` (R̂, ESS, sampler-specific
                          numerical evidence).
  - ``_gate.gt_compare``— ``_compute_gt_compare`` (z-scores, bias-sigma fields).
  - ``_gate.verdict``   — ``AutoGateVerdict``, ``_assemble_verdict``.

All currently-public names (``auto_gate``, ``AutoGateVerdict``,
``DEFAULT_THRESHOLDS``, ``Z_VERDICT_ESS_CEILING``, ``resolve_thresholds``,
``sidak_t_pass``) and the private ``_samples_to_multichain`` (imported by tests)
are re-exported from here unchanged — existing imports repo-wide work untouched.
"""

from tuningfork.calibration._gate import (
    DEFAULT_THRESHOLDS,
    Z_VERDICT_ESS_CEILING,
    AutoGateVerdict,
    _apply_vi_mode_thresholds,
    _assemble_verdict,
    _classify_metric,
    _compute_gt_compare,
    _compute_mixing_stats,
    _samples_to_multichain,
    compute_w1_realm,
    resolve_thresholds,
    sidak_t_pass,
)

__all__ = [
    "AutoGateVerdict",
    "DEFAULT_THRESHOLDS",
    "Z_VERDICT_ESS_CEILING",
    "auto_gate",
    "resolve_thresholds",
    "sidak_t_pass",
]


def auto_gate(
    samples: dict,
    info=None,
    *,
    ground_truth_summaries: dict | None = None,
    gt_draws: dict | None = None,
    posterior=None,
    n_chunks: int = 4,
    step_size: float | None = None,
    num_integration_steps: int | None = None,
    vi_sampler_mode: bool = False,
    multichain: bool | None = None,
    ess_per_grad: float | None = None,
    total_grad_evals: int | None = None,
    wall_seconds: float | None = None,
) -> AutoGateVerdict:
    """Compute MCMC quality metrics and render a 3-band verdict.

    See the module docstring and ``_gate/`` stage modules for full parameter
    documentation.  The signature and return value are identical to the
    pre-refactor monolith — this is an orchestrator only.

    Parameters
    ----------
    samples
        Post-warmup chain output, dict of arrays.
    info
        Optional sampler info. ``is_divergent`` supplies HMC divergences;
        MCLMC ``nonans`` supplies separate non-finite-proposal evidence.
    ground_truth_summaries
        Per-site summary stats (mean, std, bulk_ess, tail_ess) from the
        multichain GT.  Required for the mean-z and W1 realm stages.
    gt_draws
        Per-site raw GT draw arrays, shape ``(n_gt_chains, n_gt_draws, *event)``.
        When provided together with ``ground_truth_summaries``, triggers the
        W1/σ two-prong equivalence gate (Stage 4.5) — ONLY after R̂/ESS/div PASS.
    posterior, n_chunks, step_size, num_integration_steps, vi_sampler_mode,
    multichain, ess_per_grad, total_grad_evals, wall_seconds
        Unchanged from previous interface.

    Stages (in order):
    1. ``resolve_thresholds`` + optional VI-mode override (``_gate.bands``).
    2. ``_samples_to_multichain`` (``_gate.layout``).
    3. ``_compute_mixing_stats`` → R̂, ESS, divergences, and conditional
       non-finite-proposal evidence (``_gate.mixing``).
    4. ``_compute_gt_compare`` → max_abs_mean_z + bias-sigma fields
       (``_gate.gt_compare``), when ground truth is provided.
    4.5 ``compute_w1_realm`` → W1/σ two-prong equivalence gate (``_gate.w1_realm``),
        when ``gt_draws`` is provided AND R̂/ESS/div all PASS (second-stage).
    5. Resonance warning (inline; 3 lines; does not alter verdict).
    6. ``_assemble_verdict`` → ``AutoGateVerdict`` (``_gate.verdict``).
    """
    # --- Stage 1: resolve thresholds ---
    thresholds = resolve_thresholds(posterior)
    # VI-sampler mode: override max_abs_mean_z thresholds to the z<4.0 gate
    # (pivotal-z decision doc 2026-06-04-vi-sampler-pivotal-z-review-gate.md).
    # rhat/ESS/div thresholds are set to "always PASS" — they're computed and
    # reported in margins/evidence but never drive a non-PASS verdict.
    if vi_sampler_mode:
        thresholds = _apply_vi_mode_thresholds(thresholds)

    # --- Stage 2: ensure multichain layout ---
    mc_samples = _samples_to_multichain(samples, n_chunks, multichain=multichain)

    # --- Stage 3: R̂, bulk-ESS, divergences ---
    mixing_stats = _compute_mixing_stats(mc_samples, info)
    rhat_max = mixing_stats.rhat_max
    min_bulk_ess = mixing_stats.min_bulk_ess
    n_divergences = mixing_stats.n_divergences

    # --- Stage 4: ground-truth z-scores and bias-sigma diagnostics ---
    gt_result = None
    if ground_truth_summaries is not None and mc_samples:
        gt_result = _compute_gt_compare(
            mc_samples, ground_truth_summaries, min_bulk_ess
        )

    # --- Stage 4.5: W1/σ two-prong equivalence gate (SECOND-STAGE) ---
    # Runs only when gt_draws + ground_truth_summaries are provided AND
    # Stage-3 metrics are all NOT-FAIL (R̂/ESS/div must pass).
    # VI-sampler mode is excluded — W1 is not defined for VI in this build.
    _w1_realm_result = None
    if (
        gt_draws is not None
        and ground_truth_summaries is not None
        and mc_samples
        and not vi_sampler_mode
    ):
        _rhat_ok = rhat_max is None or (
            _classify_metric(rhat_max, thresholds.get("rhat_max", {})) != "FAIL"
        )
        _ess_ok = min_bulk_ess is None or (
            _classify_metric(min_bulk_ess, thresholds.get("min_bulk_ess", {})) != "FAIL"
        )
        _div_ok = n_divergences is None or (
            _classify_metric(float(n_divergences), thresholds.get("n_divergences", {}))
            != "FAIL"
        )
        if _rhat_ok and _ess_ok and _div_ok:
            _w1_realm_result = compute_w1_realm(
                mc_samples,
                ground_truth_summaries,
                gt_draws,
                multichain=True,
            )

    # --- Stage 5: resonance warning (fixed-L HMC only; does not alter verdict) ---
    # True danger zones are the 2kπ resonances where fixed-L HMC exhibits
    # per-dimension systematic bias.  Odd multiples of π/2 (e.g. 5π/2 ≈ 7.85)
    # are max-decorrelation points — outside the danger zones.
    #
    # Zone k=1: L·ε ∈ [5.50, 7.00]  (±12% around 2π ≈ 6.28)
    # Zone k=2: L·ε ∈ [11.05, 14.07] (±12% around 4π ≈ 12.57)
    #
    # Ratified Phase 8B.3 (2026-06-03): tightened from the previous ≥ 5.5
    # catch-all to true 2kπ intervals only.  L=25 at ε≈0.31 gives L·ε≈7.85
    # (≈ 5π/2, outside both zones → no warning); L=30 at same ε gives ≈9.4
    # (≈ 3π, also outside both zones — but produces z=2.78 bias for other
    # reasons).
    _resonance_warning: bool | None = None
    if step_size is not None and num_integration_steps is not None:
        _leps = num_integration_steps * step_size
        _in_zone1 = 5.50 <= _leps <= 7.00
        _in_zone2 = 11.05 <= _leps <= 14.07
        _resonance_warning = bool(_in_zone1 or _in_zone2)

    # --- Stage 6: classify metrics, build margins, return verdict ---
    return _assemble_verdict(
        rhat_max,
        min_bulk_ess,
        n_divergences,
        gt_result,
        thresholds,
        vi_sampler_mode=vi_sampler_mode,
        resonance_warning=_resonance_warning,
        ess_per_grad=ess_per_grad,
        total_grad_evals=total_grad_evals,
        wall_seconds=wall_seconds,
        w1_realm_result=_w1_realm_result,
        n_nonfinite_proposals=mixing_stats.n_nonfinite_proposals,
        n_proposals_evaluated=mixing_stats.n_proposals_evaluated,
        nonfinite_proposal_rate=mixing_stats.nonfinite_proposal_rate,
    )
