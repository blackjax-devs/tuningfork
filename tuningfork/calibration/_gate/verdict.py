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
"""Verdict assembly — classify metrics, build margins, return AutoGateVerdict."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .bands import _build_margin, _classify_metric, _worst, sidak_t_pass
from .constants import Z_VERDICT_ESS_CEILING
from .gt_compare import _GtCompareResult

if TYPE_CHECKING:
    from .w1_realm import W1RealmResult


@dataclass(frozen=True)
class AutoGateVerdict:
    """Output of ``auto_gate(...)``.

    Maps directly into ``Recipe.gate_evidence['auto']`` via ``to_dict()``.

    Parameters
    ----------
    rhat_max
        Maximum rank-normalised split-R̂ across all parameters and dimensions.
        ``None`` if the samples dict is empty.
    min_bulk_ess
        Minimum bulk-ESS across all parameters and dimensions.
        ``None`` if the samples dict is empty.
    n_divergences
        Total number of divergent transitions from ``info.is_divergent``.
        ``None`` if ``info`` is ``None``.
    max_abs_mean_z
        Maximum |sample_mean - gt_mean| / max(SE_sample, SE_gt) over all
        params and dimensions.  ``None`` when ``ground_truth_summaries`` is
        not provided.
    verdict
        One of ``"PASS"``, ``"REVIEW"``, or ``"FAIL"``.
    margins
        Per-metric proximity information.  Keys are metric names; values are
        dicts with at least ``{"value": float, "band": str}`` and band-limit
        keys depending on the threshold structure.  Skipped metrics (e.g.
        ``max_abs_mean_z`` when no ground truth) are absent from ``margins``.
    """

    rhat_max: float | None
    min_bulk_ess: float | None
    n_divergences: int | None
    max_abs_mean_z: float | None
    verdict: str  # "PASS" | "REVIEW" | "FAIL"
    margins: dict  # per-threshold proximity info
    resonance_warning: bool | None = (
        None  # True when L·ε ∈ 2kπ danger zone (fixed-L HMC only)
    )
    w1_realm_result: W1RealmResult | None = None  # populated when gt_draws provided

    def to_dict(self) -> dict:
        """Render in the exact shape ``Recipe.gate_evidence['auto']`` expects.

        Returns
        -------
        dict with keys:
            ``rhat_max``, ``min_bulk_ess``, ``n_divergences``,
            ``max_abs_mean_z``, ``verdict``, ``margins``.
            ``resonance_warning`` included when not ``None``.
            ``margins["w1_realm"]`` included when ``w1_realm_result`` is not ``None``.
        """
        d = {
            "rhat_max": self.rhat_max,
            "min_bulk_ess": self.min_bulk_ess,
            "n_divergences": self.n_divergences,
            "max_abs_mean_z": self.max_abs_mean_z,
            "verdict": self.verdict,
            "margins": self.margins,
        }
        if self.resonance_warning is not None:
            d["resonance_warning"] = self.resonance_warning
        return d


def _assemble_verdict(
    rhat_max: float | None,
    min_bulk_ess: float | None,
    n_divergences: int | None,
    gt_result: _GtCompareResult | None,
    thresholds: dict,
    *,
    vi_sampler_mode: bool,
    resonance_warning: bool | None,
    ess_per_grad: float | None,
    total_grad_evals: int | None,
    wall_seconds: float | None,
    w1_realm_result: W1RealmResult | None = None,
) -> AutoGateVerdict:
    """Classify each metric, assemble margins, and build AutoGateVerdict.

    Processes metrics in order: rhat_max → min_bulk_ess → n_divergences →
    max_abs_mean_z (with z-advisory demotion in ensemble realm) → cost block.

    Parameters
    ----------
    rhat_max, min_bulk_ess, n_divergences
        From the mixing stage (``_compute_mixing_stats``).
    gt_result
        From the GT-compare stage (``_compute_gt_compare``), or ``None`` when
        no ground-truth summaries were provided.
    thresholds
        Resolved (and possibly VI-mode-overridden) threshold dict.
    vi_sampler_mode
        When ``True``, the Šidák band adjustment for ``max_abs_mean_z`` is
        skipped (VI mode uses its own dedicated pivotal-z band).
    resonance_warning, ess_per_grad, total_grad_evals, wall_seconds
        Passed through verbatim into the verdict / margins.

    Returns
    -------
    AutoGateVerdict
    """
    # Unpack gt_result fields (use defaults when no GT was provided)
    max_abs_mean_z: float | None = None
    _frac_z2: float | None = None
    _n_dims: int = 0
    _bias_sigma_at_argmax_z: float | None = None
    _bias_sigma_max_at_z4: float | None = None
    _achieved_bias_bound_sigma: float | None = None
    if gt_result is not None:
        max_abs_mean_z = gt_result.max_abs_mean_z
        _frac_z2 = gt_result.frac_z2
        _n_dims = gt_result.n_dims
        _bias_sigma_at_argmax_z = gt_result.bias_sigma_at_argmax_z
        _bias_sigma_max_at_z4 = gt_result.bias_sigma_max_at_z4
        _achieved_bias_bound_sigma = gt_result.achieved_bias_bound_sigma

    # --- Classify each metric and accumulate verdict ---
    overall_verdict = "PASS"
    margins: dict = {}

    # rhat_max
    if rhat_max is not None and "rhat_max" in thresholds:
        band = _classify_metric(rhat_max, thresholds["rhat_max"])
        margins["rhat_max"] = _build_margin(rhat_max, thresholds["rhat_max"], band)
        overall_verdict = _worst(overall_verdict, band)

    # min_bulk_ess
    if min_bulk_ess is not None and "min_bulk_ess" in thresholds:
        band = _classify_metric(min_bulk_ess, thresholds["min_bulk_ess"])
        margins["min_bulk_ess"] = _build_margin(
            min_bulk_ess, thresholds["min_bulk_ess"], band
        )
        overall_verdict = _worst(overall_verdict, band)

    # n_divergences
    if n_divergences is not None and "n_divergences" in thresholds:
        band = _classify_metric(float(n_divergences), thresholds["n_divergences"])
        margins["n_divergences"] = _build_margin(
            float(n_divergences), thresholds["n_divergences"], band
        )
        overall_verdict = _worst(overall_verdict, band)

    # max_abs_mean_z with z-advisory demotion in ensemble realm
    if max_abs_mean_z is not None and "max_abs_mean_z" in thresholds:
        z_bands = thresholds["max_abs_mean_z"]
        if not vi_sampler_mode and _n_dims >= 1:
            # Dimension-aware (Šidák), loosen-only PASS band: replace the
            # fixed (0.0, 2.0) upper edge with sidak_t_pass(n_dims), which
            # is >= 2.0 for all n_dims (floored) and <= 4.0 (capped) — so the
            # PASS region only grows relative to the historical fixed band,
            # never shrinks, and the FAIL >= 4.0 boundary is untouched.
            # vi_sampler_mode uses its own dedicated pivotal-z band (z<4.0
            # PASS, no FAIL) and is deliberately left alone here.
            # See worklog/decisions/2026-07-03-dimension-aware-pass-band.md.
            t_pass = sidak_t_pass(_n_dims)
            z_bands = dict(z_bands)
            z_bands["pass"] = (0.0, t_pass)
            z_bands["review"] = (t_pass, z_bands.get("review", (2.0, 4.0))[1])
        band = _classify_metric(max_abs_mean_z, z_bands)

        # Z-advisory realm demotion: at ensemble scale (min_bulk_ess > Z_VERDICT_ESS_CEILING),
        # z-driven FAILs demote to REVIEW. The z-test rejects any fixed discrepancy
        # at this resolution, including the GT's own MC error (issue #223).
        # Small-realm (ess ≤ 6400): verdict bit-identical to previous behavior.
        z_is_advisory = False
        if (
            band == "FAIL"
            and min_bulk_ess is not None
            and min_bulk_ess > Z_VERDICT_ESS_CEILING
        ):
            # Check if z is the driver of the FAIL (no other metric is already FAIL).
            # If rhat or n_divergences is already FAIL, we keep FAIL.
            # If only z is FAIL, demote to REVIEW.
            has_other_fail = any(
                m.get("band") == "FAIL" for m in margins.values() if isinstance(m, dict)
            )
            if not has_other_fail:
                band = "REVIEW"
                z_is_advisory = True

        margins["max_abs_mean_z"] = _build_margin(max_abs_mean_z, z_bands, band)
        if z_is_advisory:
            margins["max_abs_mean_z"]["z_advisory"] = True
            margins["max_abs_mean_z"]["bias_sigma"] = (
                f"z is advisory at this resolution (min_bulk_ess={min_bulk_ess:.0f} "
                f"> {Z_VERDICT_ESS_CEILING}); see bias_sigma fields"
            )

        if _frac_z2 is not None:
            # Secondary diagnostic: fraction of sites with |z| > 2.
            # Does NOT alter the verdict — purely informational for the statistician.
            margins["max_abs_mean_z"]["frac_z2"] = _frac_z2

        # Add the new REPORTED margins (never verdict; bias effect sizes in GT-σ units)
        if _bias_sigma_at_argmax_z is not None:
            margins["max_abs_mean_z"][
                "bias_sigma_at_argmax_z"
            ] = _bias_sigma_at_argmax_z
        if _bias_sigma_max_at_z4 is not None:
            margins["max_abs_mean_z"]["bias_sigma_max_at_z4"] = _bias_sigma_max_at_z4
        if _achieved_bias_bound_sigma is not None:
            margins["max_abs_mean_z"][
                "achieved_bias_bound_sigma"
            ] = _achieved_bias_bound_sigma

        overall_verdict = _worst(overall_verdict, band)

    # --- W1 realm block (stage 4.5 — runs only when gt_draws provided and stage-1 passes) ---
    if w1_realm_result is not None:
        # _worst handles SKIP by mapping it to rank 0 (same as PASS); see _VERDICT_RANK.
        w1_verdict = w1_realm_result.verdict
        margins["w1_realm"] = {
            "verdict": w1_verdict,
            "max_w1_sigma": float(w1_realm_result.max_w1_sigma),
            "floor_of_max": float(w1_realm_result.floor_of_max),
            "frac_failing_dims": float(w1_realm_result.frac_failing_dims),
            "tau_frac": float(w1_realm_result.tau_frac),
            "n_dims": int(w1_realm_result.n_dims),
            "n_heavy_tail_dims": int(w1_realm_result.n_heavy_tail_dims),
            "max_prong_verdict": w1_realm_result.max_prong_verdict,
            "frac_prong_verdict": w1_realm_result.frac_prong_verdict,
        }
        if w1_realm_result.loo_check is not None:
            margins["w1_realm"]["loo_check"] = w1_realm_result.loo_check
        overall_verdict = _worst(overall_verdict, w1_verdict)

    # --- Add optional cost block to margins ---
    if (
        ess_per_grad is not None
        or total_grad_evals is not None
        or wall_seconds is not None
    ):
        cost: dict = {}
        if ess_per_grad is not None:
            cost["ess_per_grad"] = ess_per_grad
        if total_grad_evals is not None:
            cost["total_grad_evals"] = total_grad_evals
        if wall_seconds is not None:
            cost["wall_seconds"] = wall_seconds
        if cost:
            margins["cost"] = cost

    return AutoGateVerdict(
        rhat_max=rhat_max,
        min_bulk_ess=min_bulk_ess,
        n_divergences=n_divergences,
        max_abs_mean_z=max_abs_mean_z,
        verdict=overall_verdict,
        margins=margins,
        resonance_warning=resonance_warning,
        w1_realm_result=w1_realm_result,
    )
