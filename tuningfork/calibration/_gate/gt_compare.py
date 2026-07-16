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
"""Ground-truth comparison stage — z-scores and bias-sigma diagnostics."""

import math
from dataclasses import dataclass

import arviz as az
import numpy as np

from .bands import sidak_t_pass


@dataclass
class _GtCompareResult:
    """All outputs from the ground-truth comparison stage.

    Consumed by ``_assemble_verdict`` in ``verdict.py``.
    """

    max_abs_mean_z: float | None
    frac_z2: float | None
    n_dims: int
    bias_sigma_at_argmax_z: float | None
    bias_sigma_max_at_z4: float | None
    achieved_bias_bound_sigma: float | None


def _compute_gt_compare(
    mc_samples: dict,
    ground_truth_summaries: dict,
    min_bulk_ess: float | None,
) -> _GtCompareResult:
    """Compute z-scores and bias-sigma diagnostics against ground-truth summaries.

    Parameters
    ----------
    mc_samples
        Dict of arrays ``(n_chains, n_draws, *event_shape)`` from
        ``_samples_to_multichain``.
    ground_truth_summaries
        Dict ``{param_name: {"mean": array, "std": array, ...}}``.  Params
        absent from ``mc_samples`` are skipped.
    min_bulk_ess
        Global min bulk-ESS from the mixing stage; used only as fallback
        when per-dim ESS computation fails.

    Returns
    -------
    _GtCompareResult
        Contains ``max_abs_mean_z``, ``frac_z2``, ``n_dims``, and the three
        ``bias_sigma_*`` fields used as advisory diagnostics.
    """
    # --- max_abs_mean_z + frac_z2 + bias_sigma margins ---
    max_abs_mean_z: float | None = None
    # frac_z2: fraction of *scalar dimensions* with |z_d| > 2 across all sites,
    # flattened.  Secondary diagnostic — never alters verdict.
    # Amendment: dimension-level granularity, not site-level.
    # Site-level collapsed to {0,1} for single-vector-param models (e.g. mvn_10
    # x: 10-D = 1 site × 10 dims) — uninformative.
    _frac_z2: float | None = None
    # n_dims: count of finite per-dimension z-scores the max_abs_mean_z max is
    # taken over (across all sites) — feeds the dimension-aware Šidák PASS
    # band via sidak_t_pass(n_dims).
    _n_dims = 0
    # Margins fields: bias_sigma_* (effect sizes in GT-σ units)
    _bias_sigma_at_argmax_z: float | None = None
    _bias_sigma_max_at_z4: float | None = None
    _achieved_bias_bound_sigma: float | None = None

    z_values: list[float] = []
    # Per-dimension z-scores accumulated across all sites for frac_z2.
    _all_z_dim_scores: list[float] = []
    # Track bias_sigma values for margin computation.
    _bias_sigmas: list[float] = []
    # Track SE and GT std at argmax_z dimension for achieved_bias_bound.
    _argmax_z_idx = -1
    _se_sample_at_argmax: float | None = None
    _se_gt_at_argmax: float | None = None
    _gt_std_at_argmax: float | None = None
    for name, arr in mc_samples.items():
        if name not in ground_truth_summaries:
            continue
        gt = ground_truth_summaries[name]
        arr_np = np.asarray(arr)
        # Merge chains: (n_chains, n_draws, *event_shape) → (n_total, *event_shape)
        n_chains, n_draws = arr_np.shape[0], arr_np.shape[1]
        merged = arr_np.reshape(n_chains * n_draws, *arr_np.shape[2:])
        sample_mean = np.mean(merged, axis=0)
        sample_std = np.std(merged, axis=0)
        # Per-dimension ESS for SE computation (M1 fix).
        # Using the global min_bulk_ess for every dimension's SE was
        # inaccurate: for a model where most dims mix at ESS=2000 but
        # the worst dim has ESS=450, applying ESS=450 to all dims
        # over-inflates SE → z-scores too small → gate too lenient.
        # Fix: compute per-dim bulk ESS via az.ess(method="bulk");
        # fall back to min_bulk_ess only if the computation fails (e.g.,
        # insufficient samples).
        try:
            per_dim_ess = np.asarray(
                az.ess(arr_np, chain_axis=0, draw_axis=1, method="bulk")
            )
            per_dim_ess = np.maximum(per_dim_ess, 1.0)
            if per_dim_ess.shape == ():
                per_dim_ess = np.full(sample_mean.shape, float(per_dim_ess))
        except Exception:
            _fallback_n_eff = (
                min_bulk_ess
                if min_bulk_ess and min_bulk_ess > 0
                else float(n_chains * n_draws)
            )
            per_dim_ess = np.full(sample_mean.shape, max(_fallback_n_eff, 1.0))
        # SE of sample mean (per-dimension)
        se_sample = sample_std / np.sqrt(per_dim_ess)

        gt_mean = np.asarray(gt["mean"])
        gt_std = np.asarray(gt["std"])

        if "between_chain_se" in gt:
            # Multichain GT path (summary_v2): use per-dim between-chain SE as
            # the primary GT uncertainty estimate, floored by the ESS-capped
            # formula.  The cap (min(bulk_ess, n_gt)) is mandatory for high-ESS
            # dims where bulk_ess >> n_gt (e.g. analytic models, theta-class
            # dims with ESS > 100k draws).
            between_chain_se = np.asarray(gt["between_chain_se"])
            gt_bulk_ess = np.asarray(
                gt.get("bulk_ess", np.full_like(between_chain_se, float("inf")))
            )
            n_gt = float(gt.get("n_total", gt.get("n_samples", float("inf"))))
            se_gt_capped = gt_std / np.sqrt(np.minimum(gt_bulk_ess, max(n_gt, 1.0)))
            se_gt = np.maximum(between_chain_se, se_gt_capped)
        else:
            # Legacy GT path (summary.json, single-chain): nominal SE.
            gt_n = gt.get("n_samples", n_chains * n_draws)
            se_gt = gt_std / np.sqrt(max(float(gt_n), 1.0))

        denom = np.maximum(se_sample, se_gt)
        # Avoid division by zero
        denom = np.where(denom > 0, denom, 1.0)
        z_scores = np.abs(sample_mean - gt_mean) / denom

        # Compute bias_sigma: |mean - gt_mean| / gt_std (effect size in GT-σ units)
        # Avoid division by zero in gt_std
        gt_std_safe = np.where(gt_std > 0, gt_std, 1.0)
        bias_sigmas = np.abs(sample_mean - gt_mean) / gt_std_safe

        z_values.append(float(np.max(np.asarray(z_scores))))
        # Accumulate per-dimension z-scores for frac_z2 (dimension-level).
        _all_z_dim_scores.extend(float(z) for z in np.asarray(z_scores).ravel())
        _bias_sigmas.extend(float(b) for b in np.asarray(bias_sigmas).ravel())

        # Track the dimension with argmax z for bias_sigma_at_argmax_z.
        z_flat = np.asarray(z_scores).ravel()
        bias_flat = np.asarray(bias_sigmas).ravel()
        if len(z_flat) > 0:
            local_argmax = np.argmax(z_flat)
            if _argmax_z_idx == -1 or z_flat[local_argmax] > z_values[0]:
                _argmax_z_idx = len(_all_z_dim_scores) - len(z_flat) + local_argmax
                _bias_sigma_at_argmax_z = float(bias_flat[local_argmax])
                _se_sample_at_argmax = float(se_sample.ravel()[local_argmax])
                _se_gt_at_argmax = float(se_gt.ravel()[local_argmax])
                _gt_std_at_argmax = float(np.asarray(gt_std).ravel()[local_argmax])

    if z_values:
        max_abs_mean_z = float(max(z_values))
        # frac_z2: dimension-level (not site-level) — avoids {0,1} collapse
        # for single-vector-param models like mvn_10 (1 site × 10 dims).
        if _all_z_dim_scores:
            _frac_z2 = float(
                sum(1 for z in _all_z_dim_scores if z > 2.0) / len(_all_z_dim_scores)
            )
            _n_dims = sum(1 for z in _all_z_dim_scores if math.isfinite(z))

        # bias_sigma_max_at_z4: max bias effect size among failing dims (z >= 4.0).
        # This is what reviewers use to judge whether z≥4 excursions are material.
        if _bias_sigmas:
            z4_mask = [z >= 4.0 for z in _all_z_dim_scores]
            if any(z4_mask):
                _bias_sigma_max_at_z4 = float(
                    max(b for b, mask in zip(_bias_sigmas, z4_mask) if mask)
                )

        # achieved_bias_bound_sigma: t_pass(d) * max(se_sample, se_gt) / gt_std
        # at the argmax dimension
        if (
            _se_sample_at_argmax is not None
            and _se_gt_at_argmax is not None
            and _gt_std_at_argmax is not None
        ):
            if _n_dims >= 1 and _gt_std_at_argmax > 0:
                t_pass = sidak_t_pass(_n_dims)
                se_max = max(_se_sample_at_argmax, _se_gt_at_argmax)
                _achieved_bias_bound_sigma = float(t_pass * se_max / _gt_std_at_argmax)

    return _GtCompareResult(
        max_abs_mean_z=max_abs_mean_z,
        frac_z2=_frac_z2,
        n_dims=_n_dims,
        bias_sigma_at_argmax_z=_bias_sigma_at_argmax_z,
        bias_sigma_max_at_z4=_bias_sigma_max_at_z4,
        achieved_bias_bound_sigma=_achieved_bias_bound_sigma,
    )
