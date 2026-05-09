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
"""Statistician auto-gate — Stage 1 of the P5.0.5 quality pipeline.

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
  - ``n_divergences``  : total divergent transitions (hard-fail threshold).
  - ``max_abs_mean_z`` : max |sample_mean - gt_mean| / max(SE_sample, SE_gt)
                         across all params/dims; only when ground truth is
                         available.

Per-model threshold overrides are applied via ``Posterior.tags``; see
``resolve_thresholds`` for the recognised tag → relaxation mapping.
"""

import copy
import math
from dataclasses import dataclass

import arviz as az
import jax.numpy as jnp
import numpy as np

__all__ = [
    "AutoGateVerdict",
    "DEFAULT_THRESHOLDS",
    "auto_gate",
    "resolve_thresholds",
]

# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, dict[str, tuple]] = {
    # 3-band semantics:
    #   metric within ``pass`` band  → PASS contribution
    #   within ``review`` band       → REVIEW contribution
    #   else                          → FAIL contribution
    # Verdict is the WORST contribution across all evaluated metrics.
    # Missing metrics (e.g. max_abs_mean_z with no ground truth) are skipped.
    "rhat_max": {
        "pass": (0.0, 1.01),  # x < 1.01  → PASS
        "review": (1.01, 1.05),  # 1.01 ≤ x < 1.05 → REVIEW
        # else FAIL
    },
    "min_bulk_ess": {
        "pass": (400.0, math.inf),  # x ≥ 400 → PASS
        "review": (100.0, 400.0),  # 100 ≤ x < 400 → REVIEW
        # else FAIL
    },
    "n_divergences": {
        "pass": (0, 1),  # x == 0 → PASS (interval [0,1) i.e. x < 1)
        # else FAIL (no REVIEW band — divergences are hard)
    },
    "max_abs_mean_z": {
        "pass": (0.0, 2.0),  # x < 2 → PASS
        "review": (2.0, 4.0),  # 2 ≤ x < 4 → REVIEW
        # else FAIL
    },
}


# ---------------------------------------------------------------------------
# AutoGateVerdict
# ---------------------------------------------------------------------------


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

    def to_dict(self) -> dict:
        """Render in the exact shape ``Recipe.gate_evidence['auto']`` expects.

        Returns
        -------
        dict with keys:
            ``rhat_max``, ``min_bulk_ess``, ``n_divergences``,
            ``max_abs_mean_z``, ``verdict``, ``margins``.
        """
        return {
            "rhat_max": self.rhat_max,
            "min_bulk_ess": self.min_bulk_ess,
            "n_divergences": self.n_divergences,
            "max_abs_mean_z": self.max_abs_mean_z,
            "verdict": self.verdict,
            "margins": self.margins,
        }


# ---------------------------------------------------------------------------
# resolve_thresholds
# ---------------------------------------------------------------------------


def resolve_thresholds(posterior=None, defaults: dict | None = None) -> dict:
    """Apply per-model threshold overrides via ``posterior.tags``.

    Recognised tag → effect:

    ``"funnel"``
        Relax ``min_bulk_ess`` pass band to ``(50, inf)``, review to
        ``(10, 50)``.  Funnel geometries genuinely produce low ESS on the
        neck; the standard threshold is too strict.
    ``"multimodal"``
        Skip ``max_abs_mean_z`` entirely.  Mode-coverage is verified by a
        separate test, not by this gate.
    ``"high-correlation"``
        Relax ``rhat_max`` review band to ``(1.01, 1.10)``.

    Posteriors with no recognised tags → ``defaults`` returned unchanged.

    Parameters
    ----------
    posterior
        Optional posterior object with a ``tags: tuple[str, ...]`` attribute.
        Pass ``None`` to get the default thresholds unchanged.
    defaults
        Base threshold dict.  Defaults to ``DEFAULT_THRESHOLDS`` if ``None``.

    Returns
    -------
    dict
        A deep copy of the resolved threshold dict (never mutates ``defaults``).
    """
    if defaults is None:
        defaults = DEFAULT_THRESHOLDS
    thresholds = copy.deepcopy(defaults)
    if posterior is None:
        return thresholds
    tags = getattr(posterior, "tags", ())
    if "funnel" in tags:
        thresholds["min_bulk_ess"] = {
            "pass": (50.0, math.inf),
            "review": (10.0, 50.0),
        }
    if "multimodal" in tags:
        thresholds.pop("max_abs_mean_z", None)
    if "high-correlation" in tags:
        thresholds["rhat_max"]["review"] = (1.01, 1.10)
    return thresholds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_metric(value: float, bands: dict) -> str:
    """Return ``"PASS"``, ``"REVIEW"``, or ``"FAIL"`` for a scalar metric.

    Band intervals are half-open: ``[lo, hi)``.  The ``pass`` band is checked
    first; then ``review``; else ``"FAIL"``.

    Parameters
    ----------
    value
        The scalar metric value.
    bands
        Sub-dict from ``DEFAULT_THRESHOLDS`` for one metric, e.g.
        ``{"pass": (0.0, 1.01), "review": (1.01, 1.05)}``.

    Returns
    -------
    str
        ``"PASS"``, ``"REVIEW"``, or ``"FAIL"``.
    """
    lo_pass, hi_pass = bands["pass"]
    if lo_pass <= value < hi_pass:
        return "PASS"
    if "review" in bands:
        lo_rev, hi_rev = bands["review"]
        if lo_rev <= value < hi_rev:
            return "REVIEW"
    return "FAIL"


_VERDICT_RANK = {"PASS": 0, "REVIEW": 1, "FAIL": 2}
_RANK_VERDICT = {v: k for k, v in _VERDICT_RANK.items()}


def _worst(a: str, b: str) -> str:
    """Return the worse of two verdict strings."""
    return _RANK_VERDICT[max(_VERDICT_RANK[a], _VERDICT_RANK[b])]


def _build_margin(value: float, bands: dict, band: str) -> dict:
    """Build the margin dict for one metric.

    Parameters
    ----------
    value
        Observed metric value.
    bands
        Threshold bands for the metric.
    band
        The classified band (``"PASS"``/``"REVIEW"``/``"FAIL"``).

    Returns
    -------
    dict
        Always contains ``{"value": float, "band": str}``, plus band-limit
        keys present in ``bands``.
    """
    margin: dict = {"value": float(value), "band": band}
    if "pass" in bands:
        lo, hi = bands["pass"]
        margin["pass_lo"] = lo
        margin["pass_hi"] = hi
    if "review" in bands:
        lo, hi = bands["review"]
        margin["review_lo"] = lo
        margin["review_hi"] = hi
    return margin


def _samples_to_multichain(
    samples: dict,
    n_chunks: int,
) -> dict:
    """Ensure samples are (n_chains, n_draws, *shape); rechunk if needed.

    If the first array in ``samples`` has a shape that indicates multi-chain
    layout (ndim ≥ 2 for scalar params, ndim ≥ 3 for vector params), the
    dict is returned as-is.  Otherwise (single-chain layout with shape
    ``(n_samples, *event_shape)``), the samples are reshaped into ``n_chunks``
    contiguous segments following the Tier-A split-R̂ protocol.

    Parameters
    ----------
    samples
        Dict mapping param name → array of shape
        ``(n_chains, n_draws, *event_shape)`` or ``(n_draws, *event_shape)``.
    n_chunks
        Number of contiguous segments to reshape into when single-chain.

    Returns
    -------
    dict
        Dict where each array has shape ``(n_chains, n_draws, *event_shape)``.
    """
    if not samples:
        return samples
    first = next(iter(samples.values()))
    arr = np.asarray(first)

    # Heuristic: if the first axis is small (≤ 32), treat as n_chains.
    # Otherwise treat as single-chain (n_samples, *event_shape).
    # Single-chain: ndim == 1 (scalar param) or ndim ≥ 2 but large first dim.
    # Multi-chain: ndim ≥ 2 and first dim looks like n_chains.
    #
    # The spec says: single-chain has shape (n_samples, *event_shape);
    # multi-chain has (n_chains, n_samples, *event_shape).
    # We detect by checking if the first dim is ≤ n_chunks (heuristic).
    is_multichain = arr.ndim >= 2 and arr.shape[0] <= 64

    if is_multichain:
        return {k: np.asarray(v) for k, v in samples.items()}

    # Single-chain: reshape into n_chunks chunks
    result = {}
    for name, v in samples.items():
        v_np = np.asarray(v)
        n_total = v_np.shape[0]
        event_shape = v_np.shape[1:]
        chunk_size = n_total // n_chunks
        trimmed = v_np[: n_chunks * chunk_size]
        result[name] = trimmed.reshape(n_chunks, chunk_size, *event_shape)
    return result


# ---------------------------------------------------------------------------
# auto_gate
# ---------------------------------------------------------------------------


def auto_gate(
    samples: dict,
    info=None,
    *,
    ground_truth_summaries: dict | None = None,
    posterior=None,
    n_chunks: int = 4,
) -> AutoGateVerdict:
    """Compute MCMC quality metrics and render a 3-band verdict.

    Parameters
    ----------
    samples
        Post-warmup chain output: ``dict[param_name, jax.Array]``.

        - Multi-chain layout: ``(n_chains, n_samples, *event_shape)`` —
          used directly.
        - Single-chain layout: ``(n_samples, *event_shape)`` — reshaped
          into ``n_chunks`` contiguous segments for split-R̂.
    info
        Sampler info struct with an ``is_divergent`` boolean array attribute.
        Pass ``None`` to skip the divergence check.
    ground_truth_summaries
        Optional dict ``{param_name: {"mean": float|array, "std": float|array}}``
        providing ground-truth reference statistics.  When not ``None``, the
        ``max_abs_mean_z`` metric is computed; otherwise it is skipped and
        ``AutoGateVerdict.max_abs_mean_z`` is ``None``.
    posterior
        Optional posterior object with a ``tags: tuple[str, ...]`` attribute
        used by ``resolve_thresholds`` to apply per-model overrides.
    n_chunks
        For single-chain ``samples``, reshape into this many contiguous
        segments before computing split-R̂.  Default 4.

    Returns
    -------
    AutoGateVerdict
        Contains ``rhat_max``, ``min_bulk_ess``, ``n_divergences``,
        ``max_abs_mean_z``, ``verdict`` (``"PASS"`` / ``"REVIEW"`` /
        ``"FAIL"``), and ``margins``.

    Notes
    -----
    - R̂ uses arviz ``az.rhat`` (rank-normalised split-R̂, Vehtari et al. 2021).
    - Bulk-ESS uses ``az.ess(method="bulk")``.
    - ``n_divergences = int(jnp.sum(info.is_divergent))`` (flattened).
    - ``max_abs_mean_z = max_i |mean_i - gt_mean_i| / max(SE_i, SE_gt_i)``
      where ``SE_i = std_i / sqrt(n_eff_i)`` (approximated as
      ``std_i / sqrt(min_bulk_ess)`` for simplicity).
    - The verdict is the *worst* contribution across all evaluated metrics
      (FAIL beats REVIEW beats PASS).  Skipped metrics do not contribute.
    """
    thresholds = resolve_thresholds(posterior)

    # --- Ensure multi-chain layout ---
    mc_samples = _samples_to_multichain(samples, n_chunks)

    # --- R̂ and bulk-ESS ---
    rhat_max: float | None = None
    min_bulk_ess: float | None = None

    if mc_samples:
        rhat_values: list[float] = []
        ess_values: list[float] = []
        for arr in mc_samples.values():
            arr_np = np.asarray(arr)
            # arviz expects (n_chains, n_draws, *event_shape)
            rhat_arr = az.rhat(arr_np, chain_axis=0, draw_axis=1)
            ess_arr = az.ess(arr_np, chain_axis=0, draw_axis=1, method="bulk")
            rhat_values.append(float(np.max(np.asarray(rhat_arr))))
            ess_values.append(float(np.min(np.asarray(ess_arr))))

        rhat_max = float(max(rhat_values))
        min_bulk_ess = float(min(ess_values))

    # --- Divergences ---
    n_divergences: int | None = None
    if info is not None:
        n_divergences = int(jnp.sum(jnp.asarray(info.is_divergent)))

    # --- max_abs_mean_z ---
    max_abs_mean_z: float | None = None
    if ground_truth_summaries is not None and mc_samples:
        z_values: list[float] = []
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
            n_eff = (
                min_bulk_ess
                if min_bulk_ess and min_bulk_ess > 0
                else float(n_chains * n_draws)
            )
            # SE of sample mean
            se_sample = sample_std / np.sqrt(max(n_eff, 1.0))

            gt_mean = np.asarray(gt["mean"])
            gt_std = np.asarray(gt["std"])
            gt_n = gt.get("n_samples", n_chains * n_draws)
            se_gt = gt_std / np.sqrt(max(float(gt_n), 1.0))

            denom = np.maximum(se_sample, se_gt)
            # Avoid division by zero
            denom = np.where(denom > 0, denom, 1.0)
            z_scores = np.abs(sample_mean - gt_mean) / denom
            z_values.append(float(np.max(np.asarray(z_scores))))

        if z_values:
            max_abs_mean_z = float(max(z_values))

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

    # max_abs_mean_z
    if max_abs_mean_z is not None and "max_abs_mean_z" in thresholds:
        band = _classify_metric(max_abs_mean_z, thresholds["max_abs_mean_z"])
        margins["max_abs_mean_z"] = _build_margin(
            max_abs_mean_z, thresholds["max_abs_mean_z"], band
        )
        overall_verdict = _worst(overall_verdict, band)

    return AutoGateVerdict(
        rhat_max=rhat_max,
        min_bulk_ess=min_bulk_ess,
        n_divergences=n_divergences,
        max_abs_mean_z=max_abs_mean_z,
        verdict=overall_verdict,
        margins=margins,
    )
