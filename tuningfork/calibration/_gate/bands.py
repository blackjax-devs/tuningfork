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
"""Band helpers: Šidák threshold, metric classification, margin building.

Also hosts ``resolve_thresholds`` (per-model tag overrides) and
``_apply_vi_mode_thresholds`` (VI-as-inference override block).
"""

import copy
import math
from statistics import NormalDist

from .constants import _RANK_VERDICT, _VERDICT_RANK, DEFAULT_THRESHOLDS

# ---------------------------------------------------------------------------
# Dimension-aware (Šidák) PASS band for max_abs_mean_z
# ---------------------------------------------------------------------------


def sidak_t_pass(n_dims: int, alpha: float = 0.05) -> float:
    """Dimension-aware, loosen-only PASS threshold for ``max_abs_mean_z``.

    ``max_abs_mean_z`` is a **max over ``n_dims`` per-dimension z-scores**.
    Under a perfect sampler (H0), each z_i ~ |N(0,1)| and the max over
    ``n_dims`` of them grows with ``n_dims`` (E[max] ≈ sqrt(2 log(2 * n_dims))).
    A fixed PASS<2.0 band false-flags a perfect sampler with growing
    probability as ``n_dims`` grows.  This computes the Šidák-corrected
    per-comparison threshold for a family-wise ``alpha`` over ``n_dims``
    independent comparisons, floored/capped to ``[2.0, 4.0]`` so the band
    only ever *loosens* relative to the historical fixed PASS<2.0 boundary
    and never crosses the fixed FAIL>=4.0 boundary.

    The Šidák correction is applied to the per-dimension significance level
    so the family-wise error rate over ``n_dims`` comparisons is ``≤ alpha``.

    Parameters
    ----------
    n_dims
        Number of finite per-dimension z-scores the ``max_abs_mean_z`` max
        was taken over.  Must be ``>= 1``.
    alpha
        Family-wise significance level.  Default ``0.05``.

    Returns
    -------
    float
        The PASS threshold ``t_pass(n_dims)``, in ``[2.0, 4.0]``.
        ``sidak_t_pass(1) == 2.0`` exactly (recovers the historical fixed
        band; continuity with today's gate).

    Raises
    ------
    ValueError
        If ``n_dims < 1``.
    """
    if n_dims < 1:
        raise ValueError(f"n_dims must be >= 1, got {n_dims}")
    p = (1.0 + (1.0 - alpha) ** (1.0 / n_dims)) / 2.0
    t_pass = NormalDist().inv_cdf(p)
    return float(min(max(t_pass, 2.0), 4.0))


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
# _apply_vi_mode_thresholds
# ---------------------------------------------------------------------------


def _apply_vi_mode_thresholds(thresholds: dict) -> dict:
    """Override thresholds for VI sampler mode (pivotal-z gate).

    Called by ``auto_gate`` when ``vi_sampler_mode=True``.  Replaces
    rhat/ESS/divergence bands with infinite PASS (values still reported)
    and sets the z<4.0 pivotal gate per the 2026-06-04 decision doc.

    Parameters
    ----------
    thresholds
        Threshold dict returned by ``resolve_thresholds``; not mutated.

    Returns
    -------
    dict
        A deep copy with VI-mode overrides applied.
    """
    thresholds = copy.deepcopy(thresholds)
    thresholds["max_abs_mean_z"] = {
        "pass": (0.0, 4.0),  # z < 4 → PASS (pivotal-z gate)
        "review": (4.0, float("inf")),  # z ≥ 4 → REVIEW
        # Note: no FAIL band — even extreme z gets REVIEW (iid draws, no ESS concern)
    }
    # rhat/ESS/div: use infinite PASS bands so they never push verdict to REVIEW/FAIL.
    # The values are still computed and stored in margins for display.
    thresholds["rhat_max"] = {"pass": (0.0, float("inf"))}
    thresholds["min_bulk_ess"] = {"pass": (0.0, float("inf"))}
    thresholds["n_divergences"] = {"pass": (0, float("inf"))}
    return thresholds


# ---------------------------------------------------------------------------
# Metric classification helpers
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
    observed = float(value)
    if not math.isfinite(observed):
        raise ValueError(
            "non-finite observed metric cannot be recorded in a gate margin"
        )

    margin: dict = {"value": observed, "band": band}
    if "pass" in bands:
        lo, hi = bands["pass"]
        margin["pass_lo"] = _json_threshold_bound(lo)
        margin["pass_hi"] = _json_threshold_bound(hi)
    if "review" in bands:
        lo, hi = bands["review"]
        margin["review_lo"] = _json_threshold_bound(lo)
        margin["review_hi"] = _json_threshold_bound(hi)
    return margin


def _json_threshold_bound(value: float) -> float | dict:
    """Return a strict-JSON representation of a threshold bound.

    Infinite bounds are semantic (they mean that a band is unbounded), so
    encode them explicitly rather than relying on JSON's non-standard
    ``Infinity`` token.  NaN is never a meaningful threshold and remains an
    error.  The representation is versioned to make its meaning stable for
    persisted certification evidence.
    """
    bound = float(value)
    if math.isfinite(bound):
        return bound
    if math.isinf(bound):
        return {
            "schema": "tuningfork.gate-threshold-bound.v1",
            "kind": "unbounded",
            "sign": "positive" if bound > 0 else "negative",
        }
    raise ValueError("NaN is not a valid gate threshold bound")
