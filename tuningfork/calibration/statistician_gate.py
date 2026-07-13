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
  - ``max_abs_mean_z`` : max |sample_mean - gt_mean| / max(SE_sample, SE_gt)
                         across all params/dims; only when ground truth is
                         available. At ensemble scale (min_bulk_ess > 6400),
                         z-driven FAILs are demoted to REVIEW (advisory realm;
                         see issue #223). Small-realm (ess ≤ 6400): NHST verdict
                         unchanged.

Per-model threshold overrides are applied via ``Posterior.tags``; see
``resolve_thresholds`` for the recognised tag → relaxation mapping.
"""

import copy
import math
from dataclasses import dataclass
from statistics import NormalDist

import arviz as az
import jax.numpy as jnp
import numpy as np

__all__ = [
    "AutoGateVerdict",
    "DEFAULT_THRESHOLDS",
    "Z_VERDICT_ESS_CEILING",
    "auto_gate",
    "resolve_thresholds",
    "sidak_t_pass",
]

# ---------------------------------------------------------------------------
# Z-test verdict boundary
# ---------------------------------------------------------------------------

Z_VERDICT_ESS_CEILING: int = 6400
"""ESS ceiling above which z-test is advisory (ensemble scale), not verdict-bearing.

Rationale: above this resolution, a point-null z-test rejects any fixed discrepancy
including the GT's own MC error (issue #223). The boundary equals (4/0.05)² where
z≥4 crosses a 0.05σ effect — the gate's own MCSE at ESS=400 PASS floor.

Below Z_VERDICT_ESS_CEILING (small-realm sampling): z-test drives FAIL verdicts as
before. Above it (ensemble scale): z-driven FAILs demote to REVIEW, and z is marked
advisory in margins with human-readable bias_sigma diagnostics.
"""

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
        # Amended 2026-05-12: strict zero relaxed to small absolute count
        # for PASS (rationale per certify_reference._DIVERGENCE_RATE_TOLERANCE
        # comment + decision doc 2026-05-11-phase0-reference-protocol-
        # refinements § 8). A few divergences in a long chain reflects
        # geometry (e.g. funnel-neck visits), not adaptation failure.
        "pass": (0, 6),  # x ≤ 5 → PASS (interval [0,6) i.e. x < 6)
        "review": (6, 40),  # 6 ≤ x < 40 → REVIEW
        # else FAIL
    },
    "max_abs_mean_z": {
        # NOTE: this fixed (0.0, 2.0) PASS band is the *d=1* special case of
        # the dimension-aware Šidák band — see ``sidak_t_pass``.  ``auto_gate``
        # overrides this "pass" tuple at call time with
        # ``(0.0, sidak_t_pass(n_dims))`` before classifying, so the dict
        # value here only matters when ``max_abs_mean_z`` is classified
        # directly against ``DEFAULT_THRESHOLDS`` without going through
        # ``auto_gate`` (e.g. ``resolve_thresholds`` callers, docs, tests).
        # See worklog/decisions/2026-07-03-dimension-aware-pass-band.md.
        "pass": (0.0, 2.0),  # x < 2 → PASS (d=1 case; d>1 loosens via Šidák)
        "review": (2.0, 4.0),  # 2 ≤ x < 4 → REVIEW
        # else FAIL
    },
}


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

    See ``worklog/decisions/2026-07-03-dimension-aware-pass-band.md`` for the
    derivation, the empirical trigger, and the full verification table.

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
    resonance_warning: bool | None = (
        None  # True when L·ε ∈ 2kπ danger zone (fixed-L HMC only)
    )

    def to_dict(self) -> dict:
        """Render in the exact shape ``Recipe.gate_evidence['auto']`` expects.

        Returns
        -------
        dict with keys:
            ``rhat_max``, ``min_bulk_ess``, ``n_divergences``,
            ``max_abs_mean_z``, ``verdict``, ``margins``.
            ``resonance_warning`` included when not ``None``.
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
    multichain: bool | None = None,
) -> dict:
    """Ensure samples are (n_chains, n_draws, *shape); rechunk if needed.

    When ``multichain`` is explicitly provided, it bypasses the heuristic:
    - ``True``: treat as multichain, return as-is.
    - ``False``: treat as single-chain, reshape into n_chunks segments.
    - ``None`` (default): use heuristic to detect layout (see below).

    **Heuristic (when multichain=None)**:
    If the first array in ``samples`` has ndim ≥ 3, it is definitively
    multichain: single-chain positions are (n_samples, *event_shape);
    multichain are (n_chains, n_samples, *event_shape). For ndim < 3,
    a shape-based fallback distinguishes them conservatively (first dim
    < 64 treated as n_chains). This heuristic is permissive to avoid
    the ≤64 cliff bug (issue #217) where genuine multichain arrays with
    nc>64 were misclassified as single-chain.

    **Precondition**: Callers must call ``jax.block_until_ready(samples)`` before
    invoking this function. JAX arrays passed in are expected to be fully
    materialised; ``np.asarray`` here is used for shape inspection and conversion
    to ArviZ input format only.  See
    ``worklog/lessons/code-patterns/2026-05-28-jax-host-materialization-and-block-until-ready.md``

    Parameters
    ----------
    samples
        Dict mapping param name → array of shape
        ``(n_chains, n_draws, *event_shape)`` or ``(n_draws, *event_shape)``.
    n_chunks
        Number of contiguous segments to reshape into when single-chain.
    multichain
        Explicit layout hint. When ``True``, return as-is (cast to np).
        When ``False``, rechunk into n_chunks. When ``None`` (default),
        use the heuristic below.

    Returns
    -------
    dict
        Dict where each array has shape ``(n_chains, n_draws, *event_shape)``.
    """
    if not samples:
        return samples
    first = next(iter(samples.values()))
    arr = np.asarray(first)

    # Determine if samples are multichain
    if multichain is not None:
        is_multichain = multichain
    else:
        # Heuristic: ndim ≥ 3 is definitively multichain.
        # Single-chain: (n_samples, *event_shape) has ndim ≤ 2.
        # Multichain: (n_chains, n_samples, *event_shape) has ndim ≥ 3.
        #
        # For ndim < 3: fallback to first-axis heuristic (conservative).
        # The original ≤64 cliff caused issue #217: arrays with nc>64 were
        # misclassified as single-chain and incorrectly rechunked.
        # The new heuristic (ndim ≥ 3) avoids this cliff entirely.
        if arr.ndim >= 3:
            is_multichain = True
        else:
            # ndim <= 2: first dim ≤ 64 → treat as n_chains.
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
    step_size: float | None = None,
    num_integration_steps: int | None = None,
    vi_sampler_mode: bool = False,
    multichain: bool | None = None,
    ess_per_grad: float | None = None,
    total_grad_evals: int | None = None,
    wall_seconds: float | None = None,
) -> AutoGateVerdict:
    """Compute MCMC quality metrics and render a 3-band verdict.

    When ``vi_sampler_mode=True`` (Track A VI-as-inference recipes where
    each step draws iid from the fitted variational distribution), the gate
    semantics change:

    - ``rhat_max``, ``min_bulk_ess``, ``n_divergences`` are **computed and
      reported** in ``margins`` and ``gate_evidence`` but do **not affect
      the verdict** (their classification is overridden to ``"PASS"``).
      iid draws trivially pass these MCMC diagnostics, so gating on them
      is uninformative.
    - ``max_abs_mean_z`` uses the z<4.0 REVIEW band (per the decision doc
      ``2026-06-04-vi-sampler-pivotal-z-review-gate.md``): z is pivotal for
      iid VI draws (numerator and denominator both ∝ 1/√n, so z ~ |N(0,1)|
      regardless of sample size), making z<2.0 a false-REVIEW gate with
      37–90% false-flag rate across models with d=10–50 under exact VI.
      z<4.0 has <0.5% false-flag at d≤50 while still catching genuine bias.

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
    step_size
        Adapted step size (ε) for fixed-L HMC kernels.  When both
        ``step_size`` and ``num_integration_steps`` are provided, a resonance
        check is performed: ``L·ε ∈ [5.50,7.00] ∪ [11.05,14.07]`` (the true
        2kπ danger zones, ±12% around 2π and 4π) flags the trajectory as near
        a resonance zone where fixed-L HMC can exhibit systematic per-dimension
        bias.  Odd-π/2 multiples (e.g. 5π/2≈7.85) are max-decorrelation points
        and are NOT flagged.  See lesson
        ``2026-05-29-fixed-L-hmc-resonance-at-2pi.md``.  Does **not** affect
        the verdict — stored as ``AutoGateVerdict.resonance_warning``.
    num_integration_steps
        Fixed leapfrog step count L.  Required with ``step_size`` for the
        resonance check.  Pass ``None`` for dynamic-L kernels (NUTS/dmhmc).
    multichain
        Optional explicit hint for sample layout.  When ``True``, treat
        ``samples`` as multichain ``(n_chains, n_draws, *event_shape)`` and
        use as-is. When ``False``, treat as single-chain ``(n_draws, *event_shape)``
        and rechunk into ``n_chunks``. When ``None`` (default), auto-detect via
        the heuristic in ``_samples_to_multichain`` (ndim ≥ 3 is definitively
        multichain; ndim < 3 uses shape[0] < 64 fallback).  The emit path should
        pass ``True`` when it knowingly provides multichain positions.
    ess_per_grad
        Optional bulk-ESS per gradient evaluation (cost metric).  When provided,
        echoed to ``margins["cost"]["ess_per_grad"]``.
    total_grad_evals
        Optional total number of gradient evaluations (cost metric).  When provided,
        echoed to ``margins["cost"]["total_grad_evals"]``.
    wall_seconds
        Optional wall-clock runtime in seconds (cost metric).  When provided,
        echoed to ``margins["cost"]["wall_seconds"]``.

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
      where ``SE_i = std_i / sqrt(ess_i)`` using **per-dimension** bulk-ESS
      from ``az.ess``.  (Previously used the global ``min_bulk_ess`` for every
      dim, which over-inflated SE for well-mixing dims — M1 fix 2026-05-31.)
    - The verdict is the *worst* contribution across all evaluated metrics
      (FAIL beats REVIEW beats PASS).  Skipped metrics do not contribute.

    **Precondition**: Callers must call ``jax.block_until_ready((samples, info))``
    before invoking this function. All JAX array materialisations below (``np.asarray``,
    ``int(jnp_scalar)``, ``float(jnp_scalar)``) rely on this precondition to avoid
    buffer-pool contention. See
    ``worklog/lessons/code-patterns/2026-05-28-jax-host-materialization-and-block-until-ready.md``
    """
    thresholds = resolve_thresholds(posterior)

    # VI-sampler mode: override max_abs_mean_z thresholds to the z<4.0 gate
    # (pivotal-z decision doc 2026-06-04-vi-sampler-pivotal-z-review-gate.md).
    # rhat/ESS/div thresholds are set to "always PASS" — they're computed and
    # reported in margins/evidence but never drive a non-PASS verdict.
    if vi_sampler_mode:
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

    # --- Ensure multi-chain layout ---
    mc_samples = _samples_to_multichain(samples, n_chunks, multichain=multichain)

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
        if hasattr(info, "is_divergent"):
            # HMC/NUTS/laplace family: explicit divergence flag per step.
            n_divergences = int(jnp.sum(jnp.asarray(info.is_divergent)))
        else:
            # MCLMC family (MCLMCInfo, AdjustedMCLMCInfo): rejection-free /
            # no HMC-style divergent transition concept → 0 by definition.
            n_divergences = 0

    # --- max_abs_mean_z + frac_z2 + bias_sigma margins ---
    max_abs_mean_z: float | None = None
    # frac_z2: fraction of *scalar dimensions* with |z_d| > 2 across all sites,
    # flattened.  Secondary diagnostic — never alters verdict.
    # Amendment (2026-05-29): dimension-level granularity, not site-level.
    # Site-level collapsed to {0,1} for single-vector-param models (e.g. mvn_10
    # x: 10-D = 1 site × 10 dims) — uninformative.  See decision doc
    # 2026-05-28-max-abs-mean-z-threshold.md § Amendment.
    _frac_z2: float | None = None
    # n_dims: count of finite per-dimension z-scores the max_abs_mean_z max is
    # taken over (across all sites) — feeds the dimension-aware Šidák PASS
    # band via sidak_t_pass(n_dims).  See
    # worklog/decisions/2026-07-03-dimension-aware-pass-band.md.
    _n_dims = 0
    # Margins fields: bias_sigma_* (effect sizes in GT-σ units)
    _bias_sigma_at_argmax_z: float | None = None
    _bias_sigma_max_at_z4: float | None = None
    _achieved_bias_bound_sigma: float | None = None
    if ground_truth_summaries is not None and mc_samples:
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
            # Fix: compute az.ess per-dim; fall back to min_bulk_ess only
            # if the computation fails (e.g., insufficient samples).
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
                    sum(1 for z in _all_z_dim_scores if z > 2.0)
                    / len(_all_z_dim_scores)
                )
                _n_dims = sum(1 for z in _all_z_dim_scores if math.isfinite(z))

            # bias_sigma_max_at_z4: max bias effect size where z <= 4
            if _bias_sigmas:
                z4_mask = [z <= 4.0 for z in _all_z_dim_scores]
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
                    _achieved_bias_bound_sigma = float(
                        t_pass * se_max / _gt_std_at_argmax
                    )

    # --- Resonance warning (fixed-L HMC only; does not alter verdict) ---
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
    # reasons).  Lesson: worklog/lessons/code-patterns/
    # 2026-05-29-fixed-L-hmc-resonance-at-2pi.md
    _resonance_warning: bool | None = None
    if step_size is not None and num_integration_steps is not None:
        _leps = num_integration_steps * step_size
        _in_zone1 = 5.50 <= _leps <= 7.00
        _in_zone2 = 11.05 <= _leps <= 14.07
        _resonance_warning = bool(_in_zone1 or _in_zone2)

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
        resonance_warning=_resonance_warning,
    )
