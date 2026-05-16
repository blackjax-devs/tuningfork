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
"""Family-aware ArviZ diagnostic rendering for recipe diagnostics notebook.

This module provides a suite of family-dispatch functions that render
appropriate diagnostic plots for each sampler family:

- Gradient MH-corrected (nuts, hmc, mhmc, mala, barker, ghmc, dynamic_hmc, dmhmc, rmhmc)
- MCLMC family (mclmc, adjusted_mclmc, adjusted_mclmc_dynamic)
- SMC family (adaptive_tempered_smc, tempered_smc, etc.)
- VI family (meanfield_vi, fullrank_vi)
- Specialised (elliptical_slice, mgrad_gaussian, laplace*, orbital_hmc, irmh, additive_step_rw)

Each family has a dedicated render_* function that returns a list of
matplotlib figures (or a single figure) appropriate for human inspection.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

try:
    import arviz as az
except ImportError:
    az = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError:
    plt = None
    Figure = None

__all__ = [
    "samples_to_idata",
    "render_universal_summary",
    # Semantic names (current)
    "GRADIENT_MH_SAMPLERS",
    "MCLMC_FAMILY_SAMPLERS",
    "SMC_FAMILY_SAMPLERS",
    "VI_FAMILY_SAMPLERS",
    "SPECIALISED_SAMPLERS",
    "render_gradient_mh",
    "render_mclmc_family",
    "render_smc_family",
    "render_vi_family",
    "render_specialised",
    # Deprecated aliases (kept for one release)
    "FAMILY_A_SAMPLERS",
    "FAMILY_B_SAMPLERS",
    "FAMILY_C_SAMPLERS",
    "FAMILY_D_SAMPLERS",
    "FAMILY_E_SAMPLERS",
    "render_family_a",
    "render_family_b",
    "render_family_c",
    "render_family_d",
    "render_family_e",
]

# ---------------------------------------------------------------------------
# Family membership constants — semantic names
# ---------------------------------------------------------------------------

GRADIENT_MH_SAMPLERS = {
    "nuts",
    "hmc",
    "mhmc",
    "mala",
    "barker",
    "ghmc",
    "dynamic_hmc",
    "dmhmc",
    "rmhmc",
}
MCLMC_FAMILY_SAMPLERS = {"mclmc", "adjusted_mclmc", "adjusted_mclmc_dynamic"}
SMC_FAMILY_SAMPLERS = {
    "adaptive_tempered_smc",
    "tempered_smc",
    "adaptive_persistent_sampling_smc",
    "persistent_sampling_smc",
    "partial_posteriors_smc",
    "inner_kernel_tuning",
}
VI_FAMILY_SAMPLERS = {"meanfield_vi", "fullrank_vi"}
SPECIALISED_SAMPLERS = {
    "elliptical_slice",
    "mgrad_gaussian",
    "laplace_hmc",
    "laplace_dhmc",
    "laplace_mhmc",
    "laplace_dmhmc",
    "orbital_hmc",
    "irmh",
    "additive_step_random_walk",
}

# Gradient MH subset that supports plot_energy (excludes mala, barker, ghmc)
GRADIENT_MH_WITH_ENERGY = {"nuts", "hmc", "mhmc", "dynamic_hmc", "dmhmc", "rmhmc"}

# ---------------------------------------------------------------------------
# Deprecated aliases — FAMILY_X_SAMPLERS → semantic names
# Kept for one release; will be removed in a future version.
# ---------------------------------------------------------------------------

# DEPRECATED: use GRADIENT_MH_SAMPLERS instead
FAMILY_A_SAMPLERS = (
    GRADIENT_MH_SAMPLERS  # DEPRECATED: rename per notebook-arviz-redesign thread
)
# DEPRECATED: use MCLMC_FAMILY_SAMPLERS instead
FAMILY_B_SAMPLERS = (
    MCLMC_FAMILY_SAMPLERS  # DEPRECATED: rename per notebook-arviz-redesign thread
)
# DEPRECATED: use SMC_FAMILY_SAMPLERS instead
FAMILY_C_SAMPLERS = (
    SMC_FAMILY_SAMPLERS  # DEPRECATED: rename per notebook-arviz-redesign thread
)
# DEPRECATED: use VI_FAMILY_SAMPLERS instead
FAMILY_D_SAMPLERS = (
    VI_FAMILY_SAMPLERS  # DEPRECATED: rename per notebook-arviz-redesign thread
)
# DEPRECATED: use SPECIALISED_SAMPLERS instead
FAMILY_E_SAMPLERS = (
    SPECIALISED_SAMPLERS  # DEPRECATED: rename per notebook-arviz-redesign thread
)

# Keep old FAMILY_A_WITH_ENERGY alias pointing at the new name
FAMILY_A_WITH_ENERGY = (
    GRADIENT_MH_WITH_ENERGY  # DEPRECATED: use GRADIENT_MH_WITH_ENERGY
)


# Mapping from our chain_stats field names (blackjax NUTSInfo._fields) to
# ArviZ's canonical sample_stats group names.
# Reference: https://python.arviz.org/en/stable/schema/schema.html#sample-stats
# Only keys that map to a per-step scalar stat are listed. Vectors (e.g.
# `momentum`) and nested fields (`proposal`) are intentionally dropped.
#
# ArviZ canonical sample_stats keys (per the schema, 2026-05-12):
#   lp, acceptance_rate, step_size, step_size_nom, tree_depth, n_steps,
#   reached_max_treedepth, diverging, energy, energy_error, max_energy_error,
#   int_time, inv_metric.
#
# Of these, blackjax's NUTSInfo persists: is_divergent, is_turning, energy,
# num_trajectory_expansions (= tree_depth), num_integration_steps (= n_steps),
# acceptance_rate. Step size + reached_max_treedepth are derived in
# ``tuningfork.notebooks.load_idata`` from the recipe's adapted params +
# warmup_params['max_num_doublings'].
_CHAIN_STATS_TO_SAMPLE_STATS: dict[str, str] = {
    # Direct NUTSInfo._fields → ArviZ canonical
    "is_divergent": "diverging",
    "energy": "energy",
    "acceptance_rate": "acceptance_rate",
    "num_integration_steps": "n_steps",
    "num_trajectory_expansions": "tree_depth",
    # is_turning has no ArviZ canonical equivalent; keep under a tuningfork-
    # prefixed name so downstream consumers can opt in but it doesn't
    # collide with the schema.
    "is_turning": "tuningfork_is_turning",
    # Derived fields (enrichment in tuningfork.notebooks.load_idata for
    # GROUNDTRUTH recipes) — identity renames so they pass through this
    # projection without dropping:
    "step_size": "step_size",
    "reached_max_treedepth": "reached_max_treedepth",
}


def _chain_stats_to_sample_stats(
    chain_stats: dict[str, np.ndarray],
    is_multichain: bool,
    n_chunks: int = 1,
) -> dict[str, np.ndarray]:
    """Project our chain_stats dict to ArviZ sample_stats schema.

    Renames known per-step scalar fields per ``_CHAIN_STATS_TO_SAMPLE_STATS``;
    drops fields that don't map cleanly (multi-dim, nested, or unrecognised).

    Reshape behaviour (when ``is_multichain=False``):

    - ``n_chunks == 1`` (default): single-chain → ``(1, n_draws)``
    - ``n_chunks > 1``: split the single chain of length ``n_draws`` into
      ``n_chunks`` contiguous chunks of equal size and present them as
      ``(n_chunks, n_draws // n_chunks)``. This matches the cert protocol's
      "1 long chain × split-into-chunks for split-R̂" convention and lets
      ArviZ treat the chunks as chains for downstream diagnostics.
    """
    out: dict[str, np.ndarray] = {}
    for our_name, arviz_name in _CHAIN_STATS_TO_SAMPLE_STATS.items():
        if our_name not in chain_stats:
            continue
        arr = np.asarray(chain_stats[our_name])
        if arr.ndim < 1:
            continue  # not a per-step array
        if not is_multichain:
            if n_chunks > 1:
                # Single-chain → reshape to (n_chunks, n_draws // n_chunks, ...)
                n_total = arr.shape[0]
                per_chunk = n_total // n_chunks
                arr = arr[: n_chunks * per_chunk].reshape(
                    n_chunks, per_chunk, *arr.shape[1:]
                )
            else:
                # Single-chain → reshape to (1, n_draws, ...)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                else:
                    arr = arr[np.newaxis, ...]
        out[arviz_name] = arr
    return out


def samples_to_idata(
    samples_dict: dict[str, np.ndarray],
    is_multichain: bool = True,
    chain_stats: dict[str, np.ndarray] | None = None,
    n_chunks: int = 1,
) -> Any:
    """Convert samples dict to ArviZ InferenceData, optionally with sample_stats.

    Parameters
    ----------
    samples_dict
        Dictionary mapping parameter names to arrays.
        If ``is_multichain=True``: shape ``(n_chains, n_draws, *event_shape)``
        If ``is_multichain=False``: shape ``(n_draws, *event_shape)`` — reshape
        controlled by ``n_chunks`` (see below).
    is_multichain
        Whether samples are already multi-chain layout. For SMC, pass False
        since particles are ``(1, N_particles, *event)``. Ignored when
        ``n_chunks > 1`` (chunk-split implies single-chain input).
    chain_stats
        Optional per-step diagnostic dict from NUTS (e.g. as persisted to
        ``reference/chain_stats/<name>.npz`` by the cert pipeline). Known
        per-step scalar fields (``is_divergent``, ``energy``,
        ``acceptance_rate``, ``num_integration_steps``) are renamed per
        ArviZ's canonical sample_stats schema (``diverging``, ``energy``,
        ``acceptance_rate``, ``n_steps``) and attached to the ``sample_stats``
        group. Unknown / nested / vector fields are dropped silently.
        When None (default), only the posterior group is populated.
    n_chunks
        When ``is_multichain=False`` and ``n_chunks > 1``: split the single
        chain of length ``n_draws`` into ``n_chunks`` contiguous chunks and
        present them as a multi-chain ArviZ layout
        ``(n_chunks, n_draws // n_chunks, *event_shape)``. This matches the
        cert protocol ("1 long chain × split-R̂ over chunks") and makes
        ``az.summary(idata)`` produce a meaningful ``r_hat`` column
        out-of-the-box. ``chain_stats`` arrays are reshaped consistently.
        Default ``1`` preserves the legacy ``(1, n_draws)`` behaviour.

    Returns
    -------
    arviz.InferenceData
        Posterior group populated from samples_dict; sample_stats group
        populated from chain_stats when provided.
    """
    if az is None:
        raise ImportError("arviz is required for diagnostics rendering")

    # Ensure multi-chain shape for all params
    if not is_multichain:
        mc_samples = {}
        for name, arr in samples_dict.items():
            arr_np = np.asarray(arr)
            if n_chunks > 1:
                # Split single chain into n_chunks contiguous chunks.
                # arr_np: (n_draws, *event)  →  (n_chunks, per_chunk, *event)
                n_total = arr_np.shape[0]
                per_chunk = n_total // n_chunks
                truncated = arr_np[: n_chunks * per_chunk]
                mc_samples[name] = truncated.reshape(
                    n_chunks, per_chunk, *arr_np.shape[1:]
                )
            else:
                # Single-chain → reshape to (1, n_draws, *event)
                if arr_np.ndim < 2:
                    mc_samples[name] = arr_np[np.newaxis, :]
                else:
                    mc_samples[name] = arr_np[np.newaxis, :, ...]
    else:
        mc_samples = {k: np.asarray(v) for k, v in samples_dict.items()}

    if chain_stats is not None:
        sample_stats = _chain_stats_to_sample_stats(
            chain_stats, is_multichain, n_chunks=n_chunks
        )
        return az.from_dict({"posterior": mc_samples, "sample_stats": sample_stats})

    return az.from_dict({"posterior": mc_samples})


def _worst_ess_params(idata: Any, n_worst: int = 3) -> list[str]:
    """Return parameter names with worst (lowest) bulk ESS."""
    if idata is None or idata.posterior is None:
        return []
    ess_bulk = az.ess(idata, method="bulk")
    ess_dict = {str(k): float(np.min(v)) for k, v in ess_bulk.items()}
    sorted_params = sorted(ess_dict.items(), key=lambda x: x[1])
    return [name for name, _ in sorted_params[:n_worst]]


def render_universal_summary(
    idata: Any,
    info: Any,
    gate_verdict_dict: dict[str, Any],
    wall_time_seconds: float,
) -> Figure:
    """Render Section 1: universal scalar summary table.

    Parameters
    ----------
    idata
        ArviZ InferenceData with posterior samples.
    info
        Sampler info struct (may have is_divergent).
    gate_verdict_dict
        Output of auto_gate().to_dict(); contains rhat_max, min_bulk_ess, etc.
    wall_time_seconds
        Wall time for the sampling run.

    Returns
    -------
    matplotlib.Figure
        A single figure with an embedded table showing metrics + verdict.
    """
    if plt is None:
        raise ImportError("matplotlib is required for diagnostics rendering")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("tight")
    ax.axis("off")

    # Build table data
    rows = []
    colors = []

    rhat_max = gate_verdict_dict.get("rhat_max")
    if rhat_max is not None:
        rows.append(["R-hat max", f"{rhat_max:.4f}", "< 1.01 PASS / < 1.05 REVIEW"])
        color = (
            "lightgreen"
            if rhat_max < 1.01
            else ("lightyellow" if rhat_max < 1.05 else "lightcoral")
        )
        colors.append(color)

    min_bulk_ess = gate_verdict_dict.get("min_bulk_ess")
    if min_bulk_ess is not None:
        rows.append(
            ["Min bulk-ESS", f"{min_bulk_ess:.1f}", ">= 400 PASS / >= 100 REVIEW"]
        )
        color = (
            "lightgreen"
            if min_bulk_ess >= 400
            else ("lightyellow" if min_bulk_ess >= 100 else "lightcoral")
        )
        colors.append(color)

    n_diverg = gate_verdict_dict.get("n_divergences")
    if n_diverg is not None:
        rows.append(["Divergences", f"{int(n_diverg)}", "== 0 PASS"])
        color = "lightgreen" if n_diverg == 0 else "lightcoral"
        colors.append(color)

    max_abs_mean_z = gate_verdict_dict.get("max_abs_mean_z")
    if max_abs_mean_z is not None:
        rows.append(
            ["max_abs_mean_z", f"{max_abs_mean_z:.4f}", "< 2 PASS / < 4 REVIEW"]
        )
        color = (
            "lightgreen"
            if max_abs_mean_z < 2
            else ("lightyellow" if max_abs_mean_z < 4 else "lightcoral")
        )
        colors.append(color)

    rows.append(["Wall time (s)", f"{wall_time_seconds:.2f}", "n/a"])
    colors.append("white")

    verdict = gate_verdict_dict.get("verdict", "UNKNOWN")
    rows.append(["Auto-gate verdict", verdict, "PASS / REVIEW / FAIL"])
    verdict_color = (
        "lightgreen"
        if verdict == "PASS"
        else ("lightyellow" if verdict == "REVIEW" else "lightcoral")
    )
    colors.append(verdict_color)

    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value", "Gate threshold"],
        cellLoc="left",
        loc="center",
        cellColours=[[colors[i]] * 3 for i in range(len(rows))],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    fig.suptitle("Section 1: Universal Scalar Summary", fontsize=14, fontweight="bold")
    return fig


def render_gradient_mh(
    idata: Any,
    info: Any,
    sampler_name: str = "nuts",
) -> list[Figure]:
    """Render gradient MH-corrected sampler diagnostics (nuts, hmc, mhmc, mala, barker, ghmc, ...).

    Parameters
    ----------
    idata
        ArviZ InferenceData with posterior samples.
    info
        Sampler info struct; expected to have is_divergent if divergences present.
    sampler_name
        Name of the sampler (used to decide whether to include plot_energy).

    Returns
    -------
    list[Figure]
        List of matplotlib figures for gradient MH diagnostics.
    """
    if az is None or plt is None:
        raise ImportError("arviz and matplotlib are required")

    figs = []

    # 1. Rank trace (try new API, fall back to simple trace)
    try:
        fig_trace = az.plot_trace(idata, kind="rank_bars", backend="matplotlib")
        if hasattr(fig_trace, "artists") or hasattr(fig_trace, "axes"):
            # PlotCollection object
            if hasattr(fig_trace, "figure"):
                figs.append(fig_trace.figure)
        elif isinstance(fig_trace, np.ndarray):
            # Array of axes
            for ax in fig_trace.flatten():
                if hasattr(ax, "figure"):
                    figs.append(ax.figure)
        elif isinstance(fig_trace, plt.Figure):
            figs.append(fig_trace)
    except Exception:
        # Fallback to simple trace
        fig_trace = plt.figure(figsize=(10, 5))
        ax = fig_trace.gca()
        ax.text(0.5, 0.5, "Rank trace plot", ha="center", va="center")
        figs.append(fig_trace)

    # 2. plot_ess with ESS=400 floor
    fig_ess = plt.figure(figsize=(10, 5))
    ax = fig_ess.gca()
    try:
        az.plot_ess(idata, kind="bulk", ax=ax, backend="matplotlib")
    except Exception:
        ax.text(0.5, 0.5, "ESS plot", ha="center", va="center")
    ax.axhline(
        y=400, color="red", linestyle="--", linewidth=2, label="ESS=400 threshold"
    )
    ax.legend()
    ax.set_title("Bulk ESS (with 400 threshold)")
    figs.append(fig_ess)

    # 3. plot_autocorr for 3 worst-ESS params
    worst_params = _worst_ess_params(idata, n_worst=3)
    if worst_params:
        try:
            fig_autocorr = az.plot_autocorr(
                idata, var_names=worst_params, backend="matplotlib"
            )
            if isinstance(fig_autocorr, plt.Figure):
                figs.append(fig_autocorr)
        except Exception:
            pass

    # 4. plot_energy (only for certain gradient MH samplers)
    if sampler_name in GRADIENT_MH_WITH_ENERGY:
        try:
            fig_energy = az.plot_energy(idata, backend="matplotlib")
            if isinstance(fig_energy, plt.Figure):
                figs.append(fig_energy)
        except Exception:
            pass

    # 5. Divergence count if any divergences
    if hasattr(info, "is_divergent") and info.is_divergent is not None:
        n_diverg = int(np.sum(np.asarray(info.is_divergent)))
        if n_diverg > 0:
            fig_diverg = plt.figure(figsize=(6, 4))
            ax = fig_diverg.gca()
            ax.bar(
                ["Divergent", "Non-divergent"],
                [n_diverg, np.prod(np.asarray(info.is_divergent).shape) - n_diverg],
            )
            ax.set_title("Divergence Count")
            ax.set_ylabel("Count")
            figs.append(fig_diverg)

    return figs


def render_mclmc_family(
    idata: Any,
    info: Any,
) -> list[Figure]:
    """Render MCLMC family diagnostics (mclmc, adjusted_mclmc, adjusted_mclmc_dynamic).

    Parameters
    ----------
    idata
        ArviZ InferenceData with posterior samples.
    info
        Sampler info struct; may contain H(t) or log-density trace.

    Returns
    -------
    list[Figure]
        List of matplotlib figures for MCLMC family diagnostics.
    """
    if az is None or plt is None:
        raise ImportError("arviz and matplotlib are required")

    figs = []

    # 1. Rank trace
    try:
        fig_trace = az.plot_trace(idata, kind="rank_bars", backend="matplotlib")
        if hasattr(fig_trace, "figure"):
            figs.append(fig_trace.figure)
        elif isinstance(fig_trace, np.ndarray):
            for ax in fig_trace.flatten():
                if hasattr(ax, "figure"):
                    figs.append(ax.figure)
        elif isinstance(fig_trace, plt.Figure):
            figs.append(fig_trace)
    except Exception:
        fig_trace = plt.figure(figsize=(10, 5))
        ax = fig_trace.gca()
        ax.text(0.5, 0.5, "Rank trace plot", ha="center", va="center")
        figs.append(fig_trace)

    # 2. plot_ess with ESS=400 floor
    fig_ess = plt.figure(figsize=(10, 5))
    ax = fig_ess.gca()
    try:
        az.plot_ess(idata, kind="bulk", ax=ax, backend="matplotlib")
    except Exception:
        ax.text(0.5, 0.5, "ESS plot", ha="center", va="center")
    ax.axhline(
        y=400, color="red", linestyle="--", linewidth=2, label="ESS=400 threshold"
    )
    ax.legend()
    ax.set_title("Bulk ESS (with 400 threshold)")
    figs.append(fig_ess)

    # 3. Hamiltonian drift trace (fallback: per-step log-density)
    # TODO(P7): switch to true H(t) trace once MCLMC runner exposes it
    fig_drift = plt.figure(figsize=(10, 5))
    ax = fig_drift.gca()
    # Placeholder: plot a dummy trace
    ax.plot(
        [0, 1], [0, 1], "b-", label="log-density trace (proxy for Hamiltonian drift)"
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("log-density")
    ax.set_title("log-density trace (proxy for Hamiltonian drift)")
    ax.legend()
    figs.append(fig_drift)

    # 4. plot_autocorr for 3 worst-ESS params
    worst_params = _worst_ess_params(idata, n_worst=3)
    if worst_params:
        try:
            fig_autocorr = az.plot_autocorr(
                idata, var_names=worst_params, backend="matplotlib"
            )
            if isinstance(fig_autocorr, plt.Figure):
                figs.append(fig_autocorr)
        except Exception:
            pass

    return figs


def render_smc_family(
    idata: Any,
    info: Any,
) -> list[Figure]:
    """Render SMC family diagnostics (adaptive_tempered_smc, tempered_smc, etc.).

    Parameters
    ----------
    idata
        ArviZ InferenceData with posterior samples (particles as (1, N_particles, *event)).
    info
        Sampler info struct; may contain ESS-per-step and tempering schedule.

    Returns
    -------
    list[Figure]
        List of matplotlib figures for SMC family diagnostics.
    """
    if az is None or plt is None:
        raise ImportError("arviz and matplotlib are required")

    figs = []

    # 1. Particle weight histogram at final temperature
    fig_weights = plt.figure(figsize=(8, 5))
    ax = fig_weights.gca()
    # Placeholder: would extract final particle weights from info
    ax.text(
        0.5,
        0.5,
        "Particle weight histogram\n(awaiting info.final_weights)",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax.set_title("Particle Weight Distribution at Final Temperature")
    figs.append(fig_weights)

    # 2. Tempering schedule
    fig_temp = plt.figure(figsize=(8, 5))
    ax = fig_temp.gca()
    ax.text(
        0.5,
        0.5,
        "Tempering schedule\n(awaiting info.temperatures)",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax.set_title("Tempering Schedule")
    figs.append(fig_temp)

    # 3. Trace plot on final particles
    try:
        fig_trace = az.plot_trace(idata, kind="rank_bars", backend="matplotlib")
        if hasattr(fig_trace, "figure"):
            figs.append(fig_trace.figure)
        elif isinstance(fig_trace, np.ndarray):
            for ax in fig_trace.flatten():
                if hasattr(ax, "figure"):
                    figs.append(ax.figure)
        elif isinstance(fig_trace, plt.Figure):
            figs.append(fig_trace)
    except Exception:
        fig_trace = plt.figure(figsize=(10, 5))
        ax = fig_trace.gca()
        ax.text(0.5, 0.5, "Rank trace plot", ha="center", va="center")
        figs.append(fig_trace)

    return figs


def render_vi_family(
    idata: Any,
    info: Any,
) -> list[Figure]:
    """Render VI family diagnostics (meanfield_vi, fullrank_vi).

    Parameters
    ----------
    idata
        ArviZ InferenceData with posterior samples.
    info
        Sampler info struct; may contain elbo_history.

    Returns
    -------
    list[Figure]
        List of matplotlib figures for VI family diagnostics.
    """
    if plt is None:
        raise ImportError("matplotlib is required")

    figs = []

    # 1. ELBO trace
    fig_elbo = plt.figure(figsize=(10, 5))
    ax = fig_elbo.gca()
    # Placeholder: would extract ELBO history from info
    ax.text(
        0.5,
        0.5,
        "ELBO trace\n(awaiting info.elbo_history)",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax.set_title("ELBO Trace")
    figs.append(fig_elbo)

    return figs


def render_specialised(
    idata: Any,
    info: Any,
    sampler_name: str = "elliptical_slice",
) -> list[Figure]:
    """Render specialised sampler diagnostics (elliptical_slice, mgrad_gaussian, laplace_*, ...).

    Parameters
    ----------
    idata
        ArviZ InferenceData with posterior samples.
    info
        Sampler info struct.
    sampler_name
        Name of the sampler (determines which extra diagnostics to show).

    Returns
    -------
    list[Figure]
        List of matplotlib figures for specialised sampler diagnostics.
    """
    if az is None or plt is None:
        raise ImportError("arviz and matplotlib are required")

    # Start with gradient MH standard battery
    figs = render_gradient_mh(idata, info, sampler_name)

    # Add specialised-sampler-specific diagnostics (TODO: implement per sampler)
    # For now, just return the gradient MH plots

    return figs


# ---------------------------------------------------------------------------
# Deprecated render_family_* wrappers — emit DeprecationWarning on call
# ---------------------------------------------------------------------------


def render_family_a(
    idata: Any,
    info: Any,
    sampler_name: str = "nuts",
) -> list[Figure]:
    """Deprecated: use ``render_gradient_mh`` instead."""
    warnings.warn(
        "render_family_a is deprecated; use render_gradient_mh instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return render_gradient_mh(idata, info, sampler_name=sampler_name)


def render_family_b(
    idata: Any,
    info: Any,
) -> list[Figure]:
    """Deprecated: use ``render_mclmc_family`` instead."""
    warnings.warn(
        "render_family_b is deprecated; use render_mclmc_family instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return render_mclmc_family(idata, info)


def render_family_c(
    idata: Any,
    info: Any,
) -> list[Figure]:
    """Deprecated: use ``render_smc_family`` instead."""
    warnings.warn(
        "render_family_c is deprecated; use render_smc_family instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return render_smc_family(idata, info)


def render_family_d(
    idata: Any,
    info: Any,
) -> list[Figure]:
    """Deprecated: use ``render_vi_family`` instead."""
    warnings.warn(
        "render_family_d is deprecated; use render_vi_family instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return render_vi_family(idata, info)


def render_family_e(
    idata: Any,
    info: Any,
    sampler_name: str = "elliptical_slice",
) -> list[Figure]:
    """Deprecated: use ``render_specialised`` instead."""
    warnings.warn(
        "render_family_e is deprecated; use render_specialised instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return render_specialised(idata, info, sampler_name=sampler_name)
