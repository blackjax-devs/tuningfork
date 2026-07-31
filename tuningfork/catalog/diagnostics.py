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
    "plot_recipe_diagnostics",
    # Semantic names (current)
    "GRADIENT_MH_SAMPLERS",
    "MCLMC_FAMILY_SAMPLERS",
    "SMC_FAMILY_SAMPLERS",
    "VI_FAMILY_SAMPLERS",
    "SPECIALISED_SAMPLERS",
    "render_gradient_mh",
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


# Mapping from our chain_stats field names (blackjax NUTSInfo._fields) to
# ArviZ's canonical sample_stats group names.
# Reference: https://python.arviz.org/en/stable/schema/schema.html#sample-stats
# Known BlackJAX fields are mapped to ArviZ's canonical names. Fields not
# listed here are retained under their original names when they contain a
# per-step array.
#
# ArviZ canonical sample_stats keys (per the schema, 2026-05-12):
#   lp, acceptance_rate, step_size, step_size_nom, tree_depth, n_steps,
#   reached_max_treedepth, diverging, energy, energy_error, max_energy_error,
#   int_time, inv_metric.
#
# Of these, blackjax's NUTSInfo persists: is_divergent, is_turning, energy,
# num_trajectory_expansions (= tree_depth), num_integration_steps (= n_steps),
# acceptance_rate. Step size + reached_max_treedepth are derived in
# ``tuningfork.catalog.render.load_idata`` from the recipe's adapted params +
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
    # Laplace-family: post-accept L-BFGS iter count per step (blackjax PR #925).
    # Used to compute measured headline denominators via laplace_lbfgs_grad_evals.
    "lbfgs_iter_num": "lbfgs_iter_num",
    # Laplace-family: True when solver hit maxiter budget (non-convergence alarm).
    # lbfgs_hit_maxiter%=0 confirms maxiter budget is sufficient for the model.
    "lbfgs_hit_maxiter": "lbfgs_hit_maxiter",
    # Derived fields (enrichment in tuningfork.catalog.render.load_idata for
    # GROUNDTRUTH recipes) — identity renames so they pass through this
    # projection without dropping:
    "step_size": "step_size",
    "reached_max_treedepth": "reached_max_treedepth",
}


def _chain_stats_to_sample_stats(
    chain_stats: dict[str, np.ndarray],
    is_multichain: bool,
    n_chunks: int = 1,
    expected_draws: int | None = None,
    expected_topology: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Project our chain_stats dict to ArviZ sample_stats schema.

    Renames known fields per ``_CHAIN_STATS_TO_SAMPLE_STATS`` and retains all
    other per-step arrays under their original names. Values without a draw
    dimension are rejected rather than silently discarded. If two input keys
    resolve to the same output name, a ``ValueError`` is raised rather than
    overwriting one of the values.

    Reshape behaviour (when ``is_multichain=False``):

    - ``n_chunks == 1`` (default): single-chain → ``(1, n_draws)``
    - ``n_chunks > 1``: split the single chain of length ``n_draws`` into
      ``n_chunks`` contiguous chunks of equal size and present them as
      ``(n_chunks, n_draws // n_chunks)``. This matches the cert protocol's
      "1 long chain × split-into-chunks for split-R̂" convention and lets
      ArviZ treat the chunks as chains for downstream diagnostics.
    """
    _validate_n_chunks(n_chunks)
    out: dict[str, np.ndarray] = {}
    source_by_output_name: dict[str, str] = {}
    resolved: dict[str, tuple[str, np.ndarray]] = {}
    for our_name, value in chain_stats.items():
        arviz_name = _CHAIN_STATS_TO_SAMPLE_STATS.get(our_name, our_name)
        arr = np.asarray(value)
        if arviz_name in source_by_output_name:
            previous = source_by_output_name[arviz_name]
            raise ValueError(
                f"chain_stats fields {previous!r} and {our_name!r} both map "
                f"to sample_stats name {arviz_name!r}"
            )
        source_by_output_name[arviz_name] = our_name
        resolved[arviz_name] = (our_name, arr)

    arrays: dict[str, np.ndarray] = {}
    for arviz_name, (our_name, arr) in resolved.items():
        if arr.ndim < 1:
            raise ValueError(
                f"chain_stats field {our_name!r} must have a draw dimension"
            )
        arrays[arviz_name] = arr

    _validate_draw_topology(
        arrays,
        n_chunks,
        is_multichain=is_multichain,
        expected_draws=expected_draws,
        expected_topology=expected_topology,
    )
    for arviz_name, arr in arrays.items():
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


def _validate_n_chunks(n_chunks: int) -> None:
    """Reject invalid chunk controls before any reshape can lose data."""
    if isinstance(n_chunks, bool) or not isinstance(n_chunks, (int, np.integer)):
        raise ValueError("n_chunks must be a positive integer")
    if n_chunks <= 0:
        raise ValueError("n_chunks must be a positive integer")


def _validate_draw_topology(
    arrays: dict[str, np.ndarray],
    n_chunks: int,
    *,
    is_multichain: bool,
    expected_draws: int | None = None,
    expected_topology: tuple[int, int] | None = None,
) -> int | None:
    """Ensure all arrays share one lossless chain/draw topology."""
    if is_multichain:
        topologies = set()
        for arr in arrays.values():
            if arr.ndim < 2:
                raise ValueError(
                    "multichain arrays must have leading chain and draw dimensions"
                )
            topologies.add(tuple(arr.shape[:2]))
        if expected_topology is not None:
            topologies.add(expected_topology)
        if not topologies:
            return expected_topology[1] if expected_topology is not None else None
        if len(topologies) != 1:
            raise ValueError(
                "all arrays must have the same leading (chain, draw) topology"
            )
        topology = next(iter(topologies))
        if min(topology) < 1:
            raise ValueError("chain and draw dimensions must both be non-empty")
        return topology[1]

    lengths = {arr.shape[0] for arr in arrays.values()}
    if expected_draws is not None:
        lengths.add(expected_draws)
    if not lengths:
        return expected_draws
    if len(lengths) != 1:
        raise ValueError("all per-step arrays must have the same number of draws")
    n_draws = lengths.pop()
    if n_chunks > n_draws:
        raise ValueError("n_chunks cannot exceed the number of draws")
    if n_draws % n_chunks:
        raise ValueError(
            f"number of draws ({n_draws}) must be divisible by n_chunks ({n_chunks})"
        )
    return n_draws


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
        ``reference/<name>/chain_stats.npz`` by the cert pipeline). Known
        per-step scalar fields (``is_divergent``, ``energy``,
        ``acceptance_rate``, ``num_integration_steps``) are renamed per
        ArviZ's canonical sample_stats schema (``diverging``, ``energy``,
        ``acceptance_rate``, ``n_steps``) and attached to the ``sample_stats``
        group. Unknown and vector per-step fields are retained under their
        input names; values without a draw dimension are rejected.
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

    _validate_n_chunks(n_chunks)

    sample_arrays = {name: np.asarray(arr) for name, arr in samples_dict.items()}
    if not sample_arrays:
        raise ValueError("samples_dict must contain at least one posterior variable")
    if any(arr.ndim < 1 for arr in sample_arrays.values()):
        raise ValueError("sample arrays must have a draw dimension")
    n_draws = _validate_draw_topology(
        sample_arrays, n_chunks, is_multichain=is_multichain
    )
    posterior_topology = None
    if is_multichain:
        posterior_topology = tuple(next(iter(sample_arrays.values())).shape[:2])

    # Ensure multi-chain shape for all params
    if not is_multichain:
        mc_samples = {}
        for name, arr_np in sample_arrays.items():
            if n_chunks > 1:
                # Split single chain into n_chunks contiguous chunks.
                # arr_np: (n_draws, *event)  →  (n_chunks, per_chunk, *event)
                n_total = arr_np.shape[0]
                per_chunk = n_total // n_chunks
                mc_samples[name] = arr_np.reshape(
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
            chain_stats,
            is_multichain,
            n_chunks=n_chunks,
            expected_draws=None if is_multichain else n_draws,
            expected_topology=posterior_topology,
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


def _count_scalar_coords(posterior: Any, var_names: list[str]) -> int:
    """Total scalar coords across ``var_names`` in ``posterior``.

    For each variable ``v``, the scalar count is the product of all dims of
    ``posterior[v]`` other than ``chain`` and ``draw``.  Used to budget the
    ``arviz_plots.plot_pair`` subplot grid (which is N×N where N is this
    sum).  See ``plot_recipe_diagnostics`` for the use-site.
    """
    total = 0
    for v in var_names:
        if v not in posterior.data_vars:
            continue
        arr = posterior[v]
        event_dims = [d for d in arr.dims if d not in ("chain", "draw")]
        if not event_dims:
            total += 1
        else:
            total += int(np.prod([arr.sizes[d] for d in event_dims]))
    return total


def plot_recipe_diagnostics(
    idata: Any,
    posterior_entry: Any,
    n_forest_top: int | None = None,
) -> dict[str, Any]:
    """Render trace+pair for headline params and forest for the bulk.

    Reads `posterior_entry.headline_params` and `headline_coords` to
    decide what's "headline" vs "bulk" and renders ArviZ plots accordingly.

    Parameters
    ----------
    idata
        InferenceData (typically from `load_idata(recipe)`).
    posterior_entry
        Posterior instance from `MODELS[<model_name>]`. Reads its
        `headline_params` + `headline_coords` fields.
    n_forest_top
        Cap on the number of bulk-param coords rendered in the forest plot
        (useful for high-dim models like stoch_vol's h_raw with 500 coords).
        None = render all bulk coords.

    Returns
    -------
    dict[str, Any]
        Dict with keys 'trace', 'pair', 'forest' mapping to the corresponding
        ``arviz_plots`` PlotCollection objects (renderable inline in marimo /
        Jupyter via the cell's last-expression rule). If headline_params is
        None, 'trace' and 'pair' render all posterior sites; if no bulk params
        remain, 'forest' is None.
    """
    # Use the newer arviz_plots backend (PlotCollection-based) rather than
    # the legacy matplotlib az.plot_* path:
    #   - trace: arviz_plots.plot_trace_dist (trace + KDE side-by-side)
    #   - pair: arviz_plots.plot_pair
    #   - forest: arviz_plots.plot_forest
    # arviz_plots is auto-installed by the bench dep group (transitively via
    # arviz>=0.20). Returns plot objects that marimo renders natively.
    import arviz_plots as azp

    posterior = idata["posterior"]
    all_params = list(posterior.data_vars)
    headline_params = posterior_entry.headline_params
    headline_coords = posterior_entry.headline_coords

    # Determine headline + bulk split
    if headline_params is None:
        headline_var_names = list(all_params)
        bulk_var_names: list[str] = []
    else:
        headline_var_names = list(headline_params)
        bulk_var_names = [p for p in all_params if p not in headline_var_names]

    # Trace + pair on headline (with optional coord slicing).
    # arviz_plots.plot_pair accepts `visuals={"divergence": True}` to
    # overlay divergent transitions on the pair scatter (the new-API
    # equivalent of the legacy `divergences=True` kwarg).
    trace_kwargs: dict[str, Any] = {"var_names": headline_var_names}
    pair_kwargs: dict[str, Any] = {
        "var_names": headline_var_names,
        "visuals": {"divergence": True},
    }
    if headline_coords is not None:
        # Translate {block: [idx, ...]} to ArviZ coords format.
        # The dim name is typically "<block>_dim_0".
        coords: dict[str, Any] = {}
        for block, idx_list in headline_coords.items():
            # If headline_coords includes a block not in headline_params,
            # auto-include it in headline rendering
            if headline_params is None or block not in headline_var_names:
                if block in all_params:
                    headline_var_names.append(block)
                    if block in bulk_var_names:
                        bulk_var_names.remove(block)
            dim_name = f"{block}_dim_0"
            coords[dim_name] = idx_list
        trace_kwargs["coords"] = coords
        pair_kwargs["coords"] = coords

    trace_plot = azp.plot_trace_dist(idata, **trace_kwargs)

    # Pair plot subplot budget.  arviz_plots.plot_pair draws an N×N grid where
    # N = sum of scalar coords across `headline_var_names`.  At N>6 we exceed
    # arviz_plots's default `rcParams["plot.max_subplots"]=40` and the call
    # raises.  For high-dim "no qualitatively distinguished subset" models
    # (`headline_params is None` for ill_cond_50 / mvn_10 / german_credit per
    # the 2026-05-18 headline-params decision), the forest plot is the
    # canonical alternative — skip pair entirely in that case rather than
    # truncating to a misleading 6×6 corner.
    pair_total_coords = _count_scalar_coords(posterior, headline_var_names)
    if pair_total_coords**2 > 36:
        pair_plot = None
    else:
        pair_plot = azp.plot_pair(idata, **pair_kwargs)

    # Forest on bulk
    forest_plot = None
    if bulk_var_names:
        forest_kwargs: dict[str, Any] = {"var_names": bulk_var_names}
        if n_forest_top is not None:
            # Slice via coords on each bulk var's first dim, capping at n_forest_top
            forest_coords: dict[str, Any] = {}
            for block in bulk_var_names:
                arr = posterior[block]
                if arr.ndim > 1:  # has a coord dim beyond chain/draw
                    block_dim_size = arr.sizes.get(f"{block}_dim_0", 0)
                    if block_dim_size > n_forest_top:
                        forest_coords[f"{block}_dim_0"] = list(range(n_forest_top))
            if forest_coords:
                forest_kwargs["coords"] = forest_coords
        forest_plot = azp.plot_forest(idata, **forest_kwargs)

    return {"trace": trace_plot, "pair": pair_plot, "forest": forest_plot}


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
