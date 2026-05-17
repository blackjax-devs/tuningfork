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
"""Sample loading and ArviZ conversion helpers (load_idata, load_samples, ...).

Statistician-friendly API:

  - ``load_idata(recipe)`` — recommended one-call: returns an ``InferenceData``
    with posterior + sample_stats (divergences, energy, acceptance_rate,
    n_steps) ready for ``az.plot_trace``, ``az.summary``, ``az.plot_rank``,
    ``az.plot_energy``.
  - ``load_samples(recipe)`` — returns the raw dict[str, jax.Array] of draws
    (advanced use).
  - ``load_chain_stats(recipe)`` — returns the raw chain_stats dict
    (advanced use).
  - ``samples_to_idata(samples, chain_stats=None)`` — manual conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tuningfork._cache_io import try_load_cached_chain_stats
from tuningfork.model import MODELS

__all__ = ["load_samples", "load_chain_stats", "load_idata", "samples_to_idata"]


def load_samples(
    recipe: Any,
    *,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Load cached samples for the recipe's model.

    Cache-only in v1 — no re-run path. Calls ``recipe.load_cached_samples``
    which delegates to ``tuningfork._cache_io.try_load_cached_draws``.

    Only GROUNDTRUTH recipes have a populated cache today. For other effort
    tiers, this raises ``FileNotFoundError`` with a pointer to the Phase 0
    sweep documentation.

    Parameters
    ----------
    recipe
        A Recipe loaded via ``load_recipe``.
    cache_dir
        Override the cache directory. Defaults to the standard reference
        cache under ``tuningfork/reference/``.

    Returns
    -------
    dict[str, jax.Array]
        Mapping from parameter name to array of shape ``(n_samples, *event_shape)``.

    Raises
    ------
    FileNotFoundError
        On cache miss, with a message pointing the user at the recipe
        diagnostics notebook and Phase 0 sweep.
    """
    result = recipe.load_cached_samples(cache_dir=cache_dir)
    if result is None:
        raise FileNotFoundError(
            f"No cached samples found for recipe: model={recipe.model_name!r}, "
            f"effort={recipe.effort.value!r}, sampler={recipe.base_method_name!r}. "
            "Possible causes:\n"
            "  1. The reference cache for this model has not been generated yet.\n"
            "     Run the Phase 0 ground-truth sweep or see recipe_diagnostics.md "
            "for the full warmup+sampler path.\n"
            "  2. This recipe's effort tier is not GROUNDTRUTH. Only GROUNDTRUTH "
            "recipes have a pre-generated reference cache in v1.\n"
            "  3. The cache may be stale (version mismatch). Re-run the sweep with "
            "force_regenerate=True."
        )
    return result


# Re-export from tuningfork.catalog.diagnostics for convenience so the notebook
# only needs to import from tuningfork.catalog.render.
def load_chain_stats(
    recipe: Any,
    *,
    cache_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Load per-step chain_stats from cache, or None on miss.

    Per-step diagnostics persisted by the cert pipeline:
    ``num_integration_steps``, ``energy``, ``is_divergent``, ``acceptance_rate``,
    plus other ``NUTSInfo._fields``. Used to populate the ``sample_stats``
    group in an ArviZ ``InferenceData`` (see ``samples_to_idata``).

    Parameters
    ----------
    recipe
        A Recipe loaded via ``load_recipe``.
    cache_dir
        Override the cache directory.

    Returns
    -------
    dict[str, np.ndarray] or None
        Chain_stats dict on cache hit; None on miss. Cache misses are
        non-fatal — sample_stats are diagnostic-only, the InferenceData is
        still useful with just the posterior group.
    """
    if recipe.model_name not in MODELS:
        return None
    return try_load_cached_chain_stats(MODELS[recipe.model_name], cache_dir=cache_dir)


def _enrich_chain_stats_for_groundtruth(
    recipe: Any, chain_stats: dict[str, Any]
) -> dict[str, Any]:
    """For GROUNDTRUTH recipes, derive additional ArviZ sample_stats fields.

    Per ArviZ schema (https://python.arviz.org/en/stable/schema/schema.html):

    - ``step_size``: broadcast the recipe's adapted step_size to a per-sample
      array (it's a scalar stamped once during warmup; ArviZ expects a
      per-iteration field).
    - ``reached_max_treedepth``: derive from ``num_trajectory_expansions ==
      max_num_doublings`` — boolean per-sample indicator.

    These are GROUNDTRUTH-specific because:
    1. The adapted step_size is only stamped on GROUNDTRUTH recipes (LOW
       recipes may not have a populated cache stamp).
    2. ``max_num_doublings`` is read from ``recipe.warmup_params``, which
       GROUNDTRUTH always populates.
    """
    import numpy as _np

    enriched = dict(chain_stats)

    # step_size: broadcast adapted scalar to per-sample.
    step_size = recipe.base_method_params.get("step_size")
    if step_size is not None and "num_integration_steps" in chain_stats:
        n_samples = chain_stats["num_integration_steps"].shape[0]
        enriched["step_size"] = _np.full(n_samples, float(step_size), dtype=_np.float32)

    # reached_max_treedepth: derived from num_trajectory_expansions hitting cap.
    max_doublings = recipe.warmup_params.get("max_num_doublings")
    nte = chain_stats.get("num_trajectory_expansions")
    if max_doublings is not None and nte is not None:
        enriched["reached_max_treedepth"] = (
            _np.asarray(nte) >= int(max_doublings)
        ).astype(bool)

    return enriched


def load_idata(
    recipe: Any,
    *,
    cache_dir: Path | None = None,
) -> Any:
    """Load samples + chain_stats and build an ArviZ ``InferenceData``.

    Recommended one-call helper for the statistician check workflow:

        from tuningfork.catalog.inspect import load_recipe
        from tuningfork.catalog.render import load_idata
        import arviz as az

        recipe = load_recipe("inference/recipes/starter/.../groundtruth__nuts__stan_window.json")
        idata = load_idata(recipe)
        az.plot_trace(idata)
        az.plot_energy(idata)   # uses sample_stats.energy
        az.plot_pair(idata, divergences=True)  # uses sample_stats.diverging
        az.summary(idata)

    For **GROUNDTRUTH** recipes, the returned InferenceData's ``sample_stats``
    group is **enriched** beyond what's in chain_stats: ``step_size`` is
    broadcast from the recipe's adapted scalar to a per-sample array, and
    ``reached_max_treedepth`` is derived from ``num_trajectory_expansions``
    vs ``max_num_doublings``. For non-GROUNDTRUTH recipes, only the raw
    chain_stats fields are projected.

    Combines ``load_samples`` + ``load_chain_stats`` + ``samples_to_idata``.
    chain_stats cache miss is non-fatal: when chain_stats are absent (e.g.
    recipe pre-dates the chain_stats persistence layer), ``InferenceData``
    carries only the posterior group.

    Parameters
    ----------
    recipe
        A Recipe loaded via ``load_recipe``.
    cache_dir
        Override the cache directory.

    Returns
    -------
    arviz.InferenceData
        Posterior group + (when chain_stats available) sample_stats group.

    Raises
    ------
    FileNotFoundError
        On samples cache miss (chain_stats miss is silently ignored —
        sample_stats are optional).
    """
    samples = load_samples(recipe, cache_dir=cache_dir)
    chain_stats = load_chain_stats(recipe, cache_dir=cache_dir)

    # GROUNDTRUTH enrichment: derive ArviZ-canonical fields not in raw chain_stats
    if chain_stats is not None:
        try:
            from tuningfork.recipes._base import Effort

            if recipe.effort == Effort.GROUNDTRUTH:
                chain_stats = _enrich_chain_stats_for_groundtruth(recipe, chain_stats)
        except (ImportError, AttributeError):
            # Recipe schema mismatch or missing Effort enum — skip enrichment
            pass

    # Resolve n_chunks from the recipe's warmup params (the cert protocol's
    # split-R̂ chunking — typically 4 — recorded under warmup_params at
    # cert time). Default to 1 (no chunk split) for safety when absent.
    n_chunks = int(recipe.warmup_params.get("n_chunks", 1) or 1)

    if n_chunks > 1:
        # Transparency: print the reshape ONCE per call so users know what
        # shape they're inspecting in az.summary / az.plot_trace.
        any_site = next(iter(samples))
        n_total = int(samples[any_site].shape[0])
        per_chunk = n_total // n_chunks
        import warnings

        warnings.warn(
            f"tuningfork.catalog.render.load_idata: applied cert-protocol reshape "
            f"({n_total} draws from 1 chain) → ({n_chunks} chains × {per_chunk} draws). "
            f"This matches the certification protocol's split-R̂ chunking and lets "
            f"az.summary(idata) compute r_hat directly. Pass n_chunks=1 to "
            f"samples_to_idata if you want the raw single-chain layout.",
            stacklevel=2,
        )

    return samples_to_idata(samples, chain_stats=chain_stats, n_chunks=n_chunks)


def samples_to_idata(
    samples_dict: dict[str, Any],
    is_multichain: bool = False,
    chain_stats: dict[str, Any] | None = None,
    n_chunks: int = 1,
) -> Any:
    """Convert a samples dict to ``arviz.InferenceData``.

    Convenience re-export of ``tuningfork.catalog.diagnostics.samples_to_idata``.
    The default ``is_multichain=False`` matches the shape returned by
    ``load_samples`` (shape ``(n_samples, *event_shape)`` — single-chain
    reference draws that get promoted to ``(1, n_samples, *event_shape)``
    for ArviZ).

    Parameters
    ----------
    samples_dict
        Dictionary mapping parameter names to arrays.
        If ``is_multichain=False`` (default): shape ``(n_draws, *event_shape)``
        — reshaped to ``(1, n_draws, *event_shape)`` for ArviZ (when
        ``n_chunks == 1``), or to ``(n_chunks, n_draws // n_chunks, ...)``
        when ``n_chunks > 1``.
        If ``is_multichain=True``: shape ``(n_chains, n_draws, *event_shape)``.
    is_multichain
        Whether samples are already in multi-chain layout.
    chain_stats
        Optional per-step diagnostic dict (as returned by
        ``load_chain_stats``). When provided, known per-step scalar fields
        (``is_divergent``, ``energy``, ``acceptance_rate``,
        ``num_integration_steps``) are renamed to the ArviZ canonical
        sample_stats schema (``diverging``, ``energy``, ``acceptance_rate``,
        ``n_steps``) and attached to the ``sample_stats`` group of the
        returned InferenceData.
    n_chunks
        Cert-protocol chunk count. When ``> 1`` and ``is_multichain=False``,
        the single-chain samples are reshaped to a multi-chain ArviZ layout
        ``(n_chunks, n_draws // n_chunks, ...)``. This makes
        ``az.summary(idata)`` produce ``r_hat`` directly.
        ``load_idata`` reads this from ``recipe.warmup_params["n_chunks"]``;
        callers using this function directly should pass it explicitly when
        the recipe's cert protocol used chunked split-R̂.

    Returns
    -------
    arviz.InferenceData
        Posterior group populated from ``samples_dict``; sample_stats group
        populated from ``chain_stats`` when provided.
    """
    from tuningfork.catalog.diagnostics import samples_to_idata as _samples_to_idata

    return _samples_to_idata(
        samples_dict,
        is_multichain=is_multichain,
        chain_stats=chain_stats,
        n_chunks=n_chunks,
    )
