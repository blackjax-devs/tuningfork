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
"""On-demand resampling with caching for LOW/MEDIUM recipes.

Wrapper around ``tuningfork.recipes._recipe_runner.run_recipe_to_idata`` that
caches draws to avoid redundant re-runs in the catalog notebook.

Cache layout (per recipe):

    <catalog_root>/<model>/_cache/<recipe_stem>.draws.npz
    <catalog_root>/<model>/_cache/<recipe_stem>.chain_stats.npz

where ``<recipe_stem>`` is the recipe JSON filename without extension,
e.g. ``low__nuts__window_adaptation_diag_imm`` for a recipe at
``<catalog_root>/<model>/recipes/low__nuts__window_adaptation_diag_imm.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import arviz

__all__ = ["cached_idata_for_recipe", "regenerate_idata"]


def cached_idata_for_recipe(
    recipe: Any,
    *,
    catalog_root: Path | str | None = None,
    force_regenerate: bool = False,
) -> arviz.InferenceData:
    """Load a cached LOW/MEDIUM recipe's draws; re-sample only on explicit opt-in.

    Default behavior (``force_regenerate=False``) — **cache-only load**:
    - Cache hit  → load from ``<catalog_root>/<model>/_cache/<recipe_stem>.draws.npz``.
    - Cache miss → raises ``FileNotFoundError`` with an actionable hint.

    Explicit re-sample (``force_regenerate=True``):
    - Skips cache check; re-runs the recipe's warmup + sampling pipeline via
      ``run_recipe_to_idata``; saves to the cache; returns the InferenceData.

    Rationale: the prior default ("silent re-sample on cache miss") was a
    UX footgun — a 30+ min sampling run could be triggered by what looked
    like a cheap cache read. Re-sampling now requires explicit consent.

    For GROUNDTRUTH recipes, delegates to the standard ``load_idata`` path
    (no caching logic needed; those draws are already persisted).

    For FAILED recipes, raises RecipeFailedError (no gate-passing config to run).

    Parameters
    ----------
    recipe
        A Recipe object loaded via ``load_recipe``.
    catalog_root
        Root of the catalog directory (default: ``tuningfork/catalog/``).
    force_regenerate
        If True, ignore any cache and re-run sampling unconditionally
        (saving the result for future cache hits). Use this when the recipe
        or JAX code has changed, or to populate the cache for the first time.

    Returns
    -------
    arviz.InferenceData
        Posterior group + sample_stats group (if available).

    Raises
    ------
    FileNotFoundError
        Cache miss with ``force_regenerate=False``. Call again with
        ``force_regenerate=True`` to (re-)sample, or call
        ``run_recipe_to_idata(recipe)`` directly.
    RecipeFailedError
        If the recipe is FAILED (no gate-passing config).
    ValueError
        If the recipe model/warmup/sampler are not in the registries.
    """
    from pathlib import Path

    if catalog_root is None:
        from tuningfork.recipes._recipe_runner import _CATALOG_ROOT

        catalog_root = _CATALOG_ROOT
    else:
        catalog_root = Path(catalog_root)

    # Derive cache stem from recipe identity
    # For GROUNDTRUTH: use "groundtruth"
    # For others: use effort__sampler__warmup
    if recipe.effort.value == "groundtruth":
        recipe_stem = "groundtruth"
    else:
        recipe_stem = (
            f"{recipe.effort.value}__{recipe.base_method_name}__{recipe.warmup_name}"
        )

    # Construct cache paths
    cache_dir = catalog_root / recipe.model_name / "_cache"
    draws_cache = cache_dir / f"{recipe_stem}.draws.npz"
    stats_cache = cache_dir / f"{recipe_stem}.chain_stats.npz"

    # Cache-only mode (default): hit or raise.
    if not force_regenerate:
        if draws_cache.exists():
            return _load_from_cache(draws_cache, stats_cache)
        raise FileNotFoundError(
            f"No cache for {recipe.model_name}/"
            f"{recipe.effort.value}__{recipe.base_method_name}__"
            f"{recipe.warmup_name} at {draws_cache}.\n"
            f"Call cached_idata_for_recipe(recipe, force_regenerate=True) to "
            f"sample + populate the cache, or call run_recipe_to_idata(recipe) "
            f"directly if you don't want to persist."
        )

    # Explicit re-sample: run the pipeline and persist to cache.
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    idata = run_recipe_to_idata(recipe, catalog_root=catalog_root)
    _save_to_cache(idata, draws_cache, stats_cache)
    # Stale-cache guard (issue #244): co-write a params sidecar beside the draws
    # cache so a later `make revalidate-w1` can tell whether these draws still
    # match the recipe (path-A) or were superseded by a re-emit (degrade to a
    # fresh-draws path).  Derived from the same on-disk recipe JSON the reader
    # (classify_recipe_path) inspects, so a valid cache compares equal by
    # construction.  Groundtruth caches are not W1-eligible, so skip them.
    if recipe_stem != "groundtruth":
        _write_cache_params_sidecar(
            cache_dir, catalog_root, recipe.model_name, recipe_stem
        )
    return idata


def regenerate_idata(
    recipe: Any,
    *,
    n_samples: int = 1000,
    seed: int = 20260517,
    catalog_root: Path | str | None = None,
) -> arviz.InferenceData:
    """Re-run a recipe's warmup + sampling pipeline and return InferenceData.

    **User-triggered only** — this function runs the full warmup + sampler and
    may take several minutes.  It is intended to be called from an explicit
    user action (e.g. a ``Run`` button click in catalog_explorer) rather than
    automatically on page load.

    Primary use-case: **FAIL recipes**.  A FAIL recipe's failure mode is
    invisible without running the config — divergent transitions, stuck chains,
    and non-mixing patterns only appear in diagnostic plots.  By re-running the
    pinned config you can visually inspect *why* it failed (trace plots,
    divergence markers, rank plots) before investigating a fix.

    Also works for PASS, REVIEW, and GROUNDTRUTH recipes (for verification or
    different-seed experiments), but ``cached_idata_for_recipe`` is the
    preferred path for those since it avoids redundant compute.

    .. note::
        Does **not** use ``skip_warmup=True`` — FAIL recipes may lack valid
        skip-warmup params (e.g. divergent step_size).  Full warmup is run.

    Parameters
    ----------
    recipe
        A Recipe object loaded via ``load_recipe``.  Works for any effort level
        including FAILED.
    n_samples
        Number of post-warmup samples per chain (default 1000).  Reduce to
        ~200–400 for a quick diagnostic preview of failure modes.
    seed
        Random seed for reproducibility.  Defaults to the canonical recipe seed.
    catalog_root
        Root of the catalog directory (default: ``tuningfork/catalog/``).

    Returns
    -------
    arviz.InferenceData
        Posterior group + sample_stats group.  Pass to
        ``plot_recipe_diagnostics(idata, posterior_entry)`` to render trace,
        pair, and forest plots.

    Raises
    ------
    RuntimeError
        If the sampler raises an exception (e.g. non-terminating trajectory).

    Examples
    --------
    Inspect a FAIL recipe in catalog_explorer::

        idata = regenerate_idata(recipe, n_samples=400, seed=42)
        figs = plot_recipe_diagnostics(idata, posterior_entry)
    """
    from pathlib import Path as _Path

    if catalog_root is None:
        from tuningfork.recipes._recipe_runner import _CATALOG_ROOT

        catalog_root = _CATALOG_ROOT
    else:
        catalog_root = _Path(catalog_root)

    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    return run_recipe_to_idata(
        recipe,
        force_resample_config={"seed": seed, "n_samples": n_samples},
        catalog_root=catalog_root,
        _allow_failed_diagnostic=True,
    )


def _load_from_cache(
    draws_cache: Path,
    stats_cache: Path,
) -> arviz.InferenceData:
    """Load cached draws and chain_stats; reconstruct as InferenceData.

    Cache files store ArviZ-canonical sample_stats names (``diverging``,
    ``n_steps``, etc.) because ``_save_to_cache`` reads ``idata.sample_stats``
    after ``samples_to_idata`` has already projected raw blackjax names →
    canonical. Reverse-map them back to raw names here so that
    ``samples_to_idata``'s ``_chain_stats_to_sample_stats`` projection
    finds them and re-emits them as canonical. Without this reverse map,
    every cache hit drops ``diverging`` + ``n_steps`` silently because
    those keys aren't on the LHS of the rename map.
    """
    from tuningfork.catalog.diagnostics import (
        _CHAIN_STATS_TO_SAMPLE_STATS,
        samples_to_idata,
    )

    # Load draws
    draws_data = np.load(str(draws_cache))
    samples_dict = {k: np.asarray(draws_data[k]) for k in draws_data.files}

    # Reverse map: ArviZ canonical → raw blackjax field name.
    _SAMPLE_STATS_TO_CHAIN_STATS = {
        canonical: raw for raw, canonical in _CHAIN_STATS_TO_SAMPLE_STATS.items()
    }

    # Load chain_stats (optional; None if file missing or invalid)
    chain_stats = None
    if stats_cache.exists():
        try:
            stats_data = np.load(str(stats_cache))
            chain_stats = {
                _SAMPLE_STATS_TO_CHAIN_STATS.get(k, k): np.asarray(stats_data[k])
                for k in stats_data.files
            }
        except Exception:
            # Silently ignore corrupt/invalid stats cache
            chain_stats = None

    # Reconstruct InferenceData
    # Samples are already in multi-chain format (num_chains, n_samples, *event_shape)
    return samples_to_idata(
        samples_dict,
        is_multichain=True,
        chain_stats=chain_stats,
        n_chunks=1,
    )


def _write_cache_params_sidecar(
    cache_dir: Path,
    catalog_root: Path,
    model_name: str,
    recipe_stem: str,
) -> None:
    """Write ``_cache/<stem>.params_hash.json`` beside a freshly written draws cache.

    Records the params the cache was generated for (step_size, num_integration_steps,
    target_acceptance, IMM fingerprint) so ``classify_recipe_path`` can detect a
    re-emitted recipe whose old draws are stale (issue #244).  Derived from the same
    on-disk recipe JSON the reader inspects, via the shared ``_recipe_cache_params``
    extractor, so a matching cache compares equal by construction.  A missing or
    unreadable recipe JSON is a silent no-op — the reader then finds no sidecar and
    conservatively degrades out of path-A, which is safe.
    """
    from tuningfork.calibration.revalidation import _recipe_cache_params

    recipe_json = catalog_root / model_name / "recipes" / f"{recipe_stem}.json"
    if not recipe_json.exists():
        return
    try:
        recipe_dict = json.loads(recipe_json.read_text())
    except Exception:  # noqa: BLE001
        return
    params = _recipe_cache_params(recipe_dict)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sidecar = cache_dir / f"{recipe_stem}.params_hash.json"
    tmp = cache_dir / f"{recipe_stem}.params_hash.json.tmp"
    tmp.write_text(json.dumps(params))
    tmp.replace(sidecar)  # atomic within the same directory


def _save_to_cache(
    idata: arviz.InferenceData,
    draws_cache: Path,
    stats_cache: Path,
) -> None:
    """Save InferenceData posterior and sample_stats to cache files."""
    draws_cache.parent.mkdir(parents=True, exist_ok=True)

    # Extract posterior group: {param_name: (chains, n_draws, *event_shape)}
    posterior = idata.posterior
    draws_dict = {
        name: np.asarray(posterior[name].values) for name in posterior.data_vars
    }

    # Save draws
    np.savez_compressed(str(draws_cache), **draws_dict)

    # Save chain_stats if available
    if hasattr(idata, "sample_stats") and idata.sample_stats is not None:
        sample_stats = idata.sample_stats
        stats_dict = {
            name: np.asarray(sample_stats[name].values)
            for name in sample_stats.data_vars
        }
        np.savez_compressed(str(stats_cache), **stats_dict)
