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

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import arviz

__all__ = ["cached_idata_for_recipe"]


def cached_idata_for_recipe(
    recipe: Any,
    *,
    catalog_root: Path | str | None = None,
    force_regenerate: bool = False,
) -> arviz.InferenceData:
    """Load or re-run a LOW/MEDIUM recipe with automatic caching.

    On first call (cache miss), re-runs the recipe's warmup + sampling pipeline
    via ``run_recipe_to_idata`` and saves the positions and chain_stats to:

        <catalog_root>/<model>/_cache/<recipe_stem>.draws.npz
        <catalog_root>/<model>/_cache/<recipe_stem>.chain_stats.npz

    On subsequent calls (cache hit), loads directly from the cache.

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
        If True, skip cache check and re-run unconditionally.
        Useful for when the recipe or JAX code has changed.

    Returns
    -------
    arviz.InferenceData
        Posterior group + sample_stats group (if available).

    Raises
    ------
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

    # Try cache hit (unless force_regenerate=True)
    if not force_regenerate and draws_cache.exists():
        return _load_from_cache(draws_cache, stats_cache)

    # Cache miss: re-run the recipe
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    idata = run_recipe_to_idata(recipe, catalog_root=catalog_root)

    # Save to cache
    _save_to_cache(idata, draws_cache, stats_cache)

    return idata


def _load_from_cache(
    draws_cache: Path,
    stats_cache: Path,
) -> arviz.InferenceData:
    """Load cached draws and chain_stats; reconstruct as InferenceData."""
    from tuningfork.catalog.diagnostics import samples_to_idata

    # Load draws
    draws_data = np.load(str(draws_cache))
    samples_dict = {k: np.asarray(draws_data[k]) for k in draws_data.files}

    # Load chain_stats (optional; None if file missing or invalid)
    chain_stats = None
    if stats_cache.exists():
        try:
            stats_data = np.load(str(stats_cache))
            chain_stats = {k: np.asarray(stats_data[k]) for k in stats_data.files}
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
