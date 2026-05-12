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
"""Sample loading and ArviZ conversion helpers for tuningfork.notebooks.

Provides ``load_samples`` and ``samples_to_idata`` — the two functions a
statistician calls to get an ``arviz.InferenceData`` object ready for
``az.plot_trace``, ``az.summary``, ``az.plot_rank``, ``az.plot_energy``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["load_samples", "samples_to_idata"]


def load_samples(
    recipe: Any,
    *,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Load cached samples for the recipe's model.

    Cache-only in v1 — no re-run path. Calls ``recipe.load_cached_samples``
    which delegates to ``tuningfork.reference._io.try_load_cached_draws``.

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


# Re-export from tuningfork.diagnostics for convenience so the notebook
# only needs to import from tuningfork.notebooks.
def samples_to_idata(
    samples_dict: dict[str, Any],
    is_multichain: bool = False,
) -> Any:
    """Convert a samples dict to ``arviz.InferenceData``.

    Convenience re-export of ``tuningfork.diagnostics.samples_to_idata``.
    The default ``is_multichain=False`` matches the shape returned by
    ``load_samples`` (shape ``(n_samples, *event_shape)`` — single-chain
    reference draws that get promoted to ``(1, n_samples, *event_shape)``
    for ArviZ).

    Parameters
    ----------
    samples_dict
        Dictionary mapping parameter names to arrays.
        If ``is_multichain=False`` (default): shape ``(n_draws, *event_shape)``
        — reshaped to ``(1, n_draws, *event_shape)`` for ArviZ.
        If ``is_multichain=True``: shape ``(n_chains, n_draws, *event_shape)``.
    is_multichain
        Whether samples are already in multi-chain layout.

    Returns
    -------
    arviz.InferenceData
        Posterior group populated from ``samples_dict``.
    """
    from tuningfork.diagnostics import samples_to_idata as _samples_to_idata

    return _samples_to_idata(samples_dict, is_multichain=is_multichain)
