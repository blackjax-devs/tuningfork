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
"""Compatibility utility for recovering a headline from cached chain stats."""

from __future__ import annotations

import warnings
from collections import namedtuple
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from tuningfork.metrics.grad_counter import total_grad_evals
from tuningfork.metrics.headline import HEADLINE_ESS_ESTIMATOR
from tuningfork.recipes._base import Recipe
from tuningfork.warmup._laplace_adapter import LAPLACE_METHOD_NAMES

_CATALOG_ROOT = Path(__file__).parent.parent / "catalog"


def stamp_headline_from_chain_stats(
    recipe: Recipe,
    base_method: Any,
    catalog_root: Path = _CATALOG_ROOT,
) -> Recipe:
    """Recover a PASS/REVIEW headline from existing chain-stat evidence."""
    if recipe.headline_metric is not None:
        return recipe

    verdict = recipe.gate_evidence.get("auto", {}).get("verdict")
    if verdict == "FAIL":
        raise ValueError(
            "stamp_headline_from_chain_stats: refusing to stamp headline on a "
            f"FAIL recipe ({recipe.model_name}/{recipe.base_method_name}). "
            "FAIL recipes have no meaningful ESS and correctly carry null "
            "headline_metric. Only PASS and REVIEW recipes may be stamped."
        )

    min_bulk_ess = recipe.gate_evidence.get("auto", {}).get("min_bulk_ess")
    if min_bulk_ess is None:
        warnings.warn(
            "stamp_headline_from_chain_stats: "
            "gate_evidence.auto.min_bulk_ess is None for "
            f"{recipe.model_name}/{recipe.base_method_name}; "
            "cannot recover headline.",
            stacklevel=2,
        )
        return recipe

    chain_stats_path = catalog_root / recipe.model_name / "_cache" / "chain_stats.npz"
    if not chain_stats_path.exists():
        warnings.warn(
            "stamp_headline_from_chain_stats: chain_stats cache not found at "
            f"{chain_stats_path}; cannot recover total_grad_evals.",
            stacklevel=2,
        )
        return recipe

    with np.load(chain_stats_path, allow_pickle=False) as stats:
        fields = list(stats.files)
        proxy_type = namedtuple("_ChainStatsProxy", fields)  # type: ignore[misc]
        proxy = proxy_type(**{name: jnp.asarray(stats[name]) for name in fields})
    try:
        grad_evals = total_grad_evals(proxy, base_method.grad_count_per_step)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            "stamp_headline_from_chain_stats: failed to compute "
            f"total_grad_evals: {exc}; leaving headline unchanged.",
            stacklevel=2,
        )
        return recipe
    if grad_evals <= 0:
        warnings.warn(
            "stamp_headline_from_chain_stats: "
            f"grad_evals={grad_evals} <= 0 from chain_stats; "
            "leaving headline unchanged.",
            stacklevel=2,
        )
        return recipe

    basis = {
        "total_grad_evals": int(grad_evals),
        "min_bulk_ess": float(min_bulk_ess),
        "ess_estimator": HEADLINE_ESS_ESTIMATOR,
        "min_bulk_ess_classic_legacy": None,
        "estimator_ratio": None,
        "grad_count_convention": (
            base_method.grad_count_convention or recipe.base_method_name
        ),
        "is_lower_bound": recipe.base_method_name in LAPLACE_METHOD_NAMES,
    }
    return replace(
        recipe,
        headline_metric=float(min_bulk_ess) / float(grad_evals),
        headline_basis=basis,
    )


__all__ = ["stamp_headline_from_chain_stats"]
