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
"""Pure initialization-strategy transformations shared by recipe paths."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

_ENSEMBLE_FRIENDLY_WARMUPS = frozenset(
    {
        "window_adaptation_diag_imm",
        "window_adaptation_dense_imm",
        "window_adaptation_low_rank_imm",
        "mclmc_tuning",
        "mclmc_lrd_tuning",
        "adjusted_mclmc_tuning",
        "adjusted_mclmc_trajectory_tuning",
        "no_warmup",
        "multipathfinder_window_adaptation",
        "meads",
        "chees",
    }
)


def validate_init_strategy_warmup_compatibility(
    init_strategy: dict[str, Any] | None, warmup_name: str
) -> None:
    """Reject pre-batched initialization for single-point warmups."""
    if init_strategy is None:
        return
    strategy_type = init_strategy.get("type")
    if (
        strategy_type in {"uniform_perchain", "zero_perchain"}
        and warmup_name not in _ENSEMBLE_FRIENDLY_WARMUPS
    ):
        raise ValueError(
            f"init_strategy type={strategy_type!r} is designed for ensemble warmups "
            f"(ChEES, MEADS, window adaptations, etc.) but warmup {warmup_name!r} is "
            "a single-point method that expects scalar init positions. "
            "Use legacy types instead: {'type': 'zero'} or "
            "{'type': 'uniform', 'low': ..., 'high': ...}."
        )


def apply_init_strategy(
    strategy: dict[str, Any],
    init_position: Any,
    rng_key: Any,
    num_chains: int = 1,
) -> Any:
    """Transform an initial-position pytree according to a validated strategy."""
    type_ = strategy.get("type", "prior_sample")
    if type_ == "prior_sample":
        return init_position
    if type_ == "zero":
        return jax.tree.map(jnp.zeros_like, init_position)
    if type_ == "uniform":
        low = float(strategy["low"])
        high = float(strategy["high"])
        leaves, treedef = jax.tree_util.tree_flatten(init_position)
        keys = jax.random.split(rng_key, len(leaves))
        return treedef.unflatten(
            [
                jax.random.uniform(
                    key,
                    leaf.shape,
                    dtype=leaf.dtype,
                    minval=low,
                    maxval=high,
                )
                for key, leaf in zip(keys, leaves)
            ]
        )
    if type_ not in {"zero_perchain", "uniform_perchain"}:
        raise ValueError(f"Unknown init_strategy type: {type_!r}")  # pragma: no cover
    if (
        isinstance(num_chains, bool)
        or not isinstance(num_chains, int)
        or num_chains < 1
    ):
        raise ValueError("num_chains must be a positive integer")

    leaves, treedef = jax.tree_util.tree_flatten(init_position)
    keys = iter(jax.random.split(rng_key, num_chains * len(leaves)))
    per_leaf: list[list[Any]] = [[] for _ in leaves]
    for _ in range(num_chains):
        for leaf_index, leaf in enumerate(leaves):
            shape = (1, *leaf.shape)
            key = next(keys)
            if type_ == "zero_perchain":
                value = float(strategy.get("jitter", 0.5)) * jax.random.normal(
                    key, shape, dtype=leaf.dtype
                )
            else:
                value = jax.random.uniform(
                    key,
                    shape,
                    dtype=leaf.dtype,
                    minval=float(strategy["low"]),
                    maxval=float(strategy["high"]),
                )
            per_leaf[leaf_index].append(value)
    return treedef.unflatten([jnp.concatenate(values, axis=0) for values in per_leaf])


__all__ = [
    "apply_init_strategy",
    "validate_init_strategy_warmup_compatibility",
]
