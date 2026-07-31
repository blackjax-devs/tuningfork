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
"""Emit the standalone initial-position setup used by recipe scripts."""

from __future__ import annotations

import math
from typing import Any

__all__ = ["emit_init_strategy"]

_TYPES = {
    "prior_sample",
    "zero",
    "uniform",
    "zero_perchain",
    "uniform_perchain",
    "reference_summary",
}


def _number(strategy: dict[str, Any], key: str) -> float:
    try:
        value = float(strategy[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"init_strategy {key!r} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"init_strategy {key!r} must be finite")
    return value


def _validate(strategy: dict[str, Any] | None, num_chains: int) -> str:
    if (
        not isinstance(num_chains, int)
        or isinstance(num_chains, bool)
        or num_chains < 1
    ):
        raise ValueError("num_chains must be a positive integer")
    if strategy is None:
        return "prior_sample"
    if not isinstance(strategy, dict):
        raise ValueError("init_strategy must be a dict or None")
    type_ = strategy.get("type")
    if type_ not in _TYPES:
        raise ValueError(f"Unknown init_strategy type: {type_!r}")
    if type_ in {"uniform", "uniform_perchain"}:
        if "low" not in strategy or "high" not in strategy:
            raise ValueError(f"init_strategy type={type_!r} requires low and high")
        if _number(strategy, "low") >= _number(strategy, "high"):
            raise ValueError("init_strategy requires low < high")
    if type_ == "reference_summary":
        required = {"mean", "std", "offsets", "source_path", "source_sha256"}
        missing = required.difference(strategy)
        if missing:
            raise ValueError(f"reference_summary missing keys: {sorted(missing)!r}")
        if set(strategy["mean"]) != set(strategy["std"]):
            raise ValueError("reference_summary mean/std keys must match")
        if not isinstance(strategy["offsets"], list) or not strategy["offsets"]:
            raise ValueError("reference_summary offsets must be a non-empty list")
    if (
        type_ == "zero_perchain"
        and "jitter" in strategy
        and _number(strategy, "jitter") < 0
    ):
        raise ValueError("init_strategy jitter must be non-negative")
    return type_


def emit_init_strategy(
    strategy: dict[str, Any] | None,
    num_chains: int,
) -> str:
    """Return Python source assigning ``init_position`` for *strategy*.

    The caller must have already bound ``init_position``, ``_init_key``, and
    imported ``jax`` and ``jax.numpy as jnp``. Random strategies derive a
    deterministic child key with ``fold_in(_init_key, 42)``. Warmup
    compatibility checks and consumption of the pre-batched flag remain the
    caller's responsibility.
    """
    type_ = _validate(strategy, num_chains)
    strategy_values = {} if strategy is None else strategy
    lines = ["# === INITIAL POSITION ==="]
    if type_ == "prior_sample":
        lines += ["_init_position_is_prebatched = False"]
    elif type_ == "zero":
        lines += [
            "init_position = jax.tree.map(lambda x: jnp.zeros_like(x), init_position)",
            "_init_position_is_prebatched = False",
        ]
    elif type_ == "uniform":
        low, high = (
            _number(strategy_values, "low"),
            _number(strategy_values, "high"),
        )
        lines += [
            "_init_strategy_key = jax.random.fold_in(_init_key, 42)",
            "_init_leaves, _init_treedef = jax.tree_util.tree_flatten(init_position)",
            "_init_keys = jax.random.split(_init_strategy_key, len(_init_leaves))",
            "init_position = _init_treedef.unflatten([",
            "    jax.random.uniform(k, x.shape, dtype=x.dtype, minval=%r, maxval=%r)"
            % (low, high),
            "    for k, x in zip(_init_keys, _init_leaves)",
            "])",
            "_init_position_is_prebatched = False",
        ]
    elif type_ == "reference_summary":
        import json

        means = json.dumps(strategy_values["mean"], separators=(",", ":"))
        stds = json.dumps(strategy_values["std"], separators=(",", ":"))
        offsets = json.dumps(strategy_values["offsets"], separators=(",", ":"))
        lines += [
            f"# reference_summary source: {strategy_values['source_path']} sha256={strategy_values['source_sha256']}",
            f"_reference_summary_mean = {means}",
            f"_reference_summary_std = {stds}",
            f"_reference_summary_offsets = {offsets}",
            "_reference_summary_keys = tuple(_reference_summary_mean)",
            "_reference_summary_positions = []",
            "for _reference_summary_chain in range(num_chains):",
            "    _reference_summary_offset = _reference_summary_offsets[_reference_summary_chain % len(_reference_summary_offsets)]",
            "    _reference_summary_positions.append({",
            "        _key: jnp.asarray(_reference_summary_mean[_key], dtype=init_position[_key].dtype) + _reference_summary_offset * jnp.asarray(_reference_summary_std[_key], dtype=init_position[_key].dtype)",
            "        for _key in _reference_summary_keys",
            "    })",
            "init_position = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *_reference_summary_positions)",
            "_init_position_is_prebatched = True",
        ]
    else:
        lines.append("_init_strategy_key = jax.random.fold_in(_init_key, 42)")
        if type_ == "uniform_perchain":
            low, high = (
                _number(strategy_values, "low"),
                _number(strategy_values, "high"),
            )
            draw = (
                "jax.random.uniform(k, (1,) + _init_leaf.shape, "
                "dtype=_init_leaf.dtype, minval=%r, maxval=%r)" % (low, high)
            )
        else:
            jitter = (
                _number(strategy_values, "jitter")
                if "jitter" in strategy_values
                else 0.5
            )
            draw = (
                "(%r * jax.random.normal(k, (1,) + _init_leaf.shape, "
                "dtype=_init_leaf.dtype))" % jitter
            )
        lines += [
            "_init_leaves, _init_treedef = jax.tree_util.tree_flatten(init_position)",
            "_init_keys = jax.random.split(_init_strategy_key, num_chains * len(_init_leaves))",
            "_init_keys = _init_keys.reshape((num_chains, len(_init_leaves)))",
            "_init_new_leaves = []",
            "for _init_leaf_idx, _init_leaf in enumerate(_init_leaves):",
            "    _init_leaf_keys = _init_keys[:, _init_leaf_idx]",
            "    _init_new_leaves.append(",
            "        jnp.concatenate([%s for k in _init_leaf_keys], axis=0)" % draw,
            "    )",
            "init_position = _init_treedef.unflatten(_init_new_leaves)",
            "_init_position_is_prebatched = True",
        ]
    return "\n".join(lines) + "\n"
