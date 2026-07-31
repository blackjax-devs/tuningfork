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
"""Emit a standalone integration-step policy function.

The emitted one-argument callable is the standard ``dynamic_hmc``/``dmhmc``
policy contract.  It must not replace CHEES' adapted two-argument callable,
which receives the adaptation state as an additional argument.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

__all__ = ["emit_step_policy"]


def _as_int(spec: Mapping[str, Any], key: str) -> int:
    try:
        value = spec[key]
        if isinstance(value, bool) or int(value) != value:
            raise ValueError
        return int(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"step_policy field {key!r} must be an integer") from exc


def _as_float(spec: Mapping[str, Any], key: str) -> float:
    try:
        value = float(spec[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"step_policy field {key!r} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"step_policy field {key!r} must be finite")
    return value


def _validate(spec: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    if spec is None:
        return "uniform_int", {"low": 1, "high": 10}
    if not isinstance(spec, Mapping):
        raise ValueError("step_policy must be a mapping or None")
    kind = spec.get("kind")
    if kind == "uniform_int":
        low, high = _as_int(spec, "low"), _as_int(spec, "high")
        if low < 1 or low >= high:
            raise ValueError("uniform_int step_policy requires 1 <= low < high")
        return kind, {"low": low, "high": high}
    if kind == "empirical":
        if "values" not in spec or "weights" not in spec:
            raise ValueError("empirical step_policy requires values and weights")
        try:
            raw_values = list(spec["values"])
            if any(
                isinstance(value, (str, bytes, bool)) or int(value) != value
                for value in raw_values
            ):
                raise ValueError
            values = [int(value) for value in raw_values]
            weights = [float(weight) for weight in spec["weights"]]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "empirical values and weights must be numeric sequences"
            ) from exc
        if not values or len(values) != len(weights):
            raise ValueError(
                "empirical values and weights must be non-empty and equal length"
            )
        if any(value < 1 for value in values):
            raise ValueError("empirical values must be positive integers")
        if values != sorted(set(values)):
            raise ValueError("empirical values must be sorted and distinct")
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("empirical weights must be finite and non-negative")
        weight_sum = sum(weights)
        if weight_sum <= 0:
            raise ValueError("empirical weights must have positive sum")
        return kind, {
            "values": values,
            "weights": [weight / weight_sum for weight in weights],
        }
    if kind == "warmup_empirical":
        if len(spec) != 1:
            raise ValueError("warmup_empirical step_policy accepts only kind")
        return kind, {}
    if kind == "poisson":
        lam = _as_float(spec, "lam")
        if lam < 0:
            raise ValueError("poisson lam must be non-negative")
        low = _as_int(spec, "low") if "low" in spec else 1
        poisson_high = None if spec.get("high") is None else _as_int(spec, "high")
        if low < 1 or (poisson_high is not None and low >= poisson_high):
            raise ValueError("poisson step_policy requires low >= 1 and low < high")
        return kind, {"lam": lam, "low": low, "high": poisson_high}
    if kind == "log_uniform_int":
        low, high = _as_int(spec, "low"), _as_int(spec, "high")
        if low < 1:
            raise ValueError("log_uniform_int low must be >= 1")
        if low >= high:
            raise ValueError("log_uniform_int requires low < high")
        return kind, {"low": low, "high": high}
    if kind == "pow2_choice":
        if "options" not in spec:
            raise ValueError("pow2_choice step_policy requires options")
        try:
            raw_options = list(spec["options"])
            if any(
                isinstance(option, (str, bytes, bool)) or int(option) != option
                for option in raw_options
            ):
                raise ValueError
            options = [int(option) for option in raw_options]
        except (TypeError, ValueError) as exc:
            raise ValueError("pow2_choice options must be integers") from exc
        if not options or any(option < 1 for option in options):
            raise ValueError("pow2_choice options must be non-empty positive integers")
        if any(option & (option - 1) for option in options):
            raise ValueError("pow2_choice options must be powers of two")
        return kind, {"options": options}
    raise ValueError(f"Unknown step_policy kind {kind!r}")


def emit_step_policy(spec: Mapping[str, Any] | None) -> str:
    """Return source defining ``_integration_steps_fn`` for *spec*.

    The generated source assumes ``jax`` and ``jax.numpy as jnp`` are already
    bound by its caller and has no dependency on tuningfork.
    """
    kind, values = _validate(spec)
    if kind == "warmup_empirical":
        return (
            "\n".join(
                [
                    "_warmup_nis_raw = np.asarray(_warmup_info.info.num_integration_steps)",
                    "if _warmup_nis_raw.size == 0:",
                    "    raise ValueError('warmup_empirical requires non-empty warmup info')",
                    "try:",
                    "    _warmup_nis_float = _warmup_nis_raw.astype(np.float64, copy=False).reshape(-1)",
                    "except (TypeError, ValueError, OverflowError) as _exc:",
                    "    raise ValueError('warmup_empirical warmup info must be numeric') from _exc",
                    "if (not np.all(np.isfinite(_warmup_nis_float)) or",
                    "        np.any(_warmup_nis_float <= 0) or",
                    "        np.any(_warmup_nis_float != np.floor(_warmup_nis_float))):",
                    "    raise ValueError('warmup_empirical warmup info must contain positive integers')",
                    "_warmup_nis_int = _warmup_nis_float.astype(np.int64)",
                    "_step_policy_values_np, _step_policy_counts_np = np.unique(_warmup_nis_int, return_counts=True)",
                    "_step_policy_weights_np = (_step_policy_counts_np / _warmup_nis_int.size).astype(np.float64)",
                    "_resolved_step_policy = {'kind': 'empirical', 'values': _step_policy_values_np.tolist(), 'weights': _step_policy_weights_np.tolist()}",
                    "_step_policy_values = jnp.asarray(_step_policy_values_np, dtype=jnp.int32)",
                    "_step_policy_weights = jnp.asarray(_step_policy_weights_np, dtype=jnp.float32)",
                    "_step_policy_cdf = jnp.cumsum(_step_policy_weights)",
                    "def _integration_steps_fn(key):",
                    "    u = jax.random.uniform(key)",
                    "    idx = jnp.searchsorted(_step_policy_cdf, u, side='right')",
                    "    idx = jnp.clip(idx, 0, _step_policy_values.shape[0] - 1)",
                    "    return _step_policy_values[idx]",
                ]
            )
            + "\n"
        )
    if kind == "uniform_int":
        lines = [
            f"_resolved_step_policy = {{'kind': 'uniform_int', 'low': {values['low']!r}, 'high': {values['high']!r}}}",
            "def _integration_steps_fn(key):",
            f"    return jax.random.randint(key, (), {values['low']}, {values['high']})",
        ]
    elif kind == "empirical":
        lines = [
            f"_step_policy_values = jnp.array({values['values']!r}, dtype=jnp.int32)",
            f"_step_policy_weights = jnp.array({values['weights']!r}, dtype=jnp.float32)",
            "_step_policy_weights = _step_policy_weights / jnp.sum(_step_policy_weights)",
            "_step_policy_cdf = jnp.cumsum(_step_policy_weights)",
            f"_resolved_step_policy = {{'kind': 'empirical', 'values': {values['values']!r}, 'weights': {values['weights']!r}}}",
            "def _integration_steps_fn(key):",
            "    u = jax.random.uniform(key)",
            "    idx = jnp.searchsorted(_step_policy_cdf, u, side='right')",
            "    idx = jnp.clip(idx, 0, _step_policy_values.shape[0] - 1)",
            "    return _step_policy_values[idx]",
        ]
    elif kind == "poisson":
        lines = [
            f"_step_policy_lam = {values['lam']!r}",
            f"_step_policy_low = {values['low']!r}",
            f"_resolved_step_policy = {{'kind': 'poisson', 'lam': {values['lam']!r}, 'low': {values['low']!r}, 'high': {values['high']!r}}}",
        ]
        if values["high"] is None:
            lines += [
                "def _integration_steps_fn(key):",
                "    raw = jax.random.poisson(key, lam=_step_policy_lam)",
                "    return jnp.maximum(raw, _step_policy_low).astype(jnp.int32)",
            ]
        else:
            lines += [
                f"_step_policy_high = {values['high']!r}",
                "def _integration_steps_fn(key):",
                "    def _body(state):",
                "        key, _ = state",
                "        key, subkey = jax.random.split(key)",
                "        sample = jax.random.poisson(subkey, lam=_step_policy_lam)",
                "        sample = jnp.maximum(sample, _step_policy_low).astype(jnp.int32)",
                "        return key, sample",
                "    def _cond(state):",
                "        return state[1] >= _step_policy_high",
                "    key, subkey = jax.random.split(key)",
                "    initial = jax.random.poisson(subkey, lam=_step_policy_lam)",
                "    initial = jnp.maximum(initial, _step_policy_low).astype(jnp.int32)",
                "    _, result = jax.lax.while_loop(_cond, _body, (key, initial))",
                "    return result",
            ]
    elif kind == "log_uniform_int":
        lines = [
            f"_step_policy_log_low = {math.log(values['low'])!r}",
            f"_step_policy_log_high = {math.log(values['high'])!r}",
            f"_resolved_step_policy = {{'kind': 'log_uniform_int', 'low': {values['low']!r}, 'high': {values['high']!r}}}",
            "def _integration_steps_fn(key):",
            "    u = jax.random.uniform(key, minval=_step_policy_log_low, maxval=_step_policy_log_high)",
            "    sample = jnp.round(jnp.exp(u)).astype(jnp.int32)",
            f"    return jnp.clip(sample, {values['low']}, {values['high']})",
        ]
    else:
        lines = [
            f"_step_policy_options = jnp.array({values['options']!r}, dtype=jnp.int32)",
            f"_resolved_step_policy = {{'kind': 'pow2_choice', 'options': {values['options']!r}}}",
            "def _integration_steps_fn(key):",
            "    return jax.random.choice(key, _step_policy_options)",
        ]
    return "\n".join(lines) + "\n"
