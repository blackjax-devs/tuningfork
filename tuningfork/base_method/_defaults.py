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

"""Pure default values for declared sampler hyperparameter spaces."""

from __future__ import annotations

from typing import Any

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["default_params_for", "default_value_for_space"]


def default_value_for_space(space: HyperparamSpace) -> Any:
    """Return the deterministic default for one declared hyperparameter."""
    if space.kind == "loguniform":
        return space.low * (space.high / space.low) ** 0.7  # type: ignore[operator]
    if space.kind == "uniform":
        return (space.low + space.high) / 2  # type: ignore[operator]
    if space.kind == "int":
        return (space.low + space.high) // 2  # type: ignore[operator]
    if space.kind == "categorical":
        return space.choices[0]  # type: ignore[index]
    raise ValueError(
        f"default_value_for_space: unknown kind {space.kind!r}. "
        "Expected one of 'loguniform', 'uniform', 'int', 'categorical'."
    )


def default_params_for(entry: BaseMethod) -> dict[str, Any]:
    """Return deterministic defaults for every declared sampler parameter."""
    return {
        space.name: default_value_for_space(space) for space in entry.default_hp_space
    }
