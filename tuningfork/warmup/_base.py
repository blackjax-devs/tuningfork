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

"""Descriptor type for registered warmup procedures."""

from dataclasses import dataclass, field

from tuningfork.base_method._base import HyperparamSpace


@dataclass(frozen=True)
class Warmup:
    """Immutable description of a warmup procedure.

    Warmups are descriptors only.  Generated recipe programs own execution;
    this registry records compatibility and hyperparameter metadata.
    """

    name: str
    compatible_methods: tuple[str, ...]
    notes: str = ""
    default_hp_space: tuple[HyperparamSpace, ...] = field(default_factory=tuple)

    def is_compatible(self, base_method_name: str) -> bool:
        """Return whether this warmup supports ``base_method_name``."""
        return (
            "*" in self.compatible_methods
            or base_method_name in self.compatible_methods
        )
