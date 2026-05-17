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
"""Model registry — thin re-export of the assembly in ``_registry.py``.

The registry assembly (14 model imports + ``MODELS_BY_FAMILY`` dict)
lives in ``tuningfork/model/_registry.py``. This module is the public
entry point; users do ``from tuningfork.model import MODELS, Posterior,
build_logdensity_fn``.
"""

from tuningfork.model._registry import (
    MODELS,
    MODELS_BY_FAMILY,
    Posterior,
    ReferenceMethod,
    build_logdensity_fn,
)

__all__ = [
    "MODELS",
    "MODELS_BY_FAMILY",
    "Posterior",
    "ReferenceMethod",
    "build_logdensity_fn",
]
