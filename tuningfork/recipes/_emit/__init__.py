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
"""Python emit-functions replacing the top-level .py.tmpl template files.

These functions live INSIDE ``tuningfork/recipes/_emit/`` (internal generator).
What they OUTPUT is plain Python with no tuningfork imports in the inference
choreography (D8 compliant).

Public API
----------
emit_init_strategy(strategy, num_chains) -> str
    Emits the configured initial-position transformation.
emit_preamble(ctx) -> str
    Replaces ``_templates/preamble.py.tmpl``.
emit_laplace_preamble(ctx) -> str
    Replaces ``_templates/laplace_preamble.py.tmpl``.
emit_postamble(ctx) -> str
    Replaces ``_templates/postamble.py.tmpl``.
emit_step_policy(spec) -> str
    Emits the dynamic-HMC integration-step callable.
"""

from __future__ import annotations

from tuningfork.recipes._emit._init_strategy import emit_init_strategy
from tuningfork.recipes._emit._laplace_preamble import emit_laplace_preamble
from tuningfork.recipes._emit._postamble import emit_postamble
from tuningfork.recipes._emit._preamble import emit_preamble
from tuningfork.recipes._emit._sampler import emit_sampler
from tuningfork.recipes._emit._step_policy import emit_step_policy
from tuningfork.recipes._emit._warmup import emit_warmup

__all__ = [
    "emit_init_strategy",
    "emit_preamble",
    "emit_laplace_preamble",
    "emit_postamble",
    "emit_sampler",
    "emit_step_policy",
    "emit_warmup",
]
