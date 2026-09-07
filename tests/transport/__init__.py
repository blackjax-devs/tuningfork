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
"""Validation suite for the supplied frozen coordinate chart.

Shared scope helper only.  The chart needs float64 to exercise its exactness
claims, but enabling x64 at import time mutates global JAX state during
*collection* — before pytest's per-test guard snapshots it — which leaks
float64 into every later test in the session.  That is not hypothetical: it is
what broke two unrelated groundtruth tests.

``x64_scope`` is therefore a module-scoped fixture that enables x64 at setup and
restores the previous value in a ``finally``, so nothing escapes the module even
if a test raises.  No collection-time mutation, and no change to the shared
conftest.
"""

import jax


def x64_scope():
    """Enable x64 for one test module, restoring the prior value afterwards."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)
