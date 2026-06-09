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
"""Experimental MCLMC LRD exploration utilities.

The custom ``lrd_integrator.py`` has been retired; the upstream
``blackjax.mcmc.integrators.isokinetic_mclachlan`` now dispatches natively on
``LowRankInverseMassMatrix`` (blackjax PR #936).  The only tuningfork-side
wiring kept here is ``make_lrd_kernel``, which binds an LRD mass matrix into
the standard mclmc kernel for warmup compatibility.
"""

from tuningfork.experimental.mclmc_explore.mclmc_advanced_tuning import make_lrd_kernel

__all__ = [
    "make_lrd_kernel",
]
