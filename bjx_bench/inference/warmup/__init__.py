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
"""bjx-bench warmup registry.

Phase 3 (P3.1) lands three core warmups:
- ``stan_window``: blackjax.window_adaptation; compatible with hmc, nuts,
  barker, mala.
- ``mclmc_tuning``: blackjax.mclmc_find_L_and_step_size; compatible with
  mclmc only.
- ``no_warmup``: identity warmup returning default init state + empty params;
  compatible with all algorithms (sentinel ``"*"``).

Phase 5.4 (P5.4) adds two Pathfinder-based warmups:
- ``pathfinder``: single-path Pathfinder per chain; compatible with hmc, nuts,
  mala, rwm, barker.  Returns per-chain L-BFGS inv-Hessian diagonal as IMM.
- ``multipathfinder``: multi-path Pathfinder with PSIS importance resampling;
  compatible with hmc, nuts, mala, rwm, barker.  Returns post-PSIS empirical
  variance as the shared IMM.

Usage::

    from bjx_bench.inference.warmup import WARMUPS, Warmup

    warmup = WARMUPS["stan_window"]
    state, params = warmup.runner(rng_key, position, n_warmup, base_method,
                                  logdensity_fn=logdensity_fn)
"""

from bjx_bench.inference.warmup._base import Warmup
from bjx_bench.inference.warmup.mclmc_tuning import ENTRY as _mclmc_tuning
from bjx_bench.inference.warmup.meads import ENTRY as _meads
from bjx_bench.inference.warmup.multipathfinder import ENTRY as _multipathfinder
from bjx_bench.inference.warmup.no_warmup import ENTRY as _no_warmup
from bjx_bench.inference.warmup.pathfinder import ENTRY as _pathfinder
from bjx_bench.inference.warmup.stan_window import ENTRY as _stan_window

WARMUPS: dict[str, Warmup] = {
    _stan_window.name: _stan_window,
    _mclmc_tuning.name: _mclmc_tuning,
    _no_warmup.name: _no_warmup,
    _pathfinder.name: _pathfinder,
    _multipathfinder.name: _multipathfinder,
    _meads.name: _meads,
}

__all__ = ["WARMUPS", "Warmup"]
