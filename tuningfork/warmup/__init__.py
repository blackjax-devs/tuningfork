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
"""tuningfork warmup registry.

Core warmups:
- ``window_adaptation_diag_imm``: blackjax.window_adaptation; compatible with hmc, nuts,
  barker, mala.
- ``mclmc_tuning``: blackjax.mclmc_find_L_and_step_size; compatible with
  mclmc and adjusted_mclmc variants.
- ``no_warmup``: identity warmup returning default init state + empty params;
  compatible with all algorithms (sentinel ``"*"``).
- ``chees``: Cholesky Eigen Exploration Sampling; compatible with dynamic_hmc.
- ``meads``: Manifold-based geometric adaptation; compatible with ghmc.

Pathfinder-based warmups:
- ``pathfinder``: single-path Pathfinder per chain; compatible with hmc, nuts,
  mala, rwm, barker.  Returns per-chain L-BFGS inv-Hessian diagonal as IMM.
- ``multipathfinder``: multi-path Pathfinder with PSIS importance resampling;
  compatible with hmc, nuts, mala, rwm, barker.  Returns post-PSIS empirical
  variance as the shared IMM.
- ``multipathfinder_window_adaptation``: paper-canonical composed warmup
  (Zhang et al. 2022 § 4): multipathfinder init + window_adaptation seeded
  with dense IMM + medium pseudo-count shrinkage; compatible with hmc, nuts,
  mala, rwm, barker.

Variational inference warmups:
- ``meanfield_vi``: mean-field variational inference warmup.
- ``fullrank_vi``: full-rank variational inference warmup.

Usage::

    from tuningfork.warmup import WARMUPS, Warmup

    warmup = WARMUPS["window_adaptation_diag_imm"]
    state, params, *_ = warmup.runner(rng_key, position, n_warmup, base_method,
                                      logdensity_fn=logdensity_fn)
"""

from tuningfork.warmup._base import Warmup
from tuningfork.warmup.adjusted_mclmc_tuning import ENTRY as _adjusted_mclmc_tuning
from tuningfork.warmup.chees import ENTRY as _chees
from tuningfork.warmup.fullrank_vi import ENTRY as _fullrank_vi
from tuningfork.warmup.mclmc_lrd_tuning import ENTRY as _mclmc_lrd_tuning
from tuningfork.warmup.mclmc_tuning import ENTRY as _mclmc_tuning
from tuningfork.warmup.meads import ENTRY as _meads
from tuningfork.warmup.meanfield_vi import ENTRY as _meanfield_vi
from tuningfork.warmup.multipathfinder import ENTRY as _multipathfinder
from tuningfork.warmup.multipathfinder_window_adaptation import (
    ENTRY as _multipathfinder_window_adaptation,
)
from tuningfork.warmup.no_warmup import ENTRY as _no_warmup
from tuningfork.warmup.pathfinder import ENTRY as _pathfinder
from tuningfork.warmup.window_adaptation_dense_imm import (
    ENTRY as _window_adaptation_dense_imm,
)
from tuningfork.warmup.window_adaptation_diag_imm import (
    ENTRY as _window_adaptation_diag_imm,
)
from tuningfork.warmup.window_adaptation_low_rank_imm import (
    ENTRY as _window_adaptation_low_rank_imm,
)

WARMUPS: dict[str, Warmup] = {
    _window_adaptation_diag_imm.name: _window_adaptation_diag_imm,
    _window_adaptation_dense_imm.name: _window_adaptation_dense_imm,
    _window_adaptation_low_rank_imm.name: _window_adaptation_low_rank_imm,
    _mclmc_tuning.name: _mclmc_tuning,
    _mclmc_lrd_tuning.name: _mclmc_lrd_tuning,
    _adjusted_mclmc_tuning.name: _adjusted_mclmc_tuning,
    _no_warmup.name: _no_warmup,
    _pathfinder.name: _pathfinder,
    _multipathfinder.name: _multipathfinder,
    _multipathfinder_window_adaptation.name: _multipathfinder_window_adaptation,
    _meads.name: _meads,
    _chees.name: _chees,
    _meanfield_vi.name: _meanfield_vi,
    _fullrank_vi.name: _fullrank_vi,
}

__all__ = ["WARMUPS", "Warmup"]
