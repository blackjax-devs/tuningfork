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
"""bjx-bench algorithm registry.

``BASE_METHODS`` maps algorithm name strings to ``BaseMethod`` instances.
The runner, Optuna BO loop, and CLI all iterate over this dict to discover
available algorithms without hard-coding names.

Adding a new algorithm
----------------------
1. Create ``bjx_bench/algorithms/<name>.py`` with an ``ENTRY`` module-level
   variable of type ``BaseMethod``.
2. Import it here and add it to ``BASE_METHODS``.
"""

from bjx_bench.inference.base_method._base import BaseMethod, HyperparamSpace
from bjx_bench.inference.base_method.additive_step_random_walk import (
    ENTRY as _additive_step_random_walk_entry,
)
from bjx_bench.inference.base_method.adjusted_mclmc import (
    ENTRY as _adjusted_mclmc_entry,
)
from bjx_bench.inference.base_method.adjusted_mclmc_dynamic import (
    ENTRY as _adjusted_mclmc_dynamic_entry,
)
from bjx_bench.inference.base_method.barker import ENTRY as _barker_entry
from bjx_bench.inference.base_method.dmhmc import ENTRY as _dmhmc_entry
from bjx_bench.inference.base_method.dynamic_hmc import ENTRY as _dynamic_hmc_entry
from bjx_bench.inference.base_method.elliptical_slice import (
    ENTRY as _elliptical_slice_entry,
)
from bjx_bench.inference.base_method.fullrank_vi import ENTRY as _fullrank_vi_entry
from bjx_bench.inference.base_method.ghmc import ENTRY as _ghmc_entry
from bjx_bench.inference.base_method.hmc import ENTRY as _hmc_entry
from bjx_bench.inference.base_method.irmh import ENTRY as _irmh_entry
from bjx_bench.inference.base_method.laplace_dhmc import ENTRY as _laplace_dhmc_entry
from bjx_bench.inference.base_method.laplace_dmhmc import ENTRY as _laplace_dmhmc_entry
from bjx_bench.inference.base_method.laplace_hmc import ENTRY as _laplace_hmc_entry
from bjx_bench.inference.base_method.laplace_mhmc import ENTRY as _laplace_mhmc_entry
from bjx_bench.inference.base_method.mala import ENTRY as _mala_entry
from bjx_bench.inference.base_method.mclmc import ENTRY as _mclmc_entry
from bjx_bench.inference.base_method.meanfield_vi import ENTRY as _meanfield_vi_entry
from bjx_bench.inference.base_method.mgrad_gaussian import (
    ENTRY as _mgrad_gaussian_entry,
)
from bjx_bench.inference.base_method.mhmc import ENTRY as _mhmc_entry
from bjx_bench.inference.base_method.nuts import ENTRY as _nuts_entry
from bjx_bench.inference.base_method.orbital_hmc import ENTRY as _orbital_hmc_entry
from bjx_bench.inference.base_method.rmhmc import ENTRY as _rmhmc_entry
from bjx_bench.inference.base_method.rwm import ENTRY as _rwm_entry

BASE_METHODS: dict[str, BaseMethod] = {
    _hmc_entry.name: _hmc_entry,
    _mhmc_entry.name: _mhmc_entry,
    _nuts_entry.name: _nuts_entry,
    _mala_entry.name: _mala_entry,
    _barker_entry.name: _barker_entry,
    _rwm_entry.name: _rwm_entry,
    _mclmc_entry.name: _mclmc_entry,
    _ghmc_entry.name: _ghmc_entry,
    _dynamic_hmc_entry.name: _dynamic_hmc_entry,
    _dmhmc_entry.name: _dmhmc_entry,
    _adjusted_mclmc_entry.name: _adjusted_mclmc_entry,
    _adjusted_mclmc_dynamic_entry.name: _adjusted_mclmc_dynamic_entry,
    _elliptical_slice_entry.name: _elliptical_slice_entry,
    _mgrad_gaussian_entry.name: _mgrad_gaussian_entry,
    _irmh_entry.name: _irmh_entry,
    _meanfield_vi_entry.name: _meanfield_vi_entry,
    _fullrank_vi_entry.name: _fullrank_vi_entry,
    _laplace_hmc_entry.name: _laplace_hmc_entry,
    _laplace_dhmc_entry.name: _laplace_dhmc_entry,
    _laplace_mhmc_entry.name: _laplace_mhmc_entry,
    _laplace_dmhmc_entry.name: _laplace_dmhmc_entry,
    _orbital_hmc_entry.name: _orbital_hmc_entry,
    _rmhmc_entry.name: _rmhmc_entry,
    _additive_step_random_walk_entry.name: _additive_step_random_walk_entry,
}

__all__ = ["BaseMethod", "HyperparamSpace", "BASE_METHODS"]
