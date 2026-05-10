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
"""SMC algorithm registry — sister abstraction to BASE_METHODS.

See bjx_bench/inference/smc/_base.py for the SMCMethod dataclass.
SMC methods have a different factory contract from BaseMethod
(prior/likelihood split, inner kernel composition).
"""

from bjx_bench.inference.smc._base import SMCMethod
from bjx_bench.inference.smc.adaptive_tempered import ENTRY as _adaptive_tempered
from bjx_bench.inference.smc.inner_kernel_tuning import ENTRY as _inner_kernel_tuning
from bjx_bench.inference.smc.partial_posteriors import ENTRY as _partial_posteriors

SMC_METHODS: dict[str, SMCMethod] = {
    _adaptive_tempered.name: _adaptive_tempered,
    _partial_posteriors.name: _partial_posteriors,
    _inner_kernel_tuning.name: _inner_kernel_tuning,
}

__all__ = ["SMCMethod", "SMC_METHODS"]
