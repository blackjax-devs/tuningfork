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

See tuningfork/inference/smc/_base.py for the SMCMethod dataclass.
SMC methods have a different factory contract from BaseMethod
(prior/likelihood split, inner kernel composition).
"""

from tuningfork.inference.smc._base import SMCMethod
from tuningfork.inference.smc.adaptive_persistent_sampling import (
    ENTRY as _adaptive_persistent_sampling,
)
from tuningfork.inference.smc.adaptive_tempered import ENTRY as _adaptive_tempered
from tuningfork.inference.smc.inner_kernel_tuning import ENTRY as _inner_kernel_tuning
from tuningfork.inference.smc.partial_posteriors import ENTRY as _partial_posteriors
from tuningfork.inference.smc.persistent_sampling import ENTRY as _persistent_sampling
from tuningfork.inference.smc.tempered import ENTRY as _tempered_smc

SMC_METHODS: dict[str, SMCMethod] = {
    _adaptive_tempered.name: _adaptive_tempered,
    _partial_posteriors.name: _partial_posteriors,
    _inner_kernel_tuning.name: _inner_kernel_tuning,
    _persistent_sampling.name: _persistent_sampling,
    _adaptive_persistent_sampling.name: _adaptive_persistent_sampling,
    _tempered_smc.name: _tempered_smc,
}

__all__ = ["SMCMethod", "SMC_METHODS"]
