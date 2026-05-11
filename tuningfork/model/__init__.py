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
"""Model registry for bjx-bench.

``MODELS`` maps model name strings to ``Posterior`` instances.  Every
model file in the sub-packages exposes a module-level ``ENTRY`` constant; this
file imports them all and assembles the dict so callers never need to know
where a model lives.
"""

from tuningfork.model._base import Posterior, ReferenceMethod
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model.gaussians.ill_cond_50 import ENTRY as _ill_cond_50
from tuningfork.model.gaussians.mvn_10 import ENTRY as _mvn_10
from tuningfork.model.glm.german_credit import ENTRY as _german_credit
from tuningfork.model.glm.horseshoe import ENTRY as _horseshoe
from tuningfork.model.glm.logistic_synthetic import ENTRY as _logistic_synthetic
from tuningfork.model.hierarchical.eight_schools import ENTRY as _eight_schools_ncp
from tuningfork.model.hierarchical.irt_2pl import ENTRY as _irt_2pl
from tuningfork.model.hierarchical.radon import ENTRY as _radon
from tuningfork.model.latent_gaussian.gp_regression import ENTRY as _gp_regression
from tuningfork.model.latent_gaussian.stoch_vol import ENTRY as _stoch_vol
from tuningfork.model.ode.lotka_volterra import ENTRY as _lotka_volterra
from tuningfork.model.pathological.banana import ENTRY as _banana
from tuningfork.model.pathological.gmm_25 import ENTRY as _gmm_25
from tuningfork.model.pathological.neals_funnel import ENTRY as _neals_funnel

__all__ = ["MODELS", "Posterior", "ReferenceMethod", "build_logdensity_fn"]

MODELS: dict[str, Posterior] = {
    entry.name: entry
    for entry in [
        _mvn_10,
        _ill_cond_50,
        _neals_funnel,
        _eight_schools_ncp,
        _banana,
        _gmm_25,
        _logistic_synthetic,
        _german_credit,
        _horseshoe,
        _radon,
        _irt_2pl,
        _stoch_vol,
        _lotka_volterra,
        _gp_regression,
    ]
}
