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
"""Model registry for tuningfork.

``MODELS`` maps model name strings to ``Posterior`` instances.  Every
model file exposes a module-level ``ENTRY`` constant; this file imports
them all and assembles the dict so callers never need to know which
mathematical family a model belongs to.

``MODELS_BY_FAMILY`` preserves the original taxonomy for discoverability
and documentation. Families:

    - "gaussians"        — Gaussian baselines (mvn_10, ill_cond_50)
    - "glm"              — generalized linear models (logistic_synthetic,
                            german_credit, horseshoe)
    - "hierarchical"     — multi-level / partial-pooling models
                            (eight_schools_ncp, radon, irt_2pl)
    - "latent_gaussian"  — latent-Gaussian / state-space models
                            (gp_regression, stoch_vol)
    - "ode"              — ODE inverse problems (lotka_volterra)
    - "pathological"     — geometry-stress tests (banana, neals_funnel,
                            gmm_25)
"""

from tuningfork.model._base import Posterior, ReferenceMethod
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model.banana import ENTRY as _banana
from tuningfork.model.eight_schools import ENTRY as _eight_schools_ncp
from tuningfork.model.german_credit import ENTRY as _german_credit
from tuningfork.model.gmm_25 import ENTRY as _gmm_25
from tuningfork.model.gp_regression import ENTRY as _gp_regression
from tuningfork.model.horseshoe import ENTRY as _horseshoe
from tuningfork.model.ill_cond_50 import ENTRY as _ill_cond_50
from tuningfork.model.irt_1pl import ENTRY as _irt_1pl
from tuningfork.model.irt_2pl import ENTRY as _irt_2pl
from tuningfork.model.lgcp import ENTRY as _lgcp
from tuningfork.model.logistic_synthetic import ENTRY as _logistic_synthetic
from tuningfork.model.lotka_volterra import ENTRY as _lotka_volterra
from tuningfork.model.mvn_10 import ENTRY as _mvn_10
from tuningfork.model.neals_funnel import ENTRY as _neals_funnel
from tuningfork.model.radon import ENTRY as _radon
from tuningfork.model.stoch_vol import ENTRY as _stoch_vol

__all__ = [
    "MODELS",
    "MODELS_BY_FAMILY",
    "Posterior",
    "ReferenceMethod",
    "build_logdensity_fn",
]

MODELS_BY_FAMILY: dict[str, list[Posterior]] = {
    "gaussians": [_mvn_10, _ill_cond_50],
    "glm": [_logistic_synthetic, _german_credit, _horseshoe],
    "hierarchical": [_eight_schools_ncp, _radon, _irt_2pl, _irt_1pl],
    "latent_gaussian": [_gp_regression, _stoch_vol, _lgcp],
    "ode": [_lotka_volterra],
    "pathological": [_banana, _neals_funnel, _gmm_25],
}

MODELS: dict[str, Posterior] = {
    entry.name: entry
    for family_entries in MODELS_BY_FAMILY.values()
    for entry in family_entries
}
