"""Model registry for bjx-bench.

``MODELS`` maps model name strings to ``Posterior`` instances.  Every
model file in the sub-packages exposes a module-level ``ENTRY`` constant; this
file imports them all and assembles the dict so callers never need to know
where a model lives.
"""

from __future__ import annotations

from bjx_bench.model._base import Posterior, ReferenceMethod
from bjx_bench.model._numpyro import build_logdensity_fn
from bjx_bench.model.gaussians.ill_cond_50 import ENTRY as _ill_cond_50
from bjx_bench.model.gaussians.mvn_10 import ENTRY as _mvn_10
from bjx_bench.model.glm.german_credit import ENTRY as _german_credit
from bjx_bench.model.glm.horseshoe import ENTRY as _horseshoe
from bjx_bench.model.glm.logistic_synthetic import ENTRY as _logistic_synthetic
from bjx_bench.model.hierarchical.eight_schools import ENTRY as _eight_schools_ncp
from bjx_bench.model.hierarchical.irt_2pl import ENTRY as _irt_2pl
from bjx_bench.model.hierarchical.radon import ENTRY as _radon
from bjx_bench.model.latent_gaussian.stoch_vol import ENTRY as _stoch_vol
from bjx_bench.model.pathological.banana import ENTRY as _banana
from bjx_bench.model.pathological.gmm_25 import ENTRY as _gmm_25
from bjx_bench.model.pathological.neals_funnel import ENTRY as _neals_funnel

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
    ]
}
