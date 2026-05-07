"""Model registry for bjx-bench.

``REGISTRY`` maps model name strings to ``PosteriorEntry`` instances.  Every
model file in the sub-packages exposes a module-level ``ENTRY`` constant; this
file imports them all and assembles the dict so callers never need to know
where a model lives.
"""

from __future__ import annotations

from bjx_bench.registry._base import PosteriorEntry, ReferenceMethod
from bjx_bench.registry._numpyro import build_logdensity_fn
from bjx_bench.registry.gaussians.mvn_10 import ENTRY as _mvn_10
from bjx_bench.registry.hierarchical.eight_schools import ENTRY as _eight_schools_ncp
from bjx_bench.registry.pathological.neals_funnel import ENTRY as _neals_funnel

__all__ = ["REGISTRY", "PosteriorEntry", "ReferenceMethod", "build_logdensity_fn"]

REGISTRY: dict[str, PosteriorEntry] = {
    entry.name: entry
    for entry in [
        _mvn_10,
        _neals_funnel,
        _eight_schools_ncp,
    ]
}
