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
"""User-facing API for the tuningfork catalog (recipes + cached artifacts).

This subpackage is the consumer-side surface. Users typically do:

    from tuningfork.catalog import load_recipe, load_idata, summarize_recipe

The generator-side wiring (``model/``, ``base_method/``, ``warmup/``, ``smc/``,
``recipes/``, ``calibration/``, ``metrics/``, ``runner/``) lives outside this
subpackage and is used to PRODUCE the recipes the catalog SERVES.

Per-model artifacts live under ``tuningfork/catalog/<model>/`` (post R2,
2026-05-17):

- ``groundtruth.json`` — canonical groundtruth recipe pin
- ``groundtruth.imm.npz`` — high-dim IMM sidecar (where applicable)
- ``reference/{metadata,summary,adaptation,xcheck}.json`` — committed cert artifacts
- ``recipes/{low,medium,high,failed}__<sampler>__<warmup>.json`` — per-cell recipes
- ``_cache/{draws,chain_stats}.npz`` — gitignored runtime cache
"""

from tuningfork.catalog._rerun_inference import cached_idata_for_recipe
from tuningfork.catalog._timing import compute_total_warmup_steps, format_timing_context
from tuningfork.catalog.diagnostics import (
    plot_recipe_diagnostics,
    render_gradient_mh,
    render_mclmc_family,
    render_smc_family,
    render_specialised,
    render_universal_summary,
    render_vi_family,
    samples_to_idata,
)
from tuningfork.catalog.emit import emit_script
from tuningfork.catalog.inspect import list_recipes, load_recipe, summarize_recipe
from tuningfork.catalog.render import load_chain_stats, load_idata, load_samples

__all__ = [
    # inspect
    "load_recipe",
    "summarize_recipe",
    "list_recipes",
    # render
    "load_samples",
    "load_chain_stats",
    "load_idata",
    "cached_idata_for_recipe",
    "samples_to_idata",
    # diagnostics
    "render_universal_summary",
    "render_gradient_mh",
    "render_mclmc_family",
    "render_smc_family",
    "render_vi_family",
    "render_specialised",
    "plot_recipe_diagnostics",
    # timing
    "compute_total_warmup_steps",
    "format_timing_context",
    # emit
    "emit_script",
]
