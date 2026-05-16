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
"""tuningfork.notebooks — User-facing recipe inspection helpers.

Designed for Jupyter Lab use. Statistician-friendly: minimal wrapper code,
ArviZ-direct workflow.

Recommended quick-start (4 lines in a Jupyter cell)::

    from tuningfork.notebooks import load_recipe, load_idata
    import arviz as az

    recipe = load_recipe("tuningfork/inference/recipes/starter/eight_schools_ncp/groundtruth__nuts__stan_window.json")
    idata = load_idata(recipe)
    az.plot_trace(idata); az.plot_energy(idata); az.summary(idata)

``load_idata`` bundles ``load_samples`` + ``load_chain_stats`` +
``samples_to_idata`` — the resulting InferenceData carries both the posterior
group AND a ``sample_stats`` group with diverging/energy/acceptance_rate/n_steps
mapped to ArviZ's canonical names. This means ``az.plot_energy(idata)`` and
``az.plot_pair(idata, divergences=True)`` work out of the box.

API
---
load_recipe(path)
    Load a Recipe from a path. Resolves relative paths against the tuningfork repo root.
summarize_recipe(recipe)
    Return a 2-column (Property, Value) DataFrame. Auto-renders inline in Jupyter.
load_idata(recipe)
    One-call: returns ``arviz.InferenceData`` with posterior + sample_stats.
load_samples(recipe)
    Advanced: returns the raw dict[str, jax.Array] of draws (no ArviZ wrap).
load_chain_stats(recipe)
    Advanced: returns the raw chain_stats dict (no ArviZ wrap). None on miss.
samples_to_idata(samples, chain_stats=None)
    Manual conversion. Pass ``chain_stats`` to populate sample_stats.
"""

from tuningfork.notebooks.inspect import load_recipe, summarize_recipe
from tuningfork.notebooks.render import (
    load_chain_stats,
    load_idata,
    load_samples,
    samples_to_idata,
)

__all__ = [
    "load_recipe",
    "summarize_recipe",
    "load_samples",
    "load_chain_stats",
    "load_idata",
    "samples_to_idata",
]
