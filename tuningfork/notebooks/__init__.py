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

Quick start (5 lines in a Jupyter cell)::

    from tuningfork.notebooks import load_recipe, summarize_recipe, load_samples, samples_to_idata
    import arviz as az

    recipe = load_recipe("tuningfork/inference/recipes/starter/eight_schools_ncp/groundtruth__nuts__stan_window.json")
    samples = load_samples(recipe)
    idata = samples_to_idata(samples)
    az.plot_trace(idata); az.summary(idata)

API
---
load_recipe(path)
    Load a Recipe from a path. Resolves relative paths against the tuningfork repo root.
summarize_recipe(recipe)
    Return a 2-column (Property, Value) DataFrame. Auto-renders inline in Jupyter.
load_samples(recipe)
    Load cached samples for the recipe's model. Raises FileNotFoundError on cache miss.
samples_to_idata(samples)
    Convert a samples dict to ``arviz.InferenceData``.
"""

from tuningfork.notebooks.inspect import load_recipe, summarize_recipe
from tuningfork.notebooks.render import load_samples, samples_to_idata

__all__ = [
    "load_recipe",
    "summarize_recipe",
    "load_samples",
    "samples_to_idata",
]
