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
"""Emit-time Python function for the recipe preamble section.

Replaces ``_templates/preamble.py.tmpl`` (43 LOC, string.Template).
All slot resolution is done in Python — no $slot markers in the output.
D8 compliant: emitted string imports only tuningfork.model (allowed).
"""

from __future__ import annotations

from typing import Any


def emit_preamble(ctx: dict[str, Any]) -> str:
    """Emit the preamble section of the recipe reproduction script.

    Parameters
    ----------
    ctx : dict
        Substitution context from ``emit_script()``.
        Required keys: recipe_id, model_name, base_method_name, warmup_name,
        plan_hash, execution_manifest_json, effort, verdict, x64_config_line,
        tuning_seed, num_chains.

    Returns
    -------
    str
        Python source for the preamble block.
    """
    lines: list[str] = []
    a = lines.append

    # Module docstring
    a('"""Auto-generated recipe reproduction script.')
    a("")
    a(
        f"Source recipe: {ctx['recipe_id']} (model={ctx['model_name']}, sampler={ctx['base_method_name']}, warmup={ctx['warmup_name']})"
    )
    a(f"Execution plan hash: {ctx['plan_hash']}")
    a(f"Effort:        {ctx['effort']}")
    a(f"Verdict:       {ctx['verdict']} (expected; pinned at recipe-emission time)")
    a("")
    a("The model definition is imported from ``tuningfork.model`` (canonical NumPyro")
    a("code, no template-drift risk). The inference choreography (warmup + sampler +")
    a("loop) is hand-written in this script with no tuningfork imports -- so the")
    a("emitted choreography is auditable in one file, while the model code stays")
    a("single-sourced upstream.")
    a("")
    a("Run with::")
    a("")
    a("    uv run --with tuningfork --with jax --with blackjax --with numpyro \\\\")
    a("        python <this_script>.py")
    a('"""')
    a("")
    a(f"EXECUTION_MANIFEST_JSON = {ctx['execution_manifest_json']!r}")

    # Timing + imports
    a("import time as _recipe_time")
    a("_recipe_t0 = _recipe_time.perf_counter()")
    a("import jax")

    # x64 config (empty string for float32 models)
    if ctx["x64_config_line"]:
        a(ctx["x64_config_line"])

    a("import jax.numpy as jnp")
    a("import numpy as np")
    a("import blackjax")

    # Model import
    a("# === MODEL: imported from tuningfork (canonical NumPyro definition) ===")
    a("from tuningfork.model import MODELS")
    a("from tuningfork.model._numpyro import build_logdensity_fn")
    a("")
    a(f'posterior = MODELS["{ctx["model_name"]}"]')
    a(f"_init_key = jax.random.key({ctx['tuning_seed']})")
    a("init_position, logdensity_fn, _ = build_logdensity_fn(")
    a("    _init_key, posterior")
    a(")")
    a("")
    a("# Number of parallel chains for the vmap-scan inference loop.")
    a("# Derived from recipe metadata (warmup_params.num_chains or")
    a("# calibration_budget.num_chains); 4 for legacy non-groundtruth recipes.")
    a(f"num_chains = {ctx['num_chains']}")
    a("")
    a("# Wall-clock timer for warmup phase (reset after model/init setup).")
    a("_warmup_t0 = _recipe_time.perf_counter()")

    return "\n".join(lines)
