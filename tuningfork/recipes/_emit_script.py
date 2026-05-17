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
"""Emit a recipe-reproduction script from a Recipe.

This is the entry point for recipe portability (Principle F): given a Recipe,
emit a Python script that reproduces the recipe's inference with the wiring
code visible inline.

Templates live in ``_templates/`` and use string.Template ($slot) substitution
because Python code contains curly braces that conflict with str.format.

Design decisions
----------------
- **D8 STRICT (clarified 2026-05-17 post R3.5-MVP)**: the **inference
  choreography** (warmup + sampler + inference loop) has zero ``import
  tuningfork`` — it's auditable in one file and shows the exact BlackJAX
  call shape.  The **model** is imported via ``from tuningfork.model import
  MODELS`` (canonical NumPyro code lives upstream; not duplicated here).
  This avoids template-drift risk on the largest, most-stable code surface
  while preserving the design-smell forcing function on the actual wiring
  layer (per Principle A — heavy sampler/warmup template = upstream BlackJAX
  design issue).
- **D9**: pure function — returns a string; no side effects.  The caller
  writes to whatever path they want.
- **D10**: hand-written templates + round-trip CI gate in
  ``tests/recipes/test_emit_script.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tuningfork.recipes._base import Recipe


__all__ = ["emit_script"]

_TEMPLATES_DIR = Path(__file__).parent / "_templates"


def _load_template(relpath: str) -> Template:
    """Load a .py.tmpl file as a string.Template."""
    return Template((_TEMPLATES_DIR / relpath).read_text())


def _recipe_hash(recipe: Recipe) -> str:
    """SHA-1 of the canonical recipe JSON; first 12 chars."""
    payload = json.dumps(
        {
            "model_name": recipe.model_name,
            "base_method_name": recipe.base_method_name,
            "warmup_name": recipe.warmup_name,
            "effort": recipe.effort.value,
            "base_method_params": recipe.base_method_params,
            "warmup_params": recipe.warmup_params,
            "tuning_seed": recipe.tuning_seed,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def emit_script(
    recipe: Recipe,
    *,
    num_samples: int = 2000,
    sampler_seed: int | None = None,
) -> str:
    """Assemble a recipe-reproduction Python script.

    Per locked decision D8 (STRICT inference, 2026-05-17 clarification),
    the emitted script's **inference choreography** (warmup + sampler +
    inference loop) has zero ``import tuningfork`` and is auditable inline.
    The **model definition** is imported via ``from tuningfork.model import
    MODELS`` — canonical NumPyro code lives upstream, not duplicated as
    a per-model template.

    Parameters
    ----------
    recipe : Recipe
        The recipe to emit. Loaded via :func:`tuningfork.catalog.load_recipe`.
    num_samples : int
        Number of post-warmup samples to draw in the emitted inference loop.
        Defaults to 2000.
    sampler_seed : int, optional
        RNG seed for the post-warmup sampling. Defaults to
        ``recipe.tuning_seed + 1`` so the emitted script is deterministic
        given the recipe.

    Returns
    -------
    str
        The full Python script content. The function is pure — no side effects.
        The caller writes the returned string to whatever path they want
        (per locked decision D9).

    Raises
    ------
    FileNotFoundError
        If a required template is missing for the given
        ``(model_name, warmup_name, base_method_name)`` combo.
    KeyError
        If the recipe's ``warmup_params`` or ``base_method_params`` lack a
        required slot for the template.
    """
    if sampler_seed is None:
        sampler_seed = recipe.tuning_seed + 1

    # Normalise warmup_params key spelling: groundtruth recipes use
    # "target_acceptance" (legacy key from certify_reference.py);
    # newer recipe-generation code uses "target_acceptance_rate".
    target_acceptance_rate = recipe.warmup_params.get(
        "target_acceptance_rate",
        recipe.warmup_params.get("target_acceptance", 0.8),
    )

    # Substitution context — every $slot the templates reference must be here.
    #
    # Prefix convention (Option A — programmatic spread, R3.5b):
    #   bm_<key>  — from recipe.base_method_params  (e.g. $bm_step_size, $bm_num_integration_steps)
    #   wp_<key>  — from recipe.warmup_params        (e.g. $wp_n_warmup, $wp_target_acceptance_rate)
    #
    # These prefixed slots are in addition to the hand-unrolled top-level slots
    # (which remain for backward compatibility with existing templates).
    # New templates should prefer the prefixed $bm_* / $wp_* form so the context
    # auto-expands when new hyperparameter fields are added to recipes.
    ctx = {
        "recipe_id": (
            f"{recipe.model_name}/{recipe.effort.value}"
            f"__{recipe.base_method_name}__{recipe.warmup_name}"
        ),
        "model_name": recipe.model_name,
        "base_method_name": recipe.base_method_name,
        "warmup_name": recipe.warmup_name,
        "effort": recipe.effort.value,
        "recipe_hash": _recipe_hash(recipe),
        "verdict": recipe.gate_evidence.get("auto", {}).get("verdict", "NOT_RUN"),
        "tuning_seed": recipe.tuning_seed,
        "sampler_seed": sampler_seed,
        "num_samples": num_samples,
        # warmup_params unrolled (legacy top-level slots — backward compat)
        "target_acceptance_rate": target_acceptance_rate,
        "n_warmup": recipe.warmup_params.get("n_warmup", 1000),
        # base_method_params unrolled (legacy top-level slots — backward compat)
        "max_num_doublings": recipe.base_method_params.get("max_num_doublings", 10),
    }
    # Programmatic spread: bm_<key> from base_method_params, wp_<key> from warmup_params.
    # Values are JSON-serialised scalar types (int/float/list); templates that need
    # them reference $bm_step_size, $bm_num_integration_steps, $wp_n_warmup, etc.
    ctx.update({f"bm_{k}": v for k, v in recipe.base_method_params.items()})
    ctx.update({f"wp_{k}": v for k, v in recipe.warmup_params.items()})

    # Use safe_substitute so templates with optional $bm_*/wp_* slots that are
    # absent from the recipe (e.g. $bm_num_integration_steps in a nuts recipe)
    # leave the slot as a literal dollar-prefixed string rather than raising
    # KeyError.  Each template is responsible for using only the slots that
    # actually exist for its algorithm family.
    preamble = _load_template("preamble.py.tmpl").safe_substitute(ctx)
    warmup_body = _load_template(
        f"warmups/{recipe.warmup_name}.py.tmpl"
    ).safe_substitute(ctx)
    sampler_body = _load_template(
        f"samplers/{recipe.base_method_name}.py.tmpl"
    ).safe_substitute(ctx)
    inference_loop = _load_template("inference_loop.py.tmpl").safe_substitute(ctx)
    postamble = _load_template("postamble.py.tmpl").safe_substitute(ctx)

    # Model definition is imported from tuningfork.model in the preamble;
    # no separate model template assembled here (post R3.5-MVP clarification).
    return "\n\n".join([preamble, warmup_body, sampler_body, inference_loop, postamble])
