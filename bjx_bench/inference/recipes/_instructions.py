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
"""Auto-templated user-facing instructions for each Effort level.

Each template is a format string whose placeholders are filled from
``Recipe`` fields by ``render_instructions``.  Templates aim to give a
copy-pasteable snippet (LOW: direct kernel construction; MEDIUM: warmup +
kernel; HIGH: CLI reproduction command) plus a one-line expectation and a
"when to use" note.

LOW recipe note on ``headline_metric``: at LOW effort no MCMC is run, so the
metric is ``None``.  The LOW template avoids formatting it as a float and
instead shows ``"not measured (zero-calibration recipe)"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bjx_bench.inference.recipes._base import Effort

if TYPE_CHECKING:
    from bjx_bench.inference.recipes._base import Recipe

__all__ = ["render_instructions"]

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_LOW_TEMPLATE = """\
**Low-effort recipe** (zero calibration). To use:
  ```python
  kernel = blackjax.{base_method_name}(logdensity_fn, **{base_method_params})
  ```
Expected `min-bulk-ESS / total_grad_evals`: not measured (zero-calibration recipe).
When to use: one-off analysis, exploratory work, prototyping. \
No warmup required — these are the geometric-mean defaults from the search space.\
"""

_MEDIUM_TEMPLATE = """\
**Medium-effort recipe** ({warmup_name} adaptation). To use:
  ```python
  warmup = blackjax.window_adaptation(blackjax.{base_method_name}, \
logdensity_fn, **{warmup_params})
  (state, params), _ = warmup.run(rng_key, init_position, {n_warmup})
  kernel = blackjax.{base_method_name}(logdensity_fn, **params)
  ```
Expected `min-bulk-ESS / total_grad_evals`: {headline_metric}.
Warmup wall time: ~{warmup_seconds} s.\
"""

_HIGH_TEMPLATE = """\
**High-effort recipe** (Tier-B BO-tuned). To reproduce:
  ```bash
  bjx-bench tune {model_name} {base_method_name} \
--n-trials {n_trials} --seed {tuning_seed}
  ```
Pinned config from {n_trials} trials, seed {tuning_seed}:
  base_method_params: {base_method_params}
  warmup_params:      {warmup_params}
Expected `min-bulk-ESS / total_grad_evals`: {headline_metric}.
Total calibration time: ~{calibration_minutes} min.\
"""


def render_instructions(recipe: Recipe) -> str:
    """Render the per-effort prose template for a recipe.

    Parameters
    ----------
    recipe
        A ``Recipe`` instance.  The ``effort`` field selects the template;
        other fields supply the format arguments.

    Returns
    -------
    str
        Non-empty prose suitable for storing in ``Recipe.instructions``.

    Notes
    -----
    For ``Effort.LOW``, ``headline_metric`` is ``None`` (no MCMC run).  The
    LOW template avoids the ``.4f`` format specifier and prints a fixed
    placeholder string instead.

    For ``Effort.MEDIUM`` and ``Effort.HIGH``, the template is rendered with
    the actual metric values.  If ``headline_metric`` is still ``None`` in
    those cases (which should not happen in a correctly constructed recipe),
    the string ``"None"`` appears — a visible signal of incomplete data rather
    than a silent formatting error.
    """
    effort = recipe.effort

    if effort == Effort.LOW:
        return _LOW_TEMPLATE.format(
            base_method_name=recipe.base_method_name,
            base_method_params=recipe.base_method_params,
        )

    if effort == Effort.MEDIUM:
        headline = (
            f"{recipe.headline_metric:.4f}"
            if recipe.headline_metric is not None
            else "None"
        )
        return _MEDIUM_TEMPLATE.format(
            base_method_name=recipe.base_method_name,
            warmup_name=recipe.warmup_name,
            warmup_params=recipe.warmup_params,
            n_warmup=recipe.warmup_params.get("n_warmup", "?"),
            headline_metric=headline,
            warmup_seconds=int(
                recipe.calibration_budget.get("wall_seconds_estimate", 0)
            ),
        )

    if effort == Effort.HIGH:
        headline = (
            f"{recipe.headline_metric:.4f}"
            if recipe.headline_metric is not None
            else "None"
        )
        return _HIGH_TEMPLATE.format(
            model_name=recipe.model_name,
            base_method_name=recipe.base_method_name,
            base_method_params=recipe.base_method_params,
            warmup_params=recipe.warmup_params,
            n_trials=recipe.calibration_budget.get("trials", "?"),
            tuning_seed=recipe.tuning_seed,
            headline_metric=headline,
            calibration_minutes=int(
                recipe.calibration_budget.get("wall_seconds_estimate", 0) / 60
            ),
        )

    # Unreachable if Effort enum is exhaustive — defensive fallback.
    return f"[No template for effort={effort!r}]"
