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

Each template formats a copy-pasteable usage snippet from a ``Recipe``'s
pinned fields.  All three effort tiers produce the same kind of artifact
(pinned ``base_method_params`` + optional IMM sidecar) — what differs across
tiers is the *human + machine wall time to produce a gate-passing recipe*,
not how the user consumes it.  The prose framing reflects that:

  LOW    — conventional ``(warmup, sampler)`` pairing; library defaults passed
           the Statistician auto-gate at first emit.  Machine-only wall time.
  MEDIUM — Statistician investigation needed.  Either the default emit failed
           the gate (manual workarounds: seed change, init change, obvious-bug
           fix) or the cell explores a technically-possible-but-unconventional
           pairing (e.g., ``window_adaptation_diag_imm`` + ``mala``, ``window_adaptation_diag_imm`` + ``rmhmc``).
           Wall time = LOW + Statistician investigation.
  HIGH   — LOW and MEDIUM both failed.  The Statistician ran Bayesian workflow,
           declared warmup and sampler parameter resolution, and injected model-specific
           parameters until the gate passed. The full journey is recorded in
           ``workflow``. CI executes the generated program with the pinned
           scalars and IMM sidecar; whether warmup runs follows recipe intent.
           Wall time = MEDIUM + extra Statistician work + generated-plan evaluation.

The same "use the pinned config" snippet applies to all tiers; the tier-specific
prose explains what production effort produced those values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tuningfork.recipes._base import Effort

if TYPE_CHECKING:
    from tuningfork.recipes._base import Recipe

__all__ = ["render_instructions"]

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_LOW_TEMPLATE = """\
**Low-effort recipe** (conventional `({warmup_name}, {base_method_name})` pairing; \
library defaults passed the auto-gate at first emit).
To reproduce, emit and execute the generated program for this recipe. It
applies the resolved parameters and runs warmup only when the recipe intent
requires it.
Expected `min-bulk-ESS / total_grad_evals`: {headline_metric}.
Wall time to produce this recipe: ~{wall_seconds} s (machine-only).\
"""

_MEDIUM_TEMPLATE = """\
**Medium-effort recipe** (Statistician investigation on \
`({warmup_name}, {base_method_name})`).
The default emit either failed the auto-gate or explored a technically-possible-but-\
unconventional pairing; the Statistician applied workarounds until the gate passed. \
See `notes` for the specific intervention.
To reproduce, emit and execute the generated program for this recipe. It
applies the resolved parameters and runs warmup only when the recipe intent
requires it.
Expected `min-bulk-ESS / total_grad_evals`: {headline_metric}.
Wall time: machine + Statistician investigation (see `calibration_budget`).\
"""

_HIGH_TEMPLATE = """\
**High-effort recipe** (Bayesian workflow + generated plan + model-specific injection on \
`({warmup_name}, {base_method_name})`).
After LOW and MEDIUM both failed the auto-gate, the Statistician ran Bayesian \
workflow, resolved declared warmup and sampler parameters, and injected model-specific \
parameters. The full journey is recorded in `workflow`; CI consumes the pinned \
scalars below.
To reproduce, emit and execute the generated program for this recipe. It uses
the pinned scalars and IMM sidecar, and runs warmup only when the recipe intent
requires it.
Expected `min-bulk-ESS / total_grad_evals`: {headline_metric}.
Total calibration time: ~{calibration_minutes} min (warmup + generated plan + Statistician).\
"""

_GROUNDTRUTH_TEMPLATE = """\
**Ground-truth reference recipe** for `{model_name}`. Sampled via a single long NUTS \
chain (1×{n_samples} post-warmup, {n_warmup}-step Stan window adaptation, \
target_acceptance={target_acceptance}). Certified to split-R̂ ≤ 1.01, min per-chunk \
bulk-ESS ≥ 400, 0 divergences, E-BFMI ≥ 0.3 across {n_chunks} contiguous chunks.

The samples themselves live at `tuningfork/reference/{model_name}/draws.npz` \
(gitignored — {n_samples} samples × dim={dim}). Load via:

    from tuningfork._cache_io import get_reference_draws
    from tuningfork.model import MODELS
    draws = get_reference_draws(MODELS['{model_name}'])

Or, via the diagnostics notebook, set RECIPE_PATH to this recipe and the \
notebook's load-or-run path will short-circuit to the cache.\
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
    All three tiers run MCMC at recipe-build time, so ``headline_metric`` is
    expected to be a float for any gate-passing recipe.  When ``headline_metric``
    is ``None`` (e.g., during scaffolding or for an in-flight recipe), the
    template renders ``"not yet measured"`` rather than failing the format —
    a visible signal of incomplete data.
    """
    effort = recipe.effort

    headline = (
        f"{recipe.headline_metric:.4f}"
        if recipe.headline_metric is not None
        else "not yet measured"
    )

    if effort == Effort.LOW:
        return _LOW_TEMPLATE.format(
            base_method_name=recipe.base_method_name,
            base_method_params=recipe.base_method_params,
            warmup_name=recipe.warmup_name,
            headline_metric=headline,
            wall_seconds=int(recipe.calibration_budget.get("wall_seconds_estimate", 0)),
        )

    if effort == Effort.MEDIUM:
        return _MEDIUM_TEMPLATE.format(
            base_method_name=recipe.base_method_name,
            base_method_params=recipe.base_method_params,
            warmup_name=recipe.warmup_name,
            headline_metric=headline,
        )

    if effort == Effort.HIGH:
        return _HIGH_TEMPLATE.format(
            base_method_name=recipe.base_method_name,
            base_method_params=recipe.base_method_params,
            warmup_name=recipe.warmup_name,
            headline_metric=headline,
            calibration_minutes=int(
                recipe.calibration_budget.get("wall_seconds_estimate", 0) / 60
            ),
        )

    if effort == Effort.GROUNDTRUTH:
        n_warmup = recipe.calibration_budget.get("n_warmup", "?")
        n_samples = recipe.calibration_budget.get("n_samples", "?")
        n_chunks = recipe.warmup_params.get("n_chunks", "?")
        target_acceptance = recipe.warmup_params.get("target_acceptance", "?")
        # Attempt to look up dim from the model registry; fall back to "?"
        try:
            from tuningfork.model import MODELS

            dim = MODELS[recipe.model_name].dim if recipe.model_name in MODELS else "?"
        except Exception:  # noqa: BLE001
            dim = "?"
        return _GROUNDTRUTH_TEMPLATE.format(
            model_name=recipe.model_name,
            n_samples=n_samples,
            n_warmup=n_warmup,
            target_acceptance=target_acceptance,
            n_chunks=n_chunks,
            dim=dim,
        )

    # Unreachable if Effort enum is exhaustive — defensive fallback.
    return f"[No template for effort={effort!r}]"
