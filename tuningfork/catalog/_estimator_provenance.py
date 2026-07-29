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
"""Which ESS estimator stands behind a catalog headline.

The catalog convention is the rank-normalised split-chain bulk-ESS
(``blackjax.diagnostics.ess_bulk``), recorded per recipe in
``headline_basis["ess_estimator"]`` — see ``catalog/RECIPE_SCHEMA.md`` §4.5.

A model listed in :data:`HEADLINE_ESTIMATOR_EXCLUDED_MODELS` keeps headline
numbers produced by the older estimator, because re-measuring it costs more than
the comparability is worth.  Its headlines are therefore **not comparable to the
rest of the corpus**, which matters precisely because cross-model comparison is
what the headline is for.

The exclusion lives here rather than in a test so there is one list, readable by
the invariant tests and by anyone inspecting a recipe.  It is deliberately NOT
recorded inside the recipe JSON: recipe artifacts are written only by the emit
harness, and hand-editing one to add a marker would defeat the provenance the
marker is supposed to carry.
"""

from __future__ import annotations

from tuningfork.metrics.headline import HEADLINE_ESS_ESTIMATOR, LEGACY_ESS_ESTIMATOR

__all__ = [
    "HEADLINE_ESTIMATOR_EXCLUDED_MODELS",
    "exclusion_reason",
    "headline_estimator_of",
]

#: Models whose headlines stay on the older, non-rank-normalised estimator.
#: Keyed by model name; the value is the reason, which a reader needs in order to
#: tell "excluded deliberately on cost grounds" from "accidentally missed".
#: Adding an entry means accepting that the model's headline cannot be compared
#: to any other model's — do not add one without recording why here.
HEADLINE_ESTIMATOR_EXCLUDED_MODELS: dict[str, str] = {
    "gp_regression": (
        "Excluded on compute cost. The dense 200x200 RBF kernel makes every "
        "step roughly 50x more expensive than a peer model: its reference "
        "certification wall is about 63 hours, and its only headline-carrying "
        "recipe is HIGH effort, so a faithful re-measurement means re-running "
        "the hyperparameter search rather than a single sampler run "
        "(the sampler run alone projects to about 80 minutes). Its headline "
        "numbers remain on the older estimator and are NOT comparable to the "
        "rest of the catalog."
    ),
}


def exclusion_reason(model_name: str) -> str | None:
    """Why this model's headlines stay on the older estimator, or ``None``."""
    return HEADLINE_ESTIMATOR_EXCLUDED_MODELS.get(model_name)


def headline_estimator_of(recipe) -> tuple[str, str | None]:
    """Return ``(estimator_name, caveat)`` for a loaded recipe.

    ``caveat`` is ``None`` when the headline is on the catalog convention and
    comparable to every other recipe.  It is a sentence explaining the problem
    otherwise — either the model is excluded from the migration, or the recipe
    predates the provenance stamp and its estimator cannot be read off the
    artifact at all.

    Parameters
    ----------
    recipe
        A loaded ``Recipe``.  ``SMCRecipe`` is out of scope: its headline is an
        importance-weight ESS, not an autocorrelation one.

    Returns
    -------
    tuple of (str, str | None)
    """
    reason = exclusion_reason(getattr(recipe, "model_name", ""))
    if reason is not None:
        return LEGACY_ESS_ESTIMATOR, reason

    declared = (getattr(recipe, "headline_basis", None) or {}).get("ess_estimator")
    if declared is None:
        return (
            "unrecorded",
            "This recipe predates the estimator provenance stamp, so which ESS "
            "estimator produced its headline cannot be read off the artifact. "
            "Re-emit it to record one.",
        )
    if declared != HEADLINE_ESS_ESTIMATOR:
        return declared, (
            f"Headline is on {declared!r}, not the catalog convention "
            f"{HEADLINE_ESS_ESTIMATOR!r}; it is not comparable to other recipes."
        )
    return declared, None
