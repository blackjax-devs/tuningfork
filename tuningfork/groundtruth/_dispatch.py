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
"""Dispatch: map committed summary_v2.json to a generation method.

The single source of truth for which algorithm is used to regenerate each
model's ground-truth draws. ``_resolve_gt_method`` reads the ``generator``
string already embedded in every model's committed ``summary_v2.json`` and
maps it to a ``GTMethod`` enum value.

All 16 committed models are covered:

    analytic_iid      — banana, gmm_25, ill_cond_50, mvn_10, neals_funnel
    standard_multichain_nuts — eight_schools_ncp, german_credit, horseshoe,
                               irt_1pl, irt_2pl, lgcp, logistic_synthetic,
                               radon, stoch_vol
    explicit_positions — lotka_volterra
    closed_form_gp_marginal — gp_regression
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

__all__ = [
    "GTMethod",
    "committed_gt_dir",
    "load_committed_summary",
    "_resolve_gt_method",
]


class GTMethod(str, Enum):
    """How the ground-truth draws for a given model are produced."""

    ANALYTIC_IID = "analytic_iid"
    STANDARD_MULTICHAIN_NUTS = "standard_multichain_nuts"
    EXPLICIT_POSITIONS = "explicit_positions"
    CLOSED_FORM_GP_MARGINAL = "closed_form_gp_marginal"


# Mapping from the ``generator`` field in committed summary_v2.json files to
# the primary GTMethod.  The STANDARD_MULTICHAIN_NUTS entry is refined to
# EXPLICIT_POSITIONS in ``_resolve_gt_method`` when the provenance block
# contains ``init_positions`` (currently only lotka_volterra).
_GENERATOR_TO_METHOD: dict[str, GTMethod] = {
    "analytic_iid": GTMethod.ANALYTIC_IID,
    "nuts_perchain": GTMethod.STANDARD_MULTICHAIN_NUTS,
    "nuts_on_closed_form_gp_marginal_plus_conditional_f_reconstruction": GTMethod.CLOSED_FORM_GP_MARGINAL,
}


def _catalog_dir() -> Path:
    """Return the path to ``tuningfork/catalog/`` inside the installed package."""
    import tuningfork as _tf

    return Path(_tf.__file__).parent / "catalog"


def committed_gt_dir(model_name: str) -> Path:
    """Return ``tuningfork/catalog/<model>/groundtruth_samples/blackjax/``.

    Raises
    ------
    FileNotFoundError
        When the directory doesn't exist (model not in the catalog).
    """
    path = _catalog_dir() / model_name / "groundtruth_samples" / "blackjax"
    if not path.is_dir():
        raise FileNotFoundError(
            f"No committed GT artifacts found for model {model_name!r}. "
            f"Expected directory: {path}"
        )
    return path


def load_committed_summary(model_name: str) -> dict:
    """Load the committed ``summary_v2.json`` for ``model_name``.

    Returns
    -------
    dict
        Parsed JSON. Schema version is always ``"gt_v2_multichain"``.

    Raises
    ------
    FileNotFoundError
        When the directory or summary file doesn't exist.
    ValueError
        When the file exists but the schema version is not ``"gt_v2_multichain"``.
    """
    summary_path = committed_gt_dir(model_name) / "summary_v2.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"summary_v2.json not found for model {model_name!r}: {summary_path}"
        )
    with open(summary_path) as fh:
        summary = json.load(fh)
    schema = summary.get("schema_version", "")
    if schema != "gt_v2_multichain":
        raise ValueError(
            f"Unsupported schema_version {schema!r} for model {model_name!r}. "
            "Expected 'gt_v2_multichain'."
        )
    return summary


def _resolve_gt_method(summary: dict) -> GTMethod:
    """Resolve the generation method from a committed summary_v2.json.

    Uses the ``generator`` field as the primary dispatch key.  For
    ``nuts_perchain`` models, checks ``provenance.init_positions`` presence to
    distinguish :attr:`~GTMethod.EXPLICIT_POSITIONS` (currently only
    ``lotka_volterra``) from :attr:`~GTMethod.STANDARD_MULTICHAIN_NUTS`.

    Returns
    -------
    GTMethod
        The generation method for this model.

    Raises
    ------
    ValueError
        When the ``generator`` field doesn't match any known method.
    """
    generator = summary.get("generator", "")
    if generator not in _GENERATOR_TO_METHOD:
        known = list(_GENERATOR_TO_METHOD)
        raise ValueError(
            f"Unknown generator string {generator!r} in summary_v2.json. "
            f"Known values: {known}"
        )
    method = _GENERATOR_TO_METHOD[generator]
    if method is GTMethod.STANDARD_MULTICHAIN_NUTS:
        provenance = summary.get("provenance", {})
        if "init_positions" in provenance:
            method = GTMethod.EXPLICIT_POSITIONS
    return method
