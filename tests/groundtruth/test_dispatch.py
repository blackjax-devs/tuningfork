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
"""Tests for _dispatch: all 16 committed models resolve to the correct GTMethod.

These are pure schema-parsing tests — no JAX, no model imports.
"""

import pytest

from tuningfork.groundtruth._dispatch import (
    GTMethod,
    _resolve_gt_method,
    committed_gt_dir,
    load_committed_summary,
)

pytestmark = pytest.mark.fast

# Expected dispatch for every committed model.  Update this table if the catalog
# changes (new models, updated generator strings).
_EXPECTED: dict[str, GTMethod] = {
    # analytic i.i.d. — exact sampler, no NUTS
    "banana": GTMethod.ANALYTIC_IID,
    "gmm_25": GTMethod.ANALYTIC_IID,
    "ill_cond_50": GTMethod.ANALYTIC_IID,
    "mvn_10": GTMethod.ANALYTIC_IID,
    "neals_funnel": GTMethod.ANALYTIC_IID,
    # standard multichain NUTS — init_to_uniform_radius2 / dispersed-Gaussian
    "eight_schools_ncp": GTMethod.STANDARD_MULTICHAIN_NUTS,
    "german_credit": GTMethod.STANDARD_MULTICHAIN_NUTS,
    "horseshoe": GTMethod.STANDARD_MULTICHAIN_NUTS,
    "irt_1pl": GTMethod.STANDARD_MULTICHAIN_NUTS,
    "irt_2pl": GTMethod.STANDARD_MULTICHAIN_NUTS,
    "lgcp": GTMethod.STANDARD_MULTICHAIN_NUTS,
    "logistic_synthetic": GTMethod.STANDARD_MULTICHAIN_NUTS,
    "radon": GTMethod.STANDARD_MULTICHAIN_NUTS,
    "stoch_vol": GTMethod.STANDARD_MULTICHAIN_NUTS,
    # explicit per-chain starting positions embedded in provenance
    "lotka_volterra": GTMethod.EXPLICIT_POSITIONS,
    # closed-form GP marginal + conditional-f reconstruction
    "gp_regression": GTMethod.CLOSED_FORM_GP_MARGINAL,
}


@pytest.mark.parametrize("model_name,expected", list(_EXPECTED.items()))
def test_resolve_gt_method_all_models(model_name: str, expected: GTMethod) -> None:
    """All 16 committed models dispatch to the correct GTMethod."""
    summary = load_committed_summary(model_name)
    resolved = _resolve_gt_method(summary)
    assert resolved is expected, (
        f"Model {model_name!r}: expected {expected.value!r}, got {resolved.value!r}. "
        f"generator field = {summary.get('generator')!r}, "
        f"has init_positions = {'init_positions' in summary.get('provenance', {})}"
    )


def test_committed_gt_dir_returns_valid_path() -> None:
    """committed_gt_dir returns a real directory for every model."""
    for model_name in _EXPECTED:
        gt_dir = committed_gt_dir(model_name)
        assert gt_dir.is_dir(), f"GT dir missing for {model_name!r}: {gt_dir}"
        assert (
            gt_dir / "summary_v2.json"
        ).exists(), f"summary_v2.json missing for {model_name!r}"
        assert (gt_dir / "draws.npz").exists(), f"draws.npz missing for {model_name!r}"


def test_load_committed_summary_schema_version() -> None:
    """All committed summaries have schema_version='gt_v2_multichain'."""
    for model_name in _EXPECTED:
        summary = load_committed_summary(model_name)
        assert (
            summary["schema_version"] == "gt_v2_multichain"
        ), f"Model {model_name!r}: unexpected schema_version {summary['schema_version']!r}"


def test_resolve_gt_method_unknown_generator_raises() -> None:
    """_resolve_gt_method raises ValueError for an unknown generator string."""
    bad_summary = {
        "schema_version": "gt_v2_multichain",
        "generator": "totally_unknown_generator",
        "provenance": {},
    }
    with pytest.raises(ValueError, match="Unknown generator string"):
        _resolve_gt_method(bad_summary)


def test_nuts_perchain_without_init_positions_is_standard() -> None:
    """nuts_perchain without provenance.init_positions → STANDARD_MULTICHAIN_NUTS."""
    summary = {
        "schema_version": "gt_v2_multichain",
        "generator": "nuts_perchain",
        "provenance": {},
    }
    assert _resolve_gt_method(summary) is GTMethod.STANDARD_MULTICHAIN_NUTS


def test_nuts_perchain_with_init_positions_is_explicit() -> None:
    """nuts_perchain with provenance.init_positions → EXPLICIT_POSITIONS."""
    summary = {
        "schema_version": "gt_v2_multichain",
        "generator": "nuts_perchain",
        "provenance": {"init_positions": {"positions": {}}},
    }
    assert _resolve_gt_method(summary) is GTMethod.EXPLICIT_POSITIONS
