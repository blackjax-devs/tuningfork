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

"""Focused losslessness checks for SMC certification attempts."""

from __future__ import annotations

from typing import Any

import pytest

from tuningfork.recipes._base_smc import SMCRecipe
from tuningfork.recipes._smc_certification_record import (
    append_smc_certification_attempt,
    import_legacy_current_view,
)

pytestmark = pytest.mark.fast


def _recipe(**updates: Any) -> SMCRecipe:
    values: dict[str, Any] = dict(
        model_name="mvn_10",
        smc_method_name="tempered_smc",
        inner_method_name="hmc",
        num_particles=8,
        max_steps=4,
        smc_params={"target_ess": 0.5},
        gate_evidence={"auto": {"verdict": "FAIL"}},
        notes="historical",
        workflow="investigate",
        failure_diagnosis="poor ESS",
        _extra_fields={"future": {"ordered": (1, None)}},
    )
    values.update(updates)
    return SMCRecipe(**values)


def test_legacy_import_is_lossless_and_idempotent() -> None:
    recipe = _recipe(smc_params={"target_ess": float("inf")})
    imported = import_legacy_current_view(recipe, ground_truth={"id": "gt"})
    assert imported.attempted_configurations[0]["attempt_id"] == "legacy-current-view"
    assert imported.attempted_configurations[0]["metrics"]["legacy_current_view"][
        "smc_params"
    ]["target_ess"] == {"\u0000tuningfork_legacy_current_view_nonfinite_float": "+inf"}
    assert imported.attempted_configurations[0]["metrics"]["legacy_current_view"][
        "future"
    ]["ordered"] == [1, None]
    assert imported._extra_fields == recipe._extra_fields
    assert import_legacy_current_view(imported, ground_truth=None) == imported


def test_fresh_attempt_rejects_nonfinite_evidence() -> None:
    recipe = _recipe()
    with pytest.raises(ValueError, match="non-finite"):
        append_smc_certification_attempt(
            recipe,
            recipe,
            result=None,
            ground_truth=None,
            lifecycle_stage="EVALUATED",
            automatic_verdict="PASS",
            rationale="test",
            measurement_conditions={},
            metrics={"bad": float("nan")},
            gate_evidence=None,
            failure_evidence=None,
            recipe_updates=None,
        )


def test_append_records_materialized_updates() -> None:
    recipe = _recipe()
    updated, attempt_id = append_smc_certification_attempt(
        recipe,
        recipe,
        result=None,
        ground_truth={"id": "gt"},
        lifecycle_stage="EVALUATED",
        automatic_verdict="REVIEW",
        rationale="test",
        measurement_conditions={"particles": 8},
        metrics={"particle_ess": 7.0},
        gate_evidence={"auto": {"verdict": "REVIEW"}},
        failure_evidence=None,
        recipe_updates={"headline_metric": 1.5, "workflow": "review"},
    )
    assert attempt_id.startswith("attempt-")
    attempt = updated.attempted_configurations[-1]
    assert attempt["automatic_verdict"] == "REVIEW"
    assert attempt["measurement_conditions"] == {"particles": 8}
    assert updated.headline_metric == 1.5
    assert updated.workflow == "review"
