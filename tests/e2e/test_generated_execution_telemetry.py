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

"""End-to-end evidence checks for generated adaptation and sampling."""

from __future__ import annotations

import numpy as np
import pytest

from tuningfork.catalog import execute_recipe
from tuningfork.recipes import Effort, Recipe
from tuningfork.recipes._execution_telemetry import ExecutionTelemetry

pytestmark = pytest.mark.e2e


def test_window_adaptation_emits_exact_bound_telemetry(tmp_path) -> None:
    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1, "num_integration_steps": 3},
        warmup_params={"n_warmup": 10, "target_acceptance_rate": 0.8},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"n_samples": 3, "num_chains": 1},
        difficulty=None,
        instructions="",
        tuning_seed=7,
    )

    result = execute_recipe(
        recipe,
        tmp_path,
        num_samples=3,
        num_chains=1,
        num_warmup=10,
        warmup_num_chains=[1],
        timeout=180,
        env={"JAX_PLATFORM_NAME": "cpu"},
    )

    telemetry = result.telemetry
    assert isinstance(telemetry, ExecutionTelemetry)
    assert telemetry.plan_hash == result.manifest.plan_hash
    assert telemetry.geometry_source == "adapted"
    assert telemetry.geometry_scope == "per_chain"
    assert telemetry.warmup_grad_evals is not None
    assert telemetry.warmup_grad_evals > 0
    assert "num_integration_steps" in telemetry.warmup_grad_evals_reason
    assert np.asarray(telemetry.geometry["step_size"]).shape == (1,)
    assert np.asarray(telemetry.geometry["inverse_mass_matrix"]).shape == (1, 10)
    assert telemetry.fixed["num_integration_steps"] == 3
    assert result.telemetry_path is not None
    assert result.receipt.telemetry_sha256 == result.telemetry_sha256
