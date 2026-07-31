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

"""End-to-end execution of one tiny generated SMC program."""

import shutil
from pathlib import Path

import pytest

from tuningfork.recipes._base_smc import SMCRecipe
from tuningfork.recipes._emit_smc_script import emit_smc_script
from tuningfork.recipes._generated_smc import load_generated_smc_artifact
from tuningfork.recipes._launcher import launch_generated_program
from tuningfork.recipes._smc_certification_runner import certify_smc_recipe

pytestmark = pytest.mark.e2e

_CATALOG_ROOT = Path(__file__).parents[2] / "tuningfork" / "catalog"


def test_generated_smc_program_launches_with_bound_evidence(tmp_path) -> None:
    recipe = SMCRecipe(
        model_name="mvn_10",
        smc_method_name="adaptive_tempered_smc",
        inner_method_name="rwm",
        num_particles=8,
        max_steps=2,
        seed=17,
        smc_params={"target_ess": 0.5, "num_mcmc_steps": 1},
        inner_params_init={"sigma": 0.2},
        parameter_update_strategy="none",
    )
    result = launch_generated_program(
        emit_smc_script(recipe),
        tmp_path,
        timeout=60,
        env={"JAX_PLATFORM_NAME": "cpu"},
    )
    assert result.receipt.status == "success"
    assert result.artifact_path is not None
    assert result.telemetry is not None
    artifact = load_generated_smc_artifact(result.artifact_path, result.manifest)
    assert artifact.num_particles == 8
    assert artifact.lambda_final == pytest.approx(1.0)


def test_generated_hmc_tuning_route_launches(tmp_path) -> None:
    recipe = SMCRecipe(
        model_name="mvn_10",
        smc_method_name="inner_kernel_tuning",
        inner_method_name="hmc",
        num_particles=8,
        max_steps=2,
        seed=19,
        smc_params={
            "target_ess": 0.5,
            "num_mcmc_steps": 1,
            "num_integration_steps": 1,
        },
        inner_params_init={
            "step_size": 0.1,
            "inverse_mass_matrix": [1.0] * 10,
        },
        parameter_update_strategy="step_size_and_imm_from_particles",
        parameter_update_strategy_kwargs={"target_acceptance": 0.65},
    )
    result = launch_generated_program(
        emit_smc_script(recipe),
        tmp_path,
        timeout=60,
        env={"JAX_PLATFORM_NAME": "cpu"},
    )
    assert result.receipt.status == "success"
    assert result.artifact_path is not None
    artifact = load_generated_smc_artifact(result.artifact_path, result.manifest)
    assert artifact.final_inner_params
    assert artifact.lambda_final == pytest.approx(1.0)


def test_generated_smc_certification_persists_bound_attempt(tmp_path) -> None:
    source = _CATALOG_ROOT / "mvn_10" / "groundtruth_samples" / "blackjax"
    target = tmp_path / "mvn_10" / "groundtruth_samples" / "blackjax"
    target.mkdir(parents=True)
    shutil.copy2(source / "summary_v2.json", target / "summary_v2.json")
    shutil.copy2(source / "draws.npz", target / "draws.npz")
    recipe = SMCRecipe(
        model_name="mvn_10",
        smc_method_name="adaptive_tempered_smc",
        inner_method_name="rwm",
        num_particles=32,
        max_steps=4,
        seed=23,
        smc_params={"target_ess": 0.5, "num_mcmc_steps": 1},
        inner_params_init={"sigma": 0.2},
        parameter_update_strategy="none",
    )

    outcome = certify_smc_recipe(
        recipe,
        catalog_root=tmp_path,
        timeout=60,
    )

    assert outcome.verdict in {"PASS", "REVIEW", "FAIL"}
    assert outcome.recipe_path is not None
    saved = SMCRecipe.load(outcome.recipe_path)
    attempt = saved.attempted_configurations[-1]
    assert attempt["attempt_id"] == outcome.attempt_id
    assert attempt["execution"]["receipt"]["status"] == "success"
    assert attempt["ground_truth"]["model_name"] == "mvn_10"
