"""End-to-end execution of one tiny generated SMC program."""

import pytest

from tuningfork.recipes._base_smc import SMCRecipe
from tuningfork.recipes._emit_smc_script import emit_smc_script
from tuningfork.recipes._generated_smc import load_generated_smc_artifact
from tuningfork.recipes._launcher import launch_generated_program

pytestmark = pytest.mark.e2e


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
    artifact = load_generated_smc_artifact(result.artifact_path, result.manifest)
    assert artifact.final_inner_params
    assert artifact.lambda_final == pytest.approx(1.0)
