"""Fast contract tests for SMC execution-plan resolution."""

import dataclasses

import pytest

from tuningfork.recipes._base_smc import SMCRecipe
from tuningfork.recipes._execution_manifest import ExecutionManifest
from tuningfork.recipes._smc_execution_plan import resolve_smc_execution_plan

pytestmark = pytest.mark.fast


def _recipe(**kwargs: object) -> SMCRecipe:
    values: dict[str, object] = {
        "model_name": "mvn_10",
        "smc_method_name": "tempered_smc",
        "inner_method_name": "hmc",
        "num_particles": 16,
        "max_steps": 4,
    }
    values.update(kwargs)
    return SMCRecipe(**values)  # type: ignore[arg-type]


def test_plan_has_particle_specific_immutable_config() -> None:
    plan = resolve_smc_execution_plan(
        _recipe(
            smc_params={"target_ess": 0.5},
            inner_params_init={"step_size": 0.1},
            parameter_update_strategy="none",
        )
    )
    assert plan.config.as_dict()["execution_family"] == "smc"
    assert plan.config.as_dict()["model_name"] == "mvn_10"
    assert plan.config.as_dict()["num_particles"] == 16
    assert plan.artifact_filename == "mvn_10__smc__tempered_smc__hmc.draws.npz"
    assert plan.recipe_ref == "mvn_10/smc__tempered_smc__hmc"
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.config.num_particles = 2  # type: ignore[misc]


def test_plan_builds_a_hash_verified_manifest() -> None:
    plan = resolve_smc_execution_plan(_recipe())
    manifest = ExecutionManifest.from_plan(plan, generator_version="test")
    assert manifest.executable_config_hash == plan.executable_config_hash
    assert manifest.executable_config["execution_family"] == "smc"


def test_registry_incompatibility_is_rejected() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        resolve_smc_execution_plan(
            _recipe(smc_method_name="tempered_smc", inner_method_name="mclmc")
        )


def test_hash_changes_for_material_values_but_not_mapping_order() -> None:
    first = resolve_smc_execution_plan(_recipe(smc_params={"a": 1, "b": 2}))
    reordered = resolve_smc_execution_plan(_recipe(smc_params={"b": 2, "a": 1}))
    changed = resolve_smc_execution_plan(_recipe(smc_params={"a": 1, "b": 3}))
    assert first.plan_hash == reordered.plan_hash
    assert first.plan_hash != changed.plan_hash


def test_non_json_values_fail_before_rendering() -> None:
    with pytest.raises(TypeError, match="JSON-safe"):
        resolve_smc_execution_plan(_recipe(smc_params={"bad": object()}))
