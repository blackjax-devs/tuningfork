"""Fast contracts for particle-aware generated launching."""

import json

import numpy as np
import pytest

from tuningfork.recipes._base_smc import SMCRecipe
from tuningfork.recipes._execution_manifest import ExecutionManifest
from tuningfork.recipes._execution_plan import (
    execution_config_hash,
    execution_plan_hash,
)
from tuningfork.recipes._launcher import (
    SMCExecutionTelemetry,
    _read_telemetry,
    _validate_artifact,
)
from tuningfork.recipes._smc_execution_plan import resolve_smc_execution_plan

pytestmark = pytest.mark.fast


def _manifest() -> ExecutionManifest:
    plan = resolve_smc_execution_plan(
        SMCRecipe(
            model_name="mvn_10",
            smc_method_name="tempered_smc",
            inner_method_name="hmc",
            num_particles=4,
            max_steps=3,
        )
    )
    config = plan.config.as_dict()
    normalized = {
        "config": config,
        "recipe_ref": plan.recipe_ref,
        "artifact_filename": plan.artifact_filename,
        "telemetry_artifact_filename": plan.telemetry_artifact_filename,
    }
    return ExecutionManifest.from_dict(
        {
            "manifest_version": "tuningfork.execution-manifest.v2",
            "generator_contract": "tuningfork.execution-plan.v2",
            "generator_version": "test",
            "recipe_ref": plan.recipe_ref,
            "executable_config": config,
            "normalized_plan": normalized,
            "executable_config_hash": execution_config_hash(config),
            "plan_hash": execution_plan_hash(
                config, plan.artifact_filename, plan.telemetry_artifact_filename
            ),
        }
    )


def _artifact(path):
    np.savez(
        path,
        particle__x=np.zeros((4, 2)),
        smc__weights=np.full(4, 0.25),
        smc__lambda=np.array([0.0, 1.0]),
        smc__ess=np.array([4.0, 3.0]),
    )


def test_smc_artifact_uses_particle_contract(tmp_path):
    manifest = _manifest()
    path = tmp_path / "draws.npz"
    _artifact(path)
    assert _validate_artifact(path, manifest)


def test_smc_telemetry_does_not_require_fake_chain_fields(tmp_path):
    manifest = _manifest()
    path = tmp_path / "telemetry.json"
    payload = {
        "schema": "tuningfork.generated-smc-telemetry.v1",
        "plan_hash": manifest.plan_hash,
        "executable_config_hash": manifest.executable_config_hash,
        "draws_artifact": manifest.normalized_plan["artifact_filename"],
        "num_particles": 4,
        "num_smc_steps": 2,
        "lambda_final": 1.0,
        "timing_seconds": {
            "initialization": 0.1,
            "sampling": 0.2,
            "total": 0.3,
        },
    }
    path.write_text(json.dumps(payload))
    telemetry = _read_telemetry(path, manifest)
    assert isinstance(telemetry, SMCExecutionTelemetry)
    assert telemetry.as_dict() == payload
