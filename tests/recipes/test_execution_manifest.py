from types import SimpleNamespace

import pytest

from tuningfork.recipes._execution_manifest import MANIFEST_VERSION, ExecutionManifest
from tuningfork.recipes._execution_plan import ExecutionOverrides
from tuningfork.recipes._resolve_execution_plan import resolve_execution_plan

pytestmark = pytest.mark.fast


def _plan():
    recipe = SimpleNamespace(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="window_adaptation_diag_imm",
        effort="low",
        base_method_params={"step_size": 0.1},
        warmup_params={"n_warmup": 10},
        warmups=[{"name": "window_adaptation_diag_imm", "params": {"n_warmup": 10}}],
        calibration_budget={"n_samples": 20, "num_chains": 2},
        tuning_seed=4,
        warmup_inner_kernel=None,
        init_strategy=None,
        step_policy=None,
        variant_label=None,
    )
    return resolve_execution_plan(recipe, ExecutionOverrides(num_samples=20))


def test_manifest_is_versioned_and_deterministic():
    first = ExecutionManifest.from_plan(_plan(), generator_version="2026.07")
    second = ExecutionManifest.from_plan(_plan(), generator_version="2026.07")
    newer = ExecutionManifest.from_plan(_plan(), generator_version="2026.08")
    assert first.manifest_version == MANIFEST_VERSION
    assert first.to_json() == second.to_json()
    assert first.to_json() != newer.to_json()
    assert first.plan_hash == newer.plan_hash
    assert first.recipe_ref == "mvn_10/low__hmc__window_adaptation_diag_imm"
    assert first.executable_config_hash == _plan().executable_config_hash


def test_manifest_freezes_nested_values_and_round_trips():
    manifest = ExecutionManifest.from_plan(_plan(), generator_version="2026.07")
    with pytest.raises(TypeError):
        manifest.executable_config["num_samples"] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        manifest.normalized_plan["config"]["num_samples"] = 99  # type: ignore[index]
    exported = manifest.as_dict()
    exported["executable_config"]["num_samples"] = 99
    assert manifest.executable_config["num_samples"] == 20
    loaded = ExecutionManifest.from_dict(manifest.as_dict())
    assert loaded.to_json() == manifest.to_json()


def test_manifest_rejects_tampered_hash_or_plan():
    data = ExecutionManifest.from_plan(_plan(), generator_version="2026.07").as_dict()
    data["plan_hash"] = "0" * 64
    with pytest.raises(ValueError, match="plan_hash"):
        ExecutionManifest.from_dict(data)

    data = ExecutionManifest.from_plan(_plan(), generator_version="2026.07").as_dict()
    data["normalized_plan"]["artifact_filename"] = "other.draws.npz"
    with pytest.raises(ValueError, match="plan_hash"):
        ExecutionManifest.from_dict(data)


def test_manifest_rejects_noncanonical_values():
    data = ExecutionManifest.from_plan(_plan(), generator_version="2026.07").as_dict()
    data["executable_config"]["bad"] = object()
    with pytest.raises(TypeError):
        ExecutionManifest.from_dict(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("manifest_version", "tuningfork.execution-manifest.v0"),
        ("generator_contract", "other-contract"),
    ],
)
def test_manifest_rejects_unsupported_identity(field, value):
    data = ExecutionManifest.from_plan(_plan(), generator_version="2026.07").as_dict()
    data[field] = value
    with pytest.raises(ValueError, match="unsupported"):
        ExecutionManifest.from_dict(data)


def test_manifest_rejects_unsupported_contract_at_creation():
    with pytest.raises(ValueError, match="unsupported generator_contract"):
        ExecutionManifest.from_plan(
            _plan(),
            generator_version="2026.07",
            generator_contract="other-contract",
        )


def test_manifest_rejects_unknown_fields_and_ref_drift():
    data = ExecutionManifest.from_plan(_plan(), generator_version="2026.07").as_dict()
    data["extra"] = True
    with pytest.raises(ValueError, match="unsupported fields"):
        ExecutionManifest.from_dict(data)

    data = ExecutionManifest.from_plan(_plan(), generator_version="2026.07").as_dict()
    data["normalized_plan"]["recipe_ref"] = "different"
    with pytest.raises(ValueError, match="recipe_ref"):
        ExecutionManifest.from_dict(data)
