"""Regression tests for embedding the canonical execution manifest in scripts."""

import ast
import json
from types import SimpleNamespace

import pytest

from tuningfork._version import __version__ as CURRENT_GENERATOR_VERSION
from tuningfork.catalog import emit_script
from tuningfork.recipes._base import Effort
from tuningfork.recipes._execution_manifest import ExecutionManifest
from tuningfork.recipes._execution_plan import ExecutionOverrides
from tuningfork.recipes._resolve_execution_plan import resolve_execution_plan

pytestmark = pytest.mark.fast


def _recipe() -> SimpleNamespace:
    return SimpleNamespace(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1},
        warmup_params={"n_warmup": 10},
        warmups=[
            {
                "name": "window_adaptation_diag_imm",
                "params": {"n_warmup": 10},
            }
        ],
        calibration_budget={"n_samples": 20, "num_chains": 2},
        tuning_seed=4,
        warmup_inner_kernel=None,
        init_strategy=None,
        step_policy=None,
        variant_label=None,
        warmup_num_chains=None,
        gate_evidence={},
    )


def _embedded_manifest(source: str) -> tuple[str, ExecutionManifest]:
    tree = ast.parse(source)
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXECUTION_MANIFEST_JSON"
            for target in node.targets
        ):
            values.append(ast.literal_eval(node.value))
    assert len(values) == 1
    value = values[0]
    assert isinstance(value, str)
    return value, ExecutionManifest.from_dict(json.loads(value))


def test_emitted_source_parses_and_exposes_standalone_manifest_json():
    source = emit_script(_recipe(), sampler_seed=8, num_samples=9)
    ast.parse(source)
    embedded_json, manifest = _embedded_manifest(source)
    assert manifest.to_json() == embedded_json
    assert f"Execution plan hash: {manifest.plan_hash}" in source
    assert "Recipe hash:" not in source


def test_embedded_manifest_matches_resolved_execution_plan():
    recipe = _recipe()
    source = emit_script(recipe, sampler_seed=8, num_samples=9)
    _, embedded = _embedded_manifest(source)
    plan = resolve_execution_plan(
        recipe,
        ExecutionOverrides(sampler_seed=8, num_samples=9),
    )
    expected = ExecutionManifest.from_plan(
        plan,
        generator_version=CURRENT_GENERATOR_VERSION,
    )
    assert embedded.to_json() == expected.to_json()


def test_emit_overrides_change_identity_but_preserve_recipe_ref():
    recipe = _recipe()
    _, baseline = _embedded_manifest(emit_script(recipe, sampler_seed=8, num_samples=9))
    _, changed = _embedded_manifest(
        emit_script(recipe, sampler_seed=11, num_samples=13)
    )
    assert changed.plan_hash != baseline.plan_hash
    assert changed.executable_config_hash != baseline.executable_config_hash
    assert changed.recipe_ref == baseline.recipe_ref
    assert changed.recipe_ref == "mvn_10/low__hmc__window_adaptation_diag_imm"
