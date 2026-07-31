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

"""Fast orchestration tests for generated SMC certification attempts."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from tuningfork.calibration.smc_gate import SMCGateVerdict
from tuningfork.catalog.emit import (
    RECIPE_EVIDENCE_HASH_DOMAIN,
    RECIPE_EVIDENCE_KEY,
    RECIPE_EVIDENCE_SCHEMA,
    canonical_recipe_snapshot,
)
from tuningfork.recipes import _smc_certification_runner as runner
from tuningfork.recipes._base_smc import SMCRecipe
from tuningfork.recipes._execution_manifest import ExecutionManifest
from tuningfork.recipes._execution_plan import canonical_json
from tuningfork.recipes._generated_smc import GeneratedSMCEvaluation
from tuningfork.recipes._ground_truth_reference import GroundTruthReference
from tuningfork.recipes._launcher import (
    ExecutionTimings,
    GeneratedProgramError,
    LaunchResult,
)
from tuningfork.recipes._smc_execution_plan import resolve_smc_execution_plan
from tuningfork.recipes._smc_execution_telemetry import (
    SMC_TELEMETRY_SCHEMA,
    SMCExecutionTelemetry,
)

pytestmark = pytest.mark.fast


def _recipe(**updates: Any) -> SMCRecipe:
    values: dict[str, Any] = {
        "model_name": "mvn_10",
        "smc_method_name": "adaptive_tempered_smc",
        "inner_method_name": "rwm",
        "num_particles": 8,
        "max_steps": 2,
        "seed": 17,
        "smc_params": {"target_ess": 0.5, "num_mcmc_steps": 1},
        "inner_params_init": {"sigma": 0.2},
        "parameter_update_strategy": "none",
    }
    values.update(updates)
    return SMCRecipe(**values)


def _reference(tmp_path: Path) -> GroundTruthReference:
    return GroundTruthReference(
        model_name="mvn_10",
        summary={
            "per_site": {},
            "n_chains": 2,
            "n_draws_per_chain": 10,
            "n_total": 20,
        },
        summary_path=tmp_path / "summary_v2.json",
        draws_path=tmp_path / "draws.npz",
        identity={
            "schema": "test-ground-truth",
            "model_name": "mvn_10",
            "draws_sha256": "d" * 64,
        },
    )


def _recipe_envelope(
    recipe: SMCRecipe, reference: GroundTruthReference
) -> dict[str, Any]:
    snapshot = canonical_recipe_snapshot(recipe)
    snapshot_hash = hashlib.sha256(
        (RECIPE_EVIDENCE_HASH_DOMAIN + canonical_json(snapshot)).encode("utf-8")
    ).hexdigest()
    return {
        "schema": RECIPE_EVIDENCE_SCHEMA,
        "snapshot": snapshot,
        "snapshot_sha256": snapshot_hash,
        "caller_reference_identity": reference.identity,
    }


def _launch(
    tmp_path: Path,
    recipe: SMCRecipe,
    reference: GroundTruthReference,
    *,
    status: str = "success",
    mismatch: str | None = None,
    typed_telemetry: bool = True,
) -> LaunchResult:
    plan = resolve_smc_execution_plan(recipe)
    manifest = ExecutionManifest.from_plan(plan, generator_version="test")
    if mismatch == "plan":
        manifest = replace(manifest, plan_hash="0" * 64)

    envelope = _recipe_envelope(recipe, reference)
    if mismatch == "snapshot":
        envelope["snapshot"] = {**envelope["snapshot"], "num_particles": 999}
    if mismatch == "ground_truth":
        envelope["caller_reference_identity"] = {"model_name": "wrong"}
    identity = {RECIPE_EVIDENCE_KEY: envelope}
    receipt = SimpleNamespace(
        run_id="run-123",
        status=status,
        manifest=manifest,
        as_dict=lambda: {
            "run_id": "run-123",
            "status": status,
            "reference_identity": identity,
        },
    )
    telemetry: SMCExecutionTelemetry | None
    if status == "success" and typed_telemetry:
        telemetry = SMCExecutionTelemetry(
            schema=SMC_TELEMETRY_SCHEMA,
            plan_hash=manifest.plan_hash,
            executable_config_hash=manifest.executable_config_hash,
            draws_artifact=plan.artifact_filename,
            num_particles=recipe.num_particles,
            num_smc_steps=2,
            lambda_final=1.0,
            timing_seconds=MappingProxyType(
                {"initialization": 0.1, "sampling": 0.2, "total": 0.3}
            ),
        )
    else:
        telemetry = None
    return LaunchResult(
        run_dir=tmp_path / "run-123",
        source_path=tmp_path / "run-123" / "program.py",
        stdout_path=tmp_path / "run-123" / "stdout.log",
        stderr_path=tmp_path / "run-123" / "stderr.log",
        artifact_path=(
            tmp_path / "run-123" / plan.artifact_filename
            if status == "success"
            else None
        ),
        receipt_path=tmp_path / "run-123" / "execution_receipt.json",
        returncode=0 if status == "success" else 1,
        timed_out=False,
        source_sha256="a" * 64,
        artifact_sha256="b" * 64 if status == "success" else None,
        telemetry_path=(
            tmp_path / "run-123" / plan.telemetry_artifact_filename
            if status == "success" and typed_telemetry
            else None
        ),
        telemetry_sha256=(
            "c" * 64 if status == "success" and typed_telemetry else None
        ),
        telemetry=telemetry,
        manifest=manifest,
        receipt=cast(Any, receipt),
        timings=(ExecutionTimings(0.1, 0.2, 0.3) if status == "success" else None),
        nonfinite_stat_counts={},
    )


def _evaluation(
    verdict: str, reference: GroundTruthReference
) -> GeneratedSMCEvaluation:
    gate = SMCGateVerdict(
        verdict=verdict,
        max_abs_mean_z=1.25,
        particle_ess=4.0,
        particle_ess_fraction=0.5,
        mode_coverage_fraction=None,
        lambda_final=1.0,
    )
    return GeneratedSMCEvaluation(
        gate=gate,
        headline_metric=0.5,
        total_cost=8,
        lambda_final=1.0,
        history=MappingProxyType(
            {
                "lambda": np.array([0.0, 1.0]),
                "ess": np.array([8.0, 4.0]),
            }
        ),
        ground_truth_identity=MappingProxyType(reference.identity),
    )


def _patch_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    recipe: SMCRecipe,
    *,
    verdict: str = "PASS",
    launch: LaunchResult | None = None,
) -> tuple[GroundTruthReference, LaunchResult, dict[str, Any]]:
    reference = _reference(tmp_path)
    launch = launch or _launch(tmp_path, recipe, reference)
    seen: dict[str, Any] = {}

    monkeypatch.setattr(runner, "load_ground_truth_reference", lambda *_: reference)

    def execute(requested: SMCRecipe, *args: Any, **kwargs: Any) -> LaunchResult:
        seen["recipe"] = requested
        seen["kwargs"] = kwargs
        return launch

    monkeypatch.setattr(runner, "execute_recipe", execute)
    monkeypatch.setattr(
        runner,
        "load_generated_smc_artifact",
        lambda *args: SimpleNamespace(
            final_inner_params={"sigma": np.full(recipe.num_particles, 0.15)}
        ),
    )
    monkeypatch.setattr(
        runner,
        "evaluate_generated_smc",
        lambda *args: _evaluation(verdict, reference),
    )
    return reference, launch, seen


@pytest.mark.parametrize("verdict", ["PASS", "REVIEW", "FAIL"])
def test_gate_outcomes_persist_full_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    verdict: str,
) -> None:
    recipe = _recipe()
    _patch_success(monkeypatch, tmp_path, recipe, verdict=verdict)

    outcome = runner.certify_smc_recipe(recipe, catalog_root=tmp_path)

    assert outcome.verdict == verdict
    assert outcome.recipe_path is not None
    saved = SMCRecipe.load(outcome.recipe_path)
    attempt = saved.attempted_configurations[-1]
    assert attempt["automatic_verdict"] == verdict
    assert attempt["lifecycle_stage"] == "EVALUATED"
    assert attempt["execution"]["receipt"]["run_id"] == "run-123"
    assert attempt["metrics"]["history"] == {
        "lambda": [0.0, 1.0],
        "ess": [8.0, 4.0],
    }
    assert attempt["metrics"]["ground_truth_identity"]["draws_sha256"] == ("d" * 64)
    assert attempt["metrics"]["final_inner_param_shapes"] == {"sigma": [8]}
    assert attempt["metrics"]["full_final_inner_params"] == (
        "retained in the generated artifact"
    )
    assert saved.calibration_budget["selected_attempt_id"] == "run-123"
    assert saved.inner_params_final == ({"sigma": 0.15} if verdict == "PASS" else None)
    if verdict == "FAIL":
        assert attempt["failure_evidence"]["stage"] == "gate evaluation"
        assert saved.failure_diagnosis is not None
        assert saved.failure_diagnosis.startswith("automatic SMC gate FAIL")
    else:
        assert attempt["failure_evidence"] is None
        assert saved.failure_diagnosis is None


def test_missing_ground_truth_persists_draft_without_sampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recipe = _recipe()
    monkeypatch.setattr(
        runner,
        "load_ground_truth_reference",
        lambda *_: (_ for _ in ()).throw(FileNotFoundError("missing LFS object")),
    )
    monkeypatch.setattr(
        runner,
        "execute_recipe",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("sampling must not start")
        ),
    )

    outcome = runner.certify_smc_recipe(recipe, catalog_root=tmp_path)

    assert outcome.verdict == "ERROR"
    assert outcome.recipe_path is not None
    saved = SMCRecipe.load(outcome.recipe_path)
    attempt = saved.attempted_configurations[-1]
    assert attempt["lifecycle_stage"] == "DRAFT"
    assert attempt["failure_evidence"]["stage"] == "ground-truth preflight"


def test_codegen_error_is_a_durable_draft(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recipe = _recipe()
    reference = _reference(tmp_path)
    monkeypatch.setattr(runner, "load_ground_truth_reference", lambda *_: reference)
    monkeypatch.setattr(
        runner,
        "execute_recipe",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            NotImplementedError("missing generated call shape")
        ),
    )

    outcome = runner.certify_smc_recipe(recipe, catalog_root=tmp_path)

    assert outcome.recipe_path is not None
    attempt = SMCRecipe.load(outcome.recipe_path).attempted_configurations[-1]
    assert attempt["lifecycle_stage"] == "DRAFT"
    assert attempt["failure_evidence"]["error_type"] == "NotImplementedError"


def test_generated_program_failure_retains_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recipe = _recipe()
    reference = _reference(tmp_path)
    failed = _launch(tmp_path, recipe, reference, status="failed")
    monkeypatch.setattr(runner, "load_ground_truth_reference", lambda *_: reference)
    monkeypatch.setattr(
        runner,
        "execute_recipe",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            GeneratedProgramError("child failed", failed)
        ),
    )

    outcome = runner.certify_smc_recipe(recipe, catalog_root=tmp_path)

    assert outcome.verdict == "ERROR"
    assert outcome.receipt_path == failed.receipt_path
    assert outcome.recipe_path is not None
    attempt = SMCRecipe.load(outcome.recipe_path).attempted_configurations[-1]
    assert attempt["lifecycle_stage"] == "GENERATED"
    assert attempt["execution"]["receipt"]["run_id"] == "run-123"


@pytest.mark.parametrize("mismatch", ["plan", "snapshot", "ground_truth"])
def test_binding_mismatch_is_a_sampled_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mismatch: str,
) -> None:
    recipe = _recipe()
    reference = _reference(tmp_path)
    launch = _launch(tmp_path, recipe, reference, mismatch=mismatch)
    monkeypatch.setattr(runner, "load_ground_truth_reference", lambda *_: reference)
    monkeypatch.setattr(runner, "execute_recipe", lambda *args, **kwargs: launch)

    outcome = runner.certify_smc_recipe(recipe, catalog_root=tmp_path)

    assert outcome.verdict == "ERROR"
    assert outcome.recipe_path is not None
    attempt = SMCRecipe.load(outcome.recipe_path).attempted_configurations[-1]
    assert attempt["lifecycle_stage"] == "SAMPLED"
    assert attempt["failure_evidence"]["stage"] == "execution binding"


def test_success_without_typed_telemetry_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recipe = _recipe()
    reference = _reference(tmp_path)
    launch = _launch(tmp_path, recipe, reference, typed_telemetry=False)
    monkeypatch.setattr(runner, "load_ground_truth_reference", lambda *_: reference)
    monkeypatch.setattr(runner, "execute_recipe", lambda *args, **kwargs: launch)

    outcome = runner.certify_smc_recipe(recipe, catalog_root=tmp_path)

    assert outcome.verdict == "ERROR"
    assert "typed telemetry" in outcome.note


def test_telemetry_hash_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recipe = _recipe()
    reference = _reference(tmp_path)
    launch = _launch(tmp_path, recipe, reference)
    assert isinstance(launch.telemetry, SMCExecutionTelemetry)
    tampered = replace(launch.telemetry, plan_hash="0" * 64)
    launch = replace(launch, telemetry=tampered)
    monkeypatch.setattr(runner, "load_ground_truth_reference", lambda *_: reference)
    monkeypatch.setattr(runner, "execute_recipe", lambda *args, **kwargs: launch)

    outcome = runner.certify_smc_recipe(recipe, catalog_root=tmp_path)

    assert outcome.verdict == "ERROR"
    assert "telemetry hashes" in outcome.note


def test_binding_accepts_hmc_list_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recipe = _recipe(
        smc_method_name="inner_kernel_tuning",
        inner_method_name="hmc",
        smc_params={
            "target_ess": 0.5,
            "num_mcmc_steps": 1,
            "num_integration_steps": 2,
        },
        inner_params_init={
            "step_size": 0.1,
            "inverse_mass_matrix": [1.0] * 10,
        },
        parameter_update_strategy="step_size_and_imm_from_particles",
        parameter_update_strategy_kwargs={"target_acceptance": 0.65},
    )
    _patch_success(monkeypatch, tmp_path, recipe)

    outcome = runner.certify_smc_recipe(recipe, catalog_root=tmp_path)

    assert outcome.verdict == "PASS"
    assert outcome.recipe_path is not None


def test_new_intent_keeps_legacy_negative_and_unknown_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old = _recipe(
        notes="three seeds exposed a persistent bias",
        workflow="custom script retry sequence",
        failure_diagnosis="mode collapse",
        headline_metric=0.01,
        gate_evidence={
            "auto": {"verdict": "FAIL", "max_abs_mean_z": 7.0},
            "override": {
                "reason": "heavy-tail exception",
                "statistician_id": "human",
                "decision": "APPROVE",
            },
            "future_gate_annotation": {"keep": True},
        },
        attempted_configurations=[{"opaque_legacy_attempt": [1, None]}],
        _extra_fields={"future_top_level": {"ordered": [2, 1]}},
    )
    old.save(tmp_path)
    intent = _recipe(num_particles=16, notes="must not replace curated notes")
    _, _, seen = _patch_success(monkeypatch, tmp_path, intent)

    outcome = runner.certify_smc_recipe(intent, catalog_root=tmp_path)

    assert seen["recipe"] is intent
    assert seen["kwargs"]["reference_identity"]["draws_sha256"] == "d" * 64
    assert outcome.recipe_path is not None
    saved = SMCRecipe.load(outcome.recipe_path)
    assert saved.num_particles == 16
    assert saved.notes == old.notes
    assert saved.workflow == old.workflow
    assert saved._extra_fields == old._extra_fields
    assert saved.attempted_configurations[0] == {"opaque_legacy_attempt": [1, None]}
    legacy = next(
        item
        for item in saved.attempted_configurations
        if isinstance(item, dict) and item.get("attempt_id") == "legacy-current-view"
    )
    legacy_view = legacy["metrics"]["legacy_current_view"]
    assert legacy_view["failure_diagnosis"] == "mode collapse"
    assert legacy_view["gate_evidence"]["override"]["decision"] == "APPROVE"
    assert saved.gate_evidence["future_gate_annotation"] == {"keep": True}
    assert saved.gate_evidence["override"] == {
        "reason": "",
        "statistician_id": "",
        "decision": "",
    }


def test_malformed_existing_recipe_is_not_overwritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recipe = _recipe()
    path = runner._recipe_path(tmp_path, recipe)
    path.parent.mkdir(parents=True)
    original = b"{not valid json"
    path.write_bytes(original)
    monkeypatch.setattr(
        runner,
        "load_ground_truth_reference",
        lambda *args: (_ for _ in ()).throw(AssertionError("must not continue")),
    )

    outcome = runner.certify_smc_recipe(recipe, catalog_root=tmp_path)

    assert outcome.verdict == "ERROR"
    assert outcome.recipe_path is None
    assert path.read_bytes() == original


def test_persistence_failure_is_not_reported_as_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recipe = _recipe()
    monkeypatch.setattr(
        runner,
        "load_ground_truth_reference",
        lambda *_: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    monkeypatch.setattr(
        SMCRecipe,
        "save",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    outcome = runner.certify_smc_recipe(recipe, catalog_root=tmp_path)

    assert outcome.verdict == "ERROR"
    assert outcome.recipe_path is None
    assert "recording/persistence failed" in outcome.note


def test_wrapper_chooses_family_specific_parameter_update(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[SMCRecipe] = []

    def fake_certify(recipe: SMCRecipe, **kwargs: Any) -> runner.SMCCellResult:
        seen.append(recipe)
        return runner.SMCCellResult(
            recipe.model_name,
            recipe.smc_method_name,
            recipe.inner_method_name,
            "ERROR",
        )

    monkeypatch.setattr(
        runner,
        "certify_smc_recipe",
        fake_certify,
    )

    runner.emit_smc_recipe_for_cell(
        "mvn_10",
        "adaptive_tempered_smc",
        "rwm",
        catalog_root=tmp_path,
    )
    runner.emit_smc_recipe_for_cell(
        "mvn_10",
        "inner_kernel_tuning",
        "hmc",
        catalog_root=tmp_path,
    )

    assert seen[0].parameter_update_strategy == "none"
    assert seen[1].parameter_update_strategy == "step_size_and_imm_from_particles"
    assert seen[0].max_steps == 500
