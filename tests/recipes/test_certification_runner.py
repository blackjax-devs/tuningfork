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

"""Fast orchestration tests for generated certification attempts."""

from __future__ import annotations

import hashlib
from collections import namedtuple
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from tuningfork.catalog.emit import canonical_recipe_snapshot
from tuningfork.recipes import _certification_io as certification_io
from tuningfork.recipes import _certification_runner as runner
from tuningfork.recipes._base import Effort, Recipe
from tuningfork.recipes._certification_binding import verify_launch_binding
from tuningfork.recipes._certification_intent import CertificationIntent
from tuningfork.recipes._execution_plan import ExecutionOverrides
from tuningfork.recipes._generated_certification import GeneratedEvaluation
from tuningfork.recipes._launcher import GeneratedProgramError, LaunchResult
from tuningfork.recipes._resolve_execution_plan import resolve_execution_plan

pytestmark = pytest.mark.fast


def _recipe(**overrides) -> Recipe:
    values = dict(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="no_warmup",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1},
        warmup_params={"n_warmup": 0},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"n_samples": 2, "num_chains": 1},
        difficulty=None,
        instructions="",
        notes="",
        workflow="",
        gate_evidence={"auto": {"verdict": "NOT_RUN"}},
    )
    values.update(overrides)
    return Recipe(**cast(dict[str, Any], values))


def _intent(root: Path, recipe: Recipe | None = None) -> CertificationIntent:
    recipe = recipe or _recipe()
    path = root / recipe.model_name / "recipes" / "low__hmc__no_warmup.json"
    return CertificationIntent(recipe, None, path)


def _launch(
    tmp_path: Path,
    intent: CertificationIntent,
    reference_identity: dict[str, Any],
    *,
    status: str = "success",
    mismatch: str | None = None,
    run_id: str = "run-123",
) -> LaunchResult:
    run = tmp_path / run_id
    run.mkdir(exist_ok=True)
    expected = resolve_execution_plan(
        intent.recipe,
        ExecutionOverrides(
            tuning_seed=20260517,
            num_samples=2,
            num_chains=1,
            progress_bar=True,
            num_warmup=1000,
            warmup_num_chains=intent.recipe.warmup_num_chains,
        ),
    )
    manifest = SimpleNamespace(
        plan_hash="wrong-plan" if mismatch == "plan" else expected.plan_hash,
        executable_config_hash=expected.executable_config_hash,
        recipe_ref=expected.recipe_ref,
        as_dict=lambda: {
            "executable_config": {"num_samples": 2},
            "executable_config_hash": expected.executable_config_hash,
            "plan_hash": expected.plan_hash,
        },
    )
    identity = {
        "tuningfork_recipe_evidence": {
            "snapshot": canonical_recipe_snapshot(intent.recipe),
            "caller_reference_identity": (
                {"id": "wrong"} if mismatch == "gt" else reference_identity
            ),
        }
    }
    receipt = SimpleNamespace(
        run_id=run_id,
        status=status,
        manifest=manifest,
        reference_identity=identity,
        as_dict=lambda: {
            "run_id": "run-123",
            "status": status,
            "reference_identity": identity,
        },
    )
    telemetry = SimpleNamespace(as_dict=lambda: {"schema": "telemetry"})
    for name in (
        "program.py",
        "stdout.log",
        "stderr.log",
        "draws.npz",
        "telemetry.json",
        "receipt.json",
    ):
        (run / name).write_text(name)
    return LaunchResult(
        run_dir=run,
        source_path=run / "program.py",
        stdout_path=run / "stdout.log",
        stderr_path=run / "stderr.log",
        artifact_path=run / "draws.npz",
        receipt_path=run / "receipt.json",
        returncode=0 if status == "success" else 1,
        timed_out=False,
        source_sha256="a" * 64,
        artifact_sha256="b" * 64,
        telemetry_path=run / "telemetry.json",
        telemetry_sha256="c" * 64,
        telemetry=cast(Any, telemetry),
        manifest=cast(Any, manifest),
        receipt=cast(Any, receipt),
        timings=cast(
            Any,
            SimpleNamespace(
                warmup_seconds=1.0, sampling_seconds=2.0, total_seconds=3.0
            ),
        ),
    )


def _evaluation(verdict: str) -> GeneratedEvaluation:
    gate = {
        "verdict": verdict,
        "rhat_max": 1.1,
        "min_bulk_ess": 20.0,
        "n_divergences": 0,
    }
    return GeneratedEvaluation(
        gate_evidence={"auto": gate, "override": {}},
        headline_metric=2.0,
        headline_basis={"denominator": "draws"},
        sample_quality={"mae": 0.1},
        sampling_grad_evals=10,
        warmup_grad_evals=4,
        pinned_base_method_params={"step_size": 0.2},
        resolved_step_policy=None,
        metrics={"gate": gate},
        gt_cert_coverage="full_posterior",
    )


def _patch_common(
    monkeypatch, tmp_path, *, verdict="PASS", launch=None, evaluation=None
):
    intent = _intent(tmp_path)
    monkeypatch.setattr(runner, "build_certification_intent", lambda *a, **k: intent)
    monkeypatch.setattr(
        runner,
        "load_ground_truth_reference",
        lambda *a, **k: SimpleNamespace(identity={"id": "gt"}),
    )
    monkeypatch.setattr(
        runner,
        "execute_recipe",
        lambda *a, **k: launch or _launch(tmp_path, intent, {"id": "gt"}),
    )
    monkeypatch.setattr(runner, "load_generated_artifact", lambda *a, **k: object())
    monkeypatch.setattr(
        runner,
        "evaluate_generated_run",
        lambda *a, **k: evaluation or _evaluation(verdict),
    )
    return intent


@pytest.mark.parametrize("verdict", ["PASS", "REVIEW", "FAIL"])
def test_verdicts_save_recipe_and_rich_attempt(monkeypatch, tmp_path, verdict):
    intent = _patch_common(monkeypatch, tmp_path, verdict=verdict)
    outcome = runner.emit_low_recipe_for_cell(
        "mvn_10",
        "no_warmup",
        "hmc",
        catalog_root=tmp_path,
        outcomes_file=tmp_path / "outcomes.md",
        n_samples=2,
        num_chains=1,
    )
    assert outcome.verdict == verdict
    saved = Recipe.load(intent.recipe_path)
    attempt = saved.attempted_configurations[-1]
    assert attempt["automatic_verdict"] == verdict
    assert attempt["execution"]["receipt"]["run_id"] == "run-123"
    assert attempt["execution"]["telemetry"]["schema"] == "telemetry"
    assert attempt["metrics"]["gate"]["verdict"] == verdict


def test_generated_program_failure_retains_typed_launch_evidence(monkeypatch, tmp_path):
    intent = _patch_common(monkeypatch, tmp_path)
    launch = _launch(tmp_path, intent, {"id": "gt"}, status="failed")
    monkeypatch.setattr(
        runner,
        "execute_recipe",
        lambda *a, **k: (_ for _ in ()).throw(GeneratedProgramError("boom", launch)),
    )
    outcome = runner.emit_low_recipe_for_cell(
        "mvn_10",
        "no_warmup",
        "hmc",
        catalog_root=tmp_path,
        n_samples=2,
        num_chains=1,
    )
    attempt = Recipe.load(intent.recipe_path).attempted_configurations[-1]
    assert outcome.verdict == "ERROR"
    assert attempt["failure_evidence"]["stage"] == "generated execution"
    assert attempt["execution"]["receipt"]["status"] == "failed"
    assert attempt["execution"]["stdout_path"].endswith("stdout.log")


def test_missing_ground_truth_records_error_without_launch(monkeypatch, tmp_path):
    intent = _intent(tmp_path)
    monkeypatch.setattr(runner, "build_certification_intent", lambda *a, **k: intent)
    monkeypatch.setattr(
        runner,
        "load_ground_truth_reference",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    launched = False

    def fail_launch(*a, **k):
        nonlocal launched
        launched = True
        raise AssertionError("must not launch")

    monkeypatch.setattr(runner, "execute_recipe", fail_launch)
    outcome = runner.emit_low_recipe_for_cell(
        "mvn_10",
        "no_warmup",
        "hmc",
        catalog_root=tmp_path,
        n_samples=2,
        num_chains=1,
    )
    assert outcome.verdict == "ERROR" and not launched
    attempt = Recipe.load(intent.recipe_path).attempted_configurations[-1]
    assert attempt["lifecycle_stage"] == "DRAFT"
    assert attempt["failure_evidence"]["stage"] == "ground-truth preflight"


def test_evaluator_error_after_launch_records_receipt(monkeypatch, tmp_path):
    intent = _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "load_generated_artifact",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad artifact")),
    )
    outcome = runner.emit_low_recipe_for_cell(
        "mvn_10",
        "no_warmup",
        "hmc",
        catalog_root=tmp_path,
        n_samples=2,
        num_chains=1,
    )
    attempt = Recipe.load(intent.recipe_path).attempted_configurations[-1]
    assert outcome.verdict == "ERROR"
    assert attempt["failure_evidence"]["stage"] == "artifact evaluation"
    assert attempt["execution"]["receipt"]["run_id"] == "run-123"


@pytest.mark.parametrize("mismatch", ["gt", "plan"])
def test_execution_binding_mismatch_is_error_with_receipt(
    monkeypatch, tmp_path, mismatch
):
    intent = _intent(tmp_path)
    monkeypatch.setattr(runner, "build_certification_intent", lambda *a, **k: intent)
    monkeypatch.setattr(
        runner,
        "load_ground_truth_reference",
        lambda *a, **k: SimpleNamespace(identity={"id": "gt"}),
    )
    monkeypatch.setattr(
        runner,
        "execute_recipe",
        lambda *a, **k: _launch(tmp_path, intent, {"id": "gt"}, mismatch=mismatch),
    )
    monkeypatch.setattr(runner, "load_generated_artifact", lambda *a, **k: object())
    monkeypatch.setattr(
        runner, "evaluate_generated_run", lambda *a, **k: _evaluation("PASS")
    )

    outcome = runner.emit_low_recipe_for_cell(
        "mvn_10", "no_warmup", "hmc", catalog_root=tmp_path
    )
    attempt = Recipe.load(intent.recipe_path).attempted_configurations[-1]
    assert outcome.verdict == "ERROR"
    assert attempt["failure_evidence"]["stage"] == "execution binding"
    assert attempt["execution"]["receipt"]["run_id"] == "run-123"


def test_existing_attempts_legacy_view_and_extensions_survive(monkeypatch, tmp_path):
    existing = _recipe(
        attempted_configurations=[{"raw": {"opaque": [1, None]}}],
        headline_basis={"ess": "bulk"},
        _extra_fields={"future_extension": {"x": None}},
    )
    intent = _intent(tmp_path, existing)
    intent.recipe_path.parent.mkdir(parents=True)
    existing.save(tmp_path, filename_tag=None)
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "build_certification_intent", lambda *a, **k: intent)
    runner.emit_low_recipe_for_cell("mvn_10", "no_warmup", "hmc", catalog_root=tmp_path)
    saved = Recipe.load(intent.recipe_path)
    assert saved.attempted_configurations[0] == {"raw": {"opaque": [1, None]}}
    assert any(
        "legacy_current_view" in a.get("metrics", {})
        for a in saved.attempted_configurations
    )
    assert saved._extra_fields["future_extension"] == {"x": None}


def test_corrupt_existing_recipe_is_not_overwritten(monkeypatch, tmp_path):
    intent = _intent(tmp_path)
    intent.recipe_path.parent.mkdir(parents=True)
    intent.recipe_path.write_text("{not-json")
    before = intent.recipe_path.read_bytes()
    monkeypatch.setattr(runner, "build_certification_intent", lambda *a, **k: intent)
    monkeypatch.setattr(
        runner,
        "execute_recipe",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not launch")),
    )
    outcome = runner.emit_low_recipe_for_cell(
        "mvn_10", "no_warmup", "hmc", catalog_root=tmp_path
    )
    assert outcome.verdict == "ERROR"
    assert intent.recipe_path.read_bytes() == before


def test_large_geometry_sidecars_are_attempt_specific_and_hash_recorded(
    monkeypatch, tmp_path
):
    intent = _intent(tmp_path)
    launches = iter(
        [
            _launch(tmp_path, intent, {"id": "gt"}, run_id="run-a"),
            _launch(tmp_path, intent, {"id": "gt"}, run_id="run-b"),
        ]
    )
    monkeypatch.setattr(runner, "build_certification_intent", lambda *a, **k: intent)
    monkeypatch.setattr(
        runner,
        "load_ground_truth_reference",
        lambda *a, **k: SimpleNamespace(identity={"id": "gt"}),
    )
    monkeypatch.setattr(runner, "execute_recipe", lambda *a, **k: next(launches))
    monkeypatch.setattr(runner, "load_generated_artifact", lambda *a, **k: object())
    evaluation = _evaluation("PASS")
    evaluation = GeneratedEvaluation(
        **{
            **evaluation.__dict__,
            "pinned_base_method_params": {"inverse_mass_matrix": np.eye(51)},
        }
    )
    monkeypatch.setattr(runner, "evaluate_generated_run", lambda *a, **k: evaluation)

    for _ in range(2):
        outcome = runner.emit_low_recipe_for_cell(
            "mvn_10",
            "no_warmup",
            "hmc",
            catalog_root=tmp_path,
            n_samples=2,
            num_chains=1,
        )
        assert outcome.verdict == "PASS"
    saved = Recipe.load(intent.recipe_path)
    attempts = saved.attempted_configurations[-2:]
    paths = [a["metrics"]["derived_geometry_sidecar"]["path"] for a in attempts]
    assert paths[0] != paths[1]
    for attempt, relative in zip(attempts, paths):
        sidecar = tmp_path / relative
        assert sidecar.exists()
        assert (
            attempt["metrics"]["derived_geometry_sidecar"]["sha256"]
            == hashlib.sha256(sidecar.read_bytes()).hexdigest()
        )


def test_recipe_persistence_error_records_evaluated_attempt(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)
    captured = []
    original = runner._record_error

    def capture(**kwargs):
        captured.append(kwargs["base_recipe"])
        return original(**kwargs)

    monkeypatch.setattr(runner, "_record_error", capture)
    monkeypatch.setattr(
        certification_io,
        "persist_recipe_atomically",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    outcome = runner.emit_low_recipe_for_cell(
        "mvn_10",
        "no_warmup",
        "hmc",
        catalog_root=tmp_path,
        n_samples=2,
        num_chains=1,
    )
    assert outcome.verdict == "ERROR"
    assert captured[-1].attempted_configurations[-1]["automatic_verdict"] == "PASS"


def test_non_json_current_data_fails_closed(tmp_path):
    bad = _recipe(_extra_fields={"future": object()})
    intent = _intent(tmp_path)
    with pytest.raises(TypeError):
        certification_io.persist_recipe_atomically(bad, intent, tmp_path)


def test_binding_accepts_receipt_snapshot_canonicalization(tmp_path):
    point = namedtuple("Point", "x y")(1, 2)
    intent = _intent(tmp_path, _recipe(base_method_params={"point": point}))
    assert (
        intent.recipe.to_dict(include_legacy_warmup_fields=True)["base_method_params"][
            "point"
        ]
        == point
    )
    assert canonical_recipe_snapshot(intent.recipe)["base_method_params"]["point"] == [
        1,
        2,
    ]
    result = _launch(tmp_path, intent, {"id": "gt"})

    verify_launch_binding(
        result,
        intent,
        SimpleNamespace(identity={"id": "gt"}),
        n_warmup=1000,
        n_samples=2,
        num_chains=1,
        seed=20260517,
        progress_bar=True,
    )


def test_binding_uses_thawed_receipt_identity(tmp_path):
    intent = _intent(tmp_path)
    result = _launch(tmp_path, intent, {"id": ["gt"]})
    frozen_identity = {
        "tuningfork_recipe_evidence": {
            "snapshot": tuple(),
            "caller_reference_identity": {"id": ("gt",)},
        }
    }
    thawed_identity = {
        "tuningfork_recipe_evidence": {
            "snapshot": canonical_recipe_snapshot(intent.recipe),
            "caller_reference_identity": {"id": ["gt"]},
        }
    }
    receipt = SimpleNamespace(
        manifest=result.manifest,
        reference_identity=frozen_identity,
        as_dict=lambda: {"reference_identity": thawed_identity},
    )
    result = replace(result, receipt=receipt)

    verify_launch_binding(
        result,
        intent,
        SimpleNamespace(identity={"id": ["gt"]}),
        n_warmup=1000,
        n_samples=2,
        num_chains=1,
        seed=20260517,
        progress_bar=True,
    )
