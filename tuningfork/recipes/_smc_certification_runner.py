# Copyright 2026- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Codegen-only orchestration for SMC recipe certification."""

from __future__ import annotations

import copy
import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tuningfork.calibration.smc_gate import SMCGateVerdict
from tuningfork.catalog.emit import (
    RECIPE_EVIDENCE_HASH_DOMAIN,
    RECIPE_EVIDENCE_KEY,
    RECIPE_EVIDENCE_SCHEMA,
    canonical_recipe_snapshot,
    execute_recipe,
)
from tuningfork.recipes._base_smc import SMCRecipe
from tuningfork.recipes._execution_plan import canonical_json
from tuningfork.recipes._generated_smc import (
    GeneratedSMCEvaluation,
    evaluate_generated_smc,
    load_generated_smc_artifact,
)
from tuningfork.recipes._ground_truth_reference import (
    GroundTruthReference,
    load_ground_truth_reference,
)
from tuningfork.recipes._launcher import GeneratedProgramError, LaunchResult
from tuningfork.recipes._smc_certification_record import (
    append_smc_certification_attempt,
    import_legacy_current_view,
)
from tuningfork.recipes._smc_execution_plan import resolve_smc_execution_plan
from tuningfork.recipes._smc_execution_telemetry import SMCExecutionTelemetry

DEFAULT_CATALOG_ROOT = Path(__file__).parent.parent / "catalog"
DEFAULT_NUM_PARTICLES = 1000
DEFAULT_MAX_STEPS = 500
DEFAULT_SEED = 20260517

_EMPTY_OVERRIDE = {"reason": "", "statistician_id": "", "decision": ""}


@dataclass(frozen=True)
class SMCCellResult:
    """Outcome and durable evidence locations for one SMC attempt."""

    model_name: str
    smc_method_name: str
    inner_method_name: str
    verdict: str
    recipe_path: Path | None = None
    receipt_path: Path | None = None
    gate_verdict: SMCGateVerdict | None = None
    particle_ess: float | None = None
    max_abs_mean_z: float | None = None
    headline_metric: float | None = None
    wall_seconds: float = 0.0
    note: str = ""
    attempt_id: str | None = None


def _recipe_path(root: Path, recipe: SMCRecipe) -> Path:
    stem = f"smc__{recipe.smc_method_name}__{recipe.inner_method_name}.json"
    return root / recipe.model_name / "recipes" / stem


def _has_prior_evidence(recipe: SMCRecipe) -> bool:
    return bool(
        recipe.attempted_configurations
        or recipe.notes
        or recipe.workflow
        or recipe.failure_diagnosis
        or recipe.headline_metric is not None
        or recipe.calibration_budget
        or recipe.inner_params_final is not None
        or recipe._extra_fields
        or recipe.verdict != "NOT_RUN"
    )


def _merge_existing_recipe(
    intent: SMCRecipe,
    existing: SMCRecipe | None,
    ground_truth: Mapping[str, Any] | None,
) -> SMCRecipe:
    """Use the requested intent while retaining all earlier recipe evidence."""
    prior = existing
    if prior is None and _has_prior_evidence(intent):
        prior = intent
    if prior is None:
        return intent
    migrated = import_legacy_current_view(prior, ground_truth=ground_truth)
    return replace(
        intent,
        attempted_configurations=copy.deepcopy(migrated.attempted_configurations),
        gate_evidence=copy.deepcopy(migrated.gate_evidence),
        calibration_budget=copy.deepcopy(migrated.calibration_budget),
        notes=migrated.notes,
        workflow=migrated.workflow,
        _extra_fields=copy.deepcopy(migrated._extra_fields),
    )


def _json_value(value: Any) -> Any:
    """Return an independent JSON-shaped value from immutable/array evidence."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("evidence mappings must have string keys")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_value(tolist())
    return copy.deepcopy(value)


def _gate_evidence(base: SMCRecipe, automatic: Mapping[str, Any]) -> dict[str, Any]:
    """Start a new automatic gate without applying an older human override."""
    extensions = {
        key: copy.deepcopy(value)
        for key, value in base.gate_evidence.items()
        if key not in {"auto", "override"}
    }
    return {
        **extensions,
        "auto": copy.deepcopy(dict(automatic)),
        "override": copy.deepcopy(_EMPTY_OVERRIDE),
    }


def _telemetry_evidence(result: LaunchResult | None) -> dict[str, Any] | None:
    telemetry = None if result is None else result.telemetry
    if telemetry is None:
        return None
    as_dict = getattr(telemetry, "as_dict", None)
    if not callable(as_dict):
        return {"invalid_type": type(telemetry).__name__}
    return _json_value(as_dict())


def _measurement_conditions(
    intent: SMCRecipe, result: LaunchResult | None
) -> dict[str, Any]:
    conditions: dict[str, Any] = {
        "requested": {
            "num_particles": intent.num_particles,
            "max_steps": intent.max_steps,
            "seed": intent.seed,
        }
    }
    try:
        plan = resolve_smc_execution_plan(intent)
    except Exception:  # the error attempt itself records plan-resolution failure
        plan = None
    if plan is not None:
        conditions["resolved"] = {
            "executable_config": plan.config.as_dict(),
            "executable_config_hash": plan.executable_config_hash,
            "plan_hash": plan.plan_hash,
            "recipe_ref": plan.recipe_ref,
        }
    if result is not None:
        conditions["generated"] = {
            "manifest": _json_value(result.manifest.as_dict()),
            "telemetry": _telemetry_evidence(result),
        }
        if result.timings is not None:
            conditions["generated"]["launch_timings"] = {
                "warmup_seconds": result.timings.warmup_seconds,
                "sampling_seconds": result.timings.sampling_seconds,
                "total_seconds": result.timings.total_seconds,
            }
    return conditions


def _failure_evidence(stage: str, error: BaseException) -> dict[str, Any]:
    return {
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error),
        "diagnosis": None,
        "intervention": None,
        "learned": None,
    }


def _error_auto() -> dict[str, Any]:
    return {
        "verdict": "ERROR",
        "max_abs_mean_z": None,
        "particle_ess": None,
        "particle_ess_fraction": None,
        "mode_coverage_fraction": None,
        "lambda_final": None,
        "rhat_max": None,
        "n_divergences": None,
    }


def _lifecycle(result: LaunchResult | None, *, evaluated: bool = False) -> str:
    if result is None:
        return "DRAFT"
    if getattr(result.receipt, "status", None) != "success":
        return "GENERATED"
    return "EVALUATED" if evaluated else "SAMPLED"


def _try_persist_attempt(
    base: SMCRecipe,
    intent: SMCRecipe,
    root: Path,
    *,
    result: LaunchResult | None,
    reference: GroundTruthReference | None,
    lifecycle_stage: str,
    automatic_verdict: str,
    rationale: str,
    metrics: Mapping[str, Any] | None,
    gate_evidence: Mapping[str, Any] | None,
    failure_evidence: Mapping[str, Any] | None,
    recipe_updates: Mapping[str, Any],
) -> tuple[SMCRecipe, Path | None, str | None, str | None]:
    attempt_id: str | None = None
    try:
        updated, attempt_id = append_smc_certification_attempt(
            base,
            intent,
            result=result,
            ground_truth=None if reference is None else reference.identity,
            lifecycle_stage=lifecycle_stage,
            automatic_verdict=automatic_verdict,
            rationale=rationale,
            measurement_conditions=_measurement_conditions(intent, result),
            metrics=metrics,
            gate_evidence=gate_evidence,
            failure_evidence=failure_evidence,
            recipe_updates=recipe_updates,
        )
        budget = copy.deepcopy(updated.calibration_budget)
        budget["selected_attempt_id"] = attempt_id
        updated = replace(updated, calibration_budget=budget)
        return updated, updated.save(root), attempt_id, None
    except Exception as error:  # noqa: BLE001
        note = (
            "attempt recording/persistence failed: " f"{type(error).__name__}: {error}"
        )
        return base, None, attempt_id, note


def _error_result(
    *,
    base: SMCRecipe,
    intent: SMCRecipe,
    root: Path,
    result: LaunchResult | None,
    reference: GroundTruthReference | None,
    stage: str,
    error: BaseException,
    rationale: str,
    started_at: float,
) -> SMCCellResult:
    note = f"ERROR during {stage}: {type(error).__name__}: {error}"
    current_gate = _gate_evidence(base, _error_auto())
    budget = copy.deepcopy(base.calibration_budget)
    budget.update(
        {
            "num_particles": intent.num_particles,
            "max_steps": intent.max_steps,
            "generated_run_id": (
                None if result is None else getattr(result.receipt, "run_id", None)
            ),
        }
    )
    _, recipe_path, attempt_id, persistence_error = _try_persist_attempt(
        base,
        intent,
        root,
        result=result,
        reference=reference,
        lifecycle_stage=_lifecycle(result),
        automatic_verdict="ERROR",
        rationale=rationale,
        metrics=None,
        gate_evidence=current_gate,
        failure_evidence=_failure_evidence(stage, error),
        recipe_updates={
            "calibration_budget": budget,
            "gate_evidence": current_gate,
            "failure_diagnosis": note,
            "inner_params_final": None,
        },
    )
    if persistence_error is not None:
        note = f"{note}; {persistence_error}"
    return SMCCellResult(
        model_name=intent.model_name,
        smc_method_name=intent.smc_method_name,
        inner_method_name=intent.inner_method_name,
        verdict="ERROR",
        recipe_path=recipe_path,
        receipt_path=None if result is None else result.receipt_path,
        wall_seconds=time.perf_counter() - started_at,
        note=note,
        attempt_id=attempt_id,
    )


def _verify_launch_binding(
    result: LaunchResult,
    intent: SMCRecipe,
    reference: GroundTruthReference,
) -> None:
    expected = resolve_smc_execution_plan(intent)
    manifest = result.manifest
    if (
        manifest.plan_hash != expected.plan_hash
        or manifest.executable_config_hash != expected.executable_config_hash
        or manifest.recipe_ref != expected.recipe_ref
        or manifest.as_dict()["executable_config"] != expected.config.as_dict()
    ):
        raise ValueError(
            "generated receipt does not match the requested SMC execution plan"
        )
    if result.receipt.manifest != manifest:
        raise ValueError("launch result and receipt contain different manifests")
    if result.receipt.status != "success":
        raise ValueError("generated SMC execution did not produce a success receipt")
    if (
        result.artifact_path is None
        or result.telemetry_path is None
        or not isinstance(result.telemetry, SMCExecutionTelemetry)
    ):
        raise ValueError(
            "successful generated SMC execution lacks artifact or typed telemetry"
        )
    telemetry = result.telemetry
    if (
        telemetry.plan_hash != manifest.plan_hash
        or telemetry.executable_config_hash != manifest.executable_config_hash
        or telemetry.draws_artifact != manifest.normalized_plan["artifact_filename"]
    ):
        raise ValueError("SMC telemetry hashes do not match execution manifest")

    receipt = result.receipt.as_dict()
    identity = receipt.get("reference_identity")
    envelope = (
        identity.get(RECIPE_EVIDENCE_KEY) if isinstance(identity, Mapping) else None
    )
    if not isinstance(envelope, Mapping):
        raise ValueError("receipt lacks the SMC recipe evidence envelope")
    snapshot = canonical_recipe_snapshot(intent)
    expected_hash = hashlib.sha256(
        (RECIPE_EVIDENCE_HASH_DOMAIN + canonical_json(snapshot)).encode("utf-8")
    ).hexdigest()
    if (
        envelope.get("schema") != RECIPE_EVIDENCE_SCHEMA
        or envelope.get("snapshot") != snapshot
        or envelope.get("snapshot_sha256") != expected_hash
    ):
        raise ValueError("receipt recipe snapshot does not match SMC intent")
    if envelope.get("caller_reference_identity") != reference.identity:
        raise ValueError(
            "receipt ground-truth identity does not match SMC evaluation input"
        )


def _compact_inner_params(
    values: Mapping[str, Any], num_particles: int
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    """Keep recipe JSON small while the receipt retains the full artifact."""
    compact: dict[str, Any] = {}
    shapes: dict[str, list[int]] = {}
    for name, value in values.items():
        shape = tuple(getattr(value, "shape", ()))
        shapes[name] = list(shape)
        representative = value[0] if shape and shape[0] == num_particles else value
        compact[name] = _json_value(representative)
    return compact, shapes


def _evaluation_metrics(
    evaluation: GeneratedSMCEvaluation,
    final_inner_params: Mapping[str, Any],
    final_inner_param_shapes: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "gate": evaluation.gate.to_dict(),
        "headline_metric": evaluation.headline_metric,
        "total_cost": evaluation.total_cost,
        "lambda_final": evaluation.lambda_final,
        "history": _json_value(evaluation.history),
        "final_inner_params": _json_value(final_inner_params),
        "final_inner_param_shapes": _json_value(final_inner_param_shapes),
        "full_final_inner_params": "retained in the generated artifact",
        "ground_truth_identity": _json_value(evaluation.ground_truth_identity),
    }


def _gate_failure(
    gate: SMCGateVerdict, evaluation: GeneratedSMCEvaluation
) -> tuple[str | None, dict[str, Any] | None]:
    if gate.verdict != "FAIL":
        return None, None
    diagnosis = (
        "automatic SMC gate FAIL: "
        f"lambda_final={evaluation.lambda_final}, "
        f"max_abs_mean_z={gate.max_abs_mean_z}, "
        f"particle_ess={gate.particle_ess}, "
        f"mode_coverage_fraction={gate.mode_coverage_fraction}"
    )
    return diagnosis, {
        "stage": "gate evaluation",
        "error_type": None,
        "message": diagnosis,
        "diagnosis": None,
        "intervention": None,
        "learned": None,
    }


def certify_smc_recipe(
    recipe: SMCRecipe,
    *,
    catalog_root: Path,
    timeout: float | None = None,
    verbose: bool = False,
) -> SMCCellResult:
    """Generate, launch, evaluate, and durably record one SMC attempt."""
    del verbose
    if not isinstance(recipe, SMCRecipe):
        raise TypeError("recipe must be an SMCRecipe")
    root = Path(catalog_root)
    started_at = time.perf_counter()
    path = _recipe_path(root, recipe)

    existing: SMCRecipe | None = None
    if path.exists():
        try:
            existing = SMCRecipe.load(path)
        except Exception as error:  # noqa: BLE001
            return SMCCellResult(
                recipe.model_name,
                recipe.smc_method_name,
                recipe.inner_method_name,
                "ERROR",
                wall_seconds=time.perf_counter() - started_at,
                note=(
                    "ERROR loading existing recipe without overwriting it: "
                    f"{type(error).__name__}: {error}"
                ),
            )

    try:
        reference = load_ground_truth_reference(root, recipe.model_name)
    except Exception as error:  # noqa: BLE001
        base = _merge_existing_recipe(recipe, existing, None)
        return _error_result(
            base=base,
            intent=recipe,
            root=root,
            result=None,
            reference=None,
            stage="ground-truth preflight",
            error=error,
            rationale="Canonical ground truth is required before SMC sampling",
            started_at=started_at,
        )

    base = _merge_existing_recipe(recipe, existing, reference.identity)
    run_root = root / recipe.model_name / "_cache" / "generated_runs"
    try:
        result = execute_recipe(
            recipe,
            run_root,
            timeout=timeout,
            reference_identity=reference.identity,
        )
    except GeneratedProgramError as error:
        return _error_result(
            base=base,
            intent=recipe,
            root=root,
            result=error.result,
            reference=reference,
            stage="generated execution",
            error=error,
            rationale="Execute the requested SMC configuration through codegen",
            started_at=started_at,
        )
    except Exception as error:  # noqa: BLE001
        return _error_result(
            base=base,
            intent=recipe,
            root=root,
            result=None,
            reference=reference,
            stage="code generation or launch",
            error=error,
            rationale="Execute the requested SMC configuration through codegen",
            started_at=started_at,
        )

    try:
        _verify_launch_binding(result, recipe, reference)
    except Exception as error:  # noqa: BLE001
        return _error_result(
            base=base,
            intent=recipe,
            root=root,
            result=result,
            reference=reference,
            stage="execution binding",
            error=error,
            rationale="Bind generated evidence to exact SMC intent and ground truth",
            started_at=started_at,
        )

    try:
        if result.artifact_path is None:
            raise ValueError("successful generated SMC execution lacks an artifact")
        artifact = load_generated_smc_artifact(result.artifact_path, result.manifest)
        evaluation = evaluate_generated_smc(artifact, result.manifest, reference)
        verdict = evaluation.gate.verdict
        if verdict not in {"PASS", "REVIEW", "FAIL"}:
            raise ValueError(f"unsupported automatic verdict: {verdict!r}")
    except Exception as error:  # noqa: BLE001
        return _error_result(
            base=base,
            intent=recipe,
            root=root,
            result=result,
            reference=reference,
            stage="artifact evaluation",
            error=error,
            rationale="Evaluate only the generated SMC artifact",
            started_at=started_at,
        )

    final_inner_params, final_inner_param_shapes = _compact_inner_params(
        artifact.final_inner_params, recipe.num_particles
    )
    gate = evaluation.gate.to_dict()
    current_gate = _gate_evidence(base, gate)
    failure_diagnosis, failure = _gate_failure(evaluation.gate, evaluation)
    telemetry = result.telemetry
    assert isinstance(telemetry, SMCExecutionTelemetry)
    budget = copy.deepcopy(base.calibration_budget)
    budget.update(
        {
            "num_particles": telemetry.num_particles,
            "max_steps": recipe.max_steps,
            "n_smc_steps": telemetry.num_smc_steps,
            "lambda_final": evaluation.lambda_final,
            "total_cost": evaluation.total_cost,
            "timing_seconds": dict(telemetry.timing_seconds),
            "generated_run_id": result.receipt.run_id,
        }
    )
    _, recipe_path, attempt_id, persistence_error = _try_persist_attempt(
        base,
        recipe,
        root,
        result=result,
        reference=reference,
        lifecycle_stage="EVALUATED",
        automatic_verdict=verdict,
        rationale="Evaluate the requested SMC configuration through generated code",
        metrics=_evaluation_metrics(
            evaluation, final_inner_params, final_inner_param_shapes
        ),
        gate_evidence=current_gate,
        failure_evidence=failure,
        recipe_updates={
            "headline_metric": evaluation.headline_metric,
            "inner_params_final": (final_inner_params if verdict == "PASS" else None),
            "gate_evidence": current_gate,
            "calibration_budget": budget,
            "failure_diagnosis": failure_diagnosis,
        },
    )
    note = (
        f"{verdict} lambda={evaluation.lambda_final} "
        f"z={evaluation.gate.max_abs_mean_z} "
        f"ess={evaluation.gate.particle_ess}"
    )
    if persistence_error is not None:
        return SMCCellResult(
            recipe.model_name,
            recipe.smc_method_name,
            recipe.inner_method_name,
            "ERROR",
            receipt_path=result.receipt_path,
            gate_verdict=evaluation.gate,
            particle_ess=evaluation.gate.particle_ess,
            max_abs_mean_z=evaluation.gate.max_abs_mean_z,
            headline_metric=evaluation.headline_metric,
            wall_seconds=time.perf_counter() - started_at,
            note=f"{note}; {persistence_error}",
            attempt_id=attempt_id,
        )
    return SMCCellResult(
        recipe.model_name,
        recipe.smc_method_name,
        recipe.inner_method_name,
        verdict,
        recipe_path=recipe_path,
        receipt_path=result.receipt_path,
        gate_verdict=evaluation.gate,
        particle_ess=evaluation.gate.particle_ess,
        max_abs_mean_z=evaluation.gate.max_abs_mean_z,
        headline_metric=evaluation.headline_metric,
        wall_seconds=time.perf_counter() - started_at,
        note=note,
        attempt_id=attempt_id,
    )


def _append_outcome(path: Path, result: SMCCellResult) -> None:
    """Write the old summary log best-effort; recipe evidence is authoritative."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"- [{result.model_name}/{result.smc_method_name}/"
                f"{result.inner_method_name}] {result.verdict}: {result.note}\n"
            )
    except OSError:
        pass


def emit_smc_recipe_for_cell(
    model_name: str,
    smc_method_name: str,
    inner_method_name: str,
    *,
    num_particles: int = DEFAULT_NUM_PARTICLES,
    max_steps: int = DEFAULT_MAX_STEPS,
    smc_params: dict[str, Any] | None = None,
    inner_params_init: dict[str, Any] | None = None,
    parameter_update_strategy: str | None = None,
    parameter_update_strategy_kwargs: dict[str, Any] | None = None,
    seed: int = DEFAULT_SEED,
    catalog_root: Path = DEFAULT_CATALOG_ROOT,
    outcomes_file: Path | None = None,
    timeout: float | None = None,
    verbose: bool = True,
) -> SMCCellResult:
    """Compatibility entry point backed exclusively by generated execution."""
    if parameter_update_strategy is None:
        parameter_update_strategy = (
            "step_size_and_imm_from_particles"
            if smc_method_name == "inner_kernel_tuning"
            else "none"
        )
    recipe = SMCRecipe.from_default_config(
        model_name,
        smc_method_name,
        inner_method_name,
        num_particles=num_particles,
        max_steps=max_steps,
        seed=seed,
        smc_params=smc_params,
        inner_params_init=inner_params_init,
        parameter_update_strategy=parameter_update_strategy,
        parameter_update_strategy_kwargs=parameter_update_strategy_kwargs,
    )
    result = certify_smc_recipe(
        recipe,
        catalog_root=Path(catalog_root),
        timeout=timeout,
        verbose=verbose,
    )
    if outcomes_file is not None and result.verdict != "PASS":
        _append_outcome(Path(outcomes_file), result)
    return result


__all__ = [
    "DEFAULT_CATALOG_ROOT",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_NUM_PARTICLES",
    "DEFAULT_SEED",
    "SMCCellResult",
    "certify_smc_recipe",
    "emit_smc_recipe_for_cell",
]
