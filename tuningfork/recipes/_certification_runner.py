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
"""Codegen-only orchestration for MCMC recipe certification."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from tuningfork._machine_info import get_machine_info
from tuningfork.catalog.emit import execute_recipe
from tuningfork.recipes import _certification_io as certification_io
from tuningfork.recipes._base import Effort, Recipe
from tuningfork.recipes._certification_binding import verify_launch_binding
from tuningfork.recipes._certification_intent import (
    CertificationIntent,
    build_certification_intent,
)
from tuningfork.recipes._certification_record import append_certification_attempt
from tuningfork.recipes._execution_telemetry import ExecutionTelemetry
from tuningfork.recipes._generated_certification import (
    GeneratedEvaluation,
    evaluate_generated_run,
)
from tuningfork.recipes._generated_evaluator import load_generated_artifact
from tuningfork.recipes._ground_truth_reference import (
    GroundTruthReference,
    load_ground_truth_reference,
)
from tuningfork.recipes._laplace_config import LAPLACE_PHI_THETA_SPLITS
from tuningfork.recipes._launcher import GeneratedProgramError, LaunchResult
from tuningfork.warmup._laplace_adapter import LAPLACE_METHOD_NAMES

RECIPE_N_WARMUP = 1000
RECIPE_N_SAMPLES = 1000
RECIPE_NUM_CHAINS = 4
RECIPE_SEED = 20260517
RECIPE_N_CHUNKS = 4
RECIPE_TARGET_ACCEPTANCE = 0.8

DEFAULT_CATALOG_ROOT = Path(__file__).parent.parent / "catalog"
DEFAULT_OUTCOMES_FILE = Path("/tmp/recipe-runner-outcomes.md")


@dataclass(frozen=True)
class CellResult:
    """Outcome and durable evidence locations for one certification attempt."""

    model_name: str
    warmup_name: str
    sampler_name: str
    verdict: str
    recipe_path: Path | None = None
    imm_sidecar_path: str | None = None
    gate_rhat_max: float | None = None
    gate_min_ess: float | None = None
    gate_n_div: int | None = None
    wall_seconds: float = 0.0
    note: str = ""
    warmup_grad_evals: int | None = None
    sampling_grad_evals: int | None = None
    receipt_path: Path | None = None
    attempt_id: str | None = None

    def __repr__(self) -> str:
        return (
            f"CellResult({self.model_name}/{self.warmup_name}/{self.sampler_name} "
            f"verdict={self.verdict} wall={self.wall_seconds:.1f}s)"
        )


def _append_outcome(
    path: Path,
    model_name: str,
    warmup_name: str,
    sampler_name: str,
    message: str,
) -> None:
    """Best-effort compatibility log; the recipe attempt is authoritative."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"- {model_name} x {warmup_name} x {sampler_name}: {message}\n"
            )
    except OSError:
        # Losing this compatibility summary must not hide or invalidate the
        # durable attempt already written to the recipe.
        pass


def _measurement_conditions(
    *,
    n_warmup: int,
    n_samples: int,
    num_chains: int,
    n_chunks: int,
    seed: int,
    result: LaunchResult | None,
) -> dict[str, Any]:
    conditions: dict[str, Any] = {
        "requested": {
            "n_warmup": n_warmup,
            "n_samples": n_samples,
            "num_chains": num_chains,
            "n_total_draws": n_samples * num_chains,
            "n_chunks": n_chunks,
            "tuning_seed": seed,
        }
    }
    if result is not None:
        manifest = result.manifest.as_dict()
        conditions["generated"] = {
            "executable_config": manifest["executable_config"],
            "executable_config_hash": manifest["executable_config_hash"],
            "plan_hash": manifest["plan_hash"],
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


def _calibration_budget(
    intent: CertificationIntent,
    result: LaunchResult | None,
    *,
    wall_seconds: float,
    evaluation: GeneratedEvaluation | None,
) -> dict[str, Any]:
    budget = copy.deepcopy(intent.recipe.calibration_budget)
    timings = None if result is None else result.timings
    warmup_seconds = None if timings is None else timings.warmup_seconds
    sampling_seconds = None if timings is None else timings.sampling_seconds
    measured_total = wall_seconds if timings is None else timings.total_seconds
    n_total = budget.get("n_samples", 0) * budget.get("num_chains", 0)
    budget.update(
        {
            "wall_seconds_estimate": measured_total,
            "warmup_wall_seconds": warmup_seconds,
            "sampling_wall_seconds": sampling_seconds,
            "sampling_seconds_per_draw": (
                None
                if sampling_seconds is None or not n_total
                else sampling_seconds / n_total
            ),
            "split_source": None if timings is None else "measured",
            "machine_info": get_machine_info(),
            "warmup_grad_evals": (
                None if evaluation is None else evaluation.warmup_grad_evals
            ),
            "sampling_grad_evals": (
                None if evaluation is None else evaluation.sampling_grad_evals
            ),
            "generated_run_id": (None if result is None else result.receipt.run_id),
        }
    )
    return budget


def _result(
    *,
    model_name: str,
    warmup_name: str,
    sampler_name: str,
    verdict: str,
    wall_seconds: float,
    note: str,
    recipe_path: Path | None = None,
    imm_sidecar_path: str | None = None,
    evaluation: GeneratedEvaluation | None = None,
    launch_result: LaunchResult | None = None,
    attempt_id: str | None = None,
) -> CellResult:
    gate = {} if evaluation is None else evaluation.gate_evidence.get("auto", {})
    return CellResult(
        model_name=model_name,
        warmup_name=warmup_name,
        sampler_name=sampler_name,
        verdict=verdict,
        recipe_path=recipe_path,
        imm_sidecar_path=imm_sidecar_path,
        gate_rhat_max=gate.get("rhat_max"),
        gate_min_ess=gate.get("min_bulk_ess"),
        gate_n_div=gate.get("n_divergences"),
        wall_seconds=wall_seconds,
        note=note,
        warmup_grad_evals=(
            None if evaluation is None else evaluation.warmup_grad_evals
        ),
        sampling_grad_evals=(
            None if evaluation is None else evaluation.sampling_grad_evals
        ),
        receipt_path=(None if launch_result is None else launch_result.receipt_path),
        attempt_id=attempt_id,
    )


def _record_error(
    *,
    base_recipe: Recipe,
    intent: CertificationIntent,
    result: LaunchResult | None,
    reference: GroundTruthReference | None,
    stage: str,
    error: BaseException,
    rationale: str,
    n_warmup: int,
    n_samples: int,
    num_chains: int,
    n_chunks: int,
    seed: int,
    started_at: float,
    catalog_root: Path,
    outcomes_file: Path,
    metrics: dict[str, Any] | None = None,
    gate_evidence: dict[str, Any] | None = None,
) -> CellResult:
    wall_seconds = time.perf_counter() - started_at
    note = f"ERROR during {stage}: {type(error).__name__}: {error}"
    budget = _calibration_budget(
        intent, result, wall_seconds=wall_seconds, evaluation=None
    )
    updated, attempt_id = append_certification_attempt(
        base_recipe,
        intent.recipe,
        result=result,
        ground_truth=None if reference is None else reference.identity,
        lifecycle_stage=(
            "SAMPLED"
            if result is not None and result.receipt.status == "success"
            else "GENERATED" if result is not None else "DRAFT"
        ),
        automatic_verdict="ERROR",
        rationale=rationale,
        measurement_conditions=_measurement_conditions(
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=num_chains,
            n_chunks=n_chunks,
            seed=seed,
            result=result,
        ),
        metrics=metrics,
        gate_evidence=gate_evidence,
        failure_evidence=_failure_evidence(stage, error),
        recipe_updates={
            "calibration_budget": budget,
            "failure_diagnosis": note,
        },
    )
    budget["selected_attempt_id"] = attempt_id
    updated = replace(updated, calibration_budget=budget)
    try:
        _, recipe_path, sidecar_path = certification_io.persist_recipe_atomically(
            updated, intent, catalog_root
        )
    except Exception as save_error:  # noqa: BLE001
        note = (
            f"{note}; recipe persistence failed: "
            f"{type(save_error).__name__}: {save_error}"
        )
        recipe_path = None
        sidecar_path = None
    _append_outcome(
        outcomes_file,
        intent.recipe.model_name,
        intent.recipe.warmup_name,
        intent.recipe.base_method_name,
        note,
    )
    return _result(
        model_name=intent.recipe.model_name,
        warmup_name=intent.recipe.warmup_name,
        sampler_name=intent.recipe.base_method_name,
        verdict="ERROR",
        wall_seconds=wall_seconds,
        note=note,
        recipe_path=recipe_path,
        imm_sidecar_path=sidecar_path,
        launch_result=result,
        attempt_id=attempt_id,
    )


def emit_low_recipe_for_cell(
    model_name: str,
    warmup_name: str,
    sampler_name: str,
    *,
    n_warmup: int = RECIPE_N_WARMUP,
    n_samples: int = RECIPE_N_SAMPLES,
    num_chains: int = RECIPE_NUM_CHAINS,
    seed: int = RECIPE_SEED,
    n_chunks: int = RECIPE_N_CHUNKS,
    catalog_root: Path = DEFAULT_CATALOG_ROOT,
    outcomes_file: Path = DEFAULT_OUTCOMES_FILE,
    verbose: bool = True,
    target_acceptance: float | None = None,
    sampler_kwargs_override: dict[str, Any] | None = None,
    warmup_kwargs_override: dict[str, Any] | None = None,
    step_policy: dict[str, Any] | None = None,
    policy_tag: str | None = None,
    effort: Effort = Effort.LOW,
    warmup_inner_kernel: str | None = None,
    init_strategy: dict[str, Any] | None = None,
    variant_label: str | None = None,
    timeout: float | None = None,
) -> CellResult:
    """Generate, launch, evaluate, and durably record one recipe attempt."""
    started_at = time.perf_counter()
    root = Path(catalog_root)
    outcomes = Path(outcomes_file)
    rationale = (
        f"Evaluate requested {getattr(effort, 'value', effort)} configuration "
        "through generated execution"
    )

    try:
        intent = build_certification_intent(
            model_name,
            warmup_name,
            sampler_name,
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=num_chains,
            seed=seed,
            catalog_root=root,
            target_acceptance=target_acceptance,
            sampler_kwargs_override=sampler_kwargs_override,
            warmup_kwargs_override=warmup_kwargs_override,
            step_policy=step_policy,
            policy_tag=policy_tag,
            effort=effort,
            warmup_inner_kernel=warmup_inner_kernel,
            init_strategy=init_strategy,
            variant_label=variant_label,
        )
    except Exception as error:  # noqa: BLE001
        wall_seconds = time.perf_counter() - started_at
        note = f"ERROR during intent validation: " f"{type(error).__name__}: {error}"
        _append_outcome(outcomes, model_name, warmup_name, sampler_name, note)
        return _result(
            model_name=model_name,
            warmup_name=warmup_name,
            sampler_name=sampler_name,
            verdict="ERROR",
            wall_seconds=wall_seconds,
            note=note,
        )

    existing: Recipe | None = None
    if intent.recipe_path.exists():
        try:
            existing = Recipe.load(intent.recipe_path)
        except Exception as error:  # noqa: BLE001
            wall_seconds = time.perf_counter() - started_at
            note = (
                f"ERROR loading existing recipe without overwriting it: "
                f"{type(error).__name__}: {error}"
            )
            _append_outcome(outcomes, model_name, warmup_name, sampler_name, note)
            return _result(
                model_name=model_name,
                warmup_name=warmup_name,
                sampler_name=sampler_name,
                verdict="ERROR",
                wall_seconds=wall_seconds,
                note=note,
            )

    try:
        reference = load_ground_truth_reference(root, model_name)
    except Exception as error:  # noqa: BLE001
        base_recipe = certification_io.merge_existing_recipe(intent, existing, None)
        return _record_error(
            base_recipe=base_recipe,
            intent=intent,
            result=None,
            reference=None,
            stage="ground-truth preflight",
            error=error,
            rationale=rationale,
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=num_chains,
            n_chunks=n_chunks,
            seed=seed,
            started_at=started_at,
            catalog_root=root,
            outcomes_file=outcomes,
        )

    base_recipe = certification_io.merge_existing_recipe(
        intent, existing, reference.identity
    )
    run_root = root / model_name / "_cache" / "generated_runs"
    result: LaunchResult | None = None
    try:
        result = execute_recipe(
            intent.recipe,
            run_root,
            tuning_seed=seed,
            num_samples=n_samples,
            num_chains=num_chains,
            num_warmup=n_warmup,
            progress_bar=verbose,
            warmup_num_chains=intent.recipe.warmup_num_chains,
            timeout=timeout,
            reference_identity=reference.identity,
        )
    except GeneratedProgramError as error:
        return _record_error(
            base_recipe=base_recipe,
            intent=intent,
            result=error.result,
            reference=reference,
            stage="generated execution",
            error=error,
            rationale=rationale,
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=num_chains,
            n_chunks=n_chunks,
            seed=seed,
            started_at=started_at,
            catalog_root=root,
            outcomes_file=outcomes,
        )
    except Exception as error:  # noqa: BLE001
        return _record_error(
            base_recipe=base_recipe,
            intent=intent,
            result=result,
            reference=reference,
            stage="generated execution",
            error=error,
            rationale=rationale,
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=num_chains,
            n_chunks=n_chunks,
            seed=seed,
            started_at=started_at,
            catalog_root=root,
            outcomes_file=outcomes,
        )

    try:
        verify_launch_binding(
            result,
            intent,
            reference,
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=num_chains,
            seed=seed,
            progress_bar=verbose,
        )
    except Exception as error:  # noqa: BLE001
        return _record_error(
            base_recipe=base_recipe,
            intent=intent,
            result=result,
            reference=reference,
            stage="execution binding",
            error=error,
            rationale=rationale,
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=num_chains,
            n_chunks=n_chunks,
            seed=seed,
            started_at=started_at,
            catalog_root=root,
            outcomes_file=outcomes,
        )

    try:
        if (
            result.artifact_path is None
            or result.telemetry is None
            or result.telemetry_path is None
        ):
            raise ValueError("successful generated execution lacks draws or telemetry")
        run_data = load_generated_artifact(result.artifact_path, result.manifest)
        allowed_sites: tuple[str, ...] | None = None
        if sampler_name in LAPLACE_METHOD_NAMES:
            if model_name not in LAPLACE_PHI_THETA_SPLITS:
                raise ValueError(
                    f"Laplace model {model_name!r} has no declared " "phi/theta split"
                )
            allowed_sites = LAPLACE_PHI_THETA_SPLITS[model_name][0]
        evaluation = evaluate_generated_run(
            intent.recipe,
            run_data,
            cast(ExecutionTelemetry, result.telemetry),
            reference,
            n_chunks=n_chunks,
            allowed_sites=allowed_sites,
        )
    except Exception as error:  # noqa: BLE001
        return _record_error(
            base_recipe=base_recipe,
            intent=intent,
            result=result,
            reference=reference,
            stage="artifact evaluation",
            error=error,
            rationale=rationale,
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=num_chains,
            n_chunks=n_chunks,
            seed=seed,
            started_at=started_at,
            catalog_root=root,
            outcomes_file=outcomes,
        )

    gate_evidence = copy.deepcopy(evaluation.gate_evidence)
    auto = gate_evidence.get("auto", {})
    verdict = auto.get("verdict")
    if verdict not in {"PASS", "REVIEW", "FAIL"}:
        return _record_error(
            base_recipe=base_recipe,
            intent=intent,
            result=result,
            reference=reference,
            stage="gate evaluation",
            error=ValueError(f"unsupported automatic verdict: {verdict!r}"),
            rationale=rationale,
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=num_chains,
            n_chunks=n_chunks,
            seed=seed,
            started_at=started_at,
            catalog_root=root,
            outcomes_file=outcomes,
        )
    auto["gt_cert_coverage"] = evaluation.gt_cert_coverage
    wall_seconds = time.perf_counter() - started_at
    metrics = copy.deepcopy(evaluation.metrics)
    metrics.update(
        {
            "headline_basis": copy.deepcopy(evaluation.headline_basis),
            "resolved_step_policy": copy.deepcopy(evaluation.resolved_step_policy),
            "gt_cert_coverage": evaluation.gt_cert_coverage,
        }
    )
    try:
        pinned_params, sidecar_path, sidecar_evidence = (
            certification_io.prepare_geometry_sidecar(
                replace(
                    intent.recipe,
                    base_method_params=copy.deepcopy(
                        evaluation.pinned_base_method_params
                    ),
                ),
                intent,
                root,
                result.receipt.run_id,
            )
        )
    except Exception as error:  # noqa: BLE001
        return _record_error(
            base_recipe=base_recipe,
            intent=intent,
            result=result,
            reference=reference,
            stage="geometry sidecar persistence",
            error=error,
            rationale=rationale,
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=num_chains,
            n_chunks=n_chunks,
            seed=seed,
            started_at=started_at,
            catalog_root=root,
            outcomes_file=outcomes,
            metrics=metrics,
            gate_evidence=gate_evidence,
        )
    if sidecar_evidence is not None:
        metrics["derived_geometry_sidecar"] = sidecar_evidence
    budget = _calibration_budget(
        intent, result, wall_seconds=wall_seconds, evaluation=evaluation
    )
    failure = None
    failure_diagnosis = None
    if verdict == "FAIL":
        failure_diagnosis = (
            "automatic gate FAIL: "
            f"rhat={auto.get('rhat_max')}, "
            f"ess={auto.get('min_bulk_ess')}, "
            f"divergences={auto.get('n_divergences')}"
        )
        failure = {
            "stage": "gate evaluation",
            "error_type": None,
            "message": failure_diagnosis,
            "diagnosis": None,
            "intervention": None,
            "learned": None,
        }
    resolved_policy = evaluation.resolved_step_policy
    if resolved_policy is not None:
        selected_policy = copy.deepcopy(dict(resolved_policy))
    elif intent.recipe.step_policy is not None:
        selected_policy = copy.deepcopy(intent.recipe.step_policy)
    else:
        selected_policy = None
    updated, attempt_id = append_certification_attempt(
        base_recipe,
        intent.recipe,
        result=result,
        ground_truth=reference.identity,
        lifecycle_stage="EVALUATED",
        automatic_verdict=verdict,
        rationale=rationale,
        measurement_conditions=_measurement_conditions(
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=num_chains,
            n_chunks=n_chunks,
            seed=seed,
            result=result,
        ),
        metrics=metrics,
        gate_evidence=gate_evidence,
        failure_evidence=failure,
        recipe_updates={
            "base_method_params": pinned_params,
            "step_policy": selected_policy,
            "headline_metric": evaluation.headline_metric,
            "headline_basis": copy.deepcopy(evaluation.headline_basis),
            "sample_quality": copy.deepcopy(evaluation.sample_quality),
            "calibration_budget": budget,
            "gate_evidence": gate_evidence,
            "failure_diagnosis": failure_diagnosis,
            "inverse_mass_matrix_path": sidecar_path,
        },
    )
    budget["selected_attempt_id"] = attempt_id
    updated = replace(updated, calibration_budget=budget)
    try:
        _, recipe_path, persisted_sidecar_path = (
            certification_io.persist_recipe_atomically(updated, intent, root)
        )
    except Exception as error:  # noqa: BLE001
        return _record_error(
            base_recipe=updated,
            intent=intent,
            result=result,
            reference=reference,
            stage="recipe persistence",
            error=error,
            rationale=rationale,
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=num_chains,
            n_chunks=n_chunks,
            seed=seed,
            started_at=started_at,
            catalog_root=root,
            outcomes_file=outcomes,
            metrics=metrics,
            gate_evidence=gate_evidence,
        )
    note = (
        f"{verdict} rhat={auto.get('rhat_max')} "
        f"ess={auto.get('min_bulk_ess')} "
        f"div={auto.get('n_divergences')}"
    )
    if verdict != "PASS":
        _append_outcome(outcomes, model_name, warmup_name, sampler_name, note)
    return _result(
        model_name=model_name,
        warmup_name=warmup_name,
        sampler_name=sampler_name,
        verdict=verdict,
        wall_seconds=wall_seconds,
        note=note,
        recipe_path=recipe_path,
        imm_sidecar_path=persisted_sidecar_path,
        evaluation=evaluation,
        launch_result=result,
        attempt_id=attempt_id,
    )


__all__ = [
    "CellResult",
    "DEFAULT_CATALOG_ROOT",
    "DEFAULT_OUTCOMES_FILE",
    "RECIPE_N_CHUNKS",
    "RECIPE_N_SAMPLES",
    "RECIPE_N_WARMUP",
    "RECIPE_NUM_CHAINS",
    "RECIPE_SEED",
    "RECIPE_TARGET_ACCEPTANCE",
    "emit_low_recipe_for_cell",
]
