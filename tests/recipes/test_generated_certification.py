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

import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from tuningfork.recipes._base import Effort, Recipe
from tuningfork.recipes._execution_telemetry import ExecutionTelemetry
from tuningfork.recipes._generated_certification import (
    _json_safe,
    evaluate_generated_run,
)
from tuningfork.recipes._generated_evaluator import GeneratedRunData
from tuningfork.recipes._ground_truth_reference import GroundTruthReference

pytestmark = pytest.mark.fast


def _recipe(method="rwm", params=None):
    return Recipe(
        model_name="mvn_10",
        base_method_name=method,
        warmup_name="no_warmup",
        effort=Effort.LOW,
        base_method_params=params or {},
        warmup_params={},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
    )


def _telemetry(geometry=None, fixed=None, warmup=3, policy=None):
    return ExecutionTelemetry(
        schema="tuningfork.generated-run-telemetry.v2",
        plan_hash="p",
        executable_config_hash="e",
        draws_artifact="draws.npz",
        geometry=geometry or {},
        geometry_source="adapted" if geometry else "unavailable",
        geometry_scope="shared" if geometry else None,
        geometry_unavailable_reason=None if geometry else "not recorded",
        fixed=fixed or {},
        timing_seconds={"warmup": 0.0, "sampling": 0.0, "total": 0.0},
        warmup_grad_evals=warmup,
        warmup_grad_evals_reason="test",
        resolved_step_policy=policy,
    )


def _reference(sites=("x",)):
    stats = {
        s: {
            "mean": 0.0,
            "std": 1.0,
            "q05": -1.0,
            "q95": 1.0,
            "between_chain_se": 0.1,
            "bulk_ess": 100.0,
        }
        for s in sites
    }
    return GroundTruthReference(
        "mvn_10", {"per_site": stats, "n_total": 20}, Path("s"), Path("d"), {}
    )


def _run(method="rwm"):
    info = namedtuple("Info", ["dummy"])(np.zeros((2, 10)))
    return GeneratedRunData(
        {"x": np.zeros((2, 10))}, {}, info, "generated", 2, 10, method
    )


def _patch(monkeypatch):
    class Gate:
        def to_dict(self):
            return {
                "verdict": "PASS",
                "rhat_max": 1.0,
                "min_bulk_ess": 10.0,
                "n_divergences": 0,
                "max_abs_mean_z": 0.0,
                "margins": {},
            }

    monkeypatch.setattr(
        "tuningfork.recipes._generated_certification.auto_gate", lambda *a, **k: Gate()
    )
    monkeypatch.setattr(
        "tuningfork.recipes._generated_certification.build_headline_basis",
        lambda *a, **k: (
            1.5,
            {
                "total_grad_evals": k["total_grad_evals"],
                "grad_count_convention": k["grad_count_convention"],
                "is_lower_bound": k["is_lower_bound"],
            },
        ),
    )
    monkeypatch.setattr(
        "tuningfork.recipes._generated_certification.compute_sample_quality",
        lambda *a, **k: {"mae_vs_reference": 0.0},
    )


def test_pass_evaluation_and_geometry(monkeypatch):
    _patch(monkeypatch)
    policy = {"kind": "empirical", "values": [1, 2], "weights": [0.25, 0.75]}
    out = evaluate_generated_run(
        _recipe("rwm"),
        _run("rwm"),
        _telemetry({"step_size": 0.2}, policy=policy),
        _reference(),
        n_chunks=2,
    )
    assert out.headline_metric == 1.5
    assert out.pinned_base_method_params["step_size"] == 0.2
    assert out.warmup_grad_evals == 3
    assert out.resolved_step_policy == policy
    assert out.metrics["gate"]["verdict"] == "PASS"
    assert out.gate_evidence["override"] == {
        "reason": "",
        "statistician_id": "",
        "decision": "",
    }
    json.dumps(out.metrics)


def test_gradient_free_uses_draw_denominator(monkeypatch):
    _patch(monkeypatch)
    out = evaluate_generated_run(
        _recipe("rwm"), _run("rwm"), _telemetry(), _reference(), n_chunks=2
    )
    assert out.sampling_grad_evals == 0
    assert out.metrics["headline_denominator"] == 20


def test_missing_required_geometry_fails():
    with pytest.raises(ValueError, match="geometry"):
        evaluate_generated_run(
            _recipe("hmc"), _run("hmc"), _telemetry(), _reference(), n_chunks=2
        )


def test_ground_truth_no_overlap_fails():
    with pytest.raises(ValueError, match="overlapping"):
        evaluate_generated_run(
            _recipe("rwm"), _run("rwm"), _telemetry(), _reference(("z",)), n_chunks=2
        )


def test_json_safe_rejects_non_string_mapping_keys():
    with pytest.raises(TypeError, match="string keys"):
        _json_safe({1: "not-json-object"})


def test_json_safe_preserves_namedtuples_only_when_requested():
    point = namedtuple("Point", "x y")(1, 2)
    assert _json_safe(point) == [1, 2]
    assert _json_safe(point, preserve_namedtuple=True) == point


def test_geometry_requirement_uses_method_capability(monkeypatch):
    _patch(monkeypatch)
    out = evaluate_generated_run(
        _recipe("mala", {"step_size": np.float64(0.2)}),
        _run("mala"),
        _telemetry(),
        _reference(),
        n_chunks=2,
    )
    assert out.pinned_base_method_params["step_size"] == 0.2


def test_mclmc_nonfinite_evidence_survives_generated_evaluation():
    positions = {"x": np.zeros((2, 10, 10), dtype=np.float64)}
    info_type = namedtuple(
        "MCLMCInfo", ["logdensity", "kinetic_change", "energy_change", "nonans"]
    )
    info = info_type(
        np.zeros((2, 10)),
        np.zeros((2, 10)),
        np.zeros((2, 10)),
        np.ones((2, 10), dtype=bool),
    )
    info.nonans[0, 0] = False
    run = GeneratedRunData(positions, {}, info, "generated", 2, 10, "mclmc")

    out = evaluate_generated_run(
        _recipe("mclmc"), run, _telemetry(), _reference(), n_chunks=2
    )
    auto = out.gate_evidence["auto"]

    assert auto["n_divergences"] is None
    assert auto["n_nonfinite_proposals"] == 1
    assert auto["n_proposals_evaluated"] == 20
    assert auto["nonfinite_proposal_rate"] == 1 / 20
    assert auto["margins"]["n_nonfinite_proposals"] == {
        "value": 1,
        "band": "REVIEW",
        "n_proposals_evaluated": 20,
        "nonfinite_proposal_rate": 1 / 20,
        "policy_id": "tuningfork.nonfinite-proposal-review.v1",
        "policy_status": "provisional",
        "calibrated": False,
    }
    assert auto["verdict"] != "PASS"
