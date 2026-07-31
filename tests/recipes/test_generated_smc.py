"""Fast contract checks for generated SMC artifacts."""

import numpy as np
import pytest

from tuningfork.recipes._base_smc import SMCRecipe
from tuningfork.recipes._generated_smc import (
    GeneratedSMCArtifact,
    evaluate_generated_smc,
    load_generated_smc_artifact,
)
from tuningfork.recipes._ground_truth_reference import GroundTruthReference
from tuningfork.recipes._smc_execution_plan import resolve_smc_execution_plan

pytestmark = pytest.mark.fast


def _config(inner="rwm"):
    return {
        "execution_family": "smc",
        "model_name": "mvn_10",
        "smc_method_name": "adaptive_tempered_smc",
        "inner_method_name": inner,
        "num_particles": 4,
        "smc_params": {
            "num_mcmc_steps": 2,
            "num_integration_steps": 3,
        },
    }


def _write(path, **updates):
    data = {
        "particle__x": np.zeros((4, 2)),
        "smc__weights": np.full(4, 0.25),
        "smc__lambda": np.array([0.0, 0.5, 1.0]),
        "smc__ess": np.array([4.0, 3.0, 2.0]),
    }
    data.update(updates)
    np.savez(path, **data)


def _reference(tmp_path, model_name="mvn_10"):
    per_site = {
        "x": {
            "mean": [0.0, 0.0],
            "std": [1.0, 1.0],
            "q05": [-1.0, -1.0],
            "q95": [1.0, 1.0],
            "between_chain_se": [0.1, 0.1],
            "bulk_ess": [100.0, 100.0],
        }
    }
    return GroundTruthReference(
        model_name,
        {
            "per_site": per_site,
            "n_chains": 2,
            "n_draws_per_chain": 50,
            "n_total": 100,
        },
        tmp_path / "summary_v2.json",
        tmp_path / "draws.npz",
        {"schema": "test", "model_name": model_name},
    )


def test_valid_rwm_artifact_is_immutable_and_evaluable(tmp_path):
    path = tmp_path / "run.npz"
    _write(path)
    artifact = load_generated_smc_artifact(path, _config())
    assert isinstance(artifact, GeneratedSMCArtifact)
    with pytest.raises(ValueError):
        artifact.weights[0] = 0.5
    result = evaluate_generated_smc(artifact, _config(), _reference(tmp_path))
    assert result.gate.verdict == "PASS"
    assert result.total_cost == 24


def test_hmc_cost_uses_integration_steps(tmp_path):
    path = tmp_path / "run.npz"
    _write(path)
    result = evaluate_generated_smc(path, _config("hmc"), _reference(tmp_path))
    assert result.total_cost == 72


def test_resolved_plan_config_is_the_loader_contract(tmp_path):
    recipe = SMCRecipe(
        model_name="mvn_10",
        smc_method_name="adaptive_tempered_smc",
        inner_method_name="rwm",
        num_particles=4,
        max_steps=4,
        smc_params={"num_mcmc_steps": 2},
    )
    plan = resolve_smc_execution_plan(recipe)
    path = tmp_path / "run.npz"
    _write(path)
    artifact = load_generated_smc_artifact(path, plan.config)
    result = evaluate_generated_smc(
        artifact, plan.config, _reference(tmp_path, "mvn_10")
    )
    assert result.total_cost == 24


def test_wrong_ground_truth_model_is_rejected(tmp_path):
    path = tmp_path / "run.npz"
    _write(path)
    artifact = load_generated_smc_artifact(path, _config())
    with pytest.raises(ValueError, match="ground-truth model"):
        evaluate_generated_smc(
            artifact, _config(), _reference(tmp_path, "not-the-model")
        )


def test_incomplete_tempering_is_retained_as_gate_failure(tmp_path):
    path = tmp_path / "incomplete.npz"
    _write(
        path,
        **{
            "smc__lambda": np.array([0.0, 0.8]),
            "smc__ess": np.array([4.0, 3.0]),
        },
    )
    artifact = load_generated_smc_artifact(path, _config())
    result = evaluate_generated_smc(artifact, _config(), _reference(tmp_path))
    assert result.gate.verdict == "FAIL"
    assert result.lambda_final == 0.8


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"smc__weights": np.array([1.0, 0.0])}, "shape"),
        ({"smc__weights": np.array([0.2, 0.2, 0.2, 0.1])}, "normalized"),
        ({"smc__lambda": np.array([0.0, 0.8, 0.7])}, "monotone"),
        ({"smc__lambda": np.array([0.0, 1.1, 1.1])}, r"within \[0, 1\]"),
        ({"particle__x": np.array([[np.nan, 0.0]] * 4)}, "non-finite"),
    ],
)
def test_corrupt_artifacts_fail_closed(tmp_path, updates, message):
    path = tmp_path / "bad.npz"
    _write(path, **updates)
    with pytest.raises(ValueError, match=message):
        load_generated_smc_artifact(path, _config())


def test_missing_history_is_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(
        path, **{"particle__x": np.zeros((4, 1)), "smc__weights": np.full(4, 0.25)}
    )
    with pytest.raises(ValueError, match="missing required"):
        load_generated_smc_artifact(path, _config())
