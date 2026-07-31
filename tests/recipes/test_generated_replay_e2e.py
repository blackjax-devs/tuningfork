"""Behavioral coverage for the generated pinned-replay path."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tuningfork.catalog import emit_script, execute_recipe, prepare_pinned_replay
from tuningfork.recipes import Effort, Recipe


def _recipe(**changes: object) -> Recipe:
    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1, "inverse_mass_matrix": [1.0] * 10},
        warmup_params={"n_warmup": 2, "num_chains": 1},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"n_samples": 2, "num_chains": 1},
        difficulty=None,
        instructions="test",
    )
    return replace(recipe, **changes)


@pytest.mark.fast
def test_pinned_replay_rejects_missing_step_size() -> None:
    recipe = _recipe(base_method_params={"inverse_mass_matrix": [1.0] * 10})
    with pytest.raises(ValueError, match="pinned step_size"):
        emit_script(recipe.normalize_pinned_replay(), num_samples=2, num_chains=1)


@pytest.mark.fast
def test_pinned_replay_rejects_missing_inverse_mass_matrix() -> None:
    recipe = _recipe(base_method_params={"step_size": 0.1})
    with pytest.raises(ValueError, match="pinned inverse_mass_matrix"):
        emit_script(recipe.normalize_pinned_replay(), num_samples=2, num_chains=1)


@pytest.mark.fast
def test_pinned_laplace_replay_fails_closed() -> None:
    recipe = _recipe(
        base_method_name="laplace_hmc",
        base_method_params={"step_size": 0.1, "inverse_mass_matrix": [1.0] * 10},
    )
    with pytest.raises(NotImplementedError, match="Pinned Laplace replay"):
        emit_script(recipe.normalize_pinned_replay(), num_samples=2, num_chains=1)


@pytest.mark.e2e
def test_dense_sidecar_pinned_replay_preserves_nis_and_manifest(tmp_path: Path) -> None:
    model_root = tmp_path / "mvn_10"
    (model_root / "reference").mkdir(parents=True)
    summary = {"mean": {"x": [0.0] * 10}, "std": {"x": [1.0] * 10}}
    (model_root / "reference" / "summary.json").write_text(json.dumps(summary))
    sidecar = model_root / "dense.imm.npz"
    np.savez(sidecar, imm=np.eye(10))
    recipe = _recipe(
        base_method_name="hmc",
        base_method_params={
            "step_size": 0.1,
            "num_integration_steps": 2,
            "inverse_mass_matrix": "sidecar",
        },
        inverse_mass_matrix_path="mvn_10/dense.imm.npz",
    )
    replay = prepare_pinned_replay(recipe, catalog_root=tmp_path)
    result = execute_recipe(
        replay,
        tmp_path / "runs",
        num_samples=2,
        num_chains=1,
        progress_bar=False,
        timeout=120,
    )
    assert result.artifact_path is not None
    config = result.manifest.executable_config
    assert config["base_method_params"]["num_integration_steps"] == 2
    np.testing.assert_allclose(
        config["base_method_params"]["inverse_mass_matrix"], np.eye(10)
    )
    assert config["num_chains"] == 1
    assert config["num_samples"] == 2
