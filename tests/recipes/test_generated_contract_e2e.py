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
"""End-to-end contracts for generated recipe execution."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tuningfork.catalog import emit_script, execute_recipe, prepare_pinned_replay
from tuningfork.recipes import Effort, Recipe
from tuningfork.recipes._execution_telemetry import ExecutionTelemetry


def _recipe(**changes: Any) -> Recipe:
    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={
            "step_size": 0.1,
            "num_integration_steps": 2,
            "inverse_mass_matrix": [1.0] * 10,
        },
        warmup_params={"n_warmup": 2, "num_chains": 1},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"num_chains": 1},
        difficulty=None,
        instructions="generated contract test",
    )
    return replace(recipe, **changes)


@pytest.mark.e2e
def test_laplace_multiphase_generated_execution_uses_dense_final_imm(
    tmp_path: Path,
) -> None:
    recipe = Recipe(
        model_name="gp_regression",
        base_method_name="laplace_mhmc",
        warmup_name="window_adaptation_dense_imm",
        effort=Effort.HIGH,
        base_method_params={
            "num_integration_steps": 2,
            "step_size": 0.5,
            "inverse_mass_matrix": np.eye(3).tolist(),
            "maxiter": 2,
        },
        warmup_params={"n_warmup": 1, "num_chains": 1, "target_acceptance": 0.8},
        warmups=[
            {
                "name": "window_adaptation_diag_imm",
                "params": {
                    "n_warmup": 1,
                    "num_chains": 1,
                    "num_integration_steps": 2,
                    "maxiter": 2,
                },
            },
            {
                "name": "window_adaptation_dense_imm",
                "params": {
                    "n_warmup": 1,
                    "num_chains": 1,
                    "num_integration_steps": 2,
                    "maxiter": 2,
                    "initial_step_size_from_phase1": True,
                },
            },
        ],
        warmup_inner_kernel="laplace_hmc",
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"num_chains": 1},
        difficulty=None,
        instructions="generated contract test",
        tuning_seed=7,
    )
    result = execute_recipe(
        recipe,
        tmp_path / "runs",
        num_warmup=[1, 1],
        num_samples=1,
        num_chains=1,
        progress_bar=False,
        timeout=180,
    )
    assert result.artifact_path is not None
    assert result.manifest.executable_config["warmup_stages"][-1]["name"] == (
        "window_adaptation_dense_imm"
    )
    with np.load(result.artifact_path, allow_pickle=False) as draws:
        posterior_vars = [name for name in draws.files if not name.startswith("_ss_")]
        assert posterior_vars
        assert all(draws[name].shape[:2] == (1, 1) for name in posterior_vars)


@pytest.mark.e2e
def test_adjusted_mclmc_dynamic_generated_execution_threads_rng_and_state(
    tmp_path: Path,
) -> None:
    recipe = Recipe.load(
        Path(__file__).parents[2]
        / "tuningfork"
        / "catalog"
        / "mvn_10"
        / "recipes"
        / "low__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json"
    )
    result = execute_recipe(
        recipe,
        tmp_path / "runs",
        num_warmup=2,
        num_samples=2,
        num_chains=2,
        progress_bar=False,
        timeout=180,
        sampler_seed=11,
        reinit_seed=12,
    )
    assert result.artifact_path is not None
    source = result.source_path.read_text()
    assert "adjusted_mclmc_dynamic.init(position, logdensity_fn, rng_key)" in source
    with np.load(result.artifact_path, allow_pickle=False) as draws:
        assert draws["x"].shape[:2] == (2, 2)


@pytest.mark.e2e
def test_adjusted_mclmc_trajectory_grid_is_executed_from_recipe(
    tmp_path: Path,
) -> None:
    avg_grid = [1.0, 2.0, 4.0]
    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="adjusted_mclmc_dynamic",
        warmup_name="adjusted_mclmc_trajectory_tuning",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1, "L": 1.0},
        warmup_params={
            "n_warmup": 20,
            "num_chains": 2,
            "target_acceptance": 0.9,
            "n_pilot": 10,
            "avg_grid": avg_grid,
        },
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"n_samples": 1, "num_chains": 2},
        difficulty=None,
        instructions="generated trajectory-grid contract test",
        tuning_seed=7,
    )
    result = execute_recipe(
        recipe,
        tmp_path / "runs",
        num_warmup=20,
        num_samples=1,
        num_chains=2,
        progress_bar=False,
        timeout=180,
        env={"JAX_PLATFORM_NAME": "cpu"},
    )

    config = result.manifest.executable_config
    assert config["warmup_params"]["n_pilot"] == 10
    assert config["warmup_params"]["avg_grid"] == tuple(avg_grid)
    assert isinstance(result.telemetry, ExecutionTelemetry)
    assert result.telemetry.geometry_scope == "per_chain"
    geometry = result.telemetry.geometry
    step_size = np.asarray(geometry["step_size"])
    trajectory_length = np.asarray(geometry["L"])
    inverse_mass_matrix = np.asarray(geometry["inverse_mass_matrix"])
    assert step_size.shape == (2,)
    assert trajectory_length.shape == (2,)
    assert inverse_mass_matrix.shape == (2, 10)
    selected_avg = trajectory_length / step_size
    assert np.allclose(selected_avg, selected_avg[0])
    assert any(np.isclose(selected_avg[0], candidate) for candidate in avg_grid)

    source = result.source_path.read_text()
    assert "blackjax.diagnostics.effective_sample_size" in source
    assert "_avg_search_ess_per_grad" in source


@pytest.mark.e2e
def test_sidecar_fixed_step_mhmc_replay_preserves_nis_and_imm(tmp_path: Path) -> None:
    model_root = tmp_path / "mvn_10"
    (model_root / "reference").mkdir(parents=True)
    (model_root / "reference" / "summary.json").write_text(
        json.dumps({"mean": {"x": [0.0] * 10}, "std": {"x": [1.0] * 10}})
    )
    sidecar = model_root / "dense.imm.npz"
    imm = np.eye(10)
    np.savez(sidecar, imm=imm)
    recipe = _recipe(
        base_method_name="mhmc",
        warmup_name="",
        base_method_params={
            "step_size": 0.1,
            "num_integration_steps": 3,
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
        timeout=180,
    )
    assert result.artifact_path is not None
    params = result.manifest.executable_config["base_method_params"]
    assert params["num_integration_steps"] == 3
    np.testing.assert_allclose(params["inverse_mass_matrix"], imm)


@pytest.mark.fast
def test_x64_requirement_is_emitted_before_generated_execution() -> None:
    recipe = Recipe.load(
        Path(__file__).parents[2]
        / "tuningfork"
        / "catalog"
        / "lotka_volterra"
        / "recipes"
        / "medium__nuts__window_adaptation_diag_imm.json"
    )
    source = emit_script(recipe, num_warmup=1, num_samples=1, num_chains=1)
    assert 'jax.config.update("jax_enable_x64", True)' in source
    assert source.index("jax.config.update") < source.index("jax.random.key")


@pytest.mark.e2e
def test_mclmc_lrd_generated_execution_smoke(tmp_path: Path) -> None:
    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="mclmc",
        warmup_name="mclmc_lrd_tuning",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1, "L": 1.0},
        warmup_params={
            "n_warmup": 20,
            "num_chains": 1,
            "k_rank": 1,
            "pilot_n_warmup": 20,
            "pilot_n_samples": 20,
        },
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"num_chains": 1},
        difficulty=None,
        instructions="generated contract test",
        tuning_seed=7,
    )
    result = execute_recipe(
        recipe,
        tmp_path / "runs",
        num_warmup=20,
        num_samples=1,
        num_chains=1,
        progress_bar=False,
        timeout=180,
        env={"JAX_PLATFORM_NAME": "cpu"},
    )

    assert result.artifact_path is not None and result.artifact_path.is_file()
    assert result.manifest.executable_config["warmup_name"] == "mclmc_lrd_tuning"
    assert isinstance(result.telemetry, ExecutionTelemetry)
    geometry = result.telemetry.geometry["inverse_mass_matrix"]
    assert geometry["type"] == "low_rank_inverse_mass_matrix"
    with np.load(result.artifact_path, allow_pickle=False) as draws:
        assert draws["x"].shape == (1, 1, 10)
