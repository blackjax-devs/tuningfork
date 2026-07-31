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

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.catalog import emit_script, execute_recipe
from tuningfork.model import MODELS
from tuningfork.recipes import Effort, Recipe
from tuningfork.recipes._emit._warmup import emit_warmup
from tuningfork.recipes._execution_telemetry import ExecutionTelemetry


def _recipe(warmup: str, *, base: str = "nuts", params: dict | None = None) -> Recipe:
    recipe = Recipe.from_default_config(
        MODELS["mvn_10"],
        BASE_METHODS[base],
    )
    return replace(
        recipe,
        warmup_name=warmup,
        warmup_params={"n_warmup": 2, "num_chains": 1, **(params or {})},
        calibration_budget={"num_chains": 1},
        effort=Effort.LOW,
    )


@pytest.mark.fast
def test_generated_family_contracts() -> None:
    chees = emit_warmup(
        "chees",
        BASE_METHODS["hmc"],
        {"n_warmup": 1, "tuning_seed": 0, "wp_target_acceptance_rate": None},
    )
    assert "b1=0.0, b2=0.95" in chees
    assert "target_acceptance_rate=0.651" in chees
    explicit = emit_warmup(
        "chees",
        BASE_METHODS["hmc"],
        {"n_warmup": 1, "tuning_seed": 0, "target_acceptance_rate": 0.77},
    )
    assert "target_acceptance_rate=0.77" in explicit
    meads = emit_warmup(
        "meads",
        BASE_METHODS["ghmc"],
        {"n_warmup": 1, "tuning_seed": 0, "wp_num_folds": 2},
    )
    assert "num_chains >= num_folds" in meads and "random.normal" in meads
    lrd = emit_script(
        _recipe(
            "mclmc_lrd_tuning",
            base="mclmc",
            params={"k_rank": 1, "pilot_n_warmup": 2, "pilot_n_samples": 2},
        ),
        num_warmup=2,
        num_samples=1,
    )
    assert "_lrd_k_rank = 1" in lrd and "_lrd_pilot_n_warmup = 2" in lrd
    assert "k_rank must be" in lrd or "rank" in lrd

    vi_ctx = {
        "warmup_name": "meanfield_vi",
        "wp_num_optimization_steps": 2,
        "tuning_seed": 0,
        "target_acceptance_rate": None,
        "n_warmup": 0,
        "num_chains": 1,
        "warmup_algorithm": "blackjax.hmc",
        "warmup_extra_kwargs": "",
        "vi_prefix": "_mf",
        "vi_module": "blackjax.vi.meanfield_vi",
        "vi_imm_description": "diagonal",
        "vi_imm_extraction_block": "_mf_imm = jnp.ones((1,))",
        "vi_adapted_imm_expr": "_mf_imm",
    }
    vi_source = emit_warmup("meanfield_vi", BASE_METHODS["hmc"], vi_ctx)
    vi_da_source = emit_warmup(
        "meanfield_vi", BASE_METHODS["hmc"], {**vi_ctx, "n_warmup": 1}
    )
    assert '"_mfvi_elbo"' in vi_source and "target=0.8" in vi_da_source
    assert "dual_averaging_adaptation" not in vi_source
    assert "_mf_adapted_step_size = 1.0" in vi_source


@pytest.mark.parametrize("warmup", ["meanfield_vi", "fullrank_vi"])
@pytest.mark.e2e
def test_generated_vi_warmups_execute(tmp_path: Path, warmup: str) -> None:
    result = execute_recipe(
        _recipe(warmup, base="hmc", params={"num_optimization_steps": 2}),
        tmp_path / warmup,
        num_warmup=1,
        num_samples=1,
        num_chains=1,
        progress_bar=False,
        timeout=180,
        env={"JAX_PLATFORM_NAME": "cpu"},
    )
    assert result.artifact_path and result.artifact_path.is_file()
    assert isinstance(result.telemetry, ExecutionTelemetry)
    step = np.asarray(result.telemetry.geometry["step_size"])
    assert np.all(np.isfinite(step)) and np.all(step > 0)
    imm = np.asarray(result.telemetry.geometry["inverse_mass_matrix"])
    expected = (1, 10) if warmup == "meanfield_vi" else (1, 10, 10)
    assert imm.shape == expected and np.all(np.isfinite(imm))
    if warmup == "meanfield_vi":
        assert np.all(imm > 0)
    else:
        assert np.all(np.linalg.eigvalsh(imm[0]) > 0)
    with np.load(result.artifact_path, allow_pickle=False) as draws:
        assert draws["x"].shape[:2] == (1, 1)


@pytest.mark.parametrize("warmup", ["pathfinder", "multipathfinder"])
@pytest.mark.e2e
def test_generated_pathfinder_warmups_execute(tmp_path: Path, warmup: str) -> None:
    params = (
        {"n_paths": 2, "num_samples_per_path": 2} if warmup == "multipathfinder" else {}
    )
    result = execute_recipe(
        _recipe(warmup, params=params),
        tmp_path / warmup,
        num_warmup=1,
        num_samples=1,
        num_chains=2,
        progress_bar=False,
        timeout=180,
        env={"JAX_PLATFORM_NAME": "cpu"},
    )
    assert result.artifact_path and result.artifact_path.is_file()
    assert isinstance(result.telemetry, ExecutionTelemetry)
    step = np.asarray(result.telemetry.geometry["step_size"])
    assert np.all(np.isfinite(step)) and np.all(step > 0)
    imm = np.asarray(result.telemetry.geometry["inverse_mass_matrix"])
    assert imm.shape == (10, 10) and np.all(np.isfinite(imm))
    with np.load(result.artifact_path, allow_pickle=False) as draws:
        assert draws["x"].shape[:2] == (2, 1)
