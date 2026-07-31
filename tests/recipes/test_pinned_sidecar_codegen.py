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

"""Regression tests for JSON-safe pinned inverse-mass code generation."""

import json
from pathlib import Path

import numpy as np
import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.recipes._emit._sampler import emit_sampler
from tuningfork.recipes._emit_script import _build_inference_loop

_MARKER = {
    "type": "low_rank_inverse_mass_matrix",
    "sigma": [1.0, 2.0],
    "U": [[1.0], [0.0]],
    "lam": [3.0],
}


@pytest.mark.fast
def test_low_rank_marker_emits_blackjax_constructor_without_inference_imports() -> None:
    source = emit_sampler(
        BASE_METHODS["nuts"],
        {
            "bm_step_size": 0.1,
            "bm_inverse_mass_matrix": _MARKER,
            "is_baked_replay": True,
        },
    )
    assert "LowRankInverseMassMatrix" in source
    assert "sigma=jnp.asarray([1.0, 2.0])" in source
    assert "import tuningfork" not in source


@pytest.mark.fast
def test_malformed_low_rank_marker_fails_closed() -> None:
    malformed = {**_MARKER, "lam": [float("nan")]}
    with pytest.raises(ValueError, match="low-rank inverse mass marker"):
        emit_sampler(
            BASE_METHODS["nuts"],
            {
                "bm_step_size": 0.1,
                "bm_inverse_mass_matrix": malformed,
                "is_baked_replay": True,
            },
        )


@pytest.mark.fast
def test_prebatched_pinned_imm_uses_leafwise_broadcast() -> None:
    source = _build_inference_loop(
        num_samples=1,
        num_chains=2,
        sampler_seed=1,
        reinit_seed=2,
        use_progress_bar=False,
        warmup_is_perchain=False,
        warmup_init_is_single_chain=False,
        warmup_init_is_prebatched=True,
        needs_state_reinit=False,
        has_per_chain_L=False,
        no_warmup_step_size_expr="0.1",
        no_warmup_imm_expr="_default_imm",
    )
    assert "jax.tree.map(" in source
    assert "jnp.asarray(_default_imm).shape" not in source


@pytest.mark.fast
def test_mclmc_replay_emits_exact_geometry_and_keyed_state_init() -> None:
    source = emit_sampler(
        BASE_METHODS["mclmc"],
        {
            "bm_step_size": 0.2,
            "bm_L": 3.0,
            "bm_inverse_mass_matrix": _MARKER,
            "is_baked_replay": True,
        },
    )
    assert "_default_step_size = 0.2" in source
    assert "_default_L = 3.0" in source
    assert "L=L if L is not None else _default_L" in source
    assert "def _state_init(position, rng_key):" in source
    assert ".init(position, rng_key)" in source


@pytest.mark.fast
def test_baked_fixed_step_hmc_requires_integration_steps() -> None:
    from tuningfork.recipes import Effort, Recipe
    from tuningfork.recipes._emit_script import emit_script

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="no_warmup",
        effort=Effort.MEDIUM,
        base_method_params={
            "step_size": 0.1,
            "inverse_mass_matrix": [1.0] * 10,
        },
        warmup_params={},
        warmups=[{"name": "no_warmup", "params": {}}],
        headline_metric=None,
        sample_quality=None,
        calibration_budget={
            "trials": 0,
            "wall_seconds_estimate": 0.0,
            "baked_from": {"warmup_name": "window_adaptation_diag_imm"},
        },
        difficulty=None,
        instructions="test",
    )
    with pytest.raises(ValueError, match="num_integration_steps"):
        emit_script(recipe, num_samples=1, num_chains=1)


@pytest.mark.e2e
def test_public_mclmc_pinned_replay_with_structured_sidecar(tmp_path: Path) -> None:
    from tuningfork.catalog import execute_recipe, prepare_pinned_replay
    from tuningfork.recipes import Effort, Recipe

    model_root = tmp_path / "mvn_10"
    (model_root / "reference").mkdir(parents=True)
    summary = {
        "mean": {"x": [0.0] * 10},
        "std": {"x": [1.0] * 10},
    }
    (model_root / "reference" / "summary.json").write_text(json.dumps(summary))
    sigma = np.ones(10)
    basis = np.zeros((10, 1))
    basis[0, 0] = 1.0
    np.savez(model_root / "lrd.imm.npz", sigma=sigma, U=basis, lam=np.array([1.0]))
    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="mclmc",
        warmup_name="mclmc_tuning",
        effort=Effort.MEDIUM,
        base_method_params={"step_size": 0.05, "L": 1.0},
        warmup_params={"n_warmup": 0, "num_chains": 1},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"trials": 0, "wall_seconds_estimate": 0.0},
        difficulty=None,
        instructions="test",
        inverse_mass_matrix_path="mvn_10/lrd.imm.npz",
    )
    replay = prepare_pinned_replay(recipe, catalog_root=tmp_path)
    assert replay.base_method_params["inverse_mass_matrix"]["type"] == (
        "low_rank_inverse_mass_matrix"
    )
    result = execute_recipe(
        replay,
        tmp_path / "runs",
        num_samples=2,
        num_chains=1,
        progress_bar=False,
        timeout=120,
        sampler_seed=11,
        reinit_seed=12,
    )
    assert result.artifact_path is not None and result.artifact_path.exists()
    config = result.manifest.executable_config
    assert config["num_samples"] == 2
    assert config["num_chains"] == 1
    assert config["base_method_params"]["step_size"] == 0.05
    assert config["base_method_params"]["L"] == 1.0
    assert config["base_method_params"]["inverse_mass_matrix"]["type"] == (
        "low_rank_inverse_mass_matrix"
    )
    assert "_default_L = 1.0" in result.source_path.read_text()
