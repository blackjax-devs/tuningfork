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

"""Fast structural contracts for the standalone SMC source emitter."""

import ast

import pytest

from tuningfork.recipes._base_smc import SMCRecipe
from tuningfork.recipes._emit_smc_script import emit_smc_script

pytestmark = pytest.mark.fast


def _recipe(**updates: object) -> SMCRecipe:
    values: dict[str, object] = {
        "model_name": "mvn_10",
        "smc_method_name": "adaptive_tempered_smc",
        "inner_method_name": "rwm",
        "num_particles": 8,
        "max_steps": 3,
        "seed": 17,
        "smc_params": {"target_ess": 0.5},
        "inner_params_init": {"sigma": 0.2},
        "parameter_update_strategy": "none",
    }
    values.update(updates)
    return SMCRecipe(**values)  # type: ignore[arg-type]


def test_emitter_is_pure_and_embeds_manifest() -> None:
    recipe = _recipe()
    source = emit_smc_script(recipe)
    assert source == emit_smc_script(recipe)
    ast.parse(source)
    assert "EXECUTION_MANIFEST_JSON" in source
    assert '"execution_family":"smc"' in source
    assert '_cfg["max_steps"]' in source


def test_generated_source_owns_lifecycle_and_strict_archive() -> None:
    source = emit_smc_script(_recipe())
    assert "blackjax.adaptive_tempered_smc" in source
    assert "_algorithm.init" in source and "_algorithm.step" in source
    assert 'f"particle__{name}"' in source
    assert '"smc__weights"' in source
    assert '"smc__lambda"' in source
    assert '"smc__ess"' in source
    assert '"num_smc_steps"' in source
    assert "TUNINGFORK_TIMINGS" in source
    assert 'print("DONE")' in source
    assert "_smc_runner" not in source


def test_hmc_tuning_route_is_resolved_at_generation_time() -> None:
    source = emit_smc_script(
        _recipe(
            smc_method_name="inner_kernel_tuning",
            inner_method_name="hmc",
            inner_params_init={
                "step_size": 0.1,
                "inverse_mass_matrix": [1.0] * 10,
            },
            parameter_update_strategy="step_size_and_imm_from_particles",
        )
    )
    assert "blackjax.hmc.build_kernel" in source
    assert "inner_kernel_tuning.as_top_level_api" in source
    assert "build_rmh" not in source


@pytest.mark.parametrize(
    "strategy",
    [
        "none",
        "step_size_from_acceptance_rate",
        "imm_from_particles",
        "step_size_and_imm_from_particles",
    ],
)
def test_hmc_source_is_standalone_for_each_update_strategy(strategy: str) -> None:
    kwargs = {"target_acceptance": 0.71} if "step_size" in strategy else {}
    source = emit_smc_script(
        _recipe(
            smc_method_name="inner_kernel_tuning",
            inner_method_name="hmc",
            inner_params_init={
                "step_size": 0.1,
                "inverse_mass_matrix": [1.0] * 10,
            },
            parameter_update_strategy=strategy,
            parameter_update_strategy_kwargs=kwargs,
        )
    )
    assert "import tuningfork.smc" not in source
    ast.parse(source)
    compile(source, "generated_smc.py", "exec")
    if kwargs:
        assert "_target_acceptance = 0.71" in source


def test_update_strategy_validation_rejects_unknown_kwargs_and_bad_target() -> None:
    with pytest.raises(ValueError, match="does not allow kwargs"):
        emit_smc_script(_recipe(parameter_update_strategy_kwargs={"unexpected": 1}))
    with pytest.raises(ValueError, match="target_acceptance"):
        emit_smc_script(
            _recipe(
                smc_method_name="inner_kernel_tuning",
                inner_method_name="hmc",
                inner_params_init={
                    "step_size": 0.1,
                    "inverse_mass_matrix": [1.0] * 10,
                },
                parameter_update_strategy="step_size_from_acceptance_rate",
                parameter_update_strategy_kwargs={"target_acceptance": float("nan")},
            )
        )
    with pytest.raises(ValueError, match="require inner_kernel_tuning"):
        emit_smc_script(_recipe(parameter_update_strategy="imm_from_particles"))


def test_step_size_strategy_records_default_target_acceptance() -> None:
    source = emit_smc_script(
        _recipe(
            smc_method_name="inner_kernel_tuning",
            inner_method_name="hmc",
            inner_params_init={
                "step_size": 0.1,
                "inverse_mass_matrix": [1.0] * 10,
            },
            parameter_update_strategy="step_size_from_acceptance_rate",
        )
    )
    assert '"target_acceptance":0.65' in source
    assert "_target_acceptance = 0.65" in source


def test_missing_codegen_route_fails_at_emission() -> None:
    with pytest.raises(NotImplementedError, match="missing generated call shape"):
        emit_smc_script(_recipe(smc_method_name="tempered_smc"))


def test_incompatible_inner_method_fails_at_emission() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        emit_smc_script(_recipe(inner_method_name="mclmc"))
