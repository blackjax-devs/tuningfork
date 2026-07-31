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

from pathlib import Path

import numpy as np
import pytest

from tuningfork.recipes._base import Effort
from tuningfork.recipes._certification_intent import build_certification_intent

pytestmark = pytest.mark.fast


def _build(**kwargs):
    defaults = dict(
        model_name="mvn_10",
        warmup_name="no_warmup",
        sampler_name="hmc",
        n_warmup=10,
        n_samples=20,
        num_chains=2,
        seed=3,
        catalog_root=Path("/tmp/catalog"),
    )
    defaults.update(kwargs)
    return build_certification_intent(**defaults)


def test_defaults_and_path():
    intent = _build()
    assert intent.recipe.effort is Effort.LOW
    assert intent.recipe.warmup_params["n_warmup"] == 0
    assert intent.recipe.gate_evidence["auto"]["verdict"] == "NOT_RUN"
    assert intent.recipe_path == Path(
        "/tmp/catalog/mvn_10/recipes/low__hmc__no_warmup.json"
    )


def test_variant_label_is_stored_and_used_in_path():
    intent = _build(variant_label="hmc_tuned")
    assert intent.recipe.variant_label == "hmc_tuned"
    assert intent.recipe_path == Path(
        "/tmp/catalog/mvn_10/recipes/low__hmc_tuned__no_warmup.json"
    )


def test_overrides_target_and_empirical_policy():
    intent = _build(
        sampler_name="dynamic_hmc",
        warmup_name="window_adaptation_diag_imm",
        warmup_inner_kernel="hmc",
        target_acceptance=0.7,
        sampler_kwargs_override={"step_size": np.float64(0.2)},
    )
    assert intent.recipe.warmup_params["target_acceptance"] == 0.7
    assert intent.recipe.base_method_params["step_size"] == 0.2
    assert intent.recipe.step_policy == {"kind": "warmup_empirical"}
    assert intent.filename_tag == "inner_hmc"


def test_trajectory_tuning_defaults_are_recipe_configuration():
    intent = _build(
        sampler_name="adjusted_mclmc_dynamic",
        warmup_name="adjusted_mclmc_trajectory_tuning",
    )
    assert intent.recipe.warmup_params["target_acceptance"] == 0.9
    assert intent.recipe.warmup_params["n_pilot"] == 500
    assert intent.recipe.warmup_params["avg_grid"] == [1.0, 2.0, 4.0]


def test_rejects_non_dynamic_policy_and_bad_inputs():
    with pytest.raises(ValueError):
        _build(step_policy={"kind": "uniform_int", "low": 1, "high": 2})
    with pytest.raises(ValueError):
        _build(
            sampler_name="dynamic_hmc", warmup_name="chees", warmup_inner_kernel="nuts"
        )
    with pytest.raises(ValueError):
        _build(sampler_name="missing")
    with pytest.raises(ValueError):
        _build(policy_tag="../unsafe")
    with pytest.raises(ValueError):
        _build(variant_label="")
    with pytest.raises(ValueError):
        _build(variant_label="../unsafe")
    with pytest.raises(ValueError):
        _build(n_samples=0)


def test_rejects_non_json_override():
    with pytest.raises(TypeError):
        _build(sampler_kwargs_override={"step_size": object()})
