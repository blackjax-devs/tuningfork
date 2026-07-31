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

"""Structural coverage for emitted warmup gradient accounting telemetry."""

import ast

import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.recipes._emit._warmup import emit_warmup

pytestmark = pytest.mark.fast


def _ctx(**extra: object) -> dict[str, object]:
    ctx: dict[str, object] = {
        "target_acceptance_rate": 0.8,
        "n_warmup": 10,
        "tuning_seed": 17,
        "num_chains": 2,
        "warmup_algorithm": "blackjax.nuts",
        "warmup_extra_kwargs": "",
        "window_adaptation_fn": "blackjax.window_adaptation",
        "window_adaptation_extra_kwargs": "",
        "warmup_progress_bar": False,
        "warmup_name": "meanfield_vi",
        "wp_num_optimization_steps": 5,
        "vi_prefix": "_vi",
        "vi_module": "blackjax.vi.meanfield",
        "vi_imm_description": "test",
        "vi_imm_extraction_block": "_vi_imm = jnp.eye(1)",
        "vi_adapted_imm_expr": "_vi_imm",
        "wp_n_paths": 2,
        "wp_num_samples_per_path": 3,
        "wp_imm_shrinkage_to_previous": 0.5,
        "num_warmup_phases": 2,
        "wp0_name": "phase1",
        "wp0_target": 0.8,
        "wp0_n_warmup": 5,
        "wp1_name": "phase2",
        "wp1_target": 0.8,
        "wp1_n_warmup": 5,
    }
    ctx.update(extra)
    return ctx


@pytest.mark.parametrize(
    ("warmup", "method", "needle"),
    [
        ("no_warmup", "nuts", "_warmup_grad_evals = 0"),
        (
            "window_adaptation_diag_imm",
            "nuts",
            "_warmup_info.info.num_integration_steps",
        ),
        ("mclmc_tuning", "mclmc", "_warmup_grad_evals = int(jnp.sum"),
        ("adjusted_mclmc_tuning", "adjusted_mclmc", "_warmup_grad_evals = int(jnp.sum"),
    ],
)
def test_exact_routes_emit_accounting_once(
    warmup: str, method: str, needle: str
) -> None:
    source = emit_warmup(warmup, BASE_METHODS[method], _ctx())
    ast.parse(source)
    assert source.count("_warmup_grad_evals =") == 1
    assert source.count("_warmup_grad_evals_reason =") == 1
    assert needle in source
    if warmup == "window_adaptation_diag_imm":
        assert "one integration-step count per warmup draw" in source


@pytest.mark.parametrize(
    ("warmup", "method", "reason"),
    [
        ("pathfinder", "nuts", "pathfinder:"),
        ("multipathfinder", "nuts", "multipathfinder:"),
        ("multipathfinder_window_adaptation", "nuts", "composite warmup"),
        ("meanfield_vi", "nuts", "VI warmup:"),
        ("laplace_multiphase_warmup", "laplace_hmc", "multiphase adaptation"),
        ("chees", "nuts", "CHEES:"),
        ("meads", "ghmc", "MEADS:"),
        ("mclmc_lrd_tuning", "mclmc", "MCLMC-LRD:"),
    ],
)
def test_unavailable_routes_emit_stable_reason(
    warmup: str, method: str, reason: str
) -> None:
    source = emit_warmup(warmup, BASE_METHODS[method], _ctx())
    ast.parse(source)
    assert source.count("_warmup_grad_evals =") == 1
    assert source.count("_warmup_grad_evals_reason =") == 1
    assert "_warmup_grad_evals = None" in source
    assert reason in source
