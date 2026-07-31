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

"""Structural tests for standalone step-policy source emission."""

import ast

import pytest

from tuningfork.recipes._emit._step_policy import emit_step_policy

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    "spec",
    [
        None,
        {"kind": "uniform_int", "low": 1, "high": 10},
        {"kind": "empirical", "values": [2, 4], "weights": [1, 3]},
        {"kind": "poisson", "lam": 4, "low": 1},
        {"kind": "poisson", "lam": 4, "low": 1, "high": 20},
        {"kind": "log_uniform_int", "low": 1, "high": 64},
        {"kind": "pow2_choice", "options": [1, 2, 4, 8]},
    ],
)
def test_emits_parseable_standalone_function(spec):
    source = emit_step_policy(spec)
    tree = ast.parse(source)
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "_integration_steps_fn"
        for node in tree.body
    )
    assert "tuningfork" not in source
    assert "import " not in source


def test_empirical_emission_normalizes_weights_explicitly():
    source = emit_step_policy(
        {"kind": "empirical", "values": [10, 20], "weights": [2, 6]}
    )
    assert (
        "_step_policy_weights = _step_policy_weights / jnp.sum(_step_policy_weights)"
        in source
    )
    assert "searchsorted" in source


@pytest.mark.parametrize(
    "spec",
    [
        "uniform_int",
        {},
        {"kind": "unknown"},
        {"kind": "uniform_int", "low": 2, "high": 2},
        {"kind": "uniform_int", "low": 0, "high": 2},
        {"kind": "empirical", "values": [1], "weights": [0]},
        {"kind": "empirical", "values": [2, 1], "weights": [1, 1]},
        {"kind": "empirical", "values": [1, 1], "weights": [1, 1]},
        {"kind": "empirical", "values": [1.5], "weights": [1]},
        {"kind": "poisson", "lam": -1},
        {"kind": "poisson", "lam": 1, "low": 0},
        {"kind": "log_uniform_int", "low": 0, "high": 3},
        {"kind": "pow2_choice", "options": []},
        {"kind": "pow2_choice", "options": [1, 3]},
        {"kind": "pow2_choice", "options": [1, 4.5]},
    ],
)
def test_rejects_malformed_or_unknown_specs(spec):
    with pytest.raises(ValueError):
        emit_step_policy(spec)


def test_capped_poisson_emits_jittable_rejection_loop():
    source = emit_step_policy({"kind": "poisson", "lam": 3, "low": 1, "high": 8})
    assert "jax.lax.while_loop" in source
    assert "_step_policy_high" in source
