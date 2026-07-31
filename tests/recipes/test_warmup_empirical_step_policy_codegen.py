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

"""Fast checks for generated integration-step policy code."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tuningfork.recipes._emit._step_policy import emit_step_policy

pytestmark = pytest.mark.fast


def _run(source: str, warmup_info=None):
    namespace = {"jax": jax, "jnp": jnp, "np": np}
    if warmup_info is not None:
        namespace["_warmup_info"] = warmup_info
    exec(compile(source, "<generated-step-policy>", "exec"), namespace)
    return namespace


def test_warmup_empirical_codegen_is_structural_and_compilable():
    source = emit_step_policy({"kind": "warmup_empirical"})
    assert "_warmup_info.info.num_integration_steps" in source
    assert "np.asarray" in source and "np.unique" in source
    assert "_resolved_step_policy" in source
    ns = _run(
        source,
        type(
            "Warmup",
            (),
            {
                "info": type(
                    "Info",
                    (),
                    {"num_integration_steps": np.array([[2, 4, 2], [8, 4, 4]])},
                )
            },
        )(),
    )
    assert ns["_resolved_step_policy"] == {
        "kind": "empirical",
        "values": [2, 4, 8],
        "weights": [2 / 6, 3 / 6, 1 / 6],
    }
    assert int(ns["_integration_steps_fn"](jax.random.key(0))) in {2, 4, 8}
    json.dumps(ns["_resolved_step_policy"], allow_nan=False)


def test_warmup_empirical_rejects_missing_or_malformed_info():
    source = emit_step_policy({"kind": "warmup_empirical"})
    with pytest.raises(ValueError, match="non-empty"):
        _run(
            source,
            type(
                "W",
                (),
                {"info": type("I", (), {"num_integration_steps": np.array([])})},
            )(),
        )
    with pytest.raises(ValueError, match="positive integers"):
        _run(
            source,
            type(
                "W",
                (),
                {"info": type("I", (), {"num_integration_steps": np.array([0, 2])})},
            )(),
        )


@pytest.mark.parametrize(
    "spec, expected",
    [
        (None, {"kind": "uniform_int", "low": 1, "high": 10}),
        (
            {"kind": "uniform_int", "low": 2, "high": 7},
            {"kind": "uniform_int", "low": 2, "high": 7},
        ),
        (
            {"kind": "empirical", "values": [2, 5], "weights": [1.0, 3.0]},
            {"kind": "empirical", "values": [2, 5], "weights": [0.25, 0.75]},
        ),
        (
            {"kind": "poisson", "lam": 3.0},
            {"kind": "poisson", "lam": 3.0, "low": 1, "high": None},
        ),
        (
            {"kind": "log_uniform_int", "low": 1, "high": 8},
            {"kind": "log_uniform_int", "low": 1, "high": 8},
        ),
        (
            {"kind": "pow2_choice", "options": [1, 4]},
            {"kind": "pow2_choice", "options": [1, 4]},
        ),
    ],
)
def test_existing_kinds_bind_json_safe_resolved_policy(spec, expected):
    ns = _run(emit_step_policy(spec))
    assert ns["_resolved_step_policy"] == expected
    json.dumps(ns["_resolved_step_policy"], allow_nan=False)
