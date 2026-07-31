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

import numpy as np
import pytest

from tuningfork.recipes._execution_manifest import ExecutionManifest
from tuningfork.recipes._execution_plan import (
    execution_config_hash,
    execution_plan_hash,
)
from tuningfork.recipes._execution_telemetry import ExecutionTelemetry
from tuningfork.recipes._generated_evaluator import (
    chain0_geometry,
    load_generated_artifact,
    sampling_grad_evals,
)

pytestmark = pytest.mark.fast


def _manifest(base_method_name="mala") -> ExecutionManifest:
    config = {
        "base_method_name": base_method_name,
        "num_chains": 2,
        "num_samples": 3,
    }
    normalized = {
        "config": config,
        "recipe_ref": "r",
        "artifact_filename": "draws.npz",
        "telemetry_artifact_filename": "telemetry.json",
    }
    return ExecutionManifest(
        "tuningfork.execution-manifest.v2",
        "tuningfork.execution-plan.v2",
        "g",
        "r",
        config,
        normalized,
        execution_config_hash(config),
        execution_plan_hash(config, "draws.npz", "telemetry.json"),
    )


def _valid_stats(base_method_name):
    if base_method_name == "mala":
        return {
            "_ss_acceptance_rate": np.full((2, 3), 0.75),
            "_ss_is_accepted": np.ones((2, 3), dtype=bool),
        }
    if base_method_name == "rwm":
        return {
            "_ss_acceptance_rate": np.full((2, 3), 0.5),
            "_ss_is_accepted": np.ones((2, 3), dtype=bool),
        }
    if base_method_name == "hmc":
        return {
            "_ss_is_divergent": np.zeros((2, 3), dtype=bool),
            "_ss_energy": np.ones((2, 3)),
            "_ss_num_integration_steps": np.ones((2, 3), dtype=np.int32),
            "_ss_acceptance_rate": np.full((2, 3), 0.75),
            "_ss_is_accepted": np.ones((2, 3), dtype=bool),
        }
    if base_method_name == "meanfield_vi":
        return {}
    raise AssertionError(f"test helper lacks stats for {base_method_name}")


def _telemetry(
    *,
    geometry,
    geometry_source,
    geometry_scope,
    geometry_unavailable_reason=None,
) -> ExecutionTelemetry:
    manifest = _manifest()
    return ExecutionTelemetry.from_dict(
        {
            "schema": "tuningfork.generated-run-telemetry.v1",
            "plan_hash": manifest.plan_hash,
            "executable_config_hash": manifest.executable_config_hash,
            "draws_artifact": "draws.npz",
            "geometry": geometry,
            "geometry_source": geometry_source,
            "geometry_scope": geometry_scope,
            "geometry_unavailable_reason": geometry_unavailable_reason,
            "fixed": {},
            "timing_seconds": {"warmup": 1.0, "sampling": 2.0, "total": 3.0},
            "warmup_grad_evals": 4,
            "warmup_grad_evals_reason": "exact",
        },
        manifest,
    )


def test_load_generated_artifact_and_constant_grad_cost(tmp_path):
    path = tmp_path / "draws.npz"
    np.savez(path, x=np.ones((2, 3, 1)), **_valid_stats("mala"))
    data = load_generated_artifact(path, _manifest())
    assert data.positions["x"].flags.writeable is False
    assert data.chain_stats["acceptance_rate"].flags.writeable is False
    assert sampling_grad_evals(data) == 6


def test_artifact_preserves_arbitrary_position_names_and_boolean_stats(tmp_path):
    path = tmp_path / "draws.npz"
    np.savez(
        path,
        **{
            "theta/group": np.ones((2, 3, 1)),
            **_valid_stats("mala"),
        },
    )

    data = load_generated_artifact(path, _manifest())

    assert set(data.positions) == {"theta/group"}
    assert set(data.chain_stats) == {"acceptance_rate", "is_accepted"}
    with pytest.raises(TypeError):
        data.positions["new"] = np.ones((2, 3))  # type: ignore[index]
    with pytest.raises(ValueError):
        data.chain_stats["is_accepted"][0, 0] = False


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("x", np.full((2, 3, 1), np.nan), "non-finite"),
        ("x", np.ones((2, 2, 1)), "incorrect leading shape"),
        ("x", np.ones((2, 3), dtype=np.complex64), "real numeric"),
        ("x", np.full((2, 3), "latent"), "real numeric"),
        ("_ss_acceptance/rate", np.ones((2, 3)), "invalid statistic name"),
    ],
)
def test_artifact_rejects_invalid_arrays(tmp_path, key, value, message):
    path = tmp_path / "draws.npz"
    arrays = {"x": np.ones((2, 3)), **_valid_stats("meanfield_vi"), key: value}
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match=message):
        load_generated_artifact(path, _manifest("meanfield_vi"))


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("_ss_num_integration_steps", np.full((2, 3), 1.5), "positive integers"),
        (
            "_ss_num_integration_steps",
            np.zeros((2, 3), dtype=np.int32),
            "positive integers",
        ),
        (
            "_ss_num_integration_steps",
            -np.ones((2, 3), dtype=np.int32),
            "positive integers",
        ),
        (
            "_ss_acceptance_rate",
            np.full((2, 3), 1.1),
            "unit interval",
        ),
        (
            "_ss_is_divergent",
            np.zeros((2, 3), dtype=np.int32),
            "booleans",
        ),
    ],
)
def test_artifact_rejects_semantically_invalid_stats(tmp_path, name, value, message):
    path = tmp_path / "draws.npz"
    arrays = {"x": np.ones((2, 3)), **_valid_stats("hmc")}
    arrays[name] = value
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match=message):
        load_generated_artifact(path, _manifest("hmc"))


def test_artifact_rejects_missing_or_unexpected_stats(tmp_path):
    path = tmp_path / "draws.npz"
    stats = _valid_stats("hmc")
    stats.pop("_ss_num_integration_steps")
    stats["_ss_unknown"] = np.ones((2, 3))
    np.savez(path, x=np.ones((2, 3)), **stats)

    with pytest.raises(
        ValueError, match=r"missing=.*num_integration_steps.*unexpected"
    ):
        load_generated_artifact(path, _manifest("hmc"))


def test_artifact_requires_at_least_one_position_array(tmp_path):
    path = tmp_path / "draws.npz"
    np.savez(path, **_valid_stats("mala"))

    with pytest.raises(ValueError, match="no position arrays"):
        load_generated_artifact(path, _manifest())


def test_empty_stats_get_dummy_info(tmp_path):
    path = tmp_path / "draws.npz"
    np.savez(path, x=np.ones((2, 3)))
    data = load_generated_artifact(path, _manifest("meanfield_vi"))

    assert data.infos._fields == ("dummy",)
    assert sampling_grad_evals(data) == 6


def test_constant_zero_grad_cost(tmp_path):
    path = tmp_path / "draws.npz"
    np.savez(path, x=np.ones((2, 3)), **_valid_stats("rwm"))
    data = load_generated_artifact(path, _manifest("rwm"))

    assert sampling_grad_evals(data) == 0


def test_dynamic_nis_grad_accounting(tmp_path):
    path = tmp_path / "draws.npz"
    nis = np.array([[1, 2, 3], [2, 2, 1]])
    stats = _valid_stats("hmc")
    stats["_ss_num_integration_steps"] = nis
    np.savez(path, x=np.ones((2, 3)), **stats)

    assert (
        sampling_grad_evals(
            load_generated_artifact(path, _manifest("hmc")),
        )
        == 11
    )


def test_sampling_grad_evals_requires_valid_inputs(tmp_path):
    path = tmp_path / "draws.npz"
    np.savez(path, x=np.ones((2, 3)), **_valid_stats("rwm"))
    data = load_generated_artifact(path, _manifest("rwm"))

    with pytest.raises(TypeError, match="GeneratedRunData"):
        sampling_grad_evals(object())  # type: ignore[arg-type]
    assert sampling_grad_evals(data) == 0


def test_shared_geometry_is_preserved_and_frozen():
    telemetry = _telemetry(
        geometry={"step_size": 0.25, "inverse_mass_matrix": [1.0, 2.0]},
        geometry_source="adapted",
        geometry_scope="shared",
    )

    result = chain0_geometry(telemetry)

    assert result.source == "adapted"
    assert result.reason is None
    assert result.geometry is not None
    assert result.geometry["step_size"] == 0.25
    assert result.geometry["inverse_mass_matrix"] == (1.0, 2.0)
    with pytest.raises(TypeError):
        result.geometry["step_size"] = 0.5


def test_per_chain_geometry_selects_chain_zero():
    telemetry = _telemetry(
        geometry={
            "step_size": [0.1, 0.2],
            "inverse_mass_matrix": [[1.0, 2.0], [3.0, 4.0]],
        },
        geometry_source="adapted",
        geometry_scope="per_chain",
    )

    result = chain0_geometry(telemetry)

    assert result.geometry is not None
    assert result.geometry["step_size"] == 0.1
    assert result.geometry["inverse_mass_matrix"] == (1.0, 2.0)


@pytest.mark.parametrize(
    ("scope", "sigma", "u", "lam", "expected_sigma"),
    [
        ("shared", [1.0, 2.0], [[1.0], [0.0]], [0.5], [1.0, 2.0]),
        (
            "per_chain",
            [[1.0, 2.0], [3.0, 4.0]],
            [[[1.0], [0.0]], [[0.0], [1.0]]],
            [[0.5], [0.25]],
            [1.0, 2.0],
        ),
    ],
)
def test_low_rank_geometry_is_reconstructed(scope, sigma, u, lam, expected_sigma):
    telemetry = _telemetry(
        geometry={
            "inverse_mass_matrix": {
                "type": "low_rank_inverse_mass_matrix",
                "sigma": sigma,
                "U": u,
                "lam": lam,
            }
        },
        geometry_source="pinned",
        geometry_scope=scope,
    )

    result = chain0_geometry(telemetry)
    assert result.geometry is not None
    low_rank = result.geometry["inverse_mass_matrix"]

    np.testing.assert_allclose(low_rank.sigma, expected_sigma)
    np.testing.assert_allclose(low_rank.U, [[1.0], [0.0]])
    np.testing.assert_allclose(low_rank.lam, [0.5])


def test_unavailable_geometry_retains_reason():
    telemetry = _telemetry(
        geometry={},
        geometry_source="unavailable",
        geometry_scope=None,
        geometry_unavailable_reason="warmup does not expose reusable geometry",
    )

    result = chain0_geometry(telemetry)

    assert result.geometry is None
    assert result.source == "unavailable"
    assert result.reason == "warmup does not expose reusable geometry"


def test_chain0_geometry_requires_typed_telemetry():
    with pytest.raises(TypeError, match="ExecutionTelemetry"):
        chain0_geometry({})  # type: ignore[arg-type]
