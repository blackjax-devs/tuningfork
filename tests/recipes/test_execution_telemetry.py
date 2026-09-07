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

"""Tests for strict generated-run telemetry."""

from types import SimpleNamespace

import pytest

from tuningfork.recipes._execution_manifest import ExecutionManifest
from tuningfork.recipes._execution_plan import ExecutionOverrides
from tuningfork.recipes._execution_telemetry import (
    LEGACY_TELEMETRY_SCHEMA,
    TELEMETRY_SCHEMA,
    ExecutionTelemetry,
)
from tuningfork.recipes._resolve_execution_plan import resolve_execution_plan

pytestmark = pytest.mark.fast


def _manifest():
    recipe = SimpleNamespace(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="no_warmup",
        effort="low",
        base_method_params={},
        warmup_params={},
        warmups=[],
        calibration_budget={"n_samples": 2, "num_chains": 1},
        tuning_seed=4,
        warmup_inner_kernel=None,
        init_strategy=None,
        step_policy=None,
        variant_label=None,
    )
    return ExecutionManifest.from_plan(
        resolve_execution_plan(recipe, ExecutionOverrides(num_samples=2)),
        generator_version="test",
    )


def _raw(manifest):
    return {
        "schema": TELEMETRY_SCHEMA,
        "plan_hash": manifest.plan_hash,
        "executable_config_hash": manifest.executable_config_hash,
        "draws_artifact": manifest.normalized_plan["artifact_filename"],
        "geometry": {},
        "geometry_source": "unavailable",
        "geometry_scope": None,
        "geometry_unavailable_reason": "not recorded",
        "fixed": {},
        "timing_seconds": {"warmup": 1.0, "sampling": 2.0, "total": 3.0},
        "warmup_grad_evals": None,
        "warmup_grad_evals_reason": "not available",
        "resolved_step_policy": None,
    }


def test_round_trip_is_immutable():
    telemetry = ExecutionTelemetry.from_dict(_raw(_manifest()), _manifest())
    assert (
        telemetry.to_json()
        == ExecutionTelemetry.from_json(telemetry.to_json(), _manifest()).to_json()
    )
    with pytest.raises(TypeError):
        telemetry.geometry["x"] = 1  # type: ignore[index]


def test_legacy_round_trip_defaults_resolved_policy_to_none():
    manifest = _manifest()
    raw = _raw(manifest)
    raw["schema"] = LEGACY_TELEMETRY_SCHEMA
    raw.pop("resolved_step_policy")
    telemetry = ExecutionTelemetry.from_dict(raw, manifest)
    assert telemetry.resolved_step_policy is None
    assert "resolved_step_policy" not in telemetry.as_dict()


@pytest.mark.parametrize(
    "policy",
    [
        {"kind": "uniform_int", "low": 1, "high": 10},
        {"kind": "empirical", "values": [2, 4], "weights": [0.25, 0.75]},
        {"kind": "poisson", "lam": 3.0, "low": 1, "high": None},
        {"kind": "log_uniform_int", "low": 1, "high": 8},
        {"kind": "pow2_choice", "options": [1, 4, 8]},
    ],
)
def test_resolved_step_policy_round_trip(policy):
    manifest = _manifest()
    raw = _raw(manifest)
    raw["resolved_step_policy"] = policy
    telemetry = ExecutionTelemetry.from_dict(raw, manifest)
    assert telemetry.as_dict()["resolved_step_policy"] == policy


@pytest.mark.parametrize(
    "policy",
    [
        {"kind": "warmup_empirical"},
        {"kind": "empirical", "values": [2, 4], "weights": [1.0, 1.0]},
        {"kind": "uniform_int", "low": 10, "high": 1},
    ],
)
def test_unresolved_or_invalid_step_policy_rejected(policy):
    manifest = _manifest()
    raw = _raw(manifest)
    raw["resolved_step_policy"] = policy
    with pytest.raises(ValueError):
        ExecutionTelemetry.from_dict(raw, manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("plan_hash", "x"),
        ("executable_config_hash", "x"),
        ("draws_artifact", "wrong.npz"),
        ("extra", 1),
    ],
)
def test_cross_binding_and_shape_rejected(field, value):
    manifest = _manifest()
    raw = _raw(manifest)
    raw[field] = value
    with pytest.raises((ValueError, TypeError)):
        ExecutionTelemetry.from_dict(raw, manifest)


@pytest.mark.parametrize(
    "timing",
    [
        {"warmup": -1, "sampling": 0, "total": 0},
        {"warmup": 2, "sampling": 2, "total": 3},
        {"warmup": 0, "sampling": 0, "total": float("nan")},
    ],
)
def test_timing_invariants(timing):
    manifest = _manifest()
    raw = _raw(manifest)
    raw["timing_seconds"] = timing
    with pytest.raises(ValueError):
        ExecutionTelemetry.from_dict(raw, manifest)


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_grad_count_type(count):
    manifest = _manifest()
    raw = _raw(manifest)
    raw["warmup_grad_evals"] = count
    with pytest.raises(ValueError):
        ExecutionTelemetry.from_dict(raw, manifest)


def test_duplicate_json_keys_rejected():
    manifest = _manifest()
    with pytest.raises(ValueError, match="duplicate"):
        ExecutionTelemetry.from_json('{"schema":"x","schema":"y"}', manifest)


@pytest.mark.parametrize(
    "fixed", [{"num_integration_steps": 0}, {"num_integration_steps": True}]
)
def test_fixed_values_are_positive_integers(fixed):
    manifest = _manifest()
    raw = _raw(manifest)
    raw["fixed"] = fixed
    with pytest.raises(ValueError):
        ExecutionTelemetry.from_dict(raw, manifest)


def test_nonfinite_json_constant_rejected():
    manifest = _manifest()
    with pytest.raises(ValueError, match="non-finite"):
        ExecutionTelemetry.from_json('{"timing_seconds":NaN}', manifest)


@pytest.mark.parametrize(
    "marker",
    [
        {
            "type": "low_rank_inverse_mass_matrix",
            "sigma": [1.0, 2.0, 3.0],
            "U": [[1.0, 0.0], [0.0, 1.0]],
            "lam": [1.0, 2.0],
        },
        {
            "type": "low_rank_inverse_mass_matrix",
            "sigma": [1.0],
            "U": [[1.0, 0.0]],
            "lam": [1.0, 2.0],
        },
    ],
)
def test_low_rank_dimension_and_rank_are_bounded(marker):
    manifest = _manifest()
    raw = _raw(manifest)
    raw["geometry"] = marker
    raw["geometry_source"] = "adapted"
    raw["geometry_scope"] = "shared"
    raw["geometry_unavailable_reason"] = None
    with pytest.raises(ValueError):
        ExecutionTelemetry.from_dict(raw, manifest)


def test_batched_low_rank_chain_count_matches_manifest():
    manifest = _manifest()
    raw = _raw(manifest)
    raw["geometry"] = {
        "inverse_mass_matrix": {
            "type": "low_rank_inverse_mass_matrix",
            "sigma": [[1.0, 2.0], [1.0, 2.0]],
            "U": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
            "lam": [[1.0, 2.0], [1.0, 2.0]],
        }
    }
    raw["geometry_source"] = "adapted"
    raw["geometry_scope"] = "per_chain"
    raw["geometry_unavailable_reason"] = None
    with pytest.raises(ValueError, match="num_chains"):
        ExecutionTelemetry.from_dict(raw, manifest)


@pytest.mark.parametrize(
    ("geometry", "source", "scope", "reason"),
    [
        ({"step_size": None}, "adapted", "shared", None),
        ({"step_size": 0.1}, "unavailable", "shared", None),
        ({"step_size": 0.1}, "adapted", None, None),
        ({}, "adapted", None, "missing"),
        ({}, "unavailable", "shared", "missing"),
    ],
)
def test_geometry_provenance_is_unambiguous(geometry, source, scope, reason):
    manifest = _manifest()
    raw = _raw(manifest)
    raw.update(
        geometry=geometry,
        geometry_source=source,
        geometry_scope=scope,
        geometry_unavailable_reason=reason,
    )
    with pytest.raises(ValueError):
        ExecutionTelemetry.from_dict(raw, manifest)


def test_per_chain_geometry_requires_one_value_per_manifest_chain():
    manifest = _manifest()
    raw = _raw(manifest)
    raw.update(
        geometry={"step_size": [0.1, 0.2]},
        geometry_source="adapted",
        geometry_scope="per_chain",
        geometry_unavailable_reason=None,
    )
    with pytest.raises(ValueError, match="num_chains"):
        ExecutionTelemetry.from_dict(raw, manifest)


@pytest.mark.parametrize(
    ("geometry", "scope", "message"),
    [
        ({"step_size": "fast"}, "shared", "finite real"),
        ({"step_size": 0.0}, "shared", "positive"),
        ({"L": -1.0}, "shared", "positive"),
        ({"step_size": [0.1]}, "shared", "must be scalar"),
        ({"step_size": [[0.1]]}, "per_chain", "must contain scalars"),
        (
            {"inverse_mass_matrix": [[1.0, 0.0], [0.0]]},
            "shared",
            "rectangular",
        ),
        ({"unknown": 1.0}, "shared", "unsupported fields"),
    ],
)
def test_geometry_values_are_typed_and_finite(geometry, scope, message):
    manifest = _manifest()
    raw = _raw(manifest)
    raw.update(
        geometry=geometry,
        geometry_source="adapted",
        geometry_scope=scope,
        geometry_unavailable_reason=None,
    )

    with pytest.raises(ValueError, match=message):
        ExecutionTelemetry.from_dict(raw, manifest)


def test_shared_dense_geometry_accepts_negative_off_diagonal_entries():
    manifest = _manifest()
    raw = _raw(manifest)
    raw.update(
        geometry={
            "step_size": 0.1,
            "inverse_mass_matrix": [[2.0, -0.25], [-0.25, 1.0]],
        },
        geometry_source="adapted",
        geometry_scope="shared",
        geometry_unavailable_reason=None,
    )

    telemetry = ExecutionTelemetry.from_dict(raw, manifest)

    assert telemetry.geometry["inverse_mass_matrix"] == (
        (2.0, -0.25),
        (-0.25, 1.0),
    )


def test_fixed_rejects_unknown_fields():
    manifest = _manifest()
    raw = _raw(manifest)
    raw["fixed"] = {"undocumented_default": 1}

    with pytest.raises(ValueError, match="unsupported fields"):
        ExecutionTelemetry.from_dict(raw, manifest)


# ---------------------------------------------------------------------------
# Low-rank marker: inert (zero) columns vs deployed columns
# ---------------------------------------------------------------------------


def _low_rank(U, lam, sigma=(1.0, 1.0, 1.0)) -> dict:
    return {
        "type": "low_rank_inverse_mass_matrix",
        "sigma": list(sigma),
        "U": [list(row) for row in U],
        "lam": list(lam),
    }


def _assemble(marker: dict):
    """Assemble diag(s)(I + U(diag(lam)-I)U^T)diag(s) from the factorised form."""
    import numpy as np

    sigma = np.asarray(marker["sigma"], dtype=float)
    U = np.asarray(marker["U"], dtype=float)
    lam = np.asarray(marker["lam"], dtype=float)
    D = np.diag(sigma)
    return D @ (np.eye(len(sigma)) + U @ np.diag(lam - 1.0) @ U.T) @ D


# ---------------------------------------------------------------------------
# Low-rank marker: the active subspace is what must be orthonormal
#
# A column with lam == 1 exactly is neutral -- (lam-1) annihilates it in
# diag(s)(I + U(diag(lam)-I)U^T)diag(s) -- so its orientation is unconstrained.
# The public meta-adaptation controller publishes exactly such columns.
# ---------------------------------------------------------------------------

_T_BRANCH_SIGMA = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
_T_BRANCH_LAM = [11.917022705078125, 1.0, 1.0]
_T_BRANCH_U = [
    [-0.9999057650566101, 0.1612187623977661, -0.7918498516082764],
    [-0.004917052574455738, 0.07621265947818756, -0.20642785727977753],
    [0.009963085874915123, 0.2536298632621765, -0.3456045389175415],
    [0.003090420039370656, -0.8545103669166565, -0.08089739084243774],
    [0.0064711919985711575, -0.16315369307994843, 0.06057322397828102],
    [-0.0036927468609064817, 0.3834904134273529, 0.4480012059211731],
]


def _t_branch_marker() -> dict:
    return _low_rank(_T_BRANCH_U, _T_BRANCH_LAM, _T_BRANCH_SIGMA)


def _malformed_marker() -> dict:
    """The counterexample worth rejecting: a second ACTIVE, non-orthogonal column.

    Differs from the captured payload only in lam[1], which promotes an
    overlapping column into the active subspace and breaks the determinant
    identity.
    """
    return _low_rank(_T_BRANCH_U, [_T_BRANCH_LAM[0], 2.5, 1.0], _T_BRANCH_SIGMA)


def test_captured_payload_has_the_properties_a_whole_u_rule_gets_wrong() -> None:
    """Pin the capture, so the fixture cannot drift into something trivial."""
    import numpy as np

    U = np.asarray(_T_BRANCH_U)
    np.testing.assert_allclose(np.linalg.norm(U, axis=0), np.ones(3), atol=1e-5)
    np.testing.assert_allclose(
        np.abs(U.T @ U - np.eye(3)).max(), 0.7878345847129822, rtol=1e-6, atol=1e-6
    )
    assert sum(1 for value in _T_BRANCH_LAM if value != 1.0) == 1


def test_captured_payload_is_a_valid_metric() -> None:
    """If it is SPD and matches the logdet identity, rejecting it is our bug."""
    import numpy as np

    dense = _assemble(_t_branch_marker())
    np.testing.assert_allclose(dense, dense.T, atol=1e-6)
    assert float(np.linalg.eigvalsh(dense).min()) > 0.0
    np.testing.assert_allclose(
        float(np.linalg.slogdet(dense)[1]),
        2.0 * float(np.sum(np.log(_T_BRANCH_SIGMA)))
        + float(np.sum(np.log(_T_BRANCH_LAM))),
        rtol=1e-5,
        atol=1e-5,
    )


def test_malformed_active_subspace_breaks_the_logdet_identity() -> None:
    """Why the active check cannot be relaxed, shown rather than asserted."""
    import numpy as np

    marker = _malformed_marker()
    dense_logdet = float(np.linalg.slogdet(_assemble(marker))[1])
    closed_form = 2.0 * float(np.sum(np.log(_T_BRANCH_SIGMA))) + float(
        np.sum(np.log(marker["lam"]))
    )
    assert abs(dense_logdet - closed_form) > 1e-3


def test_all_three_consumers_accept_the_capture_and_reject_the_malformed() -> None:
    """One fixture through every consumer of this representation.

    These three drifted apart once already: telemetry accepted a payload the
    sampler-emit and pinned-replay guards rejected, so a joint attempt could be
    recorded and then never replayed.
    """
    import numpy as np

    from tuningfork.recipes._emit._sampler import _validate_low_rank_marker
    from tuningfork.recipes._execution_telemetry import _validate_low_rank

    captured, malformed = _t_branch_marker(), _malformed_marker()

    _validate_low_rank(captured)
    _validate_low_rank_marker(captured)
    with pytest.raises(ValueError, match="orthonormal"):
        _validate_low_rank(malformed)
    with pytest.raises(ValueError, match="orthonormal"):
        _validate_low_rank_marker(malformed)

    # The pinned-replay guard applies the same rule to numpy arrays.
    def _replay_guard_accepts(marker: dict) -> bool:
        basis = np.asarray(marker["U"])
        lam = np.asarray(marker["lam"])
        active = np.flatnonzero(lam != 1.0)
        if not active.size:
            return True
        columns = basis[:, active]
        return bool(
            np.allclose(columns.T @ columns, np.eye(active.size), rtol=1e-5, atol=1e-6)
        )

    assert _replay_guard_accepts(captured)
    assert not _replay_guard_accepts(malformed)


def test_rank_zero_payload_is_the_same_rule_degenerately() -> None:
    """metric="auto" pre-escalation: full-width U of zeros, lam all 1."""
    from tuningfork.recipes._execution_telemetry import _validate_low_rank

    _validate_low_rank(_low_rank([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [1.0, 1.0]))


def test_neutrality_is_exact_and_zero_active_columns_fail() -> None:
    """A near-one lam is ACTIVE; an active column must still be orthonormal."""
    from tuningfork.recipes._execution_telemetry import _validate_low_rank

    zeros = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    with pytest.raises(ValueError, match="orthonormal"):
        _validate_low_rank(_low_rank(zeros, [1.0 + 1e-9, 1.0]))
    with pytest.raises(ValueError, match="orthonormal"):
        _validate_low_rank(_low_rank(zeros, [2.0, 1.0]))


def test_neutral_columns_do_not_affect_the_assembled_matrix() -> None:
    """Why neutral columns are exempt: they are invisible to the metric."""
    import numpy as np

    from tuningfork.recipes._execution_telemetry import _validate_low_rank

    base = _low_rank([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [2.0, 1.0])
    perturbed = _low_rank([[1.0, 7.0], [0.0, -3.5], [0.0, 0.25]], [2.0, 1.0])
    np.testing.assert_allclose(_assemble(base), _assemble(perturbed))
    _validate_low_rank(base)
    _validate_low_rank(perturbed)


def test_existing_finite_positive_and_dimension_checks_are_unchanged() -> None:
    """The exemption must not have widened any other guard."""
    from tuningfork.recipes._execution_telemetry import _validate_low_rank

    zeros = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    for lam, match in (
        ([1.0, 0.0], "lam must be finite and positive"),
        ([1.0, -1.0], "lam must be finite and positive"),
        ([1.0, float("inf")], "lam must be finite and positive"),
        ([1.0], "U/lam shapes do not match"),
    ):
        with pytest.raises(ValueError, match=match):
            _validate_low_rank(_low_rank(zeros, lam))
    with pytest.raises(ValueError, match="U must be finite numeric"):
        _validate_low_rank(
            _low_rank([[0.0, float("nan")], [0.0, 0.0], [0.0, 0.0]], [1.0, 1.0])
        )
    with pytest.raises(ValueError, match="sigma must be finite and positive"):
        _validate_low_rank(_low_rank(zeros, [1.0, 1.0], sigma=(1.0, 0.0, 1.0)))
    with pytest.raises(ValueError, match="sigma dimension or rank"):
        _validate_low_rank(_low_rank(zeros, [1.0, 1.0], sigma=(1.0, 1.0)))
