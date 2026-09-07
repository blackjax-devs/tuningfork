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


def test_neutral_columns_do_not_affect_the_assembled_matrix() -> None:
    """Why lam == 1 columns are exempt: they are invisible to the metric.

    (lam[j] - 1) is exactly 0, so column j leaves U(diag(lam)-I)U^T untouched
    whatever its values or overlap.  Constraining it would constrain something
    the representation does not use.
    """
    import numpy as np

    base = _low_rank([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [2.0, 1.0])
    # Same active column; the neutral column is replaced by an arbitrary,
    # non-orthonormal, non-zero vector that overlaps the active one.
    perturbed = _low_rank([[1.0, 7.0], [0.0, -3.5], [0.0, 0.25]], [2.0, 1.0])
    np.testing.assert_allclose(_assemble(base), _assemble(perturbed))
    _validate_low_rank_ok(base)
    _validate_low_rank_ok(perturbed)


def _validate_low_rank_ok(marker: dict) -> None:
    from tuningfork.recipes._execution_telemetry import _validate_low_rank

    _validate_low_rank(marker)


def test_t_branch_shaped_payload_is_accepted() -> None:
    """The public controller's T-branch shape: lam = [x, 1, 1, ...].

    Upstream builds U as [e_dir | U_lr[:, 1:]] -- a rank-1 slow direction
    concatenated with the tail of a DIFFERENT (Fisher low-rank) basis.  Those
    blocks are under no obligation to be mutually orthonormal, and only column
    0 is active, so the full U is not orthonormal and the payload is still
    valid.  Requiring every non-zero column to be orthonormal rejected this.
    """
    from tuningfork.recipes._execution_telemetry import _validate_low_rank

    # column 0 unit-norm and active; columns 1-2 neutral, non-zero, not
    # orthogonal to column 0 nor to each other.
    U = [
        [1.0, 0.6, 0.9],
        [0.0, 0.8, 0.1],
        [0.0, 0.2, 0.4],
    ]
    _validate_low_rank(_low_rank(U, [3.25, 1.0, 1.0]))


def test_active_subspace_must_be_orthonormal() -> None:
    from tuningfork.recipes._execution_telemetry import _validate_low_rank

    # two active columns, unit-norm but not mutually orthogonal
    with pytest.raises(ValueError, match="orthonormal"):
        _validate_low_rank(_low_rank([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], [2.0, 3.0]))
    # active column that is not unit-norm
    with pytest.raises(ValueError, match="orthonormal"):
        _validate_low_rank(_low_rank([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]], [2.0, 1.0]))
    # genuinely orthonormal actives are accepted
    _validate_low_rank(_low_rank([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], [2.0, 3.0]))


def test_rank_zero_payload_is_the_degenerate_case_of_the_same_rule() -> None:
    """metric="auto" pre-escalation: full-width U of zeros, lam all 1."""
    from tuningfork.recipes._execution_telemetry import _validate_low_rank

    _validate_low_rank(_low_rank([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [1.0, 1.0]))


@pytest.mark.parametrize("lam0", [1.0 + 1e-9, 1.0 - 1e-9, 0.9999999, 1.0000001])
def test_near_one_lam_is_active_not_neutral(lam0: float) -> None:
    """Neutrality is exact.  These all pass math.isclose and are still active."""
    import math

    from tuningfork.recipes._execution_telemetry import _validate_low_rank

    assert math.isclose(lam0, 1.0, rel_tol=1e-6, abs_tol=1e-6), "control is near one"
    # a zero column is not orthonormal, so an active zero column must fail
    with pytest.raises(ValueError, match="orthonormal"):
        _validate_low_rank(_low_rank([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [lam0, 1.0]))


def test_zero_and_near_zero_active_columns_fail() -> None:
    """A column with lam != 1 is active however small it is."""
    from tuningfork.recipes._execution_telemetry import _validate_low_rank

    with pytest.raises(ValueError, match="orthonormal"):
        _validate_low_rank(_low_rank([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [2.0, 1.0]))
    with pytest.raises(ValueError, match="orthonormal"):
        _validate_low_rank(
            _low_rank([[1e-300, 0.0], [0.0, 0.0], [0.0, 0.0]], [2.0, 1.0])
        )


def test_logdet_identity_holds_exactly_when_the_rule_does() -> None:
    """The active-orthonormal rule is what makes the two readings agree.

    With active columns orthonormal and neutral ones contributing log(1) = 0,
    logdet of the assembled matrix equals 2*sum(log sigma) + sum(log lam).
    """
    import numpy as np

    from tuningfork.recipes._execution_telemetry import _validate_low_rank

    sigma = (1.5, 0.5, 2.0)
    # T-branch shape: one active unit column, two neutral non-orthonormal ones.
    marker = _low_rank(
        [[1.0, 0.6, 0.9], [0.0, 0.8, 0.1], [0.0, 0.2, 0.4]], [3.25, 1.0, 1.0], sigma
    )
    _validate_low_rank(marker)
    dense_logdet = float(np.linalg.slogdet(_assemble(marker))[1])
    factorised = 2.0 * float(np.sum(np.log(sigma))) + float(
        np.sum(np.log(np.asarray(marker["lam"])))
    )
    np.testing.assert_allclose(dense_logdet, factorised, rtol=1e-9, atol=1e-9)


def test_existing_finite_positive_and_dimension_checks_are_unchanged() -> None:
    """The generalised exemption must not have widened any other guard."""
    from tuningfork.recipes._execution_telemetry import _validate_low_rank

    zeros = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    with pytest.raises(ValueError, match="lam must be finite and positive"):
        _validate_low_rank(_low_rank(zeros, [1.0, 0.0]))
    with pytest.raises(ValueError, match="lam must be finite and positive"):
        _validate_low_rank(_low_rank(zeros, [1.0, -1.0]))
    with pytest.raises(ValueError, match="lam must be finite and positive"):
        _validate_low_rank(_low_rank(zeros, [1.0, float("inf")]))
    with pytest.raises(ValueError, match="U must be finite numeric"):
        _validate_low_rank(
            _low_rank([[0.0, float("nan")], [0.0, 0.0], [0.0, 0.0]], [1.0, 1.0])
        )
    with pytest.raises(ValueError, match="sigma must be finite and positive"):
        _validate_low_rank(_low_rank(zeros, [1.0, 1.0], sigma=(1.0, 0.0, 1.0)))
    with pytest.raises(ValueError, match="U/lam shapes do not match"):
        _validate_low_rank(_low_rank(zeros, [1.0]))
    with pytest.raises(ValueError, match="sigma dimension or rank"):
        _validate_low_rank(_low_rank(zeros, [1.0, 1.0], sigma=(1.0, 1.0)))
