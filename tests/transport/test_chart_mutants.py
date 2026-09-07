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
"""Tier 1 — seven wrong-chart mutants, each EXECUTED through the real gates.

A mutation test is only evidence if the mutant is actually run: asserting an
algebraic offset without executing a wrong provider proves nothing about the
gates.  So every mutant here is a real callable substituted into the real
:class:`Chart`, and each gate is the same function Tier 0 uses.

The table this suite pins down is not just "everything fails" — it is **which
gate fires and which stays silent**:

======  =========================================  ==================================
mutant  mutation                                   caught by
======  =========================================  ==================================
M1      drop ``alpha (d-1) t`` from ``log_det``    log-det, score
M2      sign-flip that term                        log-det, score
M3      ``alpha (d-1)`` -> ``alpha d``             log-det, score
M4      drop it from ``pullback_score`` only       **score only** — log-det is silent
M5      drop ``a (h.z)`` from the field            **score only** — log-det is silent
M6      un-normalise ``h``                         structural, round-trip, score
M7      break ``h.c = 1``                          structural, round-trip, score
======  =========================================  ==================================

M4 and M5 are the load-bearing rows: a determinant check cannot see either, so a
suite without an independent score gate would pass a wrong sampler.  M7 is the
reason the structural constraints are gates and not documentation — violating
``h.c = 1`` costs the unit clock rate and silently destroys the inverse.

M6 was expected to fail the log-det gate and does **not**; executing it is what
revealed the reason.  The Householder reflector is built from the *normalised*
direction, so the section stays orthogonal to ``h`` and ``forward`` is unchanged
to rounding when ``h`` is rescaled — only ``inverse`` and the score, which
contract against ``h`` directly, see the defect.  A predicted gate pattern is therefore
part of the evidence and not a formality.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tuningfork.transport._chart import make_chart

pytestmark = pytest.mark.fast

jax.config.update("jax_enable_x64", True)

DIM = 6
ATOL = 1e-11


def _base_chart():
    rng = np.random.default_rng(11)
    return make_chart(
        jnp.asarray(rng.normal(size=DIM)),
        jnp.asarray(rng.normal(size=DIM)),
        jnp.asarray(rng.normal(size=DIM)),
        0.4,
        jnp.asarray(rng.normal(size=DIM)),
        jnp.asarray(np.exp(0.2 * rng.normal(size=DIM))),
    )


def _points(n=4):
    rng = np.random.default_rng(5)
    return [jnp.asarray(y) for y in rng.normal(size=(n, DIM))]


def _native_logdensity(q):
    return -0.5 * jnp.sum(q * q) - 0.05 * jnp.sum(jnp.cos(2.0 * q))


# --------------------------------------------------------------------- gates
def gate_structural(chart) -> bool:
    return (
        float(jnp.abs(jnp.linalg.norm(chart.h) - 1.0)) < 1e-12
        and float(jnp.abs(jnp.dot(chart.h, chart.c) - 1.0)) < 1e-12
        and float(jnp.abs(jnp.dot(chart.h, chart.a) + chart.alpha)) < 1e-12
    )


def gate_log_det(chart, log_det) -> bool:
    for y in _points():
        _, expected = jnp.linalg.slogdet(jax.jacfwd(chart.forward)(y))
        if float(jnp.abs(log_det(y) - expected)) > ATOL:
            return False
    return True


def gate_round_trip(chart) -> bool:
    for y in _points():
        if float(jnp.max(jnp.abs(chart.inverse(chart.forward(y)) - y))) > ATOL:
            return False
    return True


def gate_score(chart, log_det, score) -> bool:
    """Supplied score vs AD of the plain, NON-HOOKED composed density."""
    grad_native = jax.grad(_native_logdensity)

    def reference(y):
        return _native_logdensity(chart.forward(y)) + log_det(y)

    for y in _points():
        supplied = score(y, grad_native(chart.forward(y)))
        if float(jnp.max(jnp.abs(supplied - jax.grad(reference)(y)))) > ATOL:
            return False
    return True


def _run_gates(chart, log_det=None, score=None):
    log_det = log_det if log_det is not None else chart.log_det
    score = score if score is not None else chart.pullback_score
    return {
        "structural": gate_structural(chart),
        "log_det": gate_log_det(chart, log_det),
        "round_trip": gate_round_trip(chart),
        "score": gate_score(chart, log_det, score),
    }


# ------------------------------------------------------------------- mutants
def _mutant(name):
    """Return (chart, log_det, score) with exactly one defect injected."""
    chart = _base_chart()
    d = DIM
    if name == "M1_drop_logdet_term":
        return chart, (lambda y: chart.log_det(y) - chart.alpha * (d - 1) * y[-1]), None
    if name == "M2_sign_flip_logdet_term":
        return (
            chart,
            (lambda y: chart.log_det(y) - 2.0 * chart.alpha * (d - 1) * y[-1]),
            None,
        )
    if name == "M3_off_by_one_trace":
        return chart, (lambda y: chart.log_det(y) + chart.alpha * y[-1]), None
    if name == "M4_drop_jacobian_term_from_score":

        def score(y, g):
            out = chart.pullback_score(y, g)
            return out.at[-1].add(-chart.alpha * (d - 1))

        return chart, None, score
    if name == "M5_drop_field_rank_one_term":

        def score(y, g):
            gz = chart._cotangent(g)
            section = chart._reflect(
                jnp.concatenate([y[:-1], jnp.zeros((1,), y.dtype)])
            )
            z = chart._flow(section, y[-1])
            broken = chart.alpha * z + chart.c  # a (h.z) dropped
            clock = jnp.dot(gz, broken) + chart.alpha * (d - 1)
            return jnp.concatenate(
                [jnp.exp(chart.alpha * y[-1]) * chart._reflect(gz)[:-1], clock[None]]
            )

        return chart, None, score
    if name == "M6_unnormalised_h":
        return chart._replace(h=chart.h * 1.7), None, None
    if name == "M7_broken_unit_clock":
        return chart._replace(c=chart.c + 0.3 * chart.h), None, None
    raise AssertionError(name)


# mutant -> (structural, log_det, round_trip, score) gate outcomes.
# log_det is SILENT for M4, M5 and M6 — see the module docstring and
# test_unnormalised_h_does_not_perturb_forward.
EXPECTED = {
    "M1_drop_logdet_term": (True, False, True, False),
    "M2_sign_flip_logdet_term": (True, False, True, False),
    "M3_off_by_one_trace": (True, False, True, False),
    "M4_drop_jacobian_term_from_score": (True, True, True, False),
    "M5_drop_field_rank_one_term": (True, True, True, False),
    "M6_unnormalised_h": (False, True, False, False),
    "M7_broken_unit_clock": (False, False, False, False),
}


def test_unmutated_chart_passes_every_gate():
    """Control: the gates are not trivially failing."""
    assert _run_gates(_base_chart()) == dict.fromkeys(
        ("structural", "log_det", "round_trip", "score"), True
    )


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_mutant_is_caught_by_the_expected_gates(name):
    """Each mutant is executed; the gate pattern must match exactly."""
    chart, log_det, score = _mutant(name)
    got = _run_gates(chart, log_det, score)
    structural, log_det_ok, round_trip, score_ok = EXPECTED[name]
    assert got == {
        "structural": structural,
        "log_det": log_det_ok,
        "round_trip": round_trip,
        "score": score_ok,
    }


@pytest.mark.parametrize(
    "name", ["M4_drop_jacobian_term_from_score", "M5_drop_field_rank_one_term"]
)
def test_score_only_mutants_are_invisible_to_the_determinant_gate(name):
    """The reason an independent score gate is mandatory, not optional."""
    chart, log_det, score = _mutant(name)
    got = _run_gates(chart, log_det, score)
    assert got["log_det"] is True and got["round_trip"] is True
    assert got["score"] is False


def test_unnormalised_h_does_not_perturb_forward():
    """Why M6 is invisible to the determinant gate.

    ``forward`` contracts ``h`` only against the section, which the reflector
    (built from the normalised direction) keeps orthogonal to ``h``.  The
    orthogonality is exact in real arithmetic and holds to one ulp in floating
    point, so rescaling ``h`` moves ``forward`` — and its Jacobian — by rounding
    only, far below the gate tolerance.  Only ``inverse`` and ``pullback_score``,
    which contract against ``h`` directly, expose the defect.
    """
    chart = _base_chart()
    mutated = chart._replace(h=chart.h * 1.7)
    for y in _points():
        section = chart._reflect(jnp.concatenate([y[:-1], jnp.zeros((1,), y.dtype)]))
        assert float(jnp.abs(jnp.dot(chart.h, section))) < 1e-15
        drift = float(jnp.max(jnp.abs(mutated.forward(y) - chart.forward(y))))
        assert drift < 1e-14, "rescaling h must not move forward beyond rounding"
    assert gate_round_trip(mutated) is False
