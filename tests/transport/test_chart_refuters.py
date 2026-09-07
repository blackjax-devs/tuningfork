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
"""Tier 2 — degenerate and special-case charts that must behave as claimed.

A validation suite that only exercises a generic chart can hide errors that
cancel.  These refuters pin the boundary cases where the answer is known
independently: the identity, a pure linear rescaling, a rotation, a non-normal
generator, and the exact funnel chart.

The funnel case (:func:`test_exact_funnel_*`) is the one that distinguishes three
quantities people conflate:

1. the **raw-field Stein residual** ``grad(log pi).V + div V``;
2. the **regression residual** of a fitter that also carries affine-in-clock
   slack ``kappa t - beta``;
3. the resulting **clock/section score structure** of the transformed density.

For the funnel these are respectively ``-t/9``, ``~0``, and an exactly Gaussian
clock independent of the section.  A non-zero raw residual is therefore *not* a
defect.

The factorisation is an **analytic** statement with a premise that these finite
evaluations corroborate but do not prove: if, on the full real line in ``t`` and
on a global Cartesian chart with global support, ``d/dt log pi_chart = -kappa t +
beta`` holds identically with ``kappa`` constant and positive, then integrating
gives ``-kappa t^2/2 + beta t + C(s)`` — a Gaussian clock independent of the
section.  Under exactly those conditions ``kappa <= 0`` fails to normalise, which
is the sense in which a zero residual means a flat, improper clock; on a bounded
clock domain that does not follow, and nothing here forbids such domains.

For the *supplied* chart tested below the premise is discharged by the algebra and
the evaluations corroborate it.  For a *learned* chart it would remain open.

What these tests do NOT establish: that any fitted chart satisfies the identity
away from its training support, that a fitted ``kappa`` is constant or positive,
or that any of this improves mixing.  Those are separate questions and none is
tested here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tuningfork.transport._chart import make_chart

pytestmark = pytest.mark.fast

jax.config.update("jax_enable_x64", True)

ATOL = 1e-11
FUNNEL_DIM = 10
FUNNEL_CLOCK_SD = 3.0


def _identity_chart(d=6):
    """alpha=0, a=0, c=h, L=I — forward is a pure Householder reflection."""
    h = jnp.zeros(d).at[-1].set(1.0)
    return make_chart(h, jnp.zeros(d), h, 0.0, jnp.zeros(d), jnp.ones(d))


def test_identity_chart_is_a_pure_reflection_with_zero_log_det():
    """No curvature, no scaling: the chart may permute but must not distort."""
    chart = _identity_chart()
    rng = np.random.default_rng(0)
    for y in rng.normal(size=(5, 6)):
        y = jnp.asarray(y)
        assert float(jnp.abs(chart.log_det(y))) < 1e-14
        # volume preserved and norm preserved (orthogonal map)
        assert (
            float(jnp.abs(jnp.linalg.norm(chart.forward(y)) - jnp.linalg.norm(y)))
            < 1e-12
        )
        assert float(jnp.max(jnp.abs(chart.inverse(chart.forward(y)) - y))) < ATOL


def test_pure_linear_chart_equals_its_preconditioner():
    """alpha=0, a=0 collapses the chart to the affine map q = center + scale*z."""
    d = 6
    rng = np.random.default_rng(2)
    h = jnp.zeros(d).at[-1].set(1.0)
    center = jnp.asarray(rng.normal(size=d))
    scale = jnp.asarray(np.exp(0.4 * rng.normal(size=d)))
    chart = make_chart(h, jnp.zeros(d), h, 0.0, center, scale)

    # log|det| is exactly the diagonal preconditioner's, with no clock term
    y = jnp.asarray(rng.normal(size=d))
    assert float(jnp.abs(chart.log_det(y) - jnp.sum(jnp.log(scale)))) < 1e-13

    # and the map is affine: forward(y1) - forward(y2) is linear in y1 - y2
    y1, y2 = jnp.asarray(rng.normal(size=d)), jnp.asarray(rng.normal(size=d))
    mid = chart.forward(0.5 * (y1 + y2))
    assert (
        float(jnp.max(jnp.abs(mid - 0.5 * (chart.forward(y1) + chart.forward(y2)))))
        < 1e-12
    )


def test_rotated_chart_preserves_volume_through_an_orthogonal_low_rank_factor():
    """A rotation-like low-rank factor with unit eigenvalues adds no volume."""
    d = 6
    rng = np.random.default_rng(3)
    basis = jnp.asarray(np.linalg.qr(rng.normal(size=(d, 3)))[0])
    eigs = jnp.ones(3)
    h = jnp.asarray(rng.normal(size=d))
    chart = make_chart(h, jnp.zeros(d), h, 0.0, jnp.zeros(d), jnp.ones(d), basis, eigs)
    y = jnp.asarray(rng.normal(size=d))
    assert float(jnp.abs(chart.log_det(y))) < 1e-13
    _, expected = jnp.linalg.slogdet(jax.jacfwd(chart.forward)(y))
    assert float(jnp.abs(chart.log_det(y) - expected)) < ATOL


def test_non_normal_generator_still_inverts_exactly():
    """`a` outside span(h) makes V non-normal; the closed-form inverse must hold."""
    d = 6
    rng = np.random.default_rng(4)
    h = jnp.zeros(d).at[-1].set(1.0)
    a = jnp.asarray(rng.normal(size=d)).at[-1].set(0.0)  # strictly transverse
    chart = make_chart(h, a, h, 0.45, jnp.zeros(d), jnp.ones(d))
    # genuinely non-normal: V's matrix part does not commute with its transpose
    m = chart.alpha * jnp.eye(d) + jnp.outer(chart.a, chart.h)
    assert float(jnp.max(jnp.abs(m @ m.T - m.T @ m))) > 1e-3
    for y in rng.normal(size=(5, d)):
        y = jnp.asarray(y)
        assert float(jnp.max(jnp.abs(chart.inverse(chart.forward(y)) - y))) < ATOL


# --------------------------------------------------------------- exact funnel
def _funnel_logdensity(q):
    """Neal's funnel, clock last: v ~ N(0, 3^2), theta_i | v ~ N(0, exp(v))."""
    v, theta = q[-1], q[:-1]
    return (
        -0.5 * (v / FUNNEL_CLOCK_SD) ** 2
        - 0.5 * jnp.sum(theta * theta) * jnp.exp(-v)
        - 0.5 * (FUNNEL_DIM - 1) * v
    )


def _funnel_chart():
    """The funnel's scaling map inside the family: alpha=1/2, h=c=e_last."""
    h = jnp.zeros(FUNNEL_DIM).at[-1].set(1.0)
    return make_chart(h, -0.5 * h, h, 0.5, jnp.zeros(FUNNEL_DIM), jnp.ones(FUNNEL_DIM))


def _funnel_reference(chart):
    """The plain, NON-HOOKED transformed density."""
    return lambda y: _funnel_logdensity(chart.forward(y)) + chart.log_det(y)


def _funnel_points(n=6, seed=13):
    rng = np.random.default_rng(seed)
    ys = rng.normal(size=(n, FUNNEL_DIM))
    ys[:, -1] *= 4.0
    return [jnp.asarray(y) for y in ys]


def test_exact_funnel_raw_field_stein_residual_is_not_zero():
    """(1) The raw field residual is -v/9, not 0 — and equals the clock score.

    ``grad(log pi).V + div V`` is exactly ``d/dt log pi_chart``.  For the funnel
    it is ``-v/9``: the conditional-scale part cancels, the N(0,9) clock prior
    does not.  A field with zero residual would correspond to a flat, improper
    clock, which is not what a useful chart produces.
    """
    chart = _funnel_chart()
    grad_native = jax.grad(_funnel_logdensity)
    div_v = jnp.trace(jax.jacfwd(chart.field)(jnp.zeros(FUNNEL_DIM)))
    assert float(jnp.abs(div_v - chart.alpha * (FUNNEL_DIM - 1))) < 1e-12

    grad_chart = jax.grad(_funnel_reference(chart))
    for y in _funnel_points():
        t = float(y[-1])
        z = chart.forward(y)  # L = I here, so native == preconditioned
        stein = float(jnp.dot(grad_native(z), chart.field(z)) + div_v)
        assert abs(stein - (-t / 9.0)) < 1e-9, "raw residual must be -v/9"
        # ... and it IS the clock component of the transformed score
        assert abs(float(grad_chart(y)[-1]) - stein) < 1e-9


def test_exact_funnel_regression_residual_vanishes_only_with_clock_slack():
    """(2) The same field has ~0 residual once affine-in-clock slack is allowed.

    Distinguishes the fitter's objective from the deployed field: with
    ``kappa = 1/9`` the residual is annihilated, so a near-zero *regression*
    residual is a statement about the objective, not a certificate of the field.
    """
    chart = _funnel_chart()
    grad_native = jax.grad(_funnel_logdensity)
    div_v = jnp.trace(jax.jacfwd(chart.field)(jnp.zeros(FUNNEL_DIM)))
    kappa, beta = 1.0 / 9.0, 0.0
    for y in _funnel_points():
        z = chart.forward(y)
        raw = float(jnp.dot(grad_native(z), chart.field(z)) + div_v)
        assert abs(raw + kappa * float(y[-1]) - beta) < 1e-9


def test_exact_funnel_yields_a_gaussian_clock_independent_of_the_section():
    """(3) The resulting structure: d_t log pi_chart = -kappa t, kappa = 1/9 > 0.

    The decisive check is the mixed partial: if the clock score is affine in the
    clock with no section dependence, the transformed density factorises as
    ``-kappa t^2/2 + C(s)`` — an exactly Gaussian clock independent of the
    section.  This is the intended useful structure, not a defect to remove.
    """
    chart = _funnel_chart()
    reference = _funnel_reference(chart)
    grad_chart = jax.grad(reference)
    clock_hessian_row = jax.jacfwd(lambda y: grad_chart(y)[-1])

    for y in _funnel_points():
        row = clock_hessian_row(y)
        # no section dependence anywhere in the clock score
        assert float(jnp.max(jnp.abs(row[:-1]))) < 1e-10
        # constant negative curvature => kappa = +1/9 > 0 (a proper clock)
        assert abs(float(row[-1]) + 1.0 / 9.0) < 1e-10

    # the integrated statement: log pi(t,s) - log pi(0,s) == -t^2 / 18
    section = _funnel_points(n=1, seed=21)[0]
    base = float(reference(section.at[-1].set(0.0)))
    for t in (-5.0, -2.0, 1.0, 3.0, 6.0):
        value = float(reference(section.at[-1].set(t))) - base
        assert abs(value + t * t / (2.0 * 9.0)) < 1e-9

    # and the section score does not depend on the clock at all
    ref_section_score = grad_chart(section.at[-1].set(0.0))[:-1]
    for t in (-4.0, 5.0):
        moved = grad_chart(section.at[-1].set(t))[:-1]
        assert float(jnp.max(jnp.abs(moved - ref_section_score))) < 1e-10
