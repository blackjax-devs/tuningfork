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
"""Special-case charts, the input contract, and the funnel factorisation.

Two earlier "refuters" tested nothing: the identity chart's ``forward`` was
literally the identity map, and the "rotated" chart's low-rank factor was
exactly ``I``, so it never entered ``_lowrank``.  Both are replaced with cases
that actually exercise the machinery.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.transport import x64_scope
from tuningfork.transport._chart import make_chart

pytestmark = pytest.mark.slow

use_x64 = pytest.fixture(autouse=True, scope="module")(x64_scope)

RTOL = 1e-12
FUNNEL_DIM = 10
FUNNEL_CLOCK_SD = 3.0


def _rel(got, want):
    return float(
        jnp.max(jnp.abs(got - want)) / jnp.maximum(jnp.max(jnp.abs(want)), 1.0)
    )


def test_pure_reflection_chart_preserves_volume_and_norm():
    """alpha=0, a=0, c=h: a genuine Householder reflection, not the identity."""
    d = 6
    h = jnp.asarray(np.random.default_rng(0).normal(size=d))
    chart = make_chart(h, jnp.zeros(d), h, 0.0, jnp.zeros(d), jnp.ones(d))
    rng = np.random.default_rng(1)
    moved = 0.0
    for y in rng.normal(size=(4, d)):
        y = jnp.asarray(y)
        assert float(jnp.abs(chart.log_det(y))) < 1e-13
        assert (
            float(jnp.abs(jnp.linalg.norm(chart.forward(y)) - jnp.linalg.norm(y)))
            < 1e-12
        )
        assert _rel(chart.inverse(chart.forward(y)), y) < RTOL
        moved = max(moved, float(jnp.max(jnp.abs(chart.forward(y) - y))))
    assert moved > 1e-3, "must not degenerate into the identity map"


def test_active_low_rank_factor_is_exercised_and_exact():
    """A genuinely active low-rank factor: lam != 1, so _lowrank does work."""
    d = 6
    rng = np.random.default_rng(3)
    basis = jnp.asarray(np.linalg.qr(rng.normal(size=(d, 3)))[0])
    eigs = jnp.asarray([4.0, 0.25, 9.0])
    chart = make_chart(
        jnp.asarray(rng.normal(size=d)),
        jnp.zeros(d),
        jnp.asarray(rng.normal(size=d)),
        0.3,
        jnp.zeros(d),
        jnp.ones(d),
        basis,
        eigs,
    )
    # the factor is not a no-op
    x = jnp.asarray(rng.normal(size=d))
    assert float(jnp.max(jnp.abs(chart._lowrank(x, 0.5) - x))) > 1e-2
    y = jnp.asarray(rng.normal(size=d))
    _, expected = jnp.linalg.slogdet(jax.jacfwd(chart.forward)(y))
    assert _rel(chart.log_det(y), expected) < RTOL
    assert _rel(chart.inverse(chart.forward(y)), y) < RTOL


def test_non_normal_generator_still_inverts_exactly():
    d = 6
    rng = np.random.default_rng(4)
    h = jnp.zeros(d).at[-1].set(1.0)
    a = jnp.asarray(rng.normal(size=d)).at[-1].set(0.0)
    chart = make_chart(h, a, h, 0.45, jnp.zeros(d), jnp.ones(d))
    m = chart.alpha * jnp.eye(d) + jnp.outer(chart.a, chart.h)
    assert float(jnp.max(jnp.abs(m @ m.T - m.T @ m))) > 1e-3, "must be non-normal"
    for y in rng.normal(size=(3, d)):
        y = jnp.asarray(y)
        assert _rel(chart.inverse(chart.forward(y)), y) < RTOL


# ----------------------------------------------------------- input contract
def test_input_contract_refuses_invalid_charts():
    """Invalid input is refused, not silently repaired into a wrong chart."""
    d = 4
    h = jnp.zeros(d).at[-1].set(1.0)
    ok = dict(a=jnp.zeros(d), c=h, alpha=0.2, center=jnp.zeros(d), scale=jnp.ones(d))
    basis = jnp.asarray(np.linalg.qr(np.random.default_rng(0).normal(size=(d, 2)))[0])

    with pytest.raises(ValueError, match="non-zero"):
        make_chart(jnp.zeros(d), **ok)
    with pytest.raises(ValueError, match="non-zero"):
        make_chart(h, **{**ok, "scale": jnp.ones(d).at[1].set(0.0)})
    with pytest.raises(ValueError, match="together"):
        make_chart(h, **ok, lr_basis=basis)
    with pytest.raises(ValueError, match="positive"):
        make_chart(h, **ok, lr_basis=basis, lr_eigenvalues=jnp.asarray([1.0, -2.0]))
    with pytest.raises(ValueError, match="orthonormal"):
        make_chart(
            h, **ok, lr_basis=basis * 2.0, lr_eigenvalues=jnp.asarray([4.0, 0.5])
        )


def test_neutral_low_rank_columns_need_no_orthogonality():
    """lam == 1 columns contribute nothing, so the contract must not reject them.

    Requiring orthonormality of every column would refuse legitimate inputs that
    carry neutral columns.
    """
    d = 5
    rng = np.random.default_rng(9)
    basis = np.linalg.qr(rng.normal(size=(d, 3)))[0]
    basis[:, 2] = rng.normal(size=d) * 2.0  # arbitrary, non-orthogonal
    h = jnp.asarray(rng.normal(size=d))
    chart = make_chart(
        h,
        jnp.zeros(d),
        h,
        0.25,
        jnp.zeros(d),
        jnp.ones(d),
        jnp.asarray(basis),
        jnp.asarray([4.0, 0.25, 1.0]),  # third is NEUTRAL
    )
    y = jnp.asarray(rng.normal(size=d))
    _, expected = jnp.linalg.slogdet(jax.jacfwd(chart.forward)(y))
    assert _rel(chart.log_det(y), expected) < RTOL
    assert _rel(chart.inverse(chart.forward(y)), y) < RTOL


# --------------------------------------------------------- funnel structure
def _funnel_logdensity(q):
    v, theta = q[-1], q[:-1]
    return (
        -0.5 * (v / FUNNEL_CLOCK_SD) ** 2
        - 0.5 * jnp.sum(theta * theta) * jnp.exp(-v)
        - 0.5 * (FUNNEL_DIM - 1) * v
    )


def test_exact_funnel_chart_factorises_the_clock():
    """The lead, stated at the strength the evidence supports.

    For the funnel's own scaling map, the transformed clock score is ``-t/9``
    with no section dependence, so the density factorises into a Gaussian clock
    times a section marginal.  This is checked on the plain, non-hooked
    transformed density via its mixed partial, which is the discriminator: a
    chart that does *not* factorise has a clock score that varies with ``s``.

    Scope: this is an analytic property of a *supplied* chart on one target.  It
    establishes nothing about fitted charts, about mixing, or about behaviour
    away from the clock range probed here.
    """
    h = jnp.zeros(FUNNEL_DIM).at[-1].set(1.0)
    chart = make_chart(h, -0.5 * h, h, 0.5, jnp.zeros(FUNNEL_DIM), jnp.ones(FUNNEL_DIM))
    reference = lambda y: _funnel_logdensity(chart.forward(y)) + chart.log_det(
        y
    )  # noqa: E731
    clock_row = jax.jacfwd(lambda y: jax.grad(reference)(y)[-1])

    rng = np.random.default_rng(13)
    for _ in range(3):
        y = jnp.asarray(
            np.concatenate([rng.normal(size=FUNNEL_DIM - 1), [rng.normal() * 3]])
        )
        row = clock_row(y)
        assert (
            float(jnp.max(jnp.abs(row[:-1]))) < 1e-10
        ), "clock score depends on section"
        assert abs(float(row[-1]) + 1.0 / 9.0) < 1e-10, "clock curvature not constant"


def test_float32_low_rank_chart_is_accepted():
    """Regression for the orthonormality tolerance.

    A genuinely orthonormal float32 basis carries Gram error around
    ``sqrt(d) * eps32``, so a fixed absolute bound rejects a VALID input.  No
    test constructed a float32 low-rank chart before, which is why that went
    unnoticed; the tolerance now scales with dtype and rank.
    """
    d = 6
    rng = np.random.default_rng(21)
    basis = jnp.asarray(np.linalg.qr(rng.normal(size=(d, 3)))[0], dtype=jnp.float32)
    chart = make_chart(
        jnp.asarray(rng.normal(size=d), dtype=jnp.float32),
        jnp.zeros(d, dtype=jnp.float32),
        jnp.asarray(rng.normal(size=d), dtype=jnp.float32),
        jnp.float32(0.3),
        jnp.zeros(d, dtype=jnp.float32),
        jnp.ones(d, dtype=jnp.float32),
        basis,
        jnp.asarray([4.0, 0.25, 9.0], dtype=jnp.float32),
    )
    y = jnp.asarray(rng.normal(size=d), dtype=jnp.float32)
    assert float(jnp.max(jnp.abs(chart.inverse(chart.forward(y)) - y))) < 1e-3


def test_non_finite_low_rank_inputs_are_refused():
    """The NaN-blind path: a NaN Gram residual would pass the `> tol` gate.

    `NaN > tol` is False, so an orthonormality check written as a bare
    comparison accepts a basis it cannot evaluate — the same defect the gates in
    this suite were repaired for. `+inf` likewise satisfies a bare `> 0` test on
    the eigenvalues. Both are refused before any spectral work, and for every
    column: `0 * NaN` is NaN, so a neutral column is not inert either.
    """
    d = 5
    rng = np.random.default_rng(31)
    h = jnp.asarray(rng.normal(size=d))
    basis = jnp.asarray(np.linalg.qr(rng.normal(size=(d, 2)))[0])
    ok = dict(a=jnp.zeros(d), c=h, alpha=0.2, center=jnp.zeros(d), scale=jnp.ones(d))

    with pytest.raises(ValueError, match="lr_basis must be finite"):
        make_chart(
            h,
            **ok,
            lr_basis=basis.at[0, 0].set(jnp.nan),
            lr_eigenvalues=jnp.asarray([4.0, 0.25]),
        )
    with pytest.raises(ValueError, match="lr_eigenvalues must be finite"):
        make_chart(h, **ok, lr_basis=basis, lr_eigenvalues=jnp.asarray([jnp.inf, 0.25]))


def test_declared_shapes_are_enforced():
    """Broadcasting would otherwise build a chart outside the declared family."""
    d = 5
    h = jnp.zeros(d).at[-1].set(1.0)
    base = dict(center=jnp.zeros(d), scale=jnp.ones(d))
    with pytest.raises(ValueError, match="a shape"):
        make_chart(h, a=jnp.asarray(0.0), c=h, alpha=0.2, **base)
    with pytest.raises(ValueError, match="alpha must be a scalar"):
        make_chart(h, a=jnp.zeros(d), c=h, alpha=jnp.zeros(2), **base)


def test_mismatched_center_and_nan_inputs_are_refused():
    """Two contract gaps: size agreement, and finiteness of a/c/alpha/center."""
    d = 5
    h = jnp.zeros(d).at[-1].set(1.0)
    ok = dict(a=jnp.zeros(d), c=h, alpha=0.2, scale=jnp.ones(d))
    with pytest.raises(ValueError, match="center shape"):
        make_chart(h, **ok, center=jnp.zeros(d - 1))
    with pytest.raises(ValueError, match="finite"):
        make_chart(
            h,
            a=jnp.zeros(d),
            c=h.at[0].set(jnp.nan),
            alpha=0.2,
            center=jnp.zeros(d),
            scale=jnp.ones(d),
        )
