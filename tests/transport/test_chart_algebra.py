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
"""Exact algebraic invariants of a supplied chart.

Every assertion is an identity in real arithmetic, so the tolerances are
rounding tolerances.  They are **relative**: the log-Jacobian grows like
``alpha (d-1) t``, so an absolute bound calibrated at ``|y| ~ 1`` rejects a
*correct* chart once ``|y|`` grows.  An earlier version of this suite used
absolute bounds and did exactly that.

The score reference is the plain, non-hooked ``native_logdensity(forward(y)) +
log_det(y)``.  Differentiating a ``custom_jvp`` density to test the score it
supplies would return the rule under test.  The inverse is likewise tested as a
supplied map: AD cannot manufacture a global inverse from ``forward`` alone.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.transport import rel_error, x64_scope
from tuningfork.transport._chart import make_chart

pytestmark = pytest.mark.slow  # traces JAX; well above the `fast` budget

use_x64 = pytest.fixture(autouse=True, scope="module")(x64_scope)

DIM = 8
RTOL = 1e-12


def _chart(seed=0, low_rank=0, alpha=0.35):
    rng = np.random.default_rng(seed)
    basis = eigs = None
    if low_rank:
        basis = jnp.asarray(np.linalg.qr(rng.normal(size=(DIM, low_rank)))[0])
        eigs = jnp.asarray(np.exp(0.5 * rng.normal(size=low_rank)))
    return make_chart(
        jnp.asarray(rng.normal(size=DIM)),
        jnp.asarray(rng.normal(size=DIM)),
        jnp.asarray(rng.normal(size=DIM)),
        alpha,
        jnp.asarray(rng.normal(size=DIM)),
        jnp.asarray(np.exp(0.3 * rng.normal(size=DIM))),
        basis,
        eigs,
    )


def _points(seed=1, n=4, spread=1.0):
    rng = np.random.default_rng(seed)
    return [jnp.asarray(y * spread) for y in rng.normal(size=(n, DIM))]


def _native_logdensity(q):
    """An arbitrary smooth non-Gaussian target; nothing is tuned to the chart."""
    return -0.5 * jnp.sum(q * q) - 0.1 * jnp.sum(jnp.cos(3.0 * q))


LOW_RANK = pytest.mark.parametrize("low_rank", [0, 3], ids=["diagonal", "lowrank3"])


@LOW_RANK
def test_structural_constraints_hold(low_rank):
    """|h|=1, h.c=1, h.a=-alpha — the identities every exactness claim rests on."""
    chart = _chart(low_rank=low_rank)
    assert float(jnp.abs(jnp.linalg.norm(chart.h) - 1.0)) < 1e-14
    assert float(jnp.abs(jnp.dot(chart.h, chart.c) - 1.0)) < 1e-14
    assert float(jnp.abs(jnp.dot(chart.h, chart.a) + chart.alpha)) < 1e-14


@LOW_RANK
@pytest.mark.parametrize("spread", [1.0, 30.0], ids=["near", "far"])
def test_log_det_matches_the_true_jacobian(low_rank, spread):
    """Closed-form log-Jacobian == slogdet of the AD Jacobian, near AND far.

    The `far` case is the one an absolute tolerance would fail on a correct
    chart: at |y| ~ 30 the log-determinant is O(100) and its rounding error
    scales with it.
    """
    chart = _chart(low_rank=low_rank)
    for y in _points(spread=spread):
        _, expected = jnp.linalg.slogdet(jax.jacfwd(chart.forward)(y))
        assert rel_error(chart.log_det(y), expected) < RTOL


@LOW_RANK
def test_inverse_is_an_exact_two_sided_inverse(low_rank):
    chart = _chart(low_rank=low_rank)
    for y in _points():
        assert rel_error(chart.inverse(chart.forward(y)), y) < RTOL
        q = chart.forward(y)
        assert rel_error(chart.forward(chart.inverse(q)), q) < RTOL


@LOW_RANK
def test_supplied_score_matches_the_non_hooked_reference(low_rank):
    chart = _chart(low_rank=low_rank)

    def reference(y):  # NOT a custom_jvp density — that would be circular
        return _native_logdensity(chart.forward(y)) + chart.log_det(y)

    grad_native = jax.grad(_native_logdensity)
    for y in _points():
        supplied = chart.pullback_score(y, grad_native(chart.forward(y)))
        assert rel_error(supplied, jax.grad(reference)(y)) < RTOL


@LOW_RANK
def test_push_score_inverts_pullback_score(low_rank):
    """The non-trivial low-rank control: a wrong preconditioner power fails here."""
    chart = _chart(low_rank=low_rank)
    grad_native = jax.grad(_native_logdensity)
    for y in _points():
        g = grad_native(chart.forward(y))
        assert rel_error(chart.push_score(y, chart.pullback_score(y, g)), g) < RTOL


def test_clock_advances_at_unit_rate():
    """d/ds (h.z) == 1 along the flow, independent of z."""
    chart = _chart()
    rng = np.random.default_rng(7)
    for _ in range(3):
        z = jnp.asarray(rng.normal(size=DIM))
        rate = jax.grad(lambda s: jnp.dot(chart.h, chart._flow(z, s)))(0.4)
        assert float(jnp.abs(rate - 1.0)) < 1e-12


def test_signed_scale_is_supported_as_a_log_absolute_determinant():
    """A negative scale entry flips orientation; |det| and every map stay exact."""
    rng = np.random.default_rng(3)
    scale = jnp.asarray(np.exp(0.2 * rng.normal(size=DIM))).at[2].multiply(-1.0)
    chart = make_chart(
        jnp.asarray(rng.normal(size=DIM)),
        jnp.asarray(rng.normal(size=DIM)),
        jnp.asarray(rng.normal(size=DIM)),
        0.3,
        jnp.zeros(DIM),
        scale,
    )
    for y in _points(n=2):
        _, expected = jnp.linalg.slogdet(jax.jacfwd(chart.forward)(y))
        assert jnp.isfinite(chart.log_det(y))
        assert rel_error(chart.log_det(y), expected) < RTOL
        assert rel_error(chart.inverse(chart.forward(y)), y) < RTOL
