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
"""Tier 0 — exact algebraic invariants of a supplied chart.

Every assertion here is an identity in real arithmetic, so the tolerances are
rounding tolerances rather than statistical ones.  No chain is run.

The score reference is deliberately the **plain, non-hooked** expression
``native_logdensity(forward(y)) + log_det(y)`` differentiated by AD.  Testing a
supplied score against the derivative of a ``custom_jvp`` density that supplies
that same score would be circular: AD would return the rule under test.  For the
same reason no inverse is derived by AD — AD cannot manufacture a global inverse
from ``forward`` alone, so :meth:`Chart.inverse` is tested as a supplied map.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tuningfork.transport._chart import make_chart

pytestmark = pytest.mark.fast

jax.config.update("jax_enable_x64", True)

DIM = 8
ATOL = 1e-11


def _chart(seed=0, low_rank=0, alpha=0.35):
    """A generic (non-degenerate) chart: no special structure, random direction."""
    rng = np.random.default_rng(seed)
    h = jnp.asarray(rng.normal(size=DIM))
    a = jnp.asarray(rng.normal(size=DIM))
    c = jnp.asarray(rng.normal(size=DIM))
    center = jnp.asarray(rng.normal(size=DIM))
    scale = jnp.asarray(np.exp(0.3 * rng.normal(size=DIM)))
    basis = eigs = None
    if low_rank:
        raw = rng.normal(size=(DIM, low_rank))
        basis = jnp.asarray(np.linalg.qr(raw)[0])
        eigs = jnp.asarray(np.exp(0.5 * rng.normal(size=low_rank)))
    return make_chart(h, a, c, alpha, center, scale, basis, eigs)


def _points(seed=1, n=6, clock_scale=1.5):
    rng = np.random.default_rng(seed)
    ys = rng.normal(size=(n, DIM))
    ys[:, -1] *= clock_scale
    return [jnp.asarray(y) for y in ys]


def _native_logdensity(q):
    """An arbitrary smooth non-Gaussian native target; nothing is tuned to it."""
    return (
        -0.5 * jnp.sum(q * q) - 0.1 * jnp.sum(jnp.cos(3.0 * q)) - 0.01 * jnp.sum(q**4)
    )


@pytest.mark.parametrize("low_rank", [0, 3], ids=["diagonal", "lowrank3"])
def test_structural_constraints_hold_by_construction(low_rank):
    """|h|=1, h.c=1, h.a=-alpha — the three identities the exactness rests on."""
    chart = _chart(low_rank=low_rank)
    assert float(jnp.abs(jnp.linalg.norm(chart.h) - 1.0)) < 1e-15
    assert float(jnp.abs(jnp.dot(chart.h, chart.c) - 1.0)) < 1e-15
    assert float(jnp.abs(jnp.dot(chart.h, chart.a) + chart.alpha)) < 1e-15


@pytest.mark.parametrize("low_rank", [0, 3], ids=["diagonal", "lowrank3"])
def test_log_det_matches_the_true_jacobian(low_rank):
    """The closed-form log-Jacobian equals slogdet of the AD Jacobian."""
    chart = _chart(low_rank=low_rank)
    for y in _points():
        jac = jax.jacfwd(chart.forward)(y)
        _, expected = jnp.linalg.slogdet(jac)
        assert float(jnp.abs(chart.log_det(y) - expected)) < ATOL


@pytest.mark.parametrize("low_rank", [0, 3], ids=["diagonal", "lowrank3"])
def test_inverse_is_an_exact_two_sided_inverse(low_rank):
    """inverse(forward(y)) == y and forward(inverse(q)) == q."""
    chart = _chart(low_rank=low_rank)
    for y in _points():
        assert float(jnp.max(jnp.abs(chart.inverse(chart.forward(y)) - y))) < ATOL
        q = chart.forward(y)
        assert float(jnp.max(jnp.abs(chart.forward(chart.inverse(q)) - q))) < ATOL


@pytest.mark.parametrize("low_rank", [0, 3], ids=["diagonal", "lowrank3"])
def test_pullback_score_matches_the_non_hooked_reference(low_rank):
    """Supplied score == AD gradient of the plain composed density."""
    chart = _chart(low_rank=low_rank)

    def reference(y):  # NOT a custom_jvp density — that would be circular
        return _native_logdensity(chart.forward(y)) + chart.log_det(y)

    grad_native = jax.grad(_native_logdensity)
    for y in _points():
        supplied = chart.pullback_score(y, grad_native(chart.forward(y)))
        assert float(jnp.max(jnp.abs(supplied - jax.grad(reference)(y)))) < ATOL


@pytest.mark.parametrize("low_rank", [0, 3], ids=["diagonal", "lowrank3"])
def test_push_score_inverts_pullback_score(low_rank):
    """The score pullback is invertible and push_score is its inverse."""
    chart = _chart(low_rank=low_rank)
    grad_native = jax.grad(_native_logdensity)
    for y in _points():
        g_native = grad_native(chart.forward(y))
        recovered = chart.push_score(y, chart.pullback_score(y, g_native))
        assert float(jnp.max(jnp.abs(recovered - g_native))) < ATOL


def test_clock_advances_at_unit_rate():
    """d/ds (h.z) == 1 along the flow — the identity every exactness claim uses."""
    chart = _chart()
    rng = np.random.default_rng(7)
    for _ in range(5):
        z = jnp.asarray(rng.normal(size=DIM))
        rate = jax.grad(lambda s: jnp.dot(chart.h, chart._flow(z, s)))(0.4)
        assert float(jnp.abs(rate - 1.0)) < 1e-12
