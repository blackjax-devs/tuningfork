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
"""The gates reject wrong providers, including non-finite ones.

The gates previously compared ``float(error) > ATOL``.  That comparison is
``False`` for NaN, so a provider returning NaN everywhere **passed every gate**.
Finiteness is now part of gate success, not an afterthought.

This module deliberately does not maintain a pass/fail vector over a family of
mutants.  Which gate a given mutation happens to perturb is a property of how
the mutation was injected, not a contract worth pinning: a score-only mutation
leaves ``forward`` and ``log_det`` untouched, so their gates cannot fail, and
asserting that they do not is a restatement of the construction rather than a
finding.  Four providers are enough — a valid chart, one wrong log-determinant,
one wrong score, and one non-finite control.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.transport import x64_scope
from tuningfork.transport._chart import Chart, make_chart

pytestmark = pytest.mark.slow

use_x64 = pytest.fixture(autouse=True, scope="module")(x64_scope)

DIM = 6
RTOL = 1e-12


def _chart():
    rng = np.random.default_rng(11)
    return make_chart(
        jnp.asarray(rng.normal(size=DIM)),
        jnp.asarray(rng.normal(size=DIM)),
        jnp.asarray(rng.normal(size=DIM)),
        0.4,
        jnp.asarray(rng.normal(size=DIM)),
        jnp.asarray(np.exp(0.2 * rng.normal(size=DIM))),
    )


def _points(n=3):
    rng = np.random.default_rng(5)
    return [jnp.asarray(y) for y in rng.normal(size=(n, DIM))]


def _native_logdensity(q):
    return -0.5 * jnp.sum(q * q) - 0.05 * jnp.sum(jnp.cos(2.0 * q))


def _agrees(got, want):
    """Finite-aware relative agreement.  NaN or inf anywhere is a failure."""
    got, want = jnp.asarray(got), jnp.asarray(want)
    if not (bool(jnp.all(jnp.isfinite(got))) and bool(jnp.all(jnp.isfinite(want)))):
        return False
    scale = float(jnp.maximum(jnp.max(jnp.abs(want)), 1.0))
    return float(jnp.max(jnp.abs(got - want))) / scale < RTOL


def gate_log_det(chart, log_det):
    for y in _points():
        _, expected = jnp.linalg.slogdet(jax.jacfwd(chart.forward)(y))
        if not _agrees(log_det(y), expected):
            return False
    return True


def gate_score(chart, log_det, score):
    """Supplied score against AD of the plain, non-hooked composed density."""
    grad_native = jax.grad(_native_logdensity)

    def reference(y):
        return _native_logdensity(chart.forward(y)) + log_det(y)

    for y in _points():
        if not _agrees(score(y, grad_native(chart.forward(y))), jax.grad(reference)(y)):
            return False
    return True


def _run(chart, log_det=None, score=None):
    log_det = log_det if log_det is not None else chart.log_det
    score = score if score is not None else chart.pullback_score
    return gate_log_det(chart, log_det), gate_score(chart, log_det, score)


def test_valid_chart_passes_both_gates():
    """Control: the gates are not trivially failing."""
    assert _run(_chart()) == (True, True)


def test_wrong_log_determinant_is_rejected():
    chart = _chart()
    wrong = lambda y: chart.log_det(y) - chart.alpha * (DIM - 1) * y[-1]  # noqa: E731
    assert gate_log_det(chart, wrong) is False


def test_wrong_score_is_rejected():
    """A score missing the log-Jacobian derivative term."""
    chart = _chart()

    def wrong(y, g):
        return chart.pullback_score(y, g).at[-1].add(-chart.alpha * (DIM - 1))

    assert gate_score(chart, chart.log_det, wrong) is False


@pytest.mark.parametrize(
    "bad", [jnp.nan, jnp.inf, -jnp.inf], ids=["nan", "+inf", "-inf"]
)
def test_non_finite_providers_are_rejected(bad):
    """The regression this module exists for: NaN used to pass every gate.

    ``float(nan) > ATOL`` is False, so a comparison-only gate reported success.
    """
    chart = _chart()
    const_logdet = lambda y: jnp.asarray(bad)  # noqa: E731
    const_score = lambda y, g: jnp.full(y.shape, bad)  # noqa: E731
    assert gate_log_det(chart, const_logdet) is False
    assert gate_score(chart, chart.log_det, const_score) is False


def _chart_without_normalising_h(h, a, c, alpha, center, scale):
    """Build a chart exactly as `make_chart` does, minus the `h` normalisation.

    This is what forgetting that one line produces: every downstream quantity —
    the projection used for `a` and `c`, and the Householder reflector — is
    derived from the RAW `h`.  Rescaling `h` on a finished chart is a different
    and milder thing, because it leaves `u`, `a` and `c` consistent with the
    normalised direction; an earlier version of this test made that mistake and
    then described itself as modelling this one.
    """
    project = lambda v: v - h * jnp.dot(h, v)  # noqa: E731
    c = project(c) + h
    a = project(a) - alpha * h
    e = jnp.zeros_like(h).at[-1].set(jnp.where(h[-1] >= 0, 1.0, -1.0))
    u = h + e
    u = u / jnp.linalg.norm(u)
    empty = jnp.zeros((h.size, 0), dtype=h.dtype)
    return Chart(h, a, c, jnp.asarray(alpha), center, scale, u, empty, empty[0])


def test_omitted_normalisation_is_rejected():
    """The real defect: a chart built without normalising h at all."""
    rng = np.random.default_rng(11)
    broken = _chart_without_normalising_h(
        jnp.asarray(rng.normal(size=DIM)) * 1.7,
        jnp.asarray(rng.normal(size=DIM)),
        jnp.asarray(rng.normal(size=DIM)),
        0.4,
        jnp.asarray(rng.normal(size=DIM)),
        jnp.asarray(np.exp(0.2 * rng.normal(size=DIM))),
    )
    assert (
        float(jnp.abs(jnp.linalg.norm(broken.h) - 1.0)) > 1e-3
    ), "must be un-normalised"
    assert _run(broken) != (True, True)
