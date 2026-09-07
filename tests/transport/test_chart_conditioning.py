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
"""The clock residual as a runtime diagnostic.

The chart's unit-clock-rate identity says ``h . z`` equals the clock coordinate
``t`` exactly.  In floating point it does not: ``h . z`` is recovered from a
difference of two ``~e^{alpha t}`` terms, so it degrades with the clock while
every other component of the flow stays at dtype precision.  The residual
``|h . z - t|`` is therefore an ``O(d)`` runtime measurement of that damage,
computable from quantities ``forward`` already has.

What this module asserts is narrow on purpose: that the residual is available
and is zero where the chart is well conditioned.  It does **not** assert that
the implementation degrades at any particular clock.  An earlier version did,
which would have made a correctness improvement fail the suite — a test that
pins today's rounding behaviour blocks tomorrow's repair.

Nor is the residual a certificate.  It detects error in the clock component and
is blind to anything leaving ``h . z`` intact, so a small value says nothing
about the accuracy of the score or the log-Jacobian.  Those are covered by the
algebra tests against an independent reference, not by this diagnostic.

Two limits, both established by review rather than assumed:

* the severity is a property of the **(chart, target) pair**, not the chart.  An
  isotropic Gaussian's score has no exponential clock dependence, so its score
  stays accurate even when the clock coordinate is destroyed.  The funnel is
  load-bearing as a probe target here and must not be swapped for a tamer one.
* a projection ``z <- z + h (t - h . z)`` is exact in real arithmetic and
  repairs the residual, but adopting it changes the implemented map and would
  need its own map/inverse/Jacobian/score verification.  It is documented here,
  not shipped.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.transport import x64_scope
from tuningfork.transport._chart import make_chart

pytestmark = pytest.mark.slow

use_x64 = pytest.fixture(autouse=True, scope="module")(x64_scope)

DIM = 10
ALPHA = 0.5


def _funnel_chart():
    h = jnp.zeros(DIM).at[-1].set(1.0)
    return make_chart(h, -ALPHA * h, h, ALPHA, jnp.zeros(DIM), jnp.ones(DIM))


def _clock_residual(chart, y):
    """|h . z - t| for the flowed point, the diagnostic itself."""
    section = chart._reflect(jnp.concatenate([y[:-1], jnp.zeros((1,), y.dtype)]))
    z = chart._flow(section, y[-1])
    return float(jnp.abs(jnp.dot(chart.h, z) - y[-1])), z


def test_clock_residual_is_zero_where_the_chart_is_well_conditioned():
    """The diagnostic reads clean in the regime the chart is meant for."""
    chart = _funnel_chart()
    rng = np.random.default_rng(2)
    section = rng.normal(size=DIM - 1)
    for clock in (0.0, 5.0, 10.0, 20.0):
        y = jnp.asarray(np.concatenate([section, [clock]]))
        residual, z = _clock_residual(chart, y)
        assert jnp.all(jnp.isfinite(z))
        assert residual == 0.0 or residual < 1e-9


def test_clock_residual_is_available_without_extra_cost():
    """It is one dot product over quantities `forward` already computes."""
    chart = _funnel_chart()
    y = jnp.asarray(np.concatenate([np.zeros(DIM - 1), [3.0]]))
    residual, z = _clock_residual(chart, y)
    assert z.shape == (DIM,)
    assert isinstance(residual, float)


def test_projection_is_a_no_op_on_a_well_conditioned_point():
    """The candidate repair does not disturb a correct value.

    ``z + h (t - h . z)`` displaces by exactly the residual, so where the
    residual is zero it changes nothing.  That is the property that makes it
    safe to consider; it is not evidence that adopting it is correct, which
    would need the full map/inverse/Jacobian/score re-verification.
    """
    chart = _funnel_chart()
    y = jnp.asarray(
        np.concatenate([np.random.default_rng(4).normal(size=DIM - 1), [8.0]])
    )
    residual, z = _clock_residual(chart, y)
    projected = z + chart.h * (y[-1] - jnp.dot(chart.h, z))
    assert float(jnp.max(jnp.abs(projected - z))) <= max(residual, 1e-15) * 1.5
