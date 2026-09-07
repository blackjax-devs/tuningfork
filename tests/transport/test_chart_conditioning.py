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
"""An EMPIRICAL conditioning indicator for the chart, and its limits.

The chart's flow carries a factor ``exp(alpha * clock)``.  Far out along the
clock the score is a contraction of exponentially large quantities whose true
value is small, so accuracy degrades long before anything overflows.  These
tests measure that degradation with a **dtype ladder** (float32 against
float64), which needs no reimplementation of the chart and therefore tests the
shipped code path rather than a stand-in.

What this establishes
---------------------
That ``exp(alpha * clock)`` *tracks* the observed loss of accuracy for this
chart, in this dtype pair, on this target — and, more importantly, that the
failure is **silent**: the chart returns finite, ordinary-looking numbers that
are wrong by orders of magnitude, at clock values well inside the representable
range.  A guard that waits for a NaN cannot catch this.

What this does NOT establish
----------------------------
It is an *indicator*, not a certified bound.  Calling it a bound would require a
derivation covering the chart parameters, the coordinates, the dtype, the
conditioning of the target's own score, and the error scale.  None of that is
attempted here.  In particular the overflow clock is a property of these
parameters and this dtype and is **not** a universal domain limit; and nothing
here says what a sampler would do with a corrupted endpoint.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tuningfork.transport._chart import make_chart

pytestmark = pytest.mark.fast

jax.config.update("jax_enable_x64", True)

DIM = 10
ALPHA = 0.5


def _funnel_logdensity(q):
    v, theta = q[-1], q[:-1]
    return (
        -0.5 * (v / 3.0) ** 2
        - 0.5 * jnp.sum(theta * theta) * jnp.exp(-v)
        - 0.5 * (DIM - 1) * v
    )


def _chart(dtype):
    h = jnp.zeros(DIM, dtype=dtype).at[-1].set(1)
    return make_chart(
        h,
        jnp.asarray(-ALPHA, dtype) * h,
        h,
        jnp.asarray(ALPHA, dtype),
        jnp.zeros(DIM, dtype=dtype),
        jnp.ones(DIM, dtype=dtype),
    )


def _ladder(clocks, seed=2):
    """Relative score discrepancy (float32 vs float64) at each clock value."""
    rng = np.random.default_rng(seed)
    section = rng.normal(size=DIM - 1)
    grad_native = jax.grad(_funnel_logdensity)
    c64, c32 = _chart(jnp.float64), _chart(jnp.float32)
    out = []
    for t in clocks:
        y64 = jnp.asarray(np.concatenate([section, [t]]), dtype=jnp.float64)
        y32 = y64.astype(jnp.float32)
        s64 = c64.pullback_score(y64, grad_native(c64.forward(y64)))
        s32 = c32.pullback_score(y32, grad_native(c32.forward(y32)).astype(jnp.float32))
        finite = bool(jnp.all(jnp.isfinite(s32)))
        scale = float(jnp.linalg.norm(s64)) or 1.0
        rel = float(jnp.max(jnp.abs(s32.astype(jnp.float64) - s64))) / scale
        out.append((float(t), rel, finite))
    return out


def test_accuracy_is_at_rounding_well_inside_the_indicator():
    """Where ``exp(alpha*clock)`` is modest, float32 and float64 agree to float32 eps."""
    for clock, rel, finite in _ladder([0.0, 5.0, 10.0, 20.0, 30.0]):
        assert finite
        assert rel < 1e-5, f"clock={clock}: rel={rel:.2e}"


def test_the_failure_is_silent_long_before_anything_overflows():
    """The load-bearing safety property: finite, plausible, and badly wrong.

    A reject-on-NaN guard is structurally incapable of catching this, because
    the corrupted values are finite over a wide band of clock values.
    """
    ladder = _ladder([30.0, 40.0, 50.0, 60.0, 80.0])
    by_clock = {c: (rel, fin) for c, rel, fin in ladder}

    # inside the silent band: badly wrong, yet every component is finite
    for clock in (40.0, 50.0, 60.0):
        rel, finite = by_clock[clock]
        assert finite, f"clock={clock} expected finite"
        assert rel > 1e-3, f"clock={clock}: expected gross error, got {rel:.2e}"

    # accuracy was still fine an octave earlier, so the band really is a band
    assert by_clock[30.0][0] < 1e-5 and by_clock[30.0][1]

    # a NaN guard only fires much later, after the silent band has been crossed
    assert not by_clock[80.0][1]


def test_indicator_orders_the_bands_but_not_individual_rounding_level_points():
    """``exp(alpha*clock)`` separates the accurate band from the corrupted one.

    Deliberately NOT asserted: monotonicity point-by-point.  Inside the accurate
    band the discrepancy is rounding noise and is genuinely non-monotone (here
    clock=20 lands below clock=10), so an assertion of pointwise monotonicity
    would be asserting noise.  The indicator's real content is the ordering of
    the *bands*, which is what a diagnostic would act on.
    """
    clocks = [10.0, 20.0, 30.0, 40.0, 50.0]
    ladder = _ladder(clocks)
    indicator = [float(np.exp(ALPHA * c)) for c in clocks]
    assert indicator == sorted(indicator)

    errors = [rel for _, rel, _ in ladder]
    accurate_band, corrupted_band = errors[:3], errors[3:]
    assert max(accurate_band) < 1e-5
    assert min(corrupted_band) > 1e-3
    # the largest indicator carries the largest error
    assert errors[-1] == max(errors)
    # and the separation is many orders of magnitude, not a marginal shift
    assert min(corrupted_band) / max(accurate_band) > 1e3


def test_representable_clock_is_not_the_usable_clock():
    """The usable range is far smaller than the representable one.

    Recorded as a property of *these* parameters and *this* dtype pair, not as a
    universal domain bound.
    """
    ladder = _ladder([30.0, 40.0, 80.0])
    usable = [c for c, rel, fin in ladder if fin and rel < 1e-5]
    representable = [c for c, _, fin in ladder if fin]
    assert max(usable) < max(representable), (
        "the chart must stay representable well past the point it stops being "
        "accurate — that gap is the silent band"
    )
