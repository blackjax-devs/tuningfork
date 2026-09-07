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
"""``phi`` against an independent oracle, at named points, values and derivatives.

An earlier version of this module asserted per-dtype "measured floors" and ran a
competition between the shipped crossover and historical ones.  Both are gone.
The floors were maxima over a sampled grid and independent review found points
exceeding them; a grid maximum is a lower bound on the worst case, so promoting
one to a "floor" was wrong in kind, not by a factor.  And a regression suite is
the wrong home for a threshold competition: it re-litigates a past decision on
every run without protecting anything.

What remains is a check that the shipped function agrees with an independent
extended-precision oracle at points chosen because they fail *differently*:
zero, small arguments in the series branch, **both sides of each supported
crossover**, and a large-argument case where the direct branch cancels.  The
tolerances below are test thresholds.  They are not claims about ``phi``.

The oracle is built from ``decimal`` at 60 digits and shares no code path with
the module: notably it does not use ``numpy.longdouble``, whose closed form
cancels worse than float64 near the origin — which is why a longdouble oracle
must be hybrid and why this one sidesteps the issue by carrying more digits.
"""

from decimal import Decimal, getcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.transport import x64_scope
from tuningfork.transport._phi import SERIES_THRESHOLD, SUPPORTED_DTYPES, phi

pytestmark = pytest.mark.slow

use_x64 = pytest.fixture(autouse=True, scope="module")(x64_scope)

getcontext().prec = 60


def _oracle(x):
    """(phi1, phi2, phi1', phi2') in 60-digit decimal, by series everywhere.

    Powers are accumulated iteratively rather than via ``**``.  ``Decimal(0) ** 0``
    raises ``InvalidOperation``, so an exponent-based series errors at exactly the
    point most worth testing.
    """
    xd = Decimal(repr(float(x)))
    p1 = p2 = d1 = d2 = Decimal(0)
    x_pow = Decimal(1)  # x**k
    x_prev = Decimal(0)  # x**(k-1), unused at k = 0
    fact = Decimal(1)  # k!
    for k in range(80):
        if k:
            fact *= k
        f1 = fact * (k + 1)  # (k+1)!
        f2 = f1 * (k + 2)  # (k+2)!
        p1 += x_pow / f1
        p2 += x_pow / f2
        if k:
            d1 += k * x_prev / f1
            d2 += k * x_prev / f2
        x_prev = x_pow
        x_pow = x_pow * xd
    return p1, p2, d1, d2


def _named_points(dtype):
    """Points chosen to fail differently, not a mesh."""
    t = SERIES_THRESHOLD[np.dtype(dtype).name]
    return [
        (0.0, "zero"),
        (1e-8, "tiny: direct form would cancel"),
        (-1e-8, "tiny negative"),
        (t * 0.5, "series side of the crossover"),
        (t * 0.99, "series side, at the crossover"),
        (t * 1.01, "direct side, at the crossover"),
        (-t * 1.01, "direct side, negative"),
        (t * 3.0, "direct side, comfortably outside"),
        (12.0, "large positive: direct form, no cancellation"),
        (-12.0, "large negative: direct form cancels against x"),
    ]


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=SUPPORTED_DTYPES)
def test_values_and_derivatives_agree_with_an_independent_oracle(dtype):
    """Values AND derivatives, both dtypes, at each named point."""
    tol = {np.float32: 2e-5, np.float64: 1e-11}[dtype]
    d1 = jax.grad(lambda z: phi(z)[0])
    d2 = jax.grad(lambda z: phi(z)[1])
    for x, why in _named_points(dtype):
        xa = jnp.asarray(x, dtype=dtype)
        got = (phi(xa)[0], phi(xa)[1], d1(xa), d2(xa))
        want = _oracle(x)
        for value, expected, label in zip(
            got, want, ("phi1", "phi2", "phi1'", "phi2'")
        ):
            assert jnp.isfinite(value), f"{label} not finite at {x} ({why})"
            scale = max(abs(float(expected)), 1e-3)
            err = abs(float(value) - float(expected)) / scale
            assert err < tol, f"{label} rel err {err:.2e} at x={x} ({why})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=SUPPORTED_DTYPES)
def test_gradients_stay_finite_where_the_value_is_finite(dtype):
    """A finite value does not imply a finite gradient.

    The series branch's Horner recurrence overflows for large ``|x|``, and under
    ``jnp.where`` an ``inf - inf`` tangent in the *unselected* branch poisons the
    selected one.  Clamping the series argument is what prevents it; without the
    clamp these points return NaN gradients on an exactly correct value.
    """
    d1 = jax.grad(lambda z: phi(z)[0])
    d2 = jax.grad(lambda z: phi(z)[1])
    big = {np.float32: [1e3, -1e3, 1e6, -1e6], np.float64: [1e6, -1e6, 1e30, -1e30]}[
        dtype
    ]
    for x in big:
        xa = jnp.asarray(x, dtype=dtype)
        assert jnp.all(jnp.isfinite(jnp.asarray(phi(xa)))), f"value not finite at {x}"
        assert jnp.isfinite(d1(xa)), f"phi1' not finite at {x}"
        assert jnp.isfinite(d2(xa)), f"phi2' not finite at {x}"


def test_unsupported_dtypes_are_refused():
    """Half precisions are rejected, not served with a float64 threshold.

    Measurement showed the float64 crossover is the *worst* available choice for
    bfloat16, so a silent fallback would be actively harmful.
    """
    for dtype in (jnp.bfloat16, jnp.float16):
        with pytest.raises(TypeError, match="supports"):
            phi(jnp.asarray(0.5, dtype=dtype))


def test_origin_is_exact_in_value_and_derivative():
    """The clamped direct branch must not poison the series branch at x = 0."""
    p1, p2 = phi(jnp.asarray(0.0))
    assert float(p1) == 1.0 and float(p2) == 0.5
    assert abs(float(jax.grad(lambda z: phi(z)[0])(0.0)) - 0.5) < 1e-15
    assert abs(float(jax.grad(lambda z: phi(z)[1])(0.0)) - 1.0 / 6.0) < 1e-15
