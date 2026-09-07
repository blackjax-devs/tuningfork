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
"""Entire functions ``phi1``/``phi2`` used by the affine-flow chart.

The chart's flow needs the two entire functions

.. math::
    \\varphi_1(x) = \\frac{e^x - 1}{x}, \\qquad
    \\varphi_2(x) = \\frac{e^x - 1 - x}{x^2},

both analytic at the origin.  Evaluated directly they lose accuracy near
``x = 0``: :math:`\\varphi_2` subtracts ``x`` from ``expm1(x) ~ x + x^2/2``, so its
relative error grows like ``eps / |x|``.  Evaluated as a truncated Taylor series
they lose accuracy for large ``|x|``.  A per-dtype threshold picks between them.

Finite-precision domain and limitations
---------------------------------------
The threshold constants below were chosen by measurement, but the accuracy they
achieve is **not** characterised here as a bound.  An earlier version of this
module quoted per-dtype "measured floors"; those numbers were maxima over a
sampled grid, and independent review found points exceeding them.  A maximum
over any finite sample is a lower bound on the true worst case, and two coarse
grids agreeing does not make either adequate — the honest diagnostic is that the
worst case stops moving as the sampling is refined, which has not been
established.  So:

* no floor, bound or guaranteed accuracy is claimed for ``phi`` at any dtype;
* the tests assert only that the shipped implementation agrees with an
  independent extended-precision oracle at named points, to tolerances that are
  documented as test thresholds and not as properties of the function.

Supported dtypes are **float32 and float64 only**.  Half precisions are refused
rather than silently served: measurement showed the float64 threshold is the
*worst* available choice for ``bfloat16`` (worst ``phi2`` relative error 6.5e-02
at threshold 0.1 against 8.8e-03 at 1.0), so a silent fallback would be actively
harmful rather than merely unsupported.

Both branches of the selection are evaluated under ``jnp.where``, so each is
guarded against the other's regime: the direct branch's denominator is clamped
away from zero, and the series branch's argument is clamped to zero outside its
own range.  Without the second clamp the series' Horner recurrence overflows to ``inf`` for
large ``|x|``; in the VJP the unselected branch then receives a zero cotangent,
and ``0 * inf`` is NaN, which contaminates the selected branch.  ``jax.grad``
returns NaN at points where the value is exactly right.
"""

import jax.numpy as jnp
from jax import Array

__all__ = ["SERIES_THRESHOLD", "SERIES_TERMS", "phi"]

SUPPORTED_DTYPES = ("float32", "float64")
"""Dtypes this module serves.  Anything else is refused, not approximated."""

SERIES_THRESHOLD = {"float32": 0.3, "float64": 0.1}
"""Per-dtype ``|x|`` below which the Taylor series replaces the direct form.

    No accuracy is claimed for either branch; see the module docstring.

The optimum is dtype-dependent and the two dtypes disagree by an order of
magnitude, so a single constant is measurably wrong for one of them.  The
direct branch's error falls like ``eps / |x|`` while the series' truncation
error grows like ``x**SERIES_TERMS``; the crossover therefore sits where the
machine epsilon puts it.
"""

SERIES_TERMS = 10
"""Number of Taylor terms retained (powers ``x^0`` through ``x^9``)."""


def phi(x: Array) -> tuple[Array, Array]:
    """Return ``(phi1(x), phi2(x))`` for float32 or float64 ``x``.

    Raises
    ------
    TypeError
        If ``x`` is not float32 or float64.  Half precisions are refused because
        the shipped threshold is measurably the wrong choice for them.
    """
    name = jnp.result_type(x).name
    if name not in SUPPORTED_DTYPES:
        raise TypeError(
            f"phi supports {SUPPORTED_DTYPES}, got {name!r}. Half precisions are "
            "refused rather than served with a threshold measured for float64."
        )
    threshold = SERIES_THRESHOLD[name]
    small = jnp.abs(x) < threshold

    # Each branch is clamped out of the other's regime, because jnp.where
    # evaluates both and a NaN/inf tangent in the unselected one contaminates
    # the selected one's gradient.
    safe_direct = jnp.where(small, jnp.ones_like(x), x)
    safe_series = jnp.where(small, x, jnp.zeros_like(x))

    em1 = jnp.expm1(x)
    direct1 = em1 / safe_direct
    direct2 = (em1 - x) / (safe_direct * safe_direct)

    series1 = jnp.zeros_like(x)
    series2 = jnp.zeros_like(x)
    for k in range(SERIES_TERMS - 1, -1, -1):
        series1 = series1 * safe_series + 1.0 / _factorial(k + 1)
        series2 = series2 * safe_series + 1.0 / _factorial(k + 2)

    return jnp.where(small, series1, direct1), jnp.where(small, series2, direct2)


def _factorial(n: int) -> float:
    out = 1.0
    for i in range(2, n + 1):
        out *= i
    return out
