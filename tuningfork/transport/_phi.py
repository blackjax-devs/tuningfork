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
``x = 0``: :math:`\\varphi_2` subtracts ``x`` from ``expm1(x) ~ x + x^2/2``, so
its relative error grows like ``eps / |x|``.  Evaluated as a truncated Taylor
series they lose accuracy for large ``|x|``.  A threshold picks between them.

Threshold and order are **selected by measurement**, not by convention.
``tests/transport/test_phi_precision.py`` re-derives the table below against a
``float128`` oracle and asserts the constants here remain the better choice.

Worst relative error of :math:`\\varphi_2` (the binding one) over
``1e-12 <= |x| <= 40``:

Worst relative error over ``1e-11 <= |x| <= 30``, measured for the shipped
implementation (10-term series) against a hybrid extended-precision oracle:

=========  ===========  ===========
threshold  float64      float32
=========  ===========  ===========
1e-4          2.54e-12      7.44e-04
1e-3          2.10e-13      1.08e-04
**0.1**   **3.77e-15**      4.16e-06
**0.3**       7.84e-14  **8.77e-07**
1.0           1.19e-08      8.77e-07
1.5           1.13e-06      1.09e-06
=========  ===========  ===========

No single threshold is right for both.  float64 wants ``0.1``; at ``0.3`` it is
already 20x worse, and at ``1.0`` seven orders worse.  float32 wants ``0.3``,
where it reaches its intrinsic floor; at ``0.1`` it is 5x worse because the
direct branch's ``eps/|x|`` cancellation dominates there.  The threshold is
therefore selected per dtype.  The first two rows are the crossovers used by the
earlier exploratory implementations (shown here at 10 terms so only the
threshold varies);
they are recorded for provenance only and no bitwise equivalence is claimed.

Both the value and the derivative are measured, because the crossover is where
the direct branch's cancellation is worst and AD inherits it:
``tests/transport/test_phi_precision.py`` pins both.
"""

import jax.numpy as jnp
from jax import Array

__all__ = ["SERIES_THRESHOLD", "SERIES_TERMS", "phi"]

SERIES_THRESHOLD = {"float32": 0.3, "float64": 0.1}
"""Per-dtype ``|x|`` below which the Taylor series replaces the direct form.

The optimum is dtype-dependent and the two dtypes disagree by an order of
magnitude, so a single constant is measurably wrong for one of them.  The
direct branch's error falls like ``eps / |x|`` while the series' truncation
error grows like ``x**SERIES_TERMS``; the crossover therefore sits where the
machine epsilon puts it.  See the table in the module docstring.
"""

SERIES_TERMS = 10
"""Number of Taylor terms retained (powers ``x^0`` through ``x^9``)."""


def phi(x: Array) -> tuple[Array, Array]:
    """Return ``(phi1(x), phi2(x))`` accurately for all finite ``x``.

    Parameters
    ----------
    x
        Argument, any shape.  The same rule is applied elementwise.

    Returns
    -------
    A pair ``(phi1, phi2)`` with the shape and dtype of ``x``.

    Notes
    -----
    Both branches are evaluated under ``jnp.where``, so the direct branch is
    computed even where the series is selected.  The denominator is therefore
    clamped away from zero: without that guard the unused branch produces
    ``inf``/``nan`` whose *tangent* contaminates the selected branch under
    ``jax.grad`` (a ``nan`` tangent multiplied by a zero cotangent is still
    ``nan``).  The clamp is why ``phi`` is differentiable at ``x = 0``.
    """
    threshold = SERIES_THRESHOLD.get(jnp.result_type(x).name, 0.1)
    small = jnp.abs(x) < threshold
    safe = jnp.where(small, jnp.ones_like(x), x)

    em1 = jnp.expm1(x)
    direct1 = em1 / safe
    direct2 = (em1 - x) / (safe * safe)

    # Horner on  phi1 = sum_k x^k / (k+1)!  and  phi2 = sum_k x^k / (k+2)!
    series1 = jnp.zeros_like(x)
    series2 = jnp.zeros_like(x)
    for k in range(SERIES_TERMS - 1, -1, -1):
        series1 = series1 * x + 1.0 / _factorial(k + 1)
        series2 = series2 * x + 1.0 / _factorial(k + 2)

    return jnp.where(small, series1, direct1), jnp.where(small, series2, direct2)


def _factorial(n: int) -> float:
    out = 1.0
    for i in range(2, n + 1):
        out *= i
    return out
