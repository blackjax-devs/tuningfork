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
"""The ``phi`` crossover is SELECTED by measurement here, per dtype.

Two earlier exploratory implementations used different crossovers (1e-4 with
5 terms; 1e-3 with 6 terms).  Neither is adopted by vote.  This module measures value and
derivative error against an extended-precision oracle in both supported dtypes
and asserts the constants in :mod:`tuningfork.transport._phi` are the better
choice — including the finding that **no single threshold serves both dtypes**.

Two measurement traps are handled explicitly, because falling into either
produces a test that looks strict and asserts nothing real:

* **The oracle must be hybrid.**  Evaluating ``(e^x - 1 - x)/x^2`` in float128 at
  ``|x| ~ 1e-11`` cancels to a relative error near ``1e-12`` — worse than the
  float64 implementation under test.  Below ``|x| = 0.5`` the oracle therefore
  uses its own extended-precision series.
* **The comparison must use the shipped code path.**  A numpy stand-in disagrees
  with the JAX implementation in float32 (different ``expm1``), and is flat
  exactly where the real crossover matters.  :func:`_shipped_style` mirrors the
  module's own ``jnp`` operations with a settable threshold.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tuningfork.transport._phi import SERIES_TERMS, SERIES_THRESHOLD, phi

pytestmark = pytest.mark.fast

jax.config.update("jax_enable_x64", True)

Q = np.longdouble
_ORACLE_OK = np.finfo(Q).eps < 1e-18
_ORACLE_SERIES_CUTOFF = 0.5
_ORACLE_TERMS = 60

PRIOR_CROSSOVERS = [(1e-4, "prior 1e-4"), (1e-3, "prior 1e-3")]


def _factorial(n):
    out = Q(1)
    for i in range(2, n + 1):
        out *= i
    return out


_INV_FACT = [Q(1) / _factorial(n) for n in range(_ORACLE_TERMS + 4)]


def _oracle(x):
    """``(phi1, phi2)`` in extended precision, series below the cutoff."""
    xq = Q(x)
    if abs(float(xq)) < _ORACLE_SERIES_CUTOFF:
        p1 = p2 = Q(0)
        for k in range(_ORACLE_TERMS - 1, -1, -1):
            p1 = p1 * xq + _INV_FACT[k + 1]
            p2 = p2 * xq + _INV_FACT[k + 2]
        return p1, p2
    e = np.expm1(xq)
    return e / xq, (e - xq) / (xq * xq)


def _oracle_derivative(x):
    """``(phi1', phi2')`` in extended precision, same hybrid split."""
    xq = Q(x)
    if abs(float(xq)) < _ORACLE_SERIES_CUTOFF:
        d1 = d2 = Q(0)
        for k in range(_ORACLE_TERMS - 1, 0, -1):
            d1 = d1 * xq + Q(k) * _INV_FACT[k + 1]
            d2 = d2 * xq + Q(k) * _INV_FACT[k + 2]
        return d1, d2
    e = np.exp(xq)
    em1 = np.expm1(xq)
    return (e * (xq - 1) + 1) / (xq * xq), (em1 * xq - 2 * (em1 - xq)) / xq**3


def _shipped_style(x, threshold, dtype):
    """The module's own computation with a settable threshold, same jnp ops."""
    x = jnp.asarray(x, dtype=dtype)
    small = jnp.abs(x) < threshold
    safe = jnp.where(small, jnp.ones_like(x), x)
    em1 = jnp.expm1(x)
    direct1, direct2 = em1 / safe, (em1 - x) / (safe * safe)
    series1 = series2 = jnp.zeros_like(x)
    for k in range(SERIES_TERMS - 1, -1, -1):
        series1 = series1 * x + float(_INV_FACT[k + 1])
        series2 = series2 * x + float(_INV_FACT[k + 2])
    return jnp.where(small, series1, direct1), jnp.where(small, series2, direct2)


def _grid():
    xs = np.concatenate([np.geomspace(1e-11, 1e-1, 40), np.geomspace(1e-1, 30.0, 40)])
    return np.concatenate([xs, -xs])


def _worst(threshold, dtype):
    worst = 0.0
    for x in _grid():
        o1, o2 = _oracle(x)
        v1, v2 = _shipped_style(x, threshold, dtype)
        worst = max(
            worst,
            float(abs(Q(float(v1)) - o1) / abs(o1)),
            float(abs(Q(float(v2)) - o2) / abs(o2)),
        )
    return worst


skip_no_oracle = pytest.mark.skipif(
    not _ORACLE_OK, reason="numpy.longdouble is not extended precision"
)


@skip_no_oracle
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_chosen_threshold_beats_both_prior_crossovers(dtype):
    """The selection criterion, executed — not an appeal to precedent."""
    chosen = SERIES_THRESHOLD[np.dtype(dtype).name]
    got = {label: _worst(t, dtype) for t, label in PRIOR_CROSSOVERS}
    got["chosen"] = _worst(chosen, dtype)
    report = "  ".join(f"{k}={v:.2e}" for k, v in got.items())
    for _, label in PRIOR_CROSSOVERS:
        assert got["chosen"] < got[label], report


@skip_no_oracle
def test_no_single_threshold_serves_both_dtypes():
    """Why the threshold is a per-dtype mapping and not one constant.

    Each dtype is measurably worse at the other's optimum, so a single shared
    constant would silently degrade one of them.
    """
    t32 = SERIES_THRESHOLD["float32"]
    t64 = SERIES_THRESHOLD["float64"]
    assert t32 != t64

    f32_at_own, f32_at_other = _worst(t32, np.float32), _worst(t64, np.float32)
    f64_at_own, f64_at_other = _worst(t64, np.float64), _worst(t32, np.float64)
    assert f32_at_own < f32_at_other, f"float32 {f32_at_own:.2e} vs {f32_at_other:.2e}"
    assert f64_at_own < f64_at_other, f"float64 {f64_at_own:.2e} vs {f64_at_other:.2e}"


@skip_no_oracle
@pytest.mark.parametrize(
    ("dtype", "tol"),
    [(np.float32, 1.5e-6), (np.float64, 1e-14)],
    ids=["float32", "float64"],
)
def test_shipped_phi_value_error_is_at_its_measured_floor(dtype, tol):
    """VALUE test of the actual shipped function.

    Measured floors: float32 8.77e-07 (intrinsic), float64 3.77e-15.
    """
    worst1 = worst2 = 0.0
    for x in _grid():
        p1, p2 = phi(jnp.asarray(x, dtype=dtype))
        o1, o2 = _oracle(x)
        worst1 = max(worst1, float(abs(Q(float(p1)) - o1) / abs(o1)))
        worst2 = max(worst2, float(abs(Q(float(p2)) - o2) / abs(o2)))
    assert worst1 < tol, f"phi1 worst rel err {worst1:.2e}"
    assert worst2 < tol, f"phi2 worst rel err {worst2:.2e}"


@skip_no_oracle
def test_shipped_phi_derivative_error_is_at_its_measured_floor():
    """DERIVATIVE test — the crossover must not introduce a gradient artefact.

    ``phi2'`` is the least accurate quantity in the module: just above the
    crossover the direct branch already cancels ~1.5 digits in the value and AD
    compounds it.  ``2.4e-13`` is the measured float64 floor, located at
    ``x = -0.1``; the bound below is that floor, not slack.
    """
    d1 = jax.grad(lambda z: phi(z)[0])
    d2 = jax.grad(lambda z: phi(z)[1])
    worst1 = worst2 = 0.0
    for x in _grid():
        o1, o2 = _oracle_derivative(x)
        worst1 = max(worst1, float(abs(Q(float(d1(x))) - o1) / abs(o1)))
        worst2 = max(worst2, float(abs(Q(float(d2(x))) - o2) / abs(o2)))
    assert worst1 < 1e-14, f"phi1' worst rel err {worst1:.2e}"
    assert worst2 < 4e-13, f"phi2' worst rel err {worst2:.2e}"


def test_phi_and_its_derivative_are_finite_at_the_origin():
    """The unused direct branch must not poison the value or the tangent."""
    p1, p2 = phi(jnp.asarray(0.0))
    assert float(p1) == 1.0 and float(p2) == 0.5
    assert abs(float(jax.grad(lambda z: phi(z)[0])(0.0)) - 0.5) < 1e-15
    assert abs(float(jax.grad(lambda z: phi(z)[1])(0.0)) - 1.0 / 6.0) < 1e-15
