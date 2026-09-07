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
"""A supplied (frozen) affine-flow coordinate chart, and its exact algebra.

Nothing in this module fits, selects or adapts anything: a :class:`Chart` is
handed complete parameters and only evaluates.

The chart
---------
Work in preconditioned coordinates ``z = L^{-1}(q - center)``, where ``L`` is a
diagonal ``scale`` optionally composed with a symmetric low-rank factor.  Fix a
constant affine vector field

.. math::  V(z) = \\alpha z + a (h \\cdot z) + c

subject to the three **structural constraints**

.. math::  \\lVert h \\rVert = 1, \\qquad h \\cdot c = 1, \\qquad h \\cdot a = -\\alpha .

Under those constraints ``d/ds (h . z) = alpha (h.z) + (h.a)(h.z) + (h.c) = 1``
*identically*, independent of ``z``.  Three exact consequences follow, and they
are the properties this module exists to expose and test:

1. the last chart coordinate (the **clock**) advances at unit rate along the
   flow, so it can be read off as ``t = h . z``;
2. :meth:`Chart.inverse` is exact — flowing back by ``t`` lands on the section
   ``h . z = 0`` with no iteration and no root find;
3. ``log|det J| = alpha (d-1) t + log|det L|`` is the **exact** log-Jacobian,
   because the section columns of the Jacobian are ``exp(alpha t)`` times an
   orthonormal basis of ``h``-perp and the clock column has unit ``h``-component.

Conventions (``CONVENTION_VERSION = "frozen-transport-chart/v1"``)
-----------------------------------------------------------------
* **clock-last**: chart coordinates are ``y = [section (d-1), clock]``;
* the Householder reflector maps ``h`` onto ``±e_{d-1}``, its sign keyed on
  ``h[-1]``, so the section coordinates span ``h``-perp;
* the ``phi`` series threshold and order come from :mod:`._phi`.

Earlier exploratory implementations used a clock-first order and a different
reflector orientation.  **No bitwise equivalence with them is claimed.**

What this module does not do
----------------------------
It supplies no fitter and no selection rule, makes no claim that any particular
chart improves mixing, and carries no adaptation or telemetry hooks.  A frozen
chart has no fitting charge in a run that uses it, but it is **not free**:
``forward``, ``pullback_score`` and the log-Jacobian are all evaluated per step.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from tuningfork.transport._phi import phi

__all__ = ["Chart", "make_chart"]


class Chart(NamedTuple):
    """A frozen affine-flow chart.

    Attributes
    ----------
    h, a, c, alpha
        The affine field ``V(z) = alpha z + a (h.z) + c`` in preconditioned
        coordinates.  Must satisfy the three structural constraints; use
        :func:`make_chart`, which enforces them by construction.
    center, scale
        Affine preconditioner ``q = center + scale * lowrank(z)``.
    u
        Unit Householder vector mapping ``h`` onto ``±e_{d-1}``.
    lr_basis, lr_eigenvalues
        Optional symmetric low-rank factor: ``lowrank(x, p) = x + U ((lam^p - 1)
        (U^T x))`` with orthonormal columns ``U``.  Rank 0 (``lam`` all ones) is
        the plain diagonal case and is bit-equivalent to omitting the factor.
    """

    h: Array
    a: Array
    c: Array
    alpha: Array
    center: Array
    scale: Array
    u: Array
    lr_basis: Array
    lr_eigenvalues: Array

    # -- preconditioner -----------------------------------------------------
    def _lowrank(self, x: Array, power: float) -> Array:
        if self.lr_basis.size == 0:
            return x
        w = self.lr_eigenvalues**power - 1.0
        return x + self.lr_basis @ (w * (self.lr_basis.T @ x))

    def _to_native(self, z: Array) -> Array:
        return self.center + self.scale * self._lowrank(z, 0.5)

    def _to_preconditioned(self, q: Array) -> Array:
        return self._lowrank((q - self.center) / self.scale, -0.5)

    def _cotangent(self, g_native: Array) -> Array:
        """Pull a native gradient back through the preconditioner: ``L^T g``."""
        return self._lowrank(self.scale * g_native, 0.5)

    # -- reflector ----------------------------------------------------------
    def _reflect(self, v: Array) -> Array:
        return v - 2.0 * self.u * jnp.dot(self.u, v)

    # -- the flow -----------------------------------------------------------
    def _flow(self, z: Array, delta: Array) -> Array:
        """Exact time-``delta`` flow of ``V`` started at ``z``."""
        x = self.alpha * delta
        p1, p2 = phi(x)
        return (
            jnp.exp(x) * z
            + delta * p1 * (self.c + self.a * jnp.dot(self.h, z))
            + delta * delta * p2 * self.a
        )

    def field(self, z: Array) -> Array:
        """The affine field ``V(z)`` in preconditioned coordinates."""
        return self.alpha * z + self.a * jnp.dot(self.h, z) + self.c

    # -- the four public maps ----------------------------------------------
    def forward(self, y: Array) -> Array:
        """Chart coordinates ``y = [section, clock]`` to a native position."""
        t = y[-1]
        section = self._reflect(jnp.concatenate([y[:-1], jnp.zeros((1,), y.dtype)]))
        return self._to_native(self._flow(section, t))

    def inverse(self, q: Array) -> Array:
        """Native position to chart coordinates.  Exact, by the unit clock rate."""
        z = self._to_preconditioned(q)
        t = jnp.dot(self.h, z)
        section = self._reflect(self._flow(z, -t))
        return jnp.concatenate([section[:-1], t[None]])

    def log_det(self, y: Array) -> Array:
        """``log|det d forward / dy|``.  Exact, not a fitted surrogate."""
        d = self.center.size
        logdet_l = jnp.sum(jnp.log(self.scale))
        if self.lr_basis.size:
            logdet_l = logdet_l + 0.5 * jnp.sum(jnp.log(self.lr_eigenvalues))
        return self.alpha * (d - 1) * y[-1] + logdet_l

    def pullback_score(self, y: Array, g_native: Array) -> Array:
        """Chart-coordinate score of ``logdensity(forward(y)) + log_det(y)``.

        ``g_native`` is the native score evaluated at ``forward(y)``.  The
        ``alpha (d-1)`` term is the derivative of the log-Jacobian and is a
        distinct contribution from the coordinate pullback.
        """
        d = self.center.size
        gz = self._cotangent(g_native)
        z = self._flow(
            self._reflect(jnp.concatenate([y[:-1], jnp.zeros((1,), y.dtype)])), y[-1]
        )
        clock = jnp.dot(gz, self.field(z)) + self.alpha * (d - 1)
        section = jnp.exp(self.alpha * y[-1]) * self._reflect(gz)[:-1]
        return jnp.concatenate([section, clock[None]])

    def push_score(self, y: Array, g_chart: Array) -> Array:
        """Inverse of :meth:`pullback_score`: chart score back to a native score."""
        d = self.center.size
        z = self._flow(
            self._reflect(jnp.concatenate([y[:-1], jnp.zeros((1,), y.dtype)])), y[-1]
        )
        transverse = self._reflect(
            jnp.concatenate(
                [jnp.exp(-self.alpha * y[-1]) * g_chart[:-1], jnp.zeros((1,), y.dtype)]
            )
        )
        longitudinal = (
            g_chart[-1] - self.alpha * (d - 1) - jnp.dot(transverse, self.field(z))
        )
        gz = transverse + longitudinal * self.h
        # Inverse of _cotangent: L^T = M D_scale with M = lowrank(., 0.5)
        # symmetric, so (L^T)^-1 g = lowrank(g, -0.5) / scale.
        return self._lowrank(gz, -0.5) / self.scale


def make_chart(
    h: Array,
    a: Array,
    c: Array,
    alpha: Array | float,
    center: Array,
    scale: Array,
    lr_basis: Array | None = None,
    lr_eigenvalues: Array | None = None,
) -> Chart:
    """Build a :class:`Chart`, projecting ``h``, ``c`` and ``a`` onto the constraints.

    The three structural constraints are imposed here rather than checked, so a
    :class:`Chart` is correct by construction:

    * ``h`` is normalised;
    * ``c`` is replaced by ``P c + h`` where ``P = I - h h^T``, giving ``h.c = 1``;
    * ``a`` is replaced by ``P a - alpha h``, giving ``h.a = -alpha``.

    Supplying parameters that already satisfy the constraints leaves them
    unchanged up to rounding.  Violating them is *not* an error the caller can
    make: to test a constraint violation, mutate the returned :class:`Chart`.
    """
    h = h / jnp.linalg.norm(h)
    alpha = jnp.asarray(alpha, dtype=h.dtype)
    project = lambda v: v - h * jnp.dot(h, v)  # noqa: E731
    c = project(c) + h
    a = project(a) - alpha * h

    # Householder taking h to +-e_{d-1}; sign keyed on h[-1] for stability.
    e = jnp.zeros_like(h).at[-1].set(jnp.where(h[-1] >= 0, 1.0, -1.0))
    u = h + e
    u = u / jnp.linalg.norm(u)

    if lr_basis is None:
        lr_basis = jnp.zeros((h.size, 0), dtype=h.dtype)
        lr_eigenvalues = jnp.zeros((0,), dtype=h.dtype)
    return Chart(h, a, c, alpha, center, scale, u, lr_basis, lr_eigenvalues)
