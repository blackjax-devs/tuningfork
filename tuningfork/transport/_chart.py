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
import numpy as np
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
        """``log|det d forward / dy|``.  Exact, not a fitted surrogate.

        This is the log-**absolute** determinant, so a negative ``scale`` entry is
        supported: it flips the map's orientation without changing the volume
        element.  ``jnp.log(jnp.abs(...))`` rather than ``jnp.log(...)`` is what
        makes that consistent — the earlier form returned NaN for a chart whose
        forward, inverse and score were all exact.
        """
        d = self.center.size
        logdet_l = jnp.sum(jnp.log(jnp.abs(self.scale)))
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
    """Build a :class:`Chart`, enforcing the supported input contract.

    Two different things happen here and it is worth not conflating them, because
    an earlier docstring claimed "correct by construction" for both.

    **Projected** (any input is accepted and made to satisfy the constraint):
    ``h`` is normalised, ``c`` becomes ``P c + h`` so ``h.c = 1``, and ``a``
    becomes ``P a - alpha h`` so ``h.a = -alpha``, with ``P = I - h h^T``.

    **Required** (invalid input is refused, not repaired): ``h`` must be
    non-zero; ``scale`` must be non-zero and the same size as ``center``;
    ``lr_basis`` and ``lr_eigenvalues`` must be supplied together with matching
    rank; ``lr_eigenvalues`` must be positive; and the **spectrally active**
    columns of ``lr_basis`` — those with ``lam != 1`` — must be orthonormal.

    The orthonormality requirement is scoped to active columns on purpose.
    Columns with ``lam == 1`` contribute exactly zero to ``_lowrank`` at every
    power and zero to the log-determinant, so they are unconstrained; requiring
    them to be orthonormal would reject legitimate inputs.  Active columns need
    orthonormality and not merely orthogonality: ``M_p M_{-p} = I`` reduces to
    ``U^T U = I`` on the active index set, which unit norm is part of.

    This function validates with Python control flow and is therefore **host-side
    only**: it is not ``jit``- or ``vmap``-traceable.  The resulting
    :class:`Chart` is a pytree and crosses those boundaries normally; only its
    construction does not.

    Raises
    ------
    ValueError
        If any required condition above is violated.
    """
    h = jnp.asarray(h)
    if h.ndim != 1:
        raise ValueError(f"h must be one-dimensional, got shape {h.shape}")
    h_norm = jnp.linalg.norm(h)
    if not bool(jnp.isfinite(h_norm)) or float(h_norm) == 0.0:
        raise ValueError("h must be finite and non-zero; it defines the clock axis")
    h = h / h_norm

    center = jnp.asarray(center)
    scale = jnp.asarray(scale)
    if center.shape != h.shape:
        raise ValueError(f"center shape {center.shape} != h shape {h.shape}")
    if scale.shape != center.shape:
        raise ValueError(f"scale shape {scale.shape} != center shape {center.shape}")
    if not bool(jnp.all(jnp.isfinite(scale))) or bool(jnp.any(scale == 0)):
        raise ValueError(
            "scale must be finite and non-zero. Negative entries ARE supported: "
            "log_det is a log-absolute determinant."
        )

    if (lr_basis is None) != (lr_eigenvalues is None):
        raise ValueError(
            "lr_basis and lr_eigenvalues must be supplied together or both omitted"
        )

    alpha = jnp.asarray(alpha, dtype=h.dtype)
    a, c = jnp.asarray(a), jnp.asarray(c)
    # Shapes are checked rather than left to broadcasting: a scalar `a`, or an
    # `alpha` with a trailing axis, would broadcast into a well-formed array
    # that is not a member of the declared family.
    if a.shape != h.shape:
        raise ValueError(f"a shape {a.shape} != h shape {h.shape}")
    if c.shape != h.shape:
        raise ValueError(f"c shape {c.shape} != h shape {h.shape}")
    if alpha.ndim != 0:
        raise ValueError(f"alpha must be a scalar, got shape {alpha.shape}")
    for name, value in (("a", a), ("c", c), ("alpha", alpha), ("center", center)):
        if not bool(jnp.all(jnp.isfinite(value))):
            raise ValueError(f"{name} must be finite; a NaN here builds a NaN chart")
    project = lambda v: v - h * jnp.dot(h, v)  # noqa: E731
    c = project(c) + h
    a = project(a) - alpha * h

    if lr_basis is None:
        lr_basis = jnp.zeros((h.size, 0), dtype=h.dtype)
        lr_eigenvalues = jnp.zeros((0,), dtype=h.dtype)
    else:
        lr_basis = jnp.asarray(lr_basis)
        lr_eigenvalues = jnp.asarray(lr_eigenvalues)
        if lr_basis.ndim != 2 or lr_basis.shape[0] != h.size:
            raise ValueError(
                f"lr_basis must have shape ({h.size}, rank), got {lr_basis.shape}"
            )
        if lr_eigenvalues.shape != (lr_basis.shape[1],):
            raise ValueError(
                f"lr_eigenvalues shape {lr_eigenvalues.shape} does not match "
                f"lr_basis rank {lr_basis.shape[1]}"
            )
        # Finiteness FIRST, and for every column including neutral ones.
        # Deferring this leaves a NaN-blind path: a NaN in an active column makes
        # the Gram residual NaN, and `NaN > tol` is False, so the orthonormality
        # gate would pass it -- the same comparison defect this suite fixed in
        # its own gates. `0 * NaN` is NaN, so neutral columns are not inert
        # either. `+inf` likewise satisfies a bare `> 0` test.
        if not bool(jnp.all(jnp.isfinite(lr_basis))):
            raise ValueError("lr_basis must be finite in every column")
        if not bool(jnp.all(jnp.isfinite(lr_eigenvalues))):
            raise ValueError("lr_eigenvalues must be finite")
        if lr_basis.shape[1] and not bool(jnp.all(lr_eigenvalues > 0)):
            raise ValueError("lr_eigenvalues must be strictly positive")
        active = np.asarray(lr_eigenvalues != 1.0)
        if active.any():
            u_active = lr_basis[:, jnp.asarray(active)]
            gram = u_active.T @ u_active
            off = float(
                jnp.max(jnp.abs(gram - jnp.eye(gram.shape[0], dtype=gram.dtype)))
            )
            # Scale with dtype and rank. A genuinely orthonormal float32 basis
            # carries Gram error ~sqrt(d)*eps32 (measured 1.19e-07 for d=6,
            # rank 3), which a fixed absolute bound rejects as invalid. This is
            # a rounding allowance, not a separation guarantee: a violation
            # smaller than the allowance is not detected, and violations can be
            # arbitrarily small.
            eps = float(jnp.finfo(lr_basis.dtype).eps)
            tol = 64.0 * eps * max(gram.shape[0], 1)
            # `not (off <= tol)` rather than `off > tol`, and the polarity is
            # load-bearing. Every INPUT here is checked finite before use, but
            # `off` is DERIVED: an off-diagonal Gram entry is a signed sum, so
            # elementwise-finite entries near sqrt(dtype max) can produce
            # (+inf) + (-inf) = NaN, which jnp.max propagates. `NaN > tol` is
            # False and would accept a basis that could not be evaluated.
            # Finiteness-before-comparison cannot protect a quantity computed
            # after the inputs are cleared; only polarity can.
            if not (off <= tol):
                raise ValueError(
                    "spectrally active lr_basis columns (lam != 1) must be "
                    f"orthonormal; max |U^T U - I| = {off:.3e} on the active set. "
                    "Neutral columns (lam == 1) are unconstrained."
                )

    # Householder taking h to +-e_{d-1}; sign keyed on h[-1] for stability.
    e = jnp.zeros_like(h).at[-1].set(jnp.where(h[-1] >= 0, 1.0, -1.0))
    u = h + e
    u = u / jnp.linalg.norm(u)
    return Chart(h, a, c, alpha, center, scale, u, lr_basis, lr_eigenvalues)
