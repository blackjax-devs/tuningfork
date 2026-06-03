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
"""Adapter: route laplace_* and rmhmc base methods through appropriate warmup kernels.

Background
----------
``blackjax.window_adaptation`` (and ``window_adaptation_low_rank_imm``) require
an algorithm object with:

- ``algorithm.build_kernel(integrator)`` — returns a raw kernel
  ``(rng_key, state, logdensity_fn, step_size, imm, **extra) -> (state, info)``
- ``algorithm.init(position, logdensity_fn)`` — returns the initial kernel
  state from a position and logdensity callable

The standard HMC-family methods (``blackjax.hmc``, ``blackjax.nuts``, …) expose
exactly this interface.  The laplace_* family does NOT — their init and kernel
take a ``LaplaceMarginal`` object instead of a plain ``logdensity_fn``, and
their ``as_top_level_api`` signature is ``(log_joint_fn, theta_init, step_size,
inverse_mass_matrix, num_integration_steps)`` rather than the standard
``(logdensity_fn, step_size, inverse_mass_matrix, …)``.

Warmup strategy for laplace_*
------------------------------
During warmup, the goal is to find a good ``(step_size, inverse_mass_matrix)``
in **phi-space** (the un-marginalised subspace, e.g. ``dim(phi) = 2`` for
``eight_schools_ncp``).  The correct warmup logdensity is the Laplace marginal
``log p̂(phi | y)`` — a plain ``phi -> float`` callable that the caller builds
via ``laplace_marginal_factory`` and passes as ``logdensity_fn``.

Given a standard ``phi -> float`` marginal logdensity, standard HMC is the
right warmup kernel: it explores phi-space, collects trajectory statistics, and
drives dual-averaging step-size + mass-matrix adaptation.  The resulting
``(step_size, IMM)`` are then consumed at sample time by the laplace_* kernel
(which also needs ``log_joint_fn`` and ``theta_init``, but those come from the
recipe/model, not from the warmup).

Warmup strategy for rmhmc
--------------------------
``blackjax.rmhmc`` uses the ``implicit_midpoint`` integrator by default at
sampling time.  ``blackjax.window_adaptation`` defaults to ``velocity_verlet``,
so naively passing ``blackjax.rmhmc`` to it adapts step_size under
``velocity_verlet`` while sampling runs under ``implicit_midpoint``.  Because
``implicit_midpoint`` is more accurate per step (same 2nd-order convergence,
symmetric form), the energy error for a given step_size is lower → higher
acceptance rate → too-conservative steps → low ESS (observed: ESS~70 vs
target ~500 for eight_schools_ncp).

Fix: ``_RMHMCImplicitMidpointAlgorithm`` wraps ``blackjax.rmhmc`` and overrides
``build_kernel`` to always use ``implicit_midpoint``, so window_adaptation
calibrates step_size under the same integrator used at sampling time.

This module provides ``resolve_warmup_algorithm``, a single function that the
three window-adaptation runners call to get the right ``(algorithm, extra_kwargs)``
pair to pass to ``blackjax.window_adaptation``:

- For laplace_* base methods: returns ``(blackjax.hmc, extra_kwargs_for_hmc)``
- For rmhmc: returns ``(_RMHMCImplicitMidpointAlgorithm(), extra_kwargs)``
  so warmup step_size is calibrated under ``implicit_midpoint``.
- For all other base methods: returns ``(base_method.factory, extra_kwargs)``
  unchanged (the existing path — no regression possible).

IMM dimensionality guarantee
-----------------------------
Because warmup runs on the marginal logdensity over **phi only**, the adapted
IMM has shape ``(dim(phi),)`` (diagonal) or ``(dim(phi), dim(phi))`` (dense).
This is the correct shape for the laplace_* kernel, which operates on phi.
It is NOT ``dim(joint) = dim(phi) + dim(theta)``.

The test in ``tests/warmup/test_laplace_marginal_warmup_smoke.py`` asserts
``imm.shape == (num_chains, dim_phi)`` to enforce this guarantee.
"""

from typing import Any

import blackjax
import blackjax.mcmc.integrators as _integrators

__all__ = [
    "LAPLACE_METHOD_NAMES",
    "RMHMC_API_METHOD_NAMES",
    "WARMUP_SUBSTITUTE_METHOD_NAMES",
    "_RMHMCImplicitMidpointAlgorithm",
    "resolve_warmup_algorithm",
]


class _RMHMCImplicitMidpointAlgorithm:
    """Warmup-only wrapper around blackjax.rmhmc with implicit_midpoint integrator.

    ``blackjax.window_adaptation`` calls ``algorithm.build_kernel(integrator)``
    with its own ``integrator`` default (``velocity_verlet``).  For rmhmc we
    must override this so the warmup calibrates step_size under the same
    ``implicit_midpoint`` integrator used at sampling time.  Without this fix
    the adapted step_size is too conservative for ``implicit_midpoint`` (which
    has lower energy error per step → higher acceptance → slow exploration →
    ESS ≈ 70 observed on eight_schools_ncp vs target 500–700).

    This class exposes the ``build_kernel`` / ``init`` interface expected by
    ``window_adaptation``, delegating to ``blackjax.rmhmc`` but always building
    the kernel with ``implicit_midpoint`` regardless of the integrator argument
    passed by the caller.  ``init`` delegates to ``blackjax.rmhmc.init``
    (= ``blackjax.hmc.init``).

    The ``build_kernel`` method accepts an ``integrator`` positional argument so
    that ``window_adaptation``'s ``inspect.signature`` check finds at least one
    parameter and calls ``build_kernel(velocity_verlet)`` rather than
    ``build_kernel()`` — the passed value is intentionally ignored.
    """

    def build_kernel(
        self, integrator: Any = None
    ) -> Any:  # integrator arg intentionally ignored
        """Build HMC kernel with implicit_midpoint (ignores caller-supplied integrator)."""
        return blackjax.rmhmc.build_kernel(_integrators.implicit_midpoint)

    def init(self, position: Any, logdensity_fn: Any) -> Any:
        """Delegate to blackjax.rmhmc.init (= blackjax.hmc.init)."""
        return blackjax.rmhmc.init(position, logdensity_fn)


# The four laplace_* base method names registered in the tuningfork registry.
LAPLACE_METHOD_NAMES: frozenset[str] = frozenset(
    ("laplace_hmc", "laplace_dhmc", "laplace_mhmc", "laplace_dmhmc")
)

# Methods whose `base_method.factory` is NOT a blackjax GenerateSamplingAPI
# object but where the direct blackjax API object must be used for
# `window_adaptation`.  For rmhmc: `base_method.factory` is a wrapper function
# that performs the IMM→mass_matrix conversion.  However,
# `blackjax.window_adaptation` calls `algorithm.build_kernel(integrator)` and
# `algorithm.init(position, logdensity_fn)` — it requires a GenerateSamplingAPI,
# not a plain Python function.  Since `blackjax.rmhmc.build_kernel =
# blackjax.hmc.build_kernel`, the raw kernel takes `inverse_mass_matrix` as a
# positional arg (same as hmc), so window_adaptation passes adapted IMM
# correctly despite rmhmc's user-facing `as_top_level_api` taking `mass_matrix`.
RMHMC_API_METHOD_NAMES: frozenset[str] = frozenset(("rmhmc",))

# Methods whose `.init` / `.build_kernel` signature requires extra kwargs that
# the `blackjax.window_adaptation` driver does not supply (laplace_* need
# `log_joint_fn` + `theta_init` and operate on a `LaplaceMarginal` object;
# dynamic_hmc / dmhmc need `random_generator_arg` at warmup step).  For warmup
# purposes we substitute standard `blackjax.nuts` — it composes naturally with
# `window_adaptation` (Stan-style canonical warmup kernel), needs no extra
# kernel kwargs at warmup time (NUTS picks its own trajectory length), and
# the adapted `(step_size, IMM)` are functionally equivalent for the
# downstream sampler.  The downstream sampler then re-inits from
# `adapted_state.position` with its own state structure (see
# `_recipe_runner.py`).
WARMUP_SUBSTITUTE_METHOD_NAMES: frozenset[str] = LAPLACE_METHOD_NAMES | frozenset(
    ("dynamic_hmc", "dmhmc")
)


def resolve_warmup_algorithm(
    base_method: Any,
    extra_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Return the warmup ``(algorithm, extra_kwargs)`` pair for ``blackjax.window_adaptation``.

    For standard HMC-family methods (hmc, nuts, mhmc, …) the existing
    ``(base_method.factory, extra_kwargs)`` pair is returned unchanged.

    For laplace_* / dynamic_hmc / dmhmc methods the function substitutes
    ``blackjax.nuts`` as the warmup algorithm.  NUTS composes naturally with
    ``window_adaptation`` (Stan convention) and needs no extra kernel kwargs at
    warmup time (NUTS picks its own trajectory length).  The adapted
    ``(step_size, IMM)`` are then passed at sample time to the actual kernel,
    which handles its own state-type / random-generator requirements.

    The caller is responsible for passing the laplace marginal logdensity
    ``log p̂(phi | y)`` as ``logdensity_fn`` to ``blackjax.window_adaptation``
    for the laplace_* case.  The adapter does NOT build or validate the
    marginal logdensity — it only selects the right algorithm object.

    Parameters
    ----------
    base_method
        A ``BaseMethod`` entry from the tuningfork registry.  The ``.name``
        attribute is used to detect substitute-family methods.
    extra_kwargs
        The ``extra_kwargs`` dict already prepared by the caller (default HP
        values injected, caller overrides applied).  For the substitute path
        these kwargs are discarded — NUTS does not need them.

    Returns
    -------
    algorithm
        Either ``base_method.factory`` (standard path) or ``blackjax.nuts``
        (substitute path).
    extra_kwargs
        ``dict(extra_kwargs)`` (standard path) or ``{}`` (substitute path —
        NUTS needs no extra kernel kwargs).
    """
    if base_method.name in RMHMC_API_METHOD_NAMES:
        # rmhmc path: base_method.factory is a wrapper function (not a
        # GenerateSamplingAPI).  window_adaptation needs an API object with
        # .build_kernel(integrator) and .init(position, logdensity_fn).
        # We return _RMHMCImplicitMidpointAlgorithm (not blackjax.rmhmc directly)
        # so the warmup uses implicit_midpoint to calibrate step_size under the
        # same integrator used at sampling time.  Without this, window_adaptation
        # uses velocity_verlet for warmup → step_size too small for implicit_midpoint
        # sampling → acceptance rate 0.98 → ESS ~70 (vs target 500–700).
        # Preserve extra_kwargs (carries num_integration_steps for the warmup kernel).
        return _RMHMCImplicitMidpointAlgorithm(), dict(extra_kwargs)

    if base_method.name not in WARMUP_SUBSTITUTE_METHOD_NAMES:
        # Standard path (hmc, nuts, mhmc, barker, mala): return unchanged — no
        # regression possible.
        return base_method.factory, dict(extra_kwargs)

    # Substitute-family path (laplace_*, dynamic_hmc, dmhmc): use standard NUTS
    # as the warmup kernel.  The laplace_* case is detailed in this module's
    # docstring (phi-only marginal — caller passes the Laplace marginal
    # logdensity as ``logdensity_fn``).  The dynamic_hmc / dmhmc case is
    # mechanical: both extend HMC with a `random_generator_arg`-driven
    # trajectory-length sampler, which `blackjax.window_adaptation` does not
    # feed at warmup time.  NUTS adapts (step_size, IMM) using its own
    # trajectory-termination criterion; the dynamic_hmc-family kernels then
    # consume those adapted params at sample time with their own state
    # structure (see ``_recipe_runner.py`` ``_run_one_chain``).
    return blackjax.nuts, {}
