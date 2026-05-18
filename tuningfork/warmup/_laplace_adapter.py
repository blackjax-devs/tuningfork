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
"""Adapter: route laplace_* base methods through standard HMC for warmup.

Background
----------
``blackjax.window_adaptation`` (and ``low_rank_window_adaptation``) require
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

This module provides ``resolve_warmup_algorithm``, a single function that the
three window-adaptation runners call to get the right ``(algorithm, extra_kwargs)``
pair to pass to ``blackjax.window_adaptation``:

- For laplace_* base methods: returns ``(blackjax.hmc, extra_kwargs_for_hmc)``
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

__all__ = ["LAPLACE_METHOD_NAMES", "resolve_warmup_algorithm"]

# The four laplace_* base method names registered in the tuningfork registry.
LAPLACE_METHOD_NAMES: frozenset[str] = frozenset(
    ("laplace_hmc", "laplace_dhmc", "laplace_mhmc", "laplace_dmhmc")
)


def resolve_warmup_algorithm(
    base_method: Any,
    extra_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Return the warmup ``(algorithm, extra_kwargs)`` pair for ``blackjax.window_adaptation``.

    For standard HMC-family methods (hmc, nuts, mhmc, …) the existing
    ``(base_method.factory, extra_kwargs)`` pair is returned unchanged.

    For laplace_* methods the function substitutes ``blackjax.hmc`` as the
    warmup algorithm (standard HMC on the marginal logdensity) and strips any
    kwargs that HMC does not understand (there are none currently, but the
    function future-proofs against laplace-specific kwargs being injected by
    callers).

    The caller is responsible for passing the laplace marginal logdensity
    ``log p̂(phi | y)`` as ``logdensity_fn`` to ``blackjax.window_adaptation``.
    The adapter does NOT build or validate the marginal logdensity — it only
    selects the right algorithm object.

    Parameters
    ----------
    base_method
        A ``BaseMethod`` entry from the tuningfork registry.  The ``.name``
        attribute is used to detect laplace_* methods.
    extra_kwargs
        The ``extra_kwargs`` dict already prepared by the caller (default HP
        values injected, caller overrides applied).  For laplace_* the
        ``num_integration_steps`` key is preserved (HMC also uses it).

    Returns
    -------
    algorithm
        Either ``base_method.factory`` (non-laplace path) or ``blackjax.hmc``
        (laplace path).
    extra_kwargs
        Possibly filtered extra_kwargs (currently a copy for both paths).
    """
    if base_method.name not in LAPLACE_METHOD_NAMES:
        # Non-laplace path: return unchanged — no regression possible.
        return base_method.factory, dict(extra_kwargs)

    # Laplace path: use standard HMC as the warmup kernel.
    # The marginal logdensity (phi → float) is already the correct logdensity_fn
    # for window adaptation — the caller is responsible for providing it.
    # HMC accepts num_integration_steps; keep it if present; use 5 as default.
    hmc_kwargs: dict[str, Any] = {}
    if "num_integration_steps" in extra_kwargs:
        hmc_kwargs["num_integration_steps"] = extra_kwargs["num_integration_steps"]
    else:
        # Sensible default for warmup — short trajectories are fine here.
        hmc_kwargs["num_integration_steps"] = 5

    return blackjax.hmc, hmc_kwargs
