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
"""Identity warmup: initialise the kernel state without any adaptation.

This is the "zero-cost" warmup used for:

- **LOW-effort recipes**: skip adaptation entirely; use default hyperparameters.
- **Gradient-free kernels** (RWM): no warmup is meaningful or needed.
- **Tier-C warmup-isolation runs** where the researcher wants to measure
  raw kernel performance before any adaptation overhead.

The runner returns an empty ``adapted_params`` dict; all kernel
hyperparameters come from the BO trial or the recipe defaults.

Runner signature (uniform across all warmups)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, **kwargs) -> (state, {})

The ``n_warmup`` argument is accepted for interface uniformity but is
not used — no gradient evaluations are performed.

MCLMC special case: ``blackjax.mclmc.init`` requires an ``rng_key``
argument to generate the initial unit-vector momentum.  All other
BlackJAX kernels' ``.init`` methods accept only a position.  This
module handles the distinction via ``base_method.name == "mclmc"``.
"""

from __future__ import annotations

from typing import Any

import jax

from bjx_bench.inference.warmup._base import Warmup

__all__ = ["ENTRY"]


def _runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,  # noqa: ARG001 — accepted for interface uniformity; not used
    base_method: Any,  # BaseMethod; not imported to avoid circular dep at module level
    *,
    logdensity_fn: Any,
    **kwargs: Any,  # noqa: ARG001 — accepted for interface uniformity; not used
) -> tuple[Any, dict[str, Any]]:
    """Initialise the kernel state using default hyperparameters; no adaptation.

    Parameters
    ----------
    rng_key
        JAX random key.  Used only when ``base_method.name == "mclmc"``
        (MCLMC ``init`` needs a key for its momentum initialisation).
        For all other kernels, the key is accepted but forwarded to init
        via the ``rng_key`` argument only if the kernel supports it.
    init_position
        Initial unconstrained parameter dict.
    n_warmup
        Accepted for interface uniformity; ignored entirely (no adaptation runs).
    base_method
        ``BaseMethod`` entry (carries ``factory`` and ``default_hp_space``).
    logdensity_fn
        BlackJAX-compatible log-density function.
    **kwargs
        Accepted for interface uniformity; ignored.

    Returns
    -------
    state
        Freshly initialised kernel state.
    adapted_params
        Always ``{}`` — no adaptation was run; all HPs come from defaults.
    """
    import jax.numpy as jnp

    from bjx_bench.calibration.tier_b import default_params_for

    defaults = default_params_for(base_method)

    # For kernels that require an inverse_mass_matrix (e.g. HMC, NUTS, Barker)
    # but don't list it in their BO HP space (since it normally comes from
    # warmup adaptation), we inject a diagonal identity preconditioner so the
    # kernel can initialise.  We derive the dimension from init_position's
    # total leaf count using jax.tree_util.tree_leaves.
    if base_method.needs_mass_matrix and "inverse_mass_matrix" not in defaults:
        leaves = jax.tree_util.tree_leaves(init_position)
        n_params = int(sum(jnp.asarray(leaf).size for leaf in leaves))
        defaults = {**defaults, "inverse_mass_matrix": jnp.ones(n_params)}

    kernel = base_method.factory(logdensity_fn, **defaults)

    # MCLMC.init requires an rng_key to sample the initial unit-vector
    # momentum; all other BlackJAX kernels' init takes only a position.
    if base_method.name == "mclmc":
        init_state = kernel.init(init_position, rng_key)
    else:
        init_state = kernel.init(init_position)

    return init_state, {}


ENTRY = Warmup(
    name="no_warmup",
    runner=_runner,
    compatible_methods=("*",),  # sentinel: works with every algorithm
    notes=(
        "Identity warmup: returns the kernel's init state with default params "
        "and an empty adapted_params dict.  Zero gradient evaluations.  "
        "Used for LOW-effort recipes, gradient-free kernels (RWM), and "
        "Tier-C warmup-isolation baselines.  "
        "MCLMC is handled specially: kernel.init(position, rng_key) rather "
        "than kernel.init(position)."
    ),
)
