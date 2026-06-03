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
- **warmup-only-isolation runs** where the researcher wants to measure
  raw kernel performance before any adaptation overhead.

The runner returns an empty ``adapted_params`` dict; all kernel
hyperparameters come from the BO trial or the recipe defaults.

Runner signature (multi-chain contract)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, num_chains: int = 4, **kwargs)
    -> (states, {})

Where:

- ``rng_key`` is a single key; split internally into ``num_chains`` keys
  (used only for MCLMC momentum initialisation; other kernels ignore it).
- ``init_position`` is a single pytree (one chain's worth); replicated
  across chains internally via ``_maybe_replicate`` unless the caller
  pre-batches it (leading dim == ``num_chains``).
- ``states`` is a batched pytree with leading dim ``num_chains``.
- ``adapted_params`` is always ``{}`` — no adaptation was run.

The ``n_warmup`` argument is accepted for interface uniformity but is
not used — no gradient evaluations are performed.

MCLMC special case: ``blackjax.mclmc.init`` requires an ``rng_key``
argument to generate the initial unit-vector momentum.  All other
BlackJAX kernels' ``.init`` methods accept only a position.  This
module handles the distinction via ``base_method.name == "mclmc"``.
"""

from typing import Any

import jax
import jax.numpy as jnp

from tuningfork.warmup._base import Warmup, _maybe_replicate

__all__ = ["ENTRY"]


def _build_prior_kwargs_from_posterior(
    base_method: Any,
    init_position: Any,
    posterior_entry: Any,
) -> dict[str, Any]:
    """Extract Gaussian prior arrays from ``posterior_entry`` for gradient-free samplers.

    Called by ``_runner`` when ``base_method.extra_required_kwargs`` is non-empty
    and ``posterior_entry`` is provided.  Currently supports
    ``extra_required_kwargs=("prior_cov", "prior_mean")`` (elliptical_slice).

    The prior is stored per site in ``posterior_entry.prior_mean`` /
    ``posterior_entry.prior_cov_diag`` as ``dict[str, list[float]]``.
    Both dicts are converted to JAX arrays and flattened via
    ``jax.flatten_util.ravel_pytree`` so their element order matches the
    flattened ``init_position`` pytree that ``blackjax.elliptical_slice``
    operates on.

    Parameters
    ----------
    base_method
        ``BaseMethod`` entry whose ``extra_required_kwargs`` lists the needed keys.
    init_position
        Single-chain position pytree (used only for structure/ordering).
    posterior_entry
        A ``Posterior`` with ``prior_mean`` and ``prior_cov_diag`` fields.

    Returns
    -------
    dict
        ``{"prior_mean": Array(d,), "prior_cov": Array(d,)}`` ready to merge
        into the factory kwargs.

    Raises
    ------
    ValueError
        If ``posterior_entry`` is missing or its prior fields are None.
    """
    from jax.flatten_util import ravel_pytree

    missing = []
    for k in ("prior_mean", "prior_cov_diag"):
        if getattr(posterior_entry, k, None) is None:
            missing.append(k)
    if missing:
        raise ValueError(
            f"no_warmup: {base_method.name!r} requires prior kwargs but "
            f"posterior_entry.{missing[0]} is None — set prior_mean and "
            f"prior_cov_diag on the Posterior entry."
        )

    # Build pytrees matching init_position structure, then flatten.
    prior_mean_pytree = {k: jnp.array(v) for k, v in posterior_entry.prior_mean.items()}
    prior_cov_pytree = {
        k: jnp.array(v) for k, v in posterior_entry.prior_cov_diag.items()
    }
    mean_flat, _ = ravel_pytree(prior_mean_pytree)
    cov_flat, _ = ravel_pytree(prior_cov_pytree)
    return {"prior_mean": mean_flat, "prior_cov": cov_flat}


def _runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,  # noqa: ARG001 — accepted for interface uniformity; not used
    base_method: Any,  # BaseMethod; not imported to avoid circular dep at module level
    *,
    logdensity_fn: Any,
    num_chains: int = 4,
    posterior_entry: Any = None,
    **kwargs: Any,  # noqa: ARG001 — accepted for interface uniformity; not used
) -> tuple[Any, dict[str, Any]]:
    """Initialise kernel states using default hyperparameters; no adaptation.

    Parameters
    ----------
    rng_key
        JAX random key.  Split into ``num_chains`` subkeys used only when
        ``base_method.name == "mclmc"`` (MCLMC ``init`` needs a key for
        its momentum initialisation).  For all other kernels, the subkeys
        are accepted but not forwarded to ``init``.
    init_position
        Initial unconstrained parameter dict.  A SINGLE pytree (one chain's
        worth).  Replicated across ``num_chains`` unless pre-batched.
    n_warmup
        Accepted for interface uniformity; ignored entirely (no adaptation runs).
    base_method
        ``BaseMethod`` entry (carries ``factory`` and ``default_hp_space``).
    logdensity_fn
        BlackJAX-compatible log-density function.  For ``elliptical_slice``
        this must be the **likelihood-only** function (joint minus prior);
        the recipe runner is responsible for the subtraction before calling
        this runner (see B3 wiring in ``_recipe_runner.py``).
    num_chains
        Number of independent chains.  Default ``4``, matching Stan/NumPyro
        convention.  The returned ``states`` has leading dim ``num_chains``
        (never squeezed, even for ``num_chains=1``).
    posterior_entry
        Optional ``Posterior`` instance.  Required when
        ``base_method.extra_required_kwargs`` is non-empty (e.g.
        ``elliptical_slice`` needs ``prior_mean``/``prior_cov`` from
        ``posterior_entry.prior_mean``/``prior_cov_diag``).
    **kwargs
        Accepted for interface uniformity; ignored.

    Returns
    -------
    states
        Freshly initialised kernel states, batched over ``num_chains``.
        ``states.position`` has shape ``(num_chains, d)``.
    adapted_params
        Always ``{}`` — no adaptation was run; all HPs come from defaults.
    """
    from tuningfork.calibration.tune import default_params_for

    if base_method.extra_required_kwargs:
        if posterior_entry is None:
            raise NotImplementedError(
                f"no_warmup runner cannot construct base_method "
                f"{base_method.name!r}: factory requires extra kwargs "
                f"{base_method.extra_required_kwargs!r}. Pass "
                f"posterior_entry=<Posterior> so the runner can extract "
                f"the prior from posterior_entry.prior_mean/prior_cov_diag."
            )
        prior_kwargs = _build_prior_kwargs_from_posterior(
            base_method, init_position, posterior_entry
        )
    else:
        prior_kwargs = {}

    defaults = {**default_params_for(base_method), **prior_kwargs}

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

    # Replicate init_position across chains.  Pass-through if pre-batched.
    init_positions = _maybe_replicate(init_position, num_chains)

    # Split rng_key for num_chains independent momentum inits (MCLMC only).
    chain_keys = jax.random.split(rng_key, num_chains)

    # MCLMC.init requires an rng_key to sample the initial unit-vector
    # momentum; all other BlackJAX kernels' init takes only a position.
    if base_method.name == "mclmc":
        # vmap over (key, position) for MCLMC
        @jax.vmap
        def init_one_mclmc(k: jax.Array, x0: Any) -> Any:
            return kernel.init(x0, k)

        states = init_one_mclmc(chain_keys, init_positions)
    else:
        # vmap over position for all other kernels (key unused)
        @jax.vmap
        def init_one(x0: Any) -> Any:
            return kernel.init(x0)

        states = init_one(init_positions)

    return states, {}


ENTRY = Warmup(
    name="no_warmup",
    runner=_runner,
    compatible_methods=("*",),  # sentinel: works with every algorithm
    notes=(
        "Identity warmup: returns the kernel's init state with default params "
        "and an empty adapted_params dict.  Zero gradient evaluations.  "
        "Used for LOW-effort recipes, gradient-free kernels (RWM), and "
        "warmup-only-isolation baselines.  "
        "MCLMC is handled specially: kernel.init(position, rng_key) rather "
        "than kernel.init(position).  "
        "multi-chain by default (num_chains=4 via jax.vmap); states "
        "batched with leading dim num_chains (never squeezed)."
    ),
)
