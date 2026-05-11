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
"""Warmup procedures.

Each Warmup wraps a BlackJAX adaptation routine into a uniform shape::

    runner(rng_key, init_position, n_warmup, base_method,
           *, logdensity_fn, num_chains: int = 4, **kwargs)
    -> (states, adapted_params)

**Multi-chain runner contract**:

Parameters
----------
rng_key
    A single key; the runner splits it into ``num_chains`` independent
    per-chain keys internally.
init_position
    A SINGLE pytree (one chain's worth of initial position), same as the
    single-chain API.  The runner is responsible for replicating it across
    ``num_chains``.  Callers that want jittered inits should pre-batch
    ``init_position`` with a leading dim of ``num_chains`` — the runner
    detects this and passes it through verbatim (see ``_maybe_replicate``).
num_chains
    Number of independent chains.  Default ``4``, matching Stan/NumPyro
    convention.  v2-P0 will study the chain-count/accuracy tradeoff per
    sampler; for now 4 is locked.  Pass ``num_chains=1`` explicitly when
    single-chain semantics are required (e.g., BO tuning trials, which
    are intentionally single-chain — chain count is orthogonal to HP tuning).

Returns
-------
states
    A batched pytree with leading dimension ``num_chains``.  For
    ``num_chains=4`` and a 10-D model, ``states.position`` has shape
    ``(4, 10)``.  When ``num_chains=1`` the leading dim is 1 (NOT
    squeezed) — callers can ``jnp.squeeze(states.position, axis=0)``
    if needed.
adapted_params
    Dict of post-warmup parameters.  Conventional keys:

    - ``"step_size"``            : ``(num_chains,)`` array OR scalar (averaged).
    - ``"inverse_mass_matrix"``  : ``(num_chains, d)`` or ``(num_chains, d, d)``
                                   for diagonal / dense (per
                                   ``is_mass_matrix_diagonal``).

    Wrappers may average across chains or return per-chain values; the
    choice is documented in the wrapper's docstring.

The Warmup dataclass supports standard registration of warmup procedures
(stan_window, mclmc_tuning, no_warmup, etc.) with compatibility guards
(``is_compatible`` method) preventing mismatched warmup-sampler pairings.
The multi-chain runner contract enables parallel warmup across multiple chains.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


def _maybe_replicate(position: Any, num_chains: int) -> Any:
    """Return ``position`` replicated to ``(num_chains, ...)`` if needed.

    If ``position`` already has a leading dimension equal to ``num_chains``,
    it is returned as-is (caller pre-batched).  Otherwise each leaf is
    broadcast to ``(num_chains, *leaf.shape)``.

    Parameters
    ----------
    position
        A JAX pytree (dict, namedtuple, or plain array).  Leaves must be
        JAX arrays.
    num_chains
        Target leading dimension.

    Returns
    -------
    Any
        A pytree with the same structure as ``position`` but with every leaf
        having a leading dimension of ``num_chains``.
    """
    leaves = jax.tree.leaves(position)
    if leaves and leaves[0].shape and leaves[0].shape[0] == num_chains:
        # Caller pre-batched — pass through verbatim.
        return position
    return jax.tree.map(
        lambda x: jnp.broadcast_to(x, (num_chains,) + jnp.asarray(x).shape), position
    )


def squeeze_single_chain(
    batched_state: Any, batched_params: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Squeeze a leading length-1 chain dim from a single-chain warmup result.

    Use this in callers that ran a warmup with ``num_chains=1`` and need the
    un-batched ``(state, params)`` shape that the rest of the codebase expects
    (e.g. BO tuning trials, MEDIUM-effort recipe builders, single-chain demos).

    Behaviour
    ---------
    - State pytree leaves: ``x`` becomes ``x[0]`` when subscriptable; non-array
      leaves pass through unchanged.
    - Param dict values: scalar Python numbers pass through; array-like values
      are coerced to ``jnp.asarray`` and indexed with ``[0]`` only when their
      first dim is exactly 1.  Anything else is returned verbatim.

    Parameters
    ----------
    batched_state
        Pytree returned by ``Warmup.runner`` with leading dim 1.
    batched_params
        Dict of adapted parameters (e.g. ``{"step_size": (1,)-array, ...}``).

    Returns
    -------
    tuple[Any, dict[str, Any]]
        ``(state, params)`` with the leading length-1 chain axis removed.
    """
    state = jax.tree.map(
        lambda x: x[0] if hasattr(x, "__getitem__") else x, batched_state
    )
    params: dict[str, Any] = {}
    for k, v in batched_params.items():
        if isinstance(v, (int, float, bool)):
            params[k] = v
        else:
            arr = jnp.asarray(v)
            if arr.shape and arr.shape[0] == 1:
                params[k] = arr[0]
            else:
                params[k] = v
    return state, params


@dataclass(frozen=True)
class Warmup:
    """A warmup procedure: produces ``(states, params)`` before sampling.

    Parameters
    ----------
    name
        Unique registry key, e.g. ``"stan_window"``, ``"mclmc_tuning"``,
        ``"no_warmup"``.
    runner
        Callable with signature::

            runner(rng_key, init_position, n_warmup, base_method,
                   *, logdensity_fn, num_chains: int = 4, **kwargs)
            -> (states, dict)

        Returns the post-warmup kernel states (batched over ``num_chains``)
        and a dict of adapted parameters.  Empty dict means "use default
        params from BO / recipe".

        See module-level docstring for the full multi-chain contract.
    compatible_methods
        Tuple of ``BaseMethod.name`` values this warmup supports.
        The special sentinel ``"*"`` means "compatible with all algorithms".
    notes
        Free-form documentation string.
    """

    name: str
    runner: Callable[..., tuple[Any, dict]]
    compatible_methods: tuple[str, ...]
    notes: str = ""

    def is_compatible(self, base_method_name: str) -> bool:
        """Return True if this warmup supports the named algorithm.

        Parameters
        ----------
        base_method_name
            ``BaseMethod.name`` string, e.g. ``"nuts"``, ``"mclmc"``.

        Returns
        -------
        bool
            ``True`` if ``"*"`` is in ``self.compatible_methods`` (meaning
            this warmup works with any algorithm) or if ``base_method_name``
            is listed explicitly in ``self.compatible_methods``.
        """
        return (
            "*" in self.compatible_methods
            or base_method_name in self.compatible_methods
        )
