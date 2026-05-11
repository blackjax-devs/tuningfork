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
"""Headline metric for tuningfork: min bulk-ESS per gradient evaluation.

The metric an algorithm developer chases:

    headline(states_position, n_grad_evals) =
        min over dims of bulk_ESS(states_position) / n_grad_evals

where bulk_ESS uses blackjax.diagnostics.effective_sample_size, which already
aggregates across chains (sum-across-chains per dimension).

The min is taken across ALL dimensions of ALL sample sites — the worst-mixing
dimension determines effective run length. n_grad_evals comes from
``tuningfork.metrics.grad_counter.total_grad_evals`` and is hardware-independent.

Empirically verified (see tests/test_headline.py):
- ``effective_sample_size((C, S, D)) → shape (D,)``: output is already
  aggregated across chains per the standard bulk-ESS definition.
- i.i.d. baseline: headline / (C*S) ≈ 0.85–1.05 for C=4, S=1000, D=5.
- AR(1) φ=0.9 suppression: headline_iid / headline_ar1 ≈ 15–22 for S=2000.
"""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from blackjax.diagnostics import effective_sample_size

__all__ = ["min_bulk_ess_per_grad"]


def min_bulk_ess_per_grad(
    states_position: Mapping[str, Any],
    n_grad_evals: int,
) -> float:
    """Headline metric: minimum bulk-ESS per gradient evaluation.

    Parameters
    ----------
    states_position
        Dict keyed by NumPyro sample-site name. Each value is an array of shape
        ``(n_chains, n_samples, *site_shape)``. The per-site shape is flattened
        before ESS computation so a site of shape ``(C, S, 4, 4)`` is treated
        as ``(C, S, 16)`` independent dimensions.
    n_grad_evals
        Total gradient evaluations across the whole run, as returned by
        ``tuningfork.metrics.grad_counter.total_grad_evals``. Must be >= 0.

    Returns
    -------
    float
        ``min over all dims of bulk-ESS / n_grad_evals``.

        Special cases:

        - ``n_grad_evals == 0``:  returns ``inf`` if ``min_ess`` is finite,
          ``nan`` if ``min_ess`` is itself non-finite (empty / degenerate).
        - Finite ``n_grad_evals > 0``:  returns ``float(min_ess / n_grad_evals)``.

    Raises
    ------
    ValueError
        If ``n_grad_evals < 0``, ``states_position`` is empty, or any site
        array has ``ndim < 2`` (must supply at least a chain and sample axis).

    Notes
    -----
    ``blackjax.diagnostics.effective_sample_size`` with ``chain_axis=0`` and
    ``sample_axis=1`` returns an array of shape ``(D,)`` for input shape
    ``(C, S, D)`` — bulk-ESS is already summed across chains per dimension.
    We therefore just take the per-site minimum and propagate the global min
    across all sites.

    The min is a pessimistic aggregation: the worst-mixing dimension in any
    sample site determines the effective run length for the whole chain.

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> from tuningfork.metrics.headline import min_bulk_ess_per_grad
    >>> key = jax.random.key(0)
    >>> samples = {"x": jax.random.normal(key, (4, 1000, 5))}
    >>> headline = min_bulk_ess_per_grad(samples, n_grad_evals=4000)
    >>> 0.5 < headline < 2.0  # i.i.d. → close to 1.0
    True
    """
    if n_grad_evals < 0:
        raise ValueError(f"n_grad_evals must be non-negative, got {n_grad_evals}")
    if not states_position:
        raise ValueError("states_position is empty; need at least one site")

    min_ess = jnp.inf
    for site, arr in states_position.items():
        if arr.ndim < 2:
            raise ValueError(
                f"site {site!r} has shape {arr.shape}; "
                "expected (n_chains, n_samples, ...) — at least 2 dimensions"
            )
        # Collapse site dims into one trailing axis: (C, S, D)
        # A scalar site (C, S) becomes (C, S, 1).
        flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
        # ess_per_dim has shape (D,) — already aggregated across chains.
        ess_per_dim = effective_sample_size(flat, chain_axis=0, sample_axis=1)
        site_min = jnp.min(ess_per_dim)
        min_ess = jnp.minimum(min_ess, site_min)

    if n_grad_evals == 0:
        # Caller asked for ESS-per-grad with zero grad budget.
        # If min_ess is finite (real samples exist), ESS/0 = +inf.
        # If min_ess is not finite (degenerate), propagate nan.
        return float("inf") if bool(jnp.isfinite(min_ess)) else float("nan")

    return float(min_ess / n_grad_evals)
