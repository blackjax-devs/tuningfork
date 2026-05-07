"""Per-site reference summary statistics.

``Summaries`` holds per-site mean/std/q05/q95 arrays computed along the
sample axis (axis=0).  ``compute_summaries`` is the only constructor users
should call.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = ["Summaries", "compute_summaries"]


@dataclass(frozen=True)
class Summaries:
    """Per-site, per-dim reference statistics.

    Each dict maps site name → array of shape ``(dim_at_site,)`` (or scalar
    for 0-D sites).  The arrays are JAX arrays so they can be compared with
    JIT-compiled code.

    Parameters
    ----------
    mean
        Empirical mean per site.
    std
        Empirical standard deviation per site.
    q05
        5th-percentile per site.
    q95
        95th-percentile per site.
    n_samples
        Number of samples used to compute these statistics.
    """

    mean: dict[str, jax.Array]
    std: dict[str, jax.Array]
    q05: dict[str, jax.Array]
    q95: dict[str, jax.Array]
    n_samples: int


def compute_summaries(draws: dict[str, jax.Array]) -> Summaries:
    """Compute mean/std/q05/q95 along axis=0 for every site.

    Parameters
    ----------
    draws
        Dict mapping site name → Array of shape ``(n_samples, *site_shape)``.
        Scalar sites should have shape ``(n_samples,)`` (1-D).

    Returns
    -------
    ``Summaries`` with per-site statistics.
    """
    n = next(iter(draws.values())).shape[0]
    mean: dict[str, jax.Array] = {}
    std: dict[str, jax.Array] = {}
    q05: dict[str, jax.Array] = {}
    q95: dict[str, jax.Array] = {}

    for site, arr in draws.items():
        # Flatten everything but the sample axis so 0-D sites become 1-D
        arr_2d = arr.reshape(n, -1) if arr.ndim > 1 else arr[:, None]
        mean[site] = jnp.mean(arr_2d, axis=0).squeeze()
        std[site] = jnp.std(arr_2d, axis=0).squeeze()
        q05[site] = jnp.quantile(arr_2d, 0.05, axis=0).squeeze()
        q95[site] = jnp.quantile(arr_2d, 0.95, axis=0).squeeze()

    return Summaries(mean=mean, std=std, q05=q05, q95=q95, n_samples=n)
