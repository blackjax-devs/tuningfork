"""Tier-A analytic path (Path A) — certify reference draws for analytic models.

For models with a known closed-form posterior, we sample directly from the
analytic distribution.  Validation is done by comparing empirical moments to
the analytic moments (when known) at a 4-σ Monte Carlo SE tolerance.  This is
a unit-test-level check, not a runtime gate — analytic samples are exact by
construction.
"""

from __future__ import annotations

import jax

from bjx_bench.calibration._summary import Summaries, compute_summaries
from bjx_bench.model._base import Posterior, ReferenceMethod

__all__ = ["certify_reference_analytic"]


def certify_reference_analytic(
    entry: Posterior,
    n: int,
    rng_key: jax.Array,
) -> tuple[dict[str, jax.Array], Summaries]:
    """Sample from entry.analytic_sampler and compute summaries.

    Parameters
    ----------
    entry
        Registry entry.  Must have ``reference_method == ANALYTIC``.
    n
        Number of i.i.d. draws to generate.
    rng_key
        JAX random key.

    Returns
    -------
    draws
        Dict mapping site name → Array of shape ``(n, *site_shape)``.
    summaries
        Per-dim mean/std/q05/q95 computed on the n draws.

    Raises
    ------
    ValueError
        If ``entry.reference_method != ANALYTIC``.
    """
    if entry.reference_method != ReferenceMethod.ANALYTIC:
        raise ValueError(
            f"{entry.name!r} uses the NUTS path; call certify_reference_nuts instead."
        )
    assert entry.analytic_sampler is not None  # guarded by reference_method check
    draws = entry.analytic_sampler(rng_key, n)
    summaries = compute_summaries(draws)
    return draws, summaries
