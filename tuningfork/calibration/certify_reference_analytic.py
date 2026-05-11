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
"""analytic reference-certification path (Path A) — certify reference draws for analytic models.

For models with a known closed-form posterior, we sample directly from the
analytic distribution.  Validation is done by comparing empirical moments to
the analytic moments (when known) at a 4-σ Monte Carlo SE tolerance.  This is
a unit-test-level check, not a runtime gate — analytic samples are exact by
construction.
"""

import jax

from tuningfork.calibration._summary import Summaries, compute_summaries
from tuningfork.model._base import Posterior, ReferenceMethod

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
