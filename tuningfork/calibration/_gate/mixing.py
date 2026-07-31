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
"""Mixing diagnostics stage — R̂, bulk-ESS, and sampler evidence counts."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from blackjax.diagnostics import ess_bulk as _bj_ess_bulk
from blackjax.diagnostics import rhat as _bj_rhat


@dataclass(frozen=True)
class MixingStats:
    """Computed mixing and sampler-evidence metrics."""

    rhat_max: float | None
    min_bulk_ess: float | None
    n_divergences: int | None
    n_nonfinite_proposals: int | None = None
    n_proposals_evaluated: int | None = None
    nonfinite_proposal_rate: float | None = None


def _compute_mixing_stats(
    mc_samples: dict,
    info,
) -> MixingStats:
    """Compute mixing and sampler-specific numerical evidence.

    Parameters
    ----------
    mc_samples
        Dict of arrays with shape ``(n_chains, n_draws, *event_shape)``,
        already returned by ``_samples_to_multichain``.
    info
        Sampler info struct. ``is_divergent`` contributes the HMC-style
        divergence count. ``nonans`` contributes a separate count, denominator,
        and rate for non-finite MCLMC proposals. An info struct with ``nonans``
        but no ``is_divergent`` leaves ``n_divergences`` unset rather than
        inventing a zero.

    Returns
    -------
    MixingStats
        ``nonans`` evidence counts false
        entries as non-finite proposals; this evidence is independent of
        HMC divergence counting.
    """
    # --- R̂ and bulk-ESS ---
    rhat_max: float | None = None
    min_bulk_ess: float | None = None

    if mc_samples:
        rhat_values: list[float] = []
        ess_values: list[float] = []
        for arr in mc_samples.values():
            arr_np = np.asarray(arr)
            rhat_arr = _bj_rhat(arr_np, chain_axis=0, sample_axis=1)
            ess_arr = _bj_ess_bulk(arr_np, chain_axis=0, sample_axis=1)
            rhat_values.append(float(np.max(np.asarray(rhat_arr))))
            ess_values.append(float(np.min(np.asarray(ess_arr))))

        rhat_max = float(max(rhat_values))
        min_bulk_ess = float(min(ess_values))

    # --- Divergences ---
    n_divergences: int | None = None
    n_nonfinite_proposals: int | None = None
    n_proposals_evaluated: int | None = None
    nonfinite_proposal_rate: float | None = None
    if info is not None:
        if hasattr(info, "is_divergent"):
            # HMC/NUTS/laplace family: explicit divergence flag per step.
            n_divergences = int(jnp.sum(jnp.asarray(info.is_divergent)))
        elif not hasattr(info, "nonans"):
            # Preserve the historical fallback for unrelated sampler infos.
            n_divergences = 0

        if hasattr(info, "nonans"):
            nonans = np.asarray(info.nonans, dtype=bool)
            n_proposals_evaluated = int(nonans.size)
            n_nonfinite_proposals = int(np.count_nonzero(~nonans))
            if n_proposals_evaluated:
                nonfinite_proposal_rate = n_nonfinite_proposals / n_proposals_evaluated

    return MixingStats(
        rhat_max,
        min_bulk_ess,
        n_divergences,
        n_nonfinite_proposals,
        n_proposals_evaluated,
        nonfinite_proposal_rate,
    )
