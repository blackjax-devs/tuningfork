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
"""Verification: gate + coherence check of generated GT vs committed GT.

The ``--verify`` flag in the CLI calls ``verify_groundtruth`` after generation
to confirm that the newly generated draws pass the quality gate and are
statistically coherent with the committed GT.

Two checks are performed:

1. **Gate check** — same thresholds as the committed summary:
   ``max_rhat ≤ 1.01``, ``min_bulk_ess ≥ 400``, ``divergence_rate ≤ 0.001``
   (NUTS models), ``min_e_bfmi ≥ 0.3`` (NUTS models, when available).

2. **Coherence check** — per-site z-score of the new mean vs the committed
   mean, normalized by ``max(between_chain_se_new, between_chain_se_committed)``.
   All sites must have ``max_z ≤ 3.0``.  This is the same framework used
   during the multichain GT migration coherence validation.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["verify_groundtruth"]


def verify_groundtruth(
    model_name: str,
    generated_summary: dict,
    generated_draws_path: Path,
    *,
    z_threshold: float = 3.0,
    print_results: bool = True,
) -> bool:
    """Check generated GT quality and coherence vs committed catalog GT.

    Parameters
    ----------
    model_name
        Registry model name.
    generated_summary
        Parsed ``summary_v2.json`` from the just-generated run.
    generated_draws_path
        Path to the generated ``draws.npz``.
    z_threshold
        Maximum allowed per-site z-score for mean coherence.  Default 3.0.
    print_results
        Print gate and coherence results to stdout.

    Returns
    -------
    bool
        ``True`` if both gate and coherence checks pass.

    Raises
    ------
    NotImplementedError
        This function will be implemented in a follow-up commit.
    """
    raise NotImplementedError(
        "verify_groundtruth is not yet implemented. " "Await the next module update."
    )
