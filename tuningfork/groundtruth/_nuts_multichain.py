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
"""Standard multi-chain NUTS and explicit-positions GT generation.

Covers two generation paths:

- **standard_multichain_nuts** — 9 models (eight_schools_ncp, german_credit,
  horseshoe, irt_1pl, irt_2pl, lgcp, logistic_synthetic, radon, stoch_vol):
  per-chain ``blackjax.window_adaptation(blackjax.nuts)`` with
  ``init_to_uniform_radius2`` starts.  stoch_vol uses
  ``target_acceptance=0.95`` (read from its committed ``sampler_config``).

- **explicit_positions** — 1 model (lotka_volterra): same NUTS path but with
  per-chain starting positions loaded directly from the committed
  ``provenance.init_positions.positions`` block, making regeneration fully
  self-contained without any reference to external summary statistics.

Both paths use vmap for parallel chain execution (sequential fallback via
``--sequential`` for diagnostics).  Progress checkpoints are emitted via
jaxtap at 25/50/75/100% of the draw scan.

Environment flags
-----------------
``GT_X64=1``
    Enable 64-bit floating point.  Required for ``lotka_volterra``
    (``requires_x64=True`` in the model registry).
``PYTHONUNBUFFERED=1``
    Recommended for long-running models (stoch_vol, lotka_volterra) to ensure
    per-step progress lines reach the log without buffering.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["generate_nuts_multichain"]


def generate_nuts_multichain(
    model_name: str,
    committed_summary: dict,
    out_dir: Path,
    *,
    seed: int | None = None,
    n_chains: int | None = None,
    n_draws: int | None = None,
    n_warmup: int | None = None,
    sequential: bool = False,
    smoke: bool = False,
) -> dict:
    """Generate multi-chain NUTS ground-truth draws.

    Handles both ``standard_multichain_nuts`` (init_to_uniform_radius2) and
    ``explicit_positions`` (init from committed provenance block) paths,
    dispatched automatically from ``committed_summary``.

    Parameters
    ----------
    model_name
        Registry model name (e.g. ``"radon"`` or ``"lotka_volterra"``).
    committed_summary
        Parsed ``summary_v2.json`` for this model.
    out_dir
        Directory where ``draws.npz`` and ``summary_v2.json`` are written.
    seed
        Master RNG seed.  Defaults to the committed seed.
    n_chains
        Number of parallel chains.  Defaults to the committed value (10).
    n_draws
        Draws per chain.  Defaults to the committed value (10000).
    n_warmup
        Warmup steps per chain.  Defaults to the committed value (2000).
    sequential
        Run chains sequentially instead of via vmap.  Useful for models with
        heterogeneous tree-depth costs (e.g. Neal's funnel variants) where vmap
        pads all chains to the maximum tree depth.
    smoke
        Run at tiny scale (2 chains × 100 draws × 100 warmup) for fast
        validation.  Overrides ``n_chains``, ``n_draws``, ``n_warmup``.

    Returns
    -------
    dict
        Parsed ``summary_v2.json`` dict for the generated GT.

    Raises
    ------
    NotImplementedError
        This path will be implemented in a follow-up commit.
    """
    raise NotImplementedError(
        "generate_nuts_multichain is not yet implemented. "
        "Use generate_analytic_iid() for analytic models, or await the "
        "next module update for NUTS models."
    )
