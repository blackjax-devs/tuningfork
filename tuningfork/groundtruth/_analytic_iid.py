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
"""Analytic i.i.d. ground-truth generation.

For models with exact analytic samplers (banana, gmm_25, ill_cond_50, mvn_10,
neals_funnel), regeneration calls the model's ``analytic_sampler`` directly
for each of ``n_chains`` independent batches of ``n_draws`` samples.

No warmup, no NUTS — output is exact.  The resulting draws are statistically
equivalent to the committed GT (in-spec) but NOT bit-identical (the RNG key
derivation from ``seed`` may differ if using a different seed).

Typical wall time: < 5 seconds for all analytic models at 10×10k.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from tuningfork.groundtruth._emit import write_gt_artifacts

__all__ = ["generate_analytic_iid"]

_SMOKE_N_CHAINS = 2
_SMOKE_N_DRAWS = 50


def generate_analytic_iid(
    model_name: str,
    committed_summary: dict,
    out_dir: Path,
    *,
    seed: int | None = None,
    n_chains: int | None = None,
    n_draws: int | None = None,
    smoke: bool = False,
) -> dict:
    """Generate analytic i.i.d. ground-truth draws.

    Calls the model's ``analytic_sampler(rng_key, n) -> dict[site, Array]``
    for each of ``n_chains`` independent batches and stacks the results into
    the canonical ``(n_chains, n_draws, *event)`` shape.

    Parameters
    ----------
    model_name
        Registry model name (e.g. ``"mvn_10"``).
    committed_summary
        Parsed ``summary_v2.json`` for this model (used to read defaults for
        ``n_chains``, ``n_draws``, and ``seed``).
    out_dir
        Directory where ``draws.npz`` and ``summary_v2.json`` are written.
    seed
        Master RNG seed.  Defaults to the committed GT seed so that the
        default invocation reproduces the original configuration.
    n_chains
        Number of independent chains.  Defaults to the committed value (10).
    n_draws
        Draws per chain.  Defaults to the committed value (10000).
    smoke
        Run at tiny scale (2 chains × 50 draws) for fast CI validation.
        Overrides ``n_chains``, ``n_draws``.

    Returns
    -------
    dict
        The parsed ``summary_v2.json`` dict for the generated GT.
    """
    import jax

    from tuningfork.model import MODELS

    entry = MODELS[model_name]
    if entry.analytic_sampler is None:
        raise ValueError(
            f"Model {model_name!r} does not have an analytic_sampler. "
            "Use generate_nuts_multichain() instead."
        )

    if smoke:
        n_chains = _SMOKE_N_CHAINS
        n_draws = _SMOKE_N_DRAWS

    _nc = n_chains if n_chains is not None else committed_summary["n_chains"]
    _nd = n_draws if n_draws is not None else committed_summary["n_draws_per_chain"]
    _seed = seed if seed is not None else committed_summary["seeds"]["master_seed"]

    key = jax.random.key(_seed)
    chain_keys = jax.random.split(key, _nc)

    t0 = time.perf_counter()
    batches = [entry.analytic_sampler(k, _nd) for k in chain_keys]
    jax.block_until_ready(batches)
    sampling_wall = time.perf_counter() - t0

    sites = list(batches[0].keys())
    positions: dict[str, np.ndarray] = {
        s: np.stack([np.asarray(b[s]) for b in batches], axis=0) for s in sites
    }

    diag: dict[str, Any] = {
        "generator": "analytic_iid",
        "total_divergences": 0,
    }
    timing = {"warmup": 0.0, "sampling": sampling_wall}
    sampler_config = dict(committed_summary["sampler_config"])

    seeds_meta = {
        "master_seed": _seed,
        "derivation": (
            "key=jax.random.key(seed); chain_keys=jax.random.split(key, n_chains)"
        ),
    }

    reproduced_from = {
        "timestamp_utc": committed_summary.get("provenance", {}).get("timestamp_utc"),
        "tuningfork_version": committed_summary.get("provenance", {}).get(
            "tuningfork_version"
        ),
    }

    _, summary_path = write_gt_artifacts(
        out_dir,
        model_name=model_name,
        positions=positions,
        diag=diag,
        timing=timing,
        generator="analytic_iid",
        space="unconstrained",
        sampler_config=sampler_config,
        seeds=seeds_meta,
        reproduced_from=reproduced_from,
        total_wall=time.perf_counter() - t0,
    )

    import json

    return json.loads(summary_path.read_text())
