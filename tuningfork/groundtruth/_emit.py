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
"""Emit canonical GT artifacts: draws.npz + summary_v2.json.

Every generation path calls ``write_gt_artifacts``, which produces output
files byte-shaped identically to the committed catalog GT.  The schema is
always ``gt_v2_multichain``.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["compute_summary_stats", "write_gt_artifacts"]


def compute_summary_stats(
    positions: dict[str, np.ndarray],
) -> tuple[dict[str, Any], float, float]:
    """Compute per-site ArviZ diagnostics + gate scalars.

    Parameters
    ----------
    positions
        Dict mapping site name to array shaped ``(n_chains, n_draws, *event)``.

    Returns
    -------
    per_site
        Dict mapping site name to stats dict (mean, std, q05, q95,
        between_chain_se, bulk_ess, tail_ess, rhat).
    max_rhat
        Worst rank-normalized split-R̂ across all sites and dimensions.
    min_bulk_ess
        Minimum bulk ESS across all sites and dimensions.
    """
    import arviz as az

    idata = az.from_dict({"posterior": positions}, sample_dims=["chain", "draw"])
    ess_bulk = az.ess(idata, method="bulk")
    ess_tail = az.ess(idata, method="tail")
    rhat = az.rhat(idata, method="rank")

    per_site: dict[str, Any] = {}
    max_rhat = 0.0
    min_bulk = float("inf")

    for site, arr in positions.items():
        a = np.asarray(arr)
        nc, ns = a.shape[0], a.shape[1]
        flat = a.reshape(nc, ns, -1)
        chain_means = flat.mean(axis=1)
        pooled = flat.reshape(-1, flat.shape[-1])
        be_se = chain_means.std(axis=0, ddof=1) / np.sqrt(nc)
        b_ess = np.atleast_1d(np.asarray(ess_bulk[site])).ravel()
        t_ess = np.atleast_1d(np.asarray(ess_tail[site])).ravel()
        r = np.atleast_1d(np.asarray(rhat[site])).ravel()
        per_site[site] = {
            "mean": pooled.mean(axis=0).tolist(),
            "std": pooled.std(axis=0, ddof=1).tolist(),
            "q05": np.quantile(pooled, 0.05, axis=0).tolist(),
            "q95": np.quantile(pooled, 0.95, axis=0).tolist(),
            "between_chain_se": be_se.tolist(),
            "bulk_ess": b_ess.tolist(),
            "tail_ess": t_ess.tolist(),
            "rhat": r.tolist(),
        }
        max_rhat = max(max_rhat, float(np.nanmax(r)))
        min_bulk = min(min_bulk, float(np.nanmin(b_ess)))

    return per_site, max_rhat, min_bulk


def write_gt_artifacts(
    out_dir: Path,
    *,
    model_name: str,
    positions: dict[str, np.ndarray],
    diag: dict[str, Any],
    timing: dict[str, float],
    generator: str,
    space: str,
    sampler_config: dict[str, Any],
    seeds: dict[str, Any],
    az_method: dict[str, str] | None = None,
    reproduced_from: dict[str, Any] | None = None,
    extra_provenance: dict[str, Any] | None = None,
    total_wall: float | None = None,
) -> tuple[Path, Path]:
    """Write ``draws.npz`` and ``summary_v2.json`` to ``out_dir``.

    Produces artifacts byte-shaped identically to the committed catalog GT.
    The ``schema_version`` is always ``"gt_v2_multichain"``.

    Parameters
    ----------
    out_dir
        Output directory (created if it does not exist).
    model_name
        Registry model name, e.g. ``"radon"``.
    positions
        Dict mapping site name to array shaped ``(n_chains, n_draws, *event)``.
    diag
        Diagnostics dict from the generation path (step_size, divergences, etc.).
    timing
        Dict with ``warmup``, ``sampling`` keys (float seconds; 0.0 if not applicable).
    generator
        The ``generator`` string to embed (e.g. ``"analytic_iid"``).
    space
        Coordinate space, always ``"unconstrained"``.
    sampler_config
        Sampler configuration dict to embed verbatim.
    seeds
        Seeds dict with ``master_seed`` and ``derivation`` keys.
    az_method
        Override for the ``az_method`` block. Defaults to the standard
        rank-normalized split-R̂ + ArviZ bulk/tail ESS description.
    reproduced_from
        Optional dict describing which committed GT this run reproduces
        (``timestamp_utc``, ``code_sha``).  Included in provenance for lineage.
    extra_provenance
        Additional key/value pairs merged into the provenance block.
    total_wall
        Total wall time in seconds. If None, derived from ``timing["warmup"] +
        timing["sampling"]``.

    Returns
    -------
    draws_path, summary_path
        Paths to the written files.
    """
    import blackjax
    import jax

    import tuningfork

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- persist draws first (pre-postprocessing, per smoke-E2E discipline) ---
    # Explicitly convert to plain numpy arrays before saving.  JAX DeviceArrays
    # saved directly via np.savez are stored as object-pickled arrays in older
    # JAX/numpy combinations; np.asarray forces concrete float/int dtype so the
    # .npz never requires allow_pickle to load.
    draws_path = out_dir / "draws.npz"
    np.savez_compressed(
        str(draws_path), **{s: np.asarray(arr) for s, arr in positions.items()}
    )

    # --- compute diagnostics ---
    per_site, max_rhat, min_bulk = compute_summary_stats(positions)

    # Determine shape from positions
    first_site = next(iter(positions))
    nc = positions[first_site].shape[0]
    nd = positions[first_site].shape[1]
    n_total = nc * nd

    is_nuts = generator not in ("analytic_iid",)
    total_div = diag.get("total_divergences", 0)
    min_ebfmi = diag.get("min_e_bfmi")

    gate_passed = (max_rhat <= 1.01) and (
        not is_nuts
        or (
            min_bulk >= 400.0
            and (total_div / n_total) <= 0.001
            and (min_ebfmi is None or min_ebfmi >= 0.3)
        )
    )

    # --- build summary ---
    if az_method is None:
        az_method = {
            "bulk_ess": "az.ess(idata, method='bulk') on raw (chain,draw) real chains",
            "tail_ess": "az.ess(idata, method='tail')",
            "rhat": "az.rhat(idata, method='rank')  # Vehtari 2021 rank-norm split-Rhat",
            "between_chain_se": "std(chain_means, ddof=1)/sqrt(n_chains)",
        }

    provenance: dict[str, Any] = {
        "tuningfork_version": tuningfork.__version__,
        "jax_version": jax.__version__,
        "blackjax_version": blackjax.__version__,
        "arviz_version": _arviz_version(),
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "device": jax.devices()[0].platform,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_sha": _git_sha(),
    }
    if reproduced_from is not None:
        provenance["reproduced_from"] = reproduced_from
    if extra_provenance:
        provenance.update(extra_provenance)

    _total = (
        total_wall
        if total_wall is not None
        else timing.get("warmup", 0.0) + timing.get("sampling", 0.0)
    )
    summary = {
        "model_name": model_name,
        "schema_version": "gt_v2_multichain",
        "generator": generator,
        "space": space,
        "n_chains": nc,
        "n_draws_per_chain": nd,
        "n_total": n_total,
        "sampler_config": sampler_config,
        "seeds": seeds,
        "az_method": az_method,
        "quality_gate": {
            "rhat_threshold": 1.01,
            "max_rhat": max_rhat,
            "min_bulk_ess": min_bulk,
            "total_divergences": total_div,
            "divergence_rate": float(total_div / n_total) if is_nuts else 0.0,
            "min_e_bfmi": min_ebfmi,
            "passed": bool(gate_passed),
        },
        "diagnostics_per_chain": diag,
        "timing_seconds": {**timing, "total": _total},
        "provenance": provenance,
        "per_site": per_site,
    }

    summary_path = out_dir / "summary_v2.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    return draws_path, summary_path


def _arviz_version() -> str:
    try:
        import arviz as az

        return az.__version__
    except ImportError:
        return "unknown"


def _git_sha() -> str:
    """Return the short HEAD SHA of the tuningfork repo, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"
