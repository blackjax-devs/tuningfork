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
"""W1 full-catalog re-validation harness.

Validates that a gate change or catalog batch does not newly break committed
recipes by running the W1/σ equivalence gate against the committed cached
draws (path A) and optionally re-generating draws on-the-fly (paths B/C).

Default mode — path-A-only (clean signal)
------------------------------------------
Only recipes with **committed cached draws** (path A) are processed.
This is the clean, unconfounded signal: the committed draws are the same
ones that received the committed gate verdict, so a W1 verdict change is
real, not a re-gen artefact.

Opt-in re-generation mode (``enable_regen=True`` / ``--regen``)
----------------------------------------------------------------
Also processes path-B/C cells (no cached draws) by re-generating draws at
``resample_n`` samples with a **deterministic per-cell seed** (stable across
Python versions and ``PYTHONHASHSEED``).

The two-stage gate is enforced on every re-generated cell:
  Stage 1 — R̂/ESS/divergences computed on the *re-generated* draws.
             If stage-1 == FAIL (R̂ ≥ 1.05, ESS < 100, or n_div ≥ 40),
             the result is reported as ``stage1_verdict=FAIL`` and W1 is
             NOT run.  A degenerate re-gen (e.g. ESS≈4.3) must not reach
             W1 and produce a spurious W1 FAIL / flip.
  Stage 2 — W1 gate; only runs when stage-1 == PASS.

The committed recipe's stage-1 verdict is **never** carried onto re-generated
draws — every re-gen gets its own fresh stage-1 check.

Stage-1 thresholds are imported from ``DEFAULT_THRESHOLDS`` in
``tuningfork.calibration._gate.constants`` — they are never copied.  Any
threshold update propagates automatically.

Path codes
----------
A   Per-recipe draws cache exists        → load + W1 gate (seconds)
B   No cache, standard MCMC              → ``run_recipe_to_idata(skip_warmup=True)``
C   No cache, MCLMC / CHEES / sidecar-IMM → ``run_recipe_to_idata(skip_warmup=False)``
SK  SMC / VI / no GT / large-nc chees   → skip (W1 N/A or infeasible on CPU)

CLI usage
---------
  # Default (path-A only — clean signal):
  JAX_PLATFORM_NAME=cpu uv run python -m tuningfork.calibration.revalidation

  # Opt-in re-gen (two-stage gate on B/C):
  JAX_PLATFORM_NAME=cpu uv run python -m tuningfork.calibration.revalidation --regen

  # Custom checkpoint:
  JAX_PLATFORM_NAME=cpu uv run python -m tuningfork.calibration.revalidation \\
      --checkpoint my_results.json

  # Via make:
  make revalidate-w1            # path-A only
  ENABLE_REGEN=1 make revalidate-w1  # include B/C cells
"""

from __future__ import annotations

__all__ = [
    "RevalidationResult",
    "_cell_regen_seed",
    "classify_recipe_path",
    "collect_eligible_cells",
    "compute_stage1_verdict",
    "process_catalog_cell",
    "run_w1_revalidation",
]

import argparse
import hashlib
import json
import os
import pathlib
import struct
import sys
import time
import traceback
import warnings

import numpy as np

# Stage-1 thresholds and classification helpers — imported from the gate so
# they stay in sync automatically (no drift from manually-copied constants).
from tuningfork.calibration._gate.bands import _classify_metric, _worst
from tuningfork.calibration._gate.constants import DEFAULT_THRESHOLDS

# ---------------------------------------------------------------------------
# Catalog location (package-relative; works for editable and built installs)
# ---------------------------------------------------------------------------

_CATALOG_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent / "catalog"
"""Default path to the per-model artifact catalog (``tuningfork/catalog/``).

Resolved relative to this module's location so it works regardless of the
caller's working directory.
"""

# ---------------------------------------------------------------------------
# Default run config
# ---------------------------------------------------------------------------

W1_B: int = int(os.environ.get("W1_B", "5000"))
"""Number of W1 bootstrap replicates.  Override via ``W1_B=500`` env for speed."""

W1_ALPHA: float = 0.05
"""Family-wise significance level for the W1/σ gate."""

W1_SEED: int = 42
"""Bootstrap seed for the W1 gate (stable across runs)."""

RESAMPLE_N: int = 500
"""Number of samples to draw per re-generated cell (paths B/C)."""

REGEN_BASE_SEED: int = 42
"""Base seed for per-cell deterministic seed derivation (paths B/C)."""

# ---------------------------------------------------------------------------
# Recipe method sets
# ---------------------------------------------------------------------------

_VI_METHODS: frozenset[str] = frozenset({"meanfield_vi", "fullrank_vi"})
_SKIP_WARMUP_METHODS: frozenset[str] = frozenset(
    {
        "dynamic_hmc",
        "nuts",
        "dmhmc",
        "mhmc",
        "hmc",
        "ghmc",
        "rmhmc",
        "barker",
        "orbital_hmc",
    }
)
_MCLMC_METHODS: frozenset[str] = frozenset(
    {"mclmc", "adjusted_mclmc", "adjusted_mclmc_dynamic"}
)
_LAPLACE_METHODS: frozenset[str] = frozenset(
    {"laplace_hmc", "laplace_dhmc", "laplace_mhmc", "laplace_dmhmc"}
)
# Warmups that require a full re-run (adapted_params hold non-serialisable callables)
_FULL_WARMUP_REQUIRED: frozenset[str] = frozenset({"chees", "meads"})
# CPU chain-count limit for chees/meads (large-nc GPU recipes → SK)
_CPU_NC_LIMIT: int = 32
# VI-warmup methods: seed-sensitive; only path A is reliable
_VI_WARMUP_METHODS: frozenset[str] = frozenset({"fullrank_vi", "meanfield_vi"})

# Canonical model order for deterministic sweep ordering
_BATCH_ORDER: list[str] = [
    "mvn_10",
    "logistic_synthetic",
    "banana",
    "eight_schools_ncp",
    "ill_cond_50",
    "german_credit",
    "lotka_volterra",
    "gmm_25",
    "irt_1pl",
    "irt_2pl",
    "radon",
    "stoch_vol",
    "horseshoe",
    "neals_funnel",
    "lgcp",
    "gp_regression",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


class RevalidationResult:
    """Summary statistics from a completed re-validation sweep.

    Attributes
    ----------
    ok_count
        Number of cells that completed processing (W1 run or stage-1 routed).
    skip_count
        Number of cells skipped (SK path, re-gen disabled, or no GT).
    error_count
        Number of cells that raised an exception.
    flip_keys
        Sorted list of recipe keys (``"model/recipe_stem"``) that flipped from
        PASS to FAIL under the W1 gate.
    stage1_fail_count
        Number of B/C cells that failed stage-1 (W1 not run).  Zero in path-A
        mode.
    results
        Full per-cell result dict keyed by ``"model/recipe_stem"``.
    """

    __slots__ = (
        "error_count",
        "flip_keys",
        "ok_count",
        "results",
        "skip_count",
        "stage1_fail_count",
    )

    def __init__(
        self,
        ok_count: int,
        skip_count: int,
        error_count: int,
        flip_keys: list[str],
        stage1_fail_count: int,
        results: dict,
    ) -> None:
        self.ok_count = ok_count
        self.skip_count = skip_count
        self.error_count = error_count
        self.flip_keys = flip_keys
        self.stage1_fail_count = stage1_fail_count
        self.results = results


# ---------------------------------------------------------------------------
# Deterministic per-cell seed
# ---------------------------------------------------------------------------


def _cell_regen_seed(key: str, base_seed: int = REGEN_BASE_SEED) -> int:
    """Deterministic per-cell seed for re-generation.

    Uses MD5 of ``regen:<base_seed>:<key>`` → 31-bit int.  Stable across
    Python versions and ``PYTHONHASHSEED`` settings.

    Parameters
    ----------
    key
        Cell key string, typically ``"<model>/<recipe_stem>"``.
    base_seed
        Base integer seed (default ``REGEN_BASE_SEED``).

    Returns
    -------
    int
        A 31-bit non-negative integer seed.
    """
    digest = hashlib.md5(f"regen:{base_seed}:{key}".encode()).digest()
    return struct.unpack("<I", digest[:4])[0] & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Stage-1 gate on re-generated draws
# ---------------------------------------------------------------------------


def compute_stage1_verdict(
    draws: dict,
    n_divergences: int | None,
) -> dict:
    """Compute stage-1 (R̂/ESS/divergences) verdict on draws.

    This gate MUST run before W1 on any re-generated draws.  A degenerate
    re-gen (e.g. ESS≈4.3 from a collapsed warmup) must be caught here and
    reported as ``stage1_verdict="FAIL"`` — it must never reach W1 and
    produce a spurious W1 FAIL / flip.

    Thresholds are imported from ``DEFAULT_THRESHOLDS`` in
    ``tuningfork.calibration._gate.constants`` and never hard-coded here.
    Any update to the gate's thresholds propagates automatically.

    Parameters
    ----------
    draws
        ``{site: np.ndarray(n_chains, n_draws[, *event_shape])}`` — samples.
        Scalar sites (2-D) are expanded to 3-D internally.
    n_divergences
        Total divergent transitions, or ``None`` when not available (MCLMC /
        info-less paths).

    Returns
    -------
    dict
        Keys: ``rhat_max``, ``min_bulk_ess``, ``n_divergences``,
        ``stage1_verdict`` (``"PASS"``, ``"REVIEW"``, or ``"FAIL"``).
    """
    from blackjax.diagnostics import ess_bulk as _bj_ess_bulk
    from blackjax.diagnostics import rhat as _bj_rhat

    rhat_vals: list[float] = []
    ess_vals: list[float] = []
    for arr in draws.values():
        a = np.asarray(arr, dtype=np.float64)
        a3 = a if a.ndim >= 3 else a[:, :, np.newaxis]
        # Suppress the "All-NaN slice encountered" RuntimeWarning that numpy
        # raises when nanmax/nanmin is applied to all-NaN arrays (e.g. from
        # zero-variance constant chains).  The NaN result is handled below by
        # _classify_metric, which correctly maps NaN → FAIL for both rhat and
        # ESS (since NaN fails all lo ≤ x < hi comparisons).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rhat_vals.append(
                float(np.nanmax(np.asarray(_bj_rhat(a3, chain_axis=0, sample_axis=1))))
            )
            ess_vals.append(
                float(
                    np.nanmin(np.asarray(_bj_ess_bulk(a3, chain_axis=0, sample_axis=1)))
                )
            )

    rhat_max: float | None = float(max(rhat_vals)) if rhat_vals else None
    min_bulk_ess: float | None = float(min(ess_vals)) if ess_vals else None

    # Worst-verdict-wins using the gate's shared classification helper and
    # thresholds (no local copies of the boundary constants).
    verdict = "PASS"
    if rhat_max is not None:
        verdict = _worst(
            verdict, _classify_metric(rhat_max, DEFAULT_THRESHOLDS["rhat_max"])
        )
    if min_bulk_ess is not None:
        verdict = _worst(
            verdict, _classify_metric(min_bulk_ess, DEFAULT_THRESHOLDS["min_bulk_ess"])
        )
    if n_divergences is not None:
        verdict = _worst(
            verdict,
            _classify_metric(n_divergences, DEFAULT_THRESHOLDS["n_divergences"]),
        )

    return {
        "rhat_max": rhat_max,
        "min_bulk_ess": min_bulk_ess,
        "n_divergences": n_divergences,
        "stage1_verdict": verdict,
    }


# ---------------------------------------------------------------------------
# GT data loading
# ---------------------------------------------------------------------------


def _load_gt_data(
    model_name: str,
    catalog_dir: pathlib.Path,
) -> tuple[dict, dict] | None:
    """Load GT draws and summary_v2 for a model.

    Returns ``(gt_draws_per_site, gt_summary_per_site)`` or ``None`` when the
    GT files are not present.

    Parameters
    ----------
    model_name
        Model directory name under ``catalog_dir``.
    catalog_dir
        Path to the catalog root (``tuningfork/catalog/``).
    """
    base = catalog_dir / model_name / "groundtruth_samples" / "blackjax"
    draws_path = base / "draws.npz"
    summ_path = base / "summary_v2.json"
    if not draws_path.exists() or not summ_path.exists():
        return None

    raw = np.load(draws_path)
    gt_draws: dict[str, np.ndarray] = {
        site: raw[site].astype(np.float64) for site in raw.files
    }
    raw.close()

    per_site = json.loads(summ_path.read_text()).get("per_site", {})
    gt_summary: dict[str, dict] = {
        site: {
            "std": np.asarray(stats["std"], dtype=np.float64),
            "bulk_ess": np.asarray(stats["bulk_ess"], dtype=np.float64),
            "tail_ess": np.asarray(stats["tail_ess"], dtype=np.float64),
        }
        for site, stats in per_site.items()
    }
    return gt_draws, gt_summary


# ---------------------------------------------------------------------------
# Recipe draw loading / re-sampling
# ---------------------------------------------------------------------------


def _load_cached_draws(recipe_path: pathlib.Path) -> dict | None:
    """Load per-recipe cached draws from ``_cache/<stem>.draws.npz``.

    Returns ``{site: np.ndarray(n_chains, n_draws[, *event])}`` or ``None``
    when no cache file exists.
    """
    cache_path = recipe_path.parent.parent / "_cache" / f"{recipe_path.stem}.draws.npz"
    if not cache_path.exists():
        return None
    raw = np.load(cache_path)
    draws = {site: raw[site].astype(np.float64) for site in raw.files}
    raw.close()
    return draws


def _resample_with_divcount(
    recipe_path: pathlib.Path,
    n_samples: int,
    *,
    skip_warmup: bool = True,
    seed: int | None = None,
) -> tuple[dict, int | None]:
    """Re-sample from a committed recipe; return ``(draws, n_divergences)``.

    Parameters
    ----------
    recipe_path
        Path to the recipe JSON.
    n_samples
        Number of post-warmup samples to draw.
    skip_warmup
        When ``True`` (path B), use stored ``step_size``/IMM.  When ``False``
        (path C), re-run the full warmup from scratch.
    seed
        RNG seed; passed as ``force_resample_config={"seed": seed}`` for
        reproducibility.

    Returns
    -------
    tuple of (draws, n_divergences)
        ``draws``: ``{site: np.ndarray(n_chains, n_draws[, *event])}``.
        ``n_divergences``: from ``idata.sample_stats``, or ``None`` when not
        available (MCLMC / info-less paths).
    """
    from tuningfork.catalog.inspect import load_recipe
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe = load_recipe(recipe_path.absolute())
    kwargs: dict = {
        "n_samples": n_samples,
        "skip_warmup": skip_warmup,
        "_suppress_print": True,
    }
    if seed is not None:
        kwargs["force_resample_config"] = {"seed": seed}

    idata = run_recipe_to_idata(recipe, **kwargs)

    post = idata.posterior
    draws: dict[str, np.ndarray] = {
        str(v): np.asarray(post[v], dtype=np.float64) for v in post.data_vars
    }

    n_div: int | None = None
    if hasattr(idata, "sample_stats"):
        ss = idata.sample_stats
        if hasattr(ss, "diverging"):
            n_div = int(np.sum(np.asarray(ss.diverging)))

    return draws, n_div


# ---------------------------------------------------------------------------
# W1 gate application
# ---------------------------------------------------------------------------


def _apply_w1_gate(
    draws: dict,
    gt_summary_per_site: dict,
    gt_draws_per_site: dict,
    *,
    b: int = W1_B,
    alpha: float = W1_ALPHA,
    seed: int = W1_SEED,
):
    """Run ``compute_w1_realm`` on draws vs GT.

    Returns a ``W1RealmResult`` namedtuple-like object from the gate.
    """
    from tuningfork.calibration._gate.w1_realm import compute_w1_realm

    return compute_w1_realm(
        samples=draws,
        ground_truth_summaries=gt_summary_per_site,
        gt_draws=gt_draws_per_site,
        B=b,
        alpha=alpha,
        seed=seed,
        multichain=True,
    )


# ---------------------------------------------------------------------------
# Path classification
# ---------------------------------------------------------------------------


def classify_recipe_path(recipe_path: pathlib.Path) -> str:
    """Classify a recipe file into a re-validation path code.

    Inspects the recipe JSON and filesystem to determine how to process the
    cell during re-validation.

    Parameters
    ----------
    recipe_path
        Path to a recipe ``.json`` file.

    Returns
    -------
    str
        One of:

        ``"A"``
            Cached draws exist — load and run W1.
        ``"B"``
            No cache; can skip warmup (standard MCMC methods).  Re-gen
            with ``skip_warmup=True``.  Requires ``enable_regen=True``.
        ``"C"``
            No cache; requires full warmup re-run (MCLMC / CHEES / sidecar-IMM
            or small-nc CHEES/MEADS recipes).  Requires ``enable_regen=True``.
        ``"SK"``
            Skip — SMC, VI, no GT, large-nc CHEES/MEADS (GPU-scale), or
            Laplace method without cached draws.
    """
    if "failed__" in recipe_path.name:
        return "SK"

    try:
        d = json.loads(recipe_path.read_text())
    except Exception:
        return "SK"

    ae = d.get("gate_evidence", {}).get("auto", {})
    if ae.get("verdict") != "PASS":
        return "SK"

    bm = d.get("base_method_name", "")
    if not bm:
        # SMC recipe — base_method_name absent or empty
        return "SK"
    if bm in _VI_METHODS:
        return "SK"

    model = recipe_path.parent.parent.name
    catalog_dir = recipe_path.parent.parent.parent  # <catalog>/<model>/recipes/<file>
    gt_base = catalog_dir / model / "groundtruth_samples" / "blackjax"
    if (
        not (gt_base / "draws.npz").exists()
        or not (gt_base / "summary_v2.json").exists()
    ):
        return "SK"

    cache_path = recipe_path.parent.parent / "_cache" / f"{recipe_path.stem}.draws.npz"
    if cache_path.exists():
        return "A"

    # Laplace methods without cached draws: MAP-init sensitive → skip
    if bm in _LAPLACE_METHODS:
        return "SK"

    # Warmup from recipe stem: ``<level>__<method>__<warmup>[__extra]``
    parts = recipe_path.stem.split("__")
    warmup_name = parts[2] if len(parts) >= 3 else ""

    if warmup_name in _FULL_WARMUP_REQUIRED:
        nc = d.get("calibration_budget", {}).get("num_chains", 0)
        if nc > _CPU_NC_LIMIT:
            return "SK"
        return "C"

    if warmup_name in _VI_WARMUP_METHODS:
        # VI-warmup cells: seed-sensitive re-runs; only path A is reliable
        return "SK"

    bmp = d.get("base_method_params", {})
    imm = bmp.get("inverse_mass_matrix") or d.get("inverse_mass_matrix")
    if imm == "sidecar":
        return "C"

    if bm in _SKIP_WARMUP_METHODS:
        return "B"
    if bm in _MCLMC_METHODS:
        return "C"
    return "SK"


# ---------------------------------------------------------------------------
# Catalog cell collection
# ---------------------------------------------------------------------------


def collect_eligible_cells(
    catalog_dir: pathlib.Path | None = None,
) -> list[tuple[str, pathlib.Path, str]]:
    """Return ``(model, recipe_path, path_code)`` for all W1-eligible PASS cells.

    Scans ``<catalog_dir>/*/recipes/*.json``, skipping ``failed__*`` files and
    cells that are not eligible for W1 re-validation.

    ``path_code`` is one of ``"A"``, ``"B"``, ``"C"``, or ``"SK"`` — see
    :func:`classify_recipe_path` for semantics.

    Parameters
    ----------
    catalog_dir
        Path to the catalog root.  Defaults to ``_CATALOG_DIR`` (the
        package-relative catalog).
    """
    if catalog_dir is None:
        catalog_dir = _CATALOG_DIR

    recipe_map: dict[str, list[pathlib.Path]] = {}
    for p in catalog_dir.glob("*/recipes/*.json"):
        if "failed__" in p.name:
            continue
        model = p.parent.parent.name
        recipe_map.setdefault(model, []).append(p)

    seen_models = set(recipe_map.keys())
    ordered: list[tuple[str, pathlib.Path, str]] = []

    for model in _BATCH_ORDER:
        for p in sorted(recipe_map.get(model, [])):
            code = classify_recipe_path(p)
            ordered.append((model, p, code))

    for model in sorted(seen_models - set(_BATCH_ORDER)):
        for p in sorted(recipe_map.get(model, [])):
            code = classify_recipe_path(p)
            ordered.append((model, p, code))

    return ordered


# ---------------------------------------------------------------------------
# Single-cell processor
# ---------------------------------------------------------------------------


def process_catalog_cell(
    model: str,
    recipe_path: pathlib.Path,
    path_code: str,
    *,
    enable_regen: bool = False,
    catalog_dir: pathlib.Path | None = None,
    w1_b: int = W1_B,
    w1_alpha: float = W1_ALPHA,
    w1_seed: int = W1_SEED,
    resample_n: int = RESAMPLE_N,
    regen_base_seed: int = REGEN_BASE_SEED,
) -> dict:
    """Process one catalog cell and return a result dict.

    Parameters
    ----------
    model
        Model name (used for GT data loading).
    recipe_path
        Path to the recipe ``.json`` file.
    path_code
        Classification from :func:`classify_recipe_path` (``"A"``, ``"B"``,
        ``"C"``, or ``"SK"``).
    enable_regen
        When ``False`` (default), path-B/C cells are returned as SKIP.
        When ``True``, path-B/C cells are re-generated and the full two-stage
        gate (stage-1 R̂/ESS/div → W1) is applied.
    catalog_dir
        Catalog root path.  Defaults to ``_CATALOG_DIR``.
    w1_b, w1_alpha, w1_seed
        W1 gate parameters.
    resample_n
        Samples to draw per re-generated cell.
    regen_base_seed
        Base seed for per-cell seed derivation.

    Returns
    -------
    dict
        Result dict with ``"status"`` key (``"OK"``, ``"SKIP"``, or
        ``"ERROR"``) and additional fields depending on the path taken.

        For path-A cells that reach W1:
            ``baseline_verdict``, ``w1_verdict``, ``flip``,
            ``max_w1_sigma``, ``floor_of_max``, ``frac_failing_dims``,
            ``tau_frac``, ``max_prong_verdict``, ``frac_prong_verdict``,
            ``n_dims``, ``n_heavy_tail_dims``, ``elapsed_s``.

        For B/C cells that fail stage-1 (W1 not run):
            ``stage1_verdict``, ``regen_rhat_max``, ``regen_min_bulk_ess``,
            ``regen_n_divergences``, ``w1_verdict=None``, ``flip=False``.
    """
    if catalog_dir is None:
        catalog_dir = _CATALOG_DIR

    if path_code == "SK":
        return {
            "status": "SKIP",
            "path_code": "SK",
            "reason": "SMC / VI / no GT / unknown",
        }

    if path_code in ("B", "C") and not enable_regen:
        return {
            "status": "SKIP",
            "path_code": path_code,
            "reason": "re-gen disabled; run with --regen to process B/C cells",
        }

    gt_data = _load_gt_data(model, catalog_dir)
    if gt_data is None:
        return {"status": "SKIP", "path_code": "SK", "reason": "GT data not found"}
    gt_draws_per_site, gt_summary_per_site = gt_data

    t0 = time.perf_counter()
    stage1_result: dict | None = None

    try:
        if path_code == "A":
            draws = _load_cached_draws(recipe_path)
            if draws is None:
                return {
                    "status": "ERROR",
                    "path_code": "A",
                    "error": "cache file disappeared",
                }
        elif path_code in ("B", "C"):
            cell_key = f"{model}/{recipe_path.stem}"
            regen_seed = _cell_regen_seed(cell_key, regen_base_seed)

            draws, n_div_regen = _resample_with_divcount(
                recipe_path,
                resample_n,
                skip_warmup=(path_code == "B"),
                seed=regen_seed,
            )

            # Stage-1 gate on the re-generated draws.  The committed recipe's
            # stage-1 verdict is NEVER carried over — every re-gen gets its own
            # fresh check.  A degenerate re-gen must not reach W1.
            stage1_result = compute_stage1_verdict(draws, n_div_regen)
            resample_elapsed = time.perf_counter() - t0

            if stage1_result["stage1_verdict"] != "PASS":
                return {
                    "status": "OK",
                    "path_code": path_code,
                    "stage1_verdict": stage1_result["stage1_verdict"],
                    "regen_rhat_max": stage1_result["rhat_max"],
                    "regen_min_bulk_ess": stage1_result["min_bulk_ess"],
                    "regen_n_divergences": stage1_result["n_divergences"],
                    "w1_verdict": None,
                    "flip": False,
                    "resample_elapsed_s": round(resample_elapsed, 1),
                    "elapsed_s": round(resample_elapsed, 1),
                    "note": (
                        f"Routed to stage-1 {stage1_result['stage1_verdict']} "
                        f"(rhat={stage1_result['rhat_max']}, "
                        f"ess={stage1_result['min_bulk_ess']}, "
                        f"ndiv={stage1_result['n_divergences']}); W1 not run"
                    ),
                }
        else:
            return {"status": "SKIP", "path_code": path_code, "reason": "unknown code"}

    except Exception as exc:
        return {
            "status": "ERROR",
            "path_code": path_code,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }

    resample_elapsed = time.perf_counter() - t0 if path_code in ("B", "C") else 0.0

    t_w1 = time.perf_counter()
    try:
        w1_result = _apply_w1_gate(
            draws,
            gt_summary_per_site,
            gt_draws_per_site,
            b=w1_b,
            alpha=w1_alpha,
            seed=w1_seed,
        )
    except Exception as exc:
        return {
            "status": "ERROR",
            "path_code": path_code,
            "error": f"W1 gate failed: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
    w1_elapsed = time.perf_counter() - t_w1
    total_elapsed = time.perf_counter() - t0

    recipe_data = json.loads(recipe_path.read_text())
    baseline_verdict = (
        recipe_data.get("gate_evidence", {}).get("auto", {}).get("verdict", "UNKNOWN")
    )
    flip = baseline_verdict == "PASS" and w1_result.verdict == "FAIL"

    result: dict = {
        "status": "OK",
        "path_code": path_code,
        "baseline_verdict": baseline_verdict,
        "w1_verdict": w1_result.verdict,
        "flip": flip,
        "max_w1_sigma": float(w1_result.max_w1_sigma),
        "floor_of_max": float(w1_result.floor_of_max),
        "frac_failing_dims": float(w1_result.frac_failing_dims),
        "tau_frac": (
            float(w1_result.tau_frac) if not np.isnan(w1_result.tau_frac) else None
        ),
        "max_prong_verdict": w1_result.max_prong_verdict,
        "frac_prong_verdict": w1_result.frac_prong_verdict,
        "n_dims": w1_result.n_dims,
        "n_heavy_tail_dims": w1_result.n_heavy_tail_dims,
        "resample_elapsed_s": (
            round(resample_elapsed, 1) if path_code in ("B", "C") else None
        ),
        "w1_elapsed_s": round(w1_elapsed, 1),
        "elapsed_s": round(total_elapsed, 1),
    }

    if stage1_result is not None:
        result["regen_rhat_max"] = stage1_result["rhat_max"]
        result["regen_min_bulk_ess"] = stage1_result["min_bulk_ess"]
        result["regen_n_divergences"] = stage1_result["n_divergences"]
        result["stage1_verdict"] = stage1_result["stage1_verdict"]

    return result


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------


def run_w1_revalidation(
    *,
    catalog_dir: pathlib.Path | None = None,
    enable_regen: bool = False,
    checkpoint_path: pathlib.Path | None = None,
    w1_b: int = W1_B,
    w1_alpha: float = W1_ALPHA,
    w1_seed: int = W1_SEED,
    resample_n: int = RESAMPLE_N,
    regen_base_seed: int = REGEN_BASE_SEED,
    verbose: bool = True,
) -> RevalidationResult:
    """Run the W1 catalog re-validation sweep.

    Parameters
    ----------
    catalog_dir
        Catalog root path.  Defaults to ``_CATALOG_DIR``.
    enable_regen
        Include path-B/C cells with two-stage gate.  Default ``False``
        (path-A-only — clean signal).
    checkpoint_path
        JSON file for checkpointing.  When the file exists, already-completed
        cells are skipped.  Updated after every cell.  Defaults to
        ``experiments/w1_revalidation_results.json`` in the current directory.
    w1_b, w1_alpha, w1_seed
        W1 gate parameters.
    resample_n
        Samples per re-generated cell.
    regen_base_seed
        Base seed for per-cell seed derivation.
    verbose
        When ``True``, print per-cell progress to stdout.

    Returns
    -------
    RevalidationResult
        Summary statistics plus the full per-cell ``results`` dict.
    """
    if catalog_dir is None:
        catalog_dir = _CATALOG_DIR
    if checkpoint_path is None:
        checkpoint_path = pathlib.Path("experiments") / "w1_revalidation_results.json"

    cells = collect_eligible_cells(catalog_dir)

    if not enable_regen:
        active_cells = [(m, p, c) for m, p, c in cells if c == "A"]
        mode_label = "path-A-only (default; use --regen to include B/C)"
    else:
        active_cells = cells
        mode_label = "full catalog (path-A + B/C with two-stage gate)"

    total = len(active_cells)
    n_a = sum(1 for _, _, c in active_cells if c == "A")
    n_bc = sum(1 for _, _, c in active_cells if c in ("B", "C"))

    if verbose:
        print(f"W1 catalog re-validation [{mode_label}]")
        print(f"  Cells: {total} total ({n_a} path-A, {n_bc} path-B/C)")
        print(f"  W1_B={w1_b}  alpha={w1_alpha}  seed={w1_seed}")
        if enable_regen:
            print(f"  RESAMPLE_N={resample_n}  REGEN_BASE_SEED={regen_base_seed}")
        print(f"  Checkpoint: {checkpoint_path}")

    results: dict = {}
    if checkpoint_path.exists():
        results = json.loads(checkpoint_path.read_text())
        done = sum(
            1
            for k, v in results.items()
            if k != "_meta" and v.get("status") in ("OK", "SKIP")
        )
        if verbose:
            print(f"  Resuming: {done}/{total} already done\n")

    results["_meta"] = {
        "W1_B": w1_b,
        "W1_ALPHA": w1_alpha,
        "W1_SEED": w1_seed,
        "RESAMPLE_N": resample_n,
        "enable_regen": enable_regen,
        "REGEN_BASE_SEED": regen_base_seed if enable_regen else None,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(results, indent=2) + "\n")

    sweep_start = time.perf_counter()
    flips: list[str] = []

    for i, (model, recipe_path, path_code) in enumerate(active_cells, 1):
        key = f"{model}/{recipe_path.stem}"

        if key in results and results[key].get("status") in ("OK", "SKIP"):
            r = results[key]
            flip = r.get("flip", False)
            w1v = r.get("w1_verdict") or r.get("stage1_verdict") or "?"
            if verbose:
                print(f"[{i:3d}/{total}] RESUME ({w1v}) [{path_code}]: {key}")
            if flip:
                flips.append(key)
            continue

        if verbose:
            print(f"\n[{i:3d}/{total}] {key} [{path_code}]", flush=True)

        result = process_catalog_cell(
            model,
            recipe_path,
            path_code,
            enable_regen=enable_regen,
            catalog_dir=catalog_dir,
            w1_b=w1_b,
            w1_alpha=w1_alpha,
            w1_seed=w1_seed,
            resample_n=resample_n,
            regen_base_seed=regen_base_seed,
        )
        results[key] = result
        checkpoint_path.write_text(json.dumps(results, indent=2) + "\n")
        sys.stdout.flush()

        if verbose:
            _print_cell_result(key, result, path_code, flips)

    total_elapsed = time.perf_counter() - sweep_start
    ok_count = sum(
        1 for k, v in results.items() if k != "_meta" and v.get("status") == "OK"
    )
    skip_count = sum(
        1 for k, v in results.items() if k != "_meta" and v.get("status") == "SKIP"
    )
    error_count = sum(
        1 for k, v in results.items() if k != "_meta" and v.get("status") == "ERROR"
    )
    stage1_fail_count = sum(
        1
        for k, v in results.items()
        if k != "_meta"
        and v.get("status") == "OK"
        and v.get("stage1_verdict") in ("FAIL", "REVIEW")
        and v.get("w1_verdict") is None
    )

    if verbose:
        _print_sweep_summary(
            total_elapsed,
            ok_count,
            skip_count,
            error_count,
            stage1_fail_count,
            flips,
            results,
            enable_regen=enable_regen,
            checkpoint_path=checkpoint_path,
        )

    return RevalidationResult(
        ok_count=ok_count,
        skip_count=skip_count,
        error_count=error_count,
        flip_keys=sorted(flips),
        stage1_fail_count=stage1_fail_count,
        results=results,
    )


# ---------------------------------------------------------------------------
# Console output helpers
# ---------------------------------------------------------------------------

_SEP = "=" * 60


def _print_cell_result(
    key: str, result: dict, path_code: str, flips: list[str]
) -> None:
    """Print a one-line summary for a single cell result (modifies ``flips``)."""
    status = result.get("status", "?")
    if status == "ERROR":
        print(f"  => ERROR: {result.get('error', '')[:120]}")
    elif status == "SKIP":
        print(f"  => SKIP ({result.get('reason', '')})")
    else:
        flip = result.get("flip", False)
        w1v = result.get("w1_verdict")
        s1v = result.get("stage1_verdict")
        mxw = result.get("max_w1_sigma")
        fom = result.get("floor_of_max")
        elapsed = result.get("elapsed_s", 0)
        if w1v is None:
            rhat_s = (
                f"{result.get('regen_rhat_max'):.4f}"
                if result.get("regen_rhat_max") is not None
                else "n/a"
            )
            ess_s = (
                f"{result.get('regen_min_bulk_ess'):.1f}"
                if result.get("regen_min_bulk_ess") is not None
                else "n/a"
            )
            print(
                f"  => STAGE1_{s1v} (rhat={rhat_s}, ess={ess_s}) "
                f"[{path_code}] {elapsed:.0f}s — W1 not run"
            )
        else:
            mxw_s = f"{mxw:.4f}" if mxw is not None else "nan"
            fom_s = f"{fom:.4f}" if fom is not None else "nan"
            flip_s = "  *** FLIP ***" if flip else ""
            print(
                f"  => W1={w1v} (max_w1σ={mxw_s} vs floor={fom_s}) "
                f"[{path_code}] {elapsed:.0f}s{flip_s}"
            )
            if flip:
                flips.append(key)


def _print_sweep_summary(
    total_elapsed: float,
    ok_count: int,
    skip_count: int,
    error_count: int,
    stage1_fail_count: int,
    flips: list[str],
    results: dict,
    *,
    enable_regen: bool,
    checkpoint_path: pathlib.Path,
) -> None:
    """Print the sweep completion summary block."""
    pass_w1 = sum(
        1
        for k, v in results.items()
        if k != "_meta"
        and v.get("status") == "OK"
        and v.get("w1_verdict") in ("PASS", "SKIP")
    )
    fail_w1 = sum(
        1
        for k, v in results.items()
        if k != "_meta" and v.get("status") == "OK" and v.get("w1_verdict") == "FAIL"
    )

    print(f"\n{_SEP}")
    print("SWEEP COMPLETE")
    print(f"Total wall: {total_elapsed / 60:.1f} min")
    print(f"  OK={ok_count}  SKIP={skip_count}  ERROR={error_count}")
    print(f"  W1-PASS={pass_w1}  W1-FAIL={fail_w1}  FLIPS={len(flips)}")
    if enable_regen and stage1_fail_count > 0:
        print(f"  STAGE1-FAIL/REVIEW={stage1_fail_count} (routed before W1; correct)")

    if flips:
        print(
            f"\n*** {len(flips)} FLIP(S): currently-PASS cells newly FAILed by W1 gate ***"
        )
        for k in flips:
            r = results.get(k, {})
            mxw = r.get("max_w1_sigma")
            fom = r.get("floor_of_max")
            mxw_s = f"{mxw:.4f}" if mxw is not None else "n/a"
            fom_s = f"{fom:.4f}" if fom is not None else "n/a"
            print(f"  {k}: max_w1σ={mxw_s} floor={fom_s}")
        print("STOP: Report to TL for diagnosis. Do not remediate.")
    else:
        print("\nGO: 0 flips — no currently-PASS cell newly FAILed from W1 gate.")

    print(f"\nCheckpoint: {checkpoint_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--regen",
        action="store_true",
        default=bool(int(os.environ.get("ENABLE_REGEN", "0"))),
        help=(
            "Enable opt-in re-generation of path-B/C draws. "
            "Applies full two-stage gate: stage-1 (R̂/ESS/div) on re-gen draws, "
            "then W1 only if stage-1 PASS. "
            "Degenerate re-gens are routed to stage-1 FAIL, not W1 flips. "
            "Also settable via ENABLE_REGEN=1 env var."
        ),
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to the JSON checkpoint file. "
            "Default: experiments/w1_revalidation_results.json in cwd."
        ),
    )
    p.add_argument(
        "--w1-b",
        type=int,
        default=W1_B,
        help=f"W1 bootstrap replicates (default {W1_B}; override via W1_B env).",
    )
    return p.parse_args()


def main() -> None:
    """CLI entry point: ``python -m tuningfork.calibration.revalidation``."""
    args = _parse_args()
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    checkpoint_path = (
        pathlib.Path(args.checkpoint)
        if args.checkpoint
        else pathlib.Path("experiments") / "w1_revalidation_results.json"
    )

    result = run_w1_revalidation(
        enable_regen=args.regen,
        checkpoint_path=checkpoint_path,
        w1_b=args.w1_b,
        verbose=True,
    )

    sys.exit(1 if result.error_count > 0 else 0)


if __name__ == "__main__":
    main()
