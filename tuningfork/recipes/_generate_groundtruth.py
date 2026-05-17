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
"""Generate ground-truth samples + recipes for all NUTS-path posteriors.

For each model in MODELS:

  * Analytic-path: call get_reference_draws to populate the cache. No recipe
    emitted (the analytic_sampler IS the ground truth).
  * NUTS-path: call get_reference_draws at production settings (n=40_000,
    n_warmup=5_000, n_chunks=4, target_acceptance=0.80) to populate the
    cache; read back metadata + adaptation; build Recipe.from_groundtruth_run
    and save to inference/recipes/starter/<model>/groundtruth__nuts__stan_window.json.
    For high-dim models (IMM.size > 50), write IMM sidecar.

Wall-time on CPU: ~30s for the 5 analytic models; ~5-15min each for the 9
NUTS-path models; ~90min total sequential.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from tuningfork.calibration.certify_reference import CertificationError
from tuningfork.model import MODELS
from tuningfork.model._base import ReferenceMethod
from tuningfork.recipes._base import Recipe
from tuningfork.reference._io import get_adaptation_params, get_reference_draws

if TYPE_CHECKING:
    from collections.abc import Callable

    from tuningfork.calibration.certify_reference import PreAdaptedWarmup
    from tuningfork.model._base import Posterior

__all__ = ["generate_groundtruth_recipe", "sweep_all"]

# Default recipe output root (relative to this file's repo).
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_DEFAULT_RECIPE_ROOT = _REPO_ROOT / "tuningfork" / "inference" / "recipes" / "starter"


def generate_groundtruth_recipe(
    entry: Posterior,
    *,
    seed: int = 0,
    n_samples: int = 40_000,
    n_warmup: int = 5_000,
    n_chunks: int = 4,
    target_acceptance: float = 0.80,
    max_num_doublings: int = 10,
    cache_dir: Path | None = None,
    recipe_root: Path | None = None,
    pre_adapted: PreAdaptedWarmup | None = None,
    checkpoint_dir: Path | None = None,
    validate_warmup_fn: Callable | None = None,
) -> Recipe | None:
    """Generate ground-truth for one model. Returns Recipe or None (analytic).

    For analytic-path models: populates the reference cache (draws, summaries,
    metadata) and returns None — the analytic sampler IS the ground truth, so
    no recipe is emitted.

    For NUTS-path models: populates the reference cache then reads back
    CertificationResult + AdaptationParams, builds a GROUNDTRUTH Recipe, saves
    it to the recipe root, and returns the Recipe. If IMM.size > 50, writes an
    IMM sidecar and attaches the path via dataclasses.replace.

    Parameters
    ----------
    entry
        Posterior registry entry.
    seed
        RNG seed for the reference run (ignored on cache hit).
    n_samples
        Number of post-warmup samples (NUTS path only).
    n_warmup
        Number of warmup steps (NUTS path only).
    n_chunks
        Number of chunks for split-R̂ certification (NUTS path only).
    target_acceptance
        Target acceptance rate for dual averaging (NUTS path only).
    cache_dir
        Override the reference cache directory.
    recipe_root
        Override the recipe output root. Defaults to
        ``<repo>/tuningfork/recipes/starter``.

    Returns
    -------
    Recipe or None
        GROUNDTRUTH Recipe for NUTS-path models; None for analytic-path models.

    Side effects
    ------------
    Populates ``reference/<name>/{draws.npz,summary.json,metadata.json}``;
    for NUTS-path also populates ``reference/<name>/adaptation.json`` and
    saves recipe JSON (and optional IMM sidecar) under ``recipe_root/<name>/``.

    Raises
    ------
    CertificationError
        If the NUTS-path reference run fails the certification gate. The caller
        (sweep_all) catches this and continues; raising allows single-model
        invocations to surface the error.
    """
    effective_recipe_root = recipe_root or _DEFAULT_RECIPE_ROOT
    rng_key = jax.random.key(seed)

    t0 = time.perf_counter()
    get_reference_draws(
        entry,
        n=n_samples,
        rng_key=rng_key,
        cache_dir=cache_dir,
        n_warmup=n_warmup,
        n_chunks=n_chunks,
        target_acceptance=target_acceptance,
        max_num_doublings=max_num_doublings,
        pre_adapted=pre_adapted,
        checkpoint_dir=checkpoint_dir,
        validate_warmup_fn=validate_warmup_fn,
    )
    wall_seconds = time.perf_counter() - t0

    if entry.reference_method == ReferenceMethod.ANALYTIC:
        # Analytic path: cache populated, no recipe emitted.
        return None

    # NUTS path: read back adaptation params and build Recipe
    adaptation = get_adaptation_params(entry, cache_dir=cache_dir)

    # Read cert diagnostics from the metadata stamp written by get_reference_draws
    from tuningfork.reference._io import _load_metadata, _resolve_cache_dir

    effective_dir = _resolve_cache_dir(cache_dir)
    meta = _load_metadata(entry.name, effective_dir)
    if meta is None:
        raise RuntimeError(
            f"metadata missing for {entry.name!r} after get_reference_draws; "
            "this is unexpected"
        )
    cert_data = meta.get("certification", {})

    # Build a lightweight CertificationResult-like object from the metadata stamp
    from tuningfork.calibration.certify_reference import CertificationResult

    cert = CertificationResult(
        passed=cert_data.get("passed", False),
        split_rhat_max=float(cert_data.get("split_rhat_max", float("nan"))),
        min_chunk_bulk_ess=float(cert_data.get("min_chunk_bulk_ess", 0.0)),
        num_divergences=int(cert_data.get("num_divergences", 0)),
        e_bfmi=float(cert_data.get("e_bfmi", float("nan"))),
    )

    imm = adaptation.inverse_mass_matrix
    imm_np = np.asarray(imm)
    large_imm = imm_np.size > 50

    recipe = Recipe.from_groundtruth_run(
        entry,
        cert=cert,
        adaptation=adaptation,
        wall_seconds=wall_seconds,
        tuning_seed=seed,
        n_warmup=n_warmup,
        n_samples=n_samples,
        n_chunks=n_chunks,
        target_acceptance=target_acceptance,
        max_num_doublings=max_num_doublings,
    )

    # Handle IMM sidecar for large models
    if large_imm:
        sidecar_rel = recipe.save_imm_sidecar(effective_recipe_root, imm)
        recipe = dataclasses.replace(recipe, inverse_mass_matrix_path=sidecar_rel)

    # Save recipe JSON
    recipe.save(effective_recipe_root)

    return recipe


def sweep_all(
    *,
    seed: int = 0,
    n_samples: int = 40_000,
    n_warmup: int = 5_000,
    n_chunks: int = 4,
    target_acceptance: float = 0.80,
    models: list[str] | None = None,
    cache_dir: Path | None = None,
    recipe_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run generate_groundtruth_recipe across MODELS. Returns per-model summary.

    Parameters
    ----------
    seed
        RNG seed forwarded to generate_groundtruth_recipe for each model.
    n_samples
        Number of post-warmup samples (NUTS path only).
    n_warmup
        Number of warmup steps (NUTS path only).
    n_chunks
        Number of chunks for split-R̂ certification (NUTS path only).
    target_acceptance
        Target acceptance rate for dual averaging (NUTS path only).
    models
        List of model names to process. Default: all 14 models in MODELS.
    cache_dir
        Override the reference cache directory.
    recipe_root
        Override the recipe output root.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-model summary dict with keys:
        ``passed`` (bool), ``wall_seconds`` (float), ``generator``
        (``"analytic"`` or ``"nuts"``), ``cert_diagnostics`` (dict or None),
        ``recipe_path`` (str or None).

    Notes
    -----
    If a single model's NUTS certification fails (CertificationError), the
    summary records ``passed=False`` and the sweep continues. The caller decides
    whether to retry at higher n_samples.
    """
    model_names = models if models is not None else list(MODELS.keys())
    results: dict[str, dict[str, Any]] = {}

    # Strictly sequential per the user-policy + decision doc § 4 (no jax.pmap
    # across cells; within-model JAX multi-core is fine). A single NUTS chain
    # saturates the available cores; cross-model parallelism thrashes.
    for name in model_names:
        entry = MODELS[name]
        generator = entry.reference_method.value  # "analytic" or "nuts"
        summary: dict[str, Any] = {
            "passed": False,
            "wall_seconds": 0.0,
            "generator": generator,
            "cert_diagnostics": None,
            "recipe_path": None,
        }

        t0 = time.perf_counter()
        try:
            recipe = generate_groundtruth_recipe(
                entry,
                seed=seed,
                n_samples=n_samples,
                n_warmup=n_warmup,
                n_chunks=n_chunks,
                target_acceptance=target_acceptance,
                cache_dir=cache_dir,
                recipe_root=recipe_root,
            )
            summary["wall_seconds"] = time.perf_counter() - t0
            summary["passed"] = True

            if recipe is not None:
                auto = recipe.gate_evidence["auto"]
                summary["cert_diagnostics"] = {
                    "rhat_max": auto.get("rhat_max"),
                    "min_bulk_ess": auto.get("min_bulk_ess"),
                    "n_divergences": auto.get("n_divergences"),
                }
                # Determine recipe path
                effective_root = recipe_root or _DEFAULT_RECIPE_ROOT
                recipe_filename = (
                    f"{recipe.effort.value}__{recipe.base_method_name}"
                    f"__{recipe.warmup_name}.json"
                )
                summary["recipe_path"] = str(effective_root / name / recipe_filename)
            else:
                # Analytic: no cert diagnostics, no recipe file
                summary["cert_diagnostics"] = None
                summary["recipe_path"] = None

        except CertificationError as exc:
            summary["wall_seconds"] = time.perf_counter() - t0
            summary["passed"] = False
            cert = exc.cert
            summary["cert_diagnostics"] = {
                "rhat_max": cert.split_rhat_max,
                "min_bulk_ess": cert.min_chunk_bulk_ess,
                "n_divergences": cert.num_divergences,
            }
            print(
                f"[WARN] {name}: certification FAILED — "
                f"rhat={cert.split_rhat_max:.4f}, "
                f"min_ess={cert.min_chunk_bulk_ess:.1f}, "
                f"n_div={cert.num_divergences}"
            )
            sys.stdout.flush()
        except Exception as exc:  # noqa: BLE001
            summary["wall_seconds"] = time.perf_counter() - t0
            summary["passed"] = False
            print(f"[ERROR] {name}: unexpected error — {exc!r}")
            sys.stdout.flush()
        else:
            # Success path — print a brief per-model passed line so progress
            # is visible in the log (PYTHONUNBUFFERED=1 + explicit flush is
            # belt-and-suspenders per decision doc § 6).
            cert_d = summary.get("cert_diagnostics") or {}
            rhat = cert_d.get("rhat_max")
            min_ess = cert_d.get("min_bulk_ess")
            n_div = cert_d.get("n_divergences")
            if rhat is not None:
                # NUTS path
                print(
                    f"[ OK ] {name}: PASSED in {summary['wall_seconds']:.1f}s — "
                    f"rhat={rhat:.4f}, min_ess={min_ess:.1f}, n_div={n_div}"
                )
            else:
                # Analytic path — no chain diagnostics
                print(
                    f"[ OK ] {name}: PASSED (analytic) in "
                    f"{summary['wall_seconds']:.2f}s"
                )
            sys.stdout.flush()

        results[name] = summary

    # Print wall-time table
    print()
    print(
        f"{'model':<30} {'generator':<10} {'wall_s':>8} {'passed':>8} "
        f"{'rhat':>8} {'min_ess':>10}"
    )
    print("-" * 80)
    for name, s in results.items():
        cd = s.get("cert_diagnostics") or {}
        rhat_str = f"{cd['rhat_max']:.4f}" if cd.get("rhat_max") is not None else "N/A"
        ess_str = (
            f"{cd['min_bulk_ess']:.1f}" if cd.get("min_bulk_ess") is not None else "N/A"
        )
        print(
            f"{name:<30} {s['generator']:<10} {s['wall_seconds']:>8.1f} "
            f"{'PASS' if s['passed'] else 'FAIL':>8} {rhat_str:>8} {ess_str:>10}"
        )

    return results
