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
"""On-demand resampling with caching for LOW/MEDIUM recipes.

Helpers for loading generated recipe artifacts and caching draws to avoid
redundant re-runs in the catalog notebook.

Cache layout (per recipe):

    <catalog_root>/<model>/_cache/<recipe_stem>.draws.npz
    <catalog_root>/<model>/_cache/<recipe_stem>.chain_stats.npz

where ``<recipe_stem>`` is the recipe JSON filename without extension,
e.g. ``low__nuts__window_adaptation_diag_imm`` for a recipe at
``<catalog_root>/<model>/recipes/low__nuts__window_adaptation_diag_imm.json``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import arviz

    from tuningfork.recipes._base import Recipe

_CATALOG_ROOT = Path(__file__).parent

__all__ = [
    "cached_idata_for_recipe",
    "load_generated_idata",
    "prepare_pinned_replay",
    "regenerate_idata",
]


def prepare_pinned_replay(recipe: Recipe, *, catalog_root: Path | str) -> Recipe:
    """Return ``recipe`` normalized for no-warmup replay from its reference summary.

    The parsed summary values are embedded without statistical transformation,
    together with a catalog-relative path and the raw file's SHA-256 hash.
    """
    from tuningfork.recipes._base import Effort

    if recipe.effort == Effort.GROUNDTRUTH:
        raise ValueError(
            "prepare_pinned_replay does not execute Effort.GROUNDTRUTH recipes; "
            "groundtruth is load-only"
        )

    catalog_root = Path(catalog_root)
    model_name = recipe.model_name
    if (
        not isinstance(model_name, str)
        or not model_name
        or Path(model_name).name != model_name
    ):
        raise ValueError("recipe.model_name must be one catalog directory name")
    summary_path = catalog_root / model_name / "reference" / "summary.json"
    try:
        raw_summary = summary_path.read_bytes()
        summary = json.loads(raw_summary)
        means = summary["mean"]
        stds = summary["std"]
        if not isinstance(means, dict) or not isinstance(stds, dict):
            raise ValueError("mean/std must be JSON objects")
        if set(means) != set(stds) or not means:
            raise ValueError("mean/std must have matching non-empty keys")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Malformed or missing reference summary at {summary_path}: {exc}"
        ) from exc

    strategy = {
        "type": "reference_summary",
        "mean": means,
        "std": stds,
        "offsets": [0.1, -0.1, 0.05, -0.05],
        "source_path": str(summary_path.relative_to(catalog_root)),
        "source_sha256": hashlib.sha256(raw_summary).hexdigest(),
    }
    from tuningfork.recipes._base import validate_init_strategy

    try:
        validate_init_strategy(strategy)
    except ValueError as exc:
        raise ValueError(
            f"Malformed or missing reference summary at {summary_path}: {exc}"
        ) from exc
    runtime_recipe = recipe
    params = dict(recipe.base_method_params)
    executable_imm = params.get("inverse_mass_matrix")
    is_sidecar_sentinel = (
        isinstance(executable_imm, str) and executable_imm == "sidecar"
    )
    sidecar_path_declared = recipe.inverse_mass_matrix_path is not None
    # An explicit inline value is authoritative.  A declared path remains
    # provenance, but must never silently replace that executable value.
    if is_sidecar_sentinel and not sidecar_path_declared:
        raise ValueError(
            "Pinned replay sidecar sentinel requires inverse_mass_matrix_path"
        )
    if sidecar_path_declared and (executable_imm is None or is_sidecar_sentinel):
        if not recipe.inverse_mass_matrix_path:
            raise ValueError(
                "Pinned replay sidecar sentinel requires inverse_mass_matrix_path"
            )
        root = catalog_root.resolve()
        relative_path = Path(recipe.inverse_mass_matrix_path)
        if relative_path.is_absolute():
            raise ValueError(
                "Pinned replay inverse_mass_matrix_path must be relative to catalog_root"
            )
        sidecar_path = (root / relative_path).resolve()
        try:
            sidecar_rel = sidecar_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "Pinned replay inverse_mass_matrix_path escapes catalog_root"
            ) from exc
        if not sidecar_path.is_file():
            raise ValueError(f"Pinned replay sidecar is missing: {sidecar_rel}")
        raw_sidecar = sidecar_path.read_bytes()
        try:
            imm = recipe.load_imm_sidecar(root)
        except Exception as exc:  # noqa: BLE001 - normalize archive failures
            raise ValueError(
                f"Malformed inverse-mass-matrix sidecar at {sidecar_rel}: {exc}"
            ) from exc
        if imm is None:
            raise ValueError(f"Pinned replay sidecar is unavailable: {sidecar_rel}")
        if hasattr(imm, "sigma") and hasattr(imm, "U") and hasattr(imm, "lam"):
            try:
                sigma = np.asarray(imm.sigma)
                basis = np.asarray(imm.U)
                lam = np.asarray(imm.lam)
                if sigma.ndim != 1 or basis.ndim != 2 or lam.ndim != 1:
                    raise ValueError("sigma/U/lam must be rank-1/rank-2/rank-1")
                if sigma.size == 0 or basis.shape[0] == 0 or lam.size == 0:
                    raise ValueError("sigma/U/lam must be non-empty")
                if basis.shape[0] != sigma.shape[0] or basis.shape[1] != lam.shape[0]:
                    raise ValueError("sigma/U/lam shapes are inconsistent")
                if lam.shape[0] > sigma.shape[0]:
                    raise ValueError("low-rank dimension exceeds parameter dimension")
                if (
                    not np.issubdtype(sigma.dtype, np.number)
                    or not np.issubdtype(basis.dtype, np.number)
                    or not np.issubdtype(lam.dtype, np.number)
                    or np.iscomplexobj(sigma)
                    or np.iscomplexobj(basis)
                    or np.iscomplexobj(lam)
                ):
                    raise TypeError("sigma/U/lam must be real numeric arrays")
                if not (
                    np.all(np.isfinite(sigma))
                    and np.all(np.isfinite(basis))
                    and np.all(np.isfinite(lam))
                ):
                    raise ValueError("sigma/U/lam contain non-finite values")
                if np.any(sigma <= 0) or np.any(lam <= 0):
                    raise ValueError("sigma and lam must be strictly positive")
                if not np.allclose(
                    basis.T @ basis,
                    np.eye(lam.shape[0]),
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    raise ValueError("U columns must be orthonormal")
                inline_imm = {
                    "type": "low_rank_inverse_mass_matrix",
                    "sigma": sigma.tolist(),
                    "U": basis.tolist(),
                    "lam": lam.tolist(),
                }
                json.dumps(inline_imm, allow_nan=False)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Malformed inverse-mass-matrix sidecar at {sidecar_rel}: {exc}"
                ) from exc
        else:
            try:
                imm_array = np.asarray(imm)
                if not np.issubdtype(imm_array.dtype, np.number) or np.iscomplexobj(
                    imm_array
                ):
                    raise TypeError(f"unsupported dtype {imm_array.dtype}")
                if not np.all(np.isfinite(imm_array)):
                    raise ValueError("contains non-finite values")
                if imm_array.ndim == 0:
                    raise ValueError("must have rank 1 or 2; scalar IMMs are invalid")
                if imm_array.ndim > 2:
                    raise ValueError("must have rank 1 (diagonal) or rank 2 (dense)")
                if imm_array.size == 0:
                    raise ValueError("must be non-empty")
                if imm_array.ndim == 1:
                    if np.any(imm_array <= 0):
                        raise ValueError("diagonal entries must be strictly positive")
                else:
                    if imm_array.shape[0] != imm_array.shape[1]:
                        raise ValueError("dense matrix must be square")
                    if not np.allclose(imm_array, imm_array.T, rtol=1e-5, atol=1e-6):
                        raise ValueError("dense matrix must be symmetric")
                    try:
                        eigenvalues = np.linalg.eigvalsh(imm_array)
                    except np.linalg.LinAlgError as exc:
                        raise ValueError(
                            "dense matrix eigendecomposition failed"
                        ) from exc
                    if not np.all(eigenvalues > 0):
                        raise ValueError("dense matrix must be positive definite")
                inline_imm = imm_array.tolist()
                json.dumps(inline_imm, allow_nan=False)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Malformed inverse-mass-matrix sidecar at {sidecar_rel}: {exc}"
                ) from exc
        params["inverse_mass_matrix"] = inline_imm
        budget = dict(runtime_recipe.calibration_budget or {})
        baked_from = budget.get("baked_from")
        if not isinstance(baked_from, dict):
            baked_from = {} if baked_from is None else {"legacy": baked_from}
        else:
            baked_from = dict(baked_from)
        digest = hashlib.sha256(raw_sidecar).hexdigest()
        # Keep all provenance keys write-once: callers may already have a
        # stronger source record from an earlier normalization pass.
        for key, value in {
            "inverse_mass_matrix": "sidecar",
            "inverse_mass_matrix_path": recipe.inverse_mass_matrix_path,
            "inverse_mass_matrix_source_path": str(sidecar_rel),
            "inverse_mass_matrix_source_sha256": digest,
        }.items():
            baked_from.setdefault(key, value)
        budget["baked_from"] = baked_from
        runtime_recipe = replace(
            runtime_recipe,
            base_method_params=params,
            calibration_budget=budget,
        )
    else:
        runtime_recipe = replace(runtime_recipe, base_method_params=params)

    return replace(runtime_recipe.normalize_pinned_replay(), init_strategy=strategy)


def cached_idata_for_recipe(
    recipe: Any,
    *,
    catalog_root: Path | str | None = None,
    force_regenerate: bool = False,
) -> arviz.InferenceData:
    """Load a cached LOW/MEDIUM recipe's draws; re-sample only on explicit opt-in.

    Default behavior (``force_regenerate=False``) — **cache-only load**:
    - Cache hit  → load from ``<catalog_root>/<model>/_cache/<recipe_stem>.draws.npz``.
    - Cache miss → raises ``FileNotFoundError`` with an actionable hint.

    Explicit re-sample (``force_regenerate=True``):
    - Skips cache check; re-runs the recipe's warmup + sampling pipeline via
      the recipe pipeline; saves to the cache; returns the InferenceData.

    Rationale: the prior default ("silent re-sample on cache miss") was a
    UX footgun — a 30+ min sampling run could be triggered by what looked
    like a cheap cache read. Re-sampling now requires explicit consent.

    For GROUNDTRUTH recipes, delegates to the standard ``load_idata`` path
    (no caching logic needed; those draws are already persisted).

    For FAILED recipes, raises RecipeFailedError (no gate-passing config to run).

    Parameters
    ----------
    recipe
        A Recipe object loaded via ``load_recipe``.
    catalog_root
        Root of the catalog directory (default: ``tuningfork/catalog/``).
    force_regenerate
        If True, ignore any cache and re-run sampling unconditionally
        (saving the result for future cache hits). Use this when the recipe
        or JAX code has changed, or to populate the cache for the first time.

    Returns
    -------
    arviz.InferenceData
        Posterior group + sample_stats group (if available).

    Raises
    ------
    FileNotFoundError
        Cache miss with ``force_regenerate=False``. Call again with
        ``force_regenerate=True`` to (re-)sample, or call
        the recipe pipeline directly.
    RecipeFailedError
        If the recipe is FAILED (no gate-passing config).
    ValueError
        If the recipe model/warmup/sampler are not in the registries.
    """
    if catalog_root is None:
        catalog_root = _CATALOG_ROOT
    else:
        catalog_root = Path(catalog_root)

    # Groundtruth is an LFS-backed artifact, not a recipe to execute.  Keep
    # this path load-only even when callers request force regeneration; in
    # particular, never create a derived cache entry for groundtruth.
    if getattr(getattr(recipe, "effort", None), "value", None) == "groundtruth":
        from tuningfork.catalog.render import load_idata

        return load_idata(recipe, cache_dir=catalog_root)

    # FAILED recipes intentionally remain non-executable through this cache
    # helper.  ``regenerate_idata`` is a separate explicit diagnostic API that
    # permits failed-config execution, but populating the ordinary cache must
    # preserve the fail-closed contract.
    if getattr(getattr(recipe, "effort", None), "value", None) == "failed":
        from tuningfork.recipes._base import RecipeFailedError

        raise RecipeFailedError(recipe)

    # Keep cache identity aligned with Recipe.save(), including baked warmup
    # provenance and variant labels.
    recipe_stem = recipe.catalog_stem()

    # Construct cache paths
    cache_dir = catalog_root / recipe.model_name / "_cache"
    draws_cache = cache_dir / f"{recipe_stem}.draws.npz"
    stats_cache = cache_dir / f"{recipe_stem}.chain_stats.npz"

    # Cache-only mode (default): hit or raise.
    if not force_regenerate:
        if draws_cache.exists():
            return _load_from_cache(draws_cache, stats_cache)
        raise FileNotFoundError(
            f"No cache for {recipe.model_name}/"
            f"{recipe.effort.value}__{recipe.base_method_name}__"
            f"{recipe.warmup_name} at {draws_cache}.\n"
            f"Call cached_idata_for_recipe(recipe, force_regenerate=True) to "
            f"sample + populate the cache, or call regenerate_idata(recipe) "
            f"directly if you don't want to persist."
        )

    # Explicit re-sample: execute the public generated program and persist its
    # verified artifact to the ordinary cache.  Resolve the recipe's configured
    # sample count rather than silently replacing it with a helper default.
    configured_samples = (
        (recipe.calibration_budget or {}).get("n_samples")
        or (recipe.warmup_params or {}).get("n_samples")
        or 1000
    )
    regeneration_options: dict[str, Any] = {
        "n_samples": int(configured_samples),
        "catalog_root": catalog_root,
    }
    regeneration_options["seed"] = recipe.tuning_seed
    idata = regenerate_idata(recipe, **regeneration_options)
    _save_to_cache(idata, draws_cache, stats_cache)
    # Stale-cache guard (issue #244): co-write a params sidecar beside the draws
    # cache so a later `make revalidate-w1` can tell whether these draws still
    # match the recipe (path-A) or were superseded by a re-emit (degrade to a
    # fresh-draws path).  Derived from the same on-disk recipe JSON the reader
    # (classify_recipe_path) inspects, so a valid cache compares equal by
    # construction.  Groundtruth caches are not W1-eligible, so skip them.
    if recipe_stem != "groundtruth":
        _write_cache_params_sidecar(
            cache_dir, catalog_root, recipe.model_name, recipe_stem
        )
    return idata


def regenerate_idata(
    recipe: Any,
    *,
    n_samples: int = 1000,
    seed: int = 20260517,
    catalog_root: Path | str | None = None,
    replay_pinned: bool = False,
) -> arviz.InferenceData:
    """Re-run a recipe's warmup + sampling pipeline and return InferenceData.

    **User-triggered only** — this function runs the full warmup + sampler and
    may take several minutes.  It is intended to be called from an explicit
    user action (e.g. a ``Run`` button click in catalog_explorer) rather than
    automatically on page load.

    Primary use-case: **FAIL recipes**.  A FAIL recipe's failure mode is
    invisible without running the config — divergent transitions, stuck chains,
    and non-mixing patterns only appear in diagnostic plots.  By re-running the
    pinned config you can visually inspect *why* it failed (trace plots,
    divergence markers, rank plots) before investigating a fix.

    Also works for PASS and REVIEW recipes, but
    ``cached_idata_for_recipe`` is the preferred path for those since it avoids
    redundant compute. GROUNDTRUTH recipes keep their existing LFS-backed load
    path and do not launch a new sampling run.

    .. note::
        The default runs the recipe's generated warmup and sampling program.
        Set ``replay_pinned=True`` only when the canonical no-warmup replay
        from the reference summary is intended; failed recipes may lack valid
        pinned parameters.

    Parameters
    ----------
    recipe
        A Recipe object loaded via ``load_recipe``.  Works for any effort level
        including FAILED.
    n_samples
        Number of post-warmup samples per chain (default 1000).  Reduce to
        ~200–400 for a quick diagnostic preview of failure modes.
    seed
        Master random seed for reproducibility. It is applied to an immutable
        copy of the recipe before code generation; the input recipe is not
        modified. Defaults to the canonical diagnostic seed.
    catalog_root
        Root of the catalog directory (default: ``tuningfork/catalog/``).
        Generated source, logs, artifacts, and receipts are retained under
        ``<catalog_root>/<model>/_cache/generated_runs``.
    replay_pinned
        If true, replace warmup with the canonical no-warmup stage and embed
        the model's reference-summary initialization plus its provenance before
        generation.  This does not mutate the input recipe.

    Returns
    -------
    arviz.InferenceData
        Posterior group + sample_stats group.  Pass to
        ``plot_recipe_diagnostics(idata, posterior_entry)`` to render trace,
        pair, and forest plots.

    Raises
    ------
    GeneratedProgramError
        If generated execution fails. The exception carries its failed
        ``LaunchResult`` and receipt path.
    RuntimeError
        If a successful execution does not expose its verified artifact.

    Examples
    --------
    Inspect a FAIL recipe in catalog_explorer::

        idata = regenerate_idata(recipe, n_samples=400, seed=42)
        figs = plot_recipe_diagnostics(idata, posterior_entry)
    """
    if not isinstance(replay_pinned, bool):
        raise TypeError("replay_pinned must be a bool")

    if catalog_root is None:
        catalog_root = _CATALOG_ROOT
    else:
        catalog_root = Path(catalog_root)

    if getattr(getattr(recipe, "effort", None), "value", None) == "groundtruth":
        from tuningfork.catalog.render import load_idata

        return load_idata(recipe, cache_dir=catalog_root)

    if replay_pinned:
        recipe = prepare_pinned_replay(recipe, catalog_root=catalog_root)

    from tuningfork.catalog.emit import execute_recipe

    run_root = catalog_root / recipe.model_name / "_cache" / "generated_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    result = execute_recipe(
        recipe,
        run_root,
        tuning_seed=seed,
        num_samples=n_samples,
    )
    if result.artifact_path is None:
        raise RuntimeError(
            "Generated recipe execution succeeded without a verified artifact"
        )
    return load_generated_idata(result.artifact_path)


def load_generated_idata(artifact_path: Path | str) -> arviz.InferenceData:
    """Load a verified generated ``.npz`` artifact into InferenceData."""
    from tuningfork.catalog.diagnostics import samples_to_idata

    posterior: dict[str, np.ndarray] = {}
    chain_stats: dict[str, np.ndarray] = {}
    with np.load(str(artifact_path), allow_pickle=False) as archive:
        for key in archive.files:
            value = np.asarray(archive[key])
            if key.startswith("_ss_"):
                stat_name = key[4:]
                if not stat_name:
                    raise ValueError(
                        "Generated artifact contains an empty statistic name"
                    )
                if stat_name in chain_stats:
                    raise ValueError(
                        f"Generated artifact contains duplicate chain statistic {stat_name!r}"
                    )
                chain_stats[stat_name] = value
            else:
                posterior[key] = value
    if not posterior:
        raise ValueError("Generated artifact contains no posterior variables")
    posterior_shapes: set[tuple[int, int]] = set()
    for name, value in posterior.items():
        if value.ndim < 2:
            raise ValueError(
                f"Generated artifact posterior variable {name!r} must have "
                f"at least two dimensions, got shape {value.shape!r}"
            )
        posterior_shapes.add(tuple(value.shape[:2]))
    if len(posterior_shapes) != 1:
        raise ValueError(
            "Generated artifact posterior variables have inconsistent leading "
            f"shapes: {sorted(posterior_shapes)!r}"
        )
    expected_shape = next(iter(posterior_shapes))
    for name, value in chain_stats.items():
        if value.ndim < 2:
            raise ValueError(
                f"Generated artifact statistic {name!r} must have at least two "
                f"dimensions, got shape {value.shape!r}"
            )
        if tuple(value.shape[:2]) != expected_shape:
            raise ValueError(
                f"Generated artifact statistic {name!r} has leading shape "
                f"{value.shape[:2]!r}; expected {expected_shape!r}"
            )
    return samples_to_idata(
        posterior,
        is_multichain=True,
        chain_stats=chain_stats or None,
        n_chunks=1,
    )


# Private compatibility name for callers written before the artifact adapter
# became part of the generated-execution API.
_artifact_to_idata = load_generated_idata


def _load_from_cache(
    draws_cache: Path,
    stats_cache: Path,
) -> arviz.InferenceData:
    """Load cached draws and chain_stats; reconstruct as InferenceData.

    Cache files store ArviZ-canonical sample_stats names (``diverging``,
    ``n_steps``, etc.) because ``_save_to_cache`` reads ``idata.sample_stats``
    after ``samples_to_idata`` has already projected raw blackjax names →
    canonical. Reverse-map them back to raw names here so that
    ``samples_to_idata``'s ``_chain_stats_to_sample_stats`` projection
    finds them and re-emits them as canonical. Without this reverse map,
    every cache hit drops ``diverging`` + ``n_steps`` silently because
    those keys aren't on the LHS of the rename map.
    """
    from tuningfork.catalog.diagnostics import (
        _CHAIN_STATS_TO_SAMPLE_STATS,
        samples_to_idata,
    )

    with np.load(str(draws_cache), allow_pickle=False) as draws_data:
        samples_dict = {k: np.asarray(draws_data[k]) for k in draws_data.files}

    # Reverse map: ArviZ canonical → raw blackjax field name.
    _SAMPLE_STATS_TO_CHAIN_STATS = {
        canonical: raw for raw, canonical in _CHAIN_STATS_TO_SAMPLE_STATS.items()
    }

    # Load chain_stats (optional; missing means absent, invalid fails loudly).
    chain_stats = None
    if stats_cache.exists():
        try:
            with np.load(str(stats_cache), allow_pickle=False) as stats_data:
                chain_stats = {}
                for name in stats_data.files:
                    raw_name = _SAMPLE_STATS_TO_CHAIN_STATS.get(name, name)
                    if raw_name in chain_stats:
                        raise ValueError(
                            f"chain-stats cache keys collide after canonical "
                            f"name resolution at {raw_name!r}"
                        )
                    chain_stats[raw_name] = np.asarray(stats_data[name])
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Could not load chain-stats cache {stats_cache}: {exc}"
            ) from exc

    # Reconstruct InferenceData
    # Samples are already in multi-chain format (num_chains, n_samples, *event_shape)
    return samples_to_idata(
        samples_dict,
        is_multichain=True,
        chain_stats=chain_stats,
        n_chunks=1,
    )


def _write_cache_params_sidecar(
    cache_dir: Path,
    catalog_root: Path,
    model_name: str,
    recipe_stem: str,
) -> None:
    """Write ``_cache/<stem>.params_hash.json`` beside a freshly written draws cache.

    Records the params the cache was generated for (step_size, num_integration_steps,
    target_acceptance, IMM fingerprint) so ``classify_recipe_path`` can detect a
    re-emitted recipe whose old draws are stale (issue #244).  Derived from the same
    on-disk recipe JSON the reader inspects, via the shared ``_recipe_cache_params``
    extractor, so a matching cache compares equal by construction.  A missing or
    unreadable recipe JSON is a silent no-op — the reader then finds no sidecar and
    conservatively degrades out of path-A, which is safe.
    """
    from tuningfork.calibration.revalidation import _recipe_cache_params

    recipe_json = catalog_root / model_name / "recipes" / f"{recipe_stem}.json"
    if not recipe_json.exists():
        return
    try:
        recipe_dict = json.loads(recipe_json.read_text())
    except Exception:  # noqa: BLE001
        return
    params = _recipe_cache_params(recipe_dict)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sidecar = cache_dir / f"{recipe_stem}.params_hash.json"
    tmp = cache_dir / f"{recipe_stem}.params_hash.json.tmp"
    tmp.write_text(json.dumps(params))
    tmp.replace(sidecar)  # atomic within the same directory


def _save_to_cache(
    idata: arviz.InferenceData,
    draws_cache: Path,
    stats_cache: Path,
) -> None:
    """Save InferenceData posterior and sample_stats to cache files."""
    draws_cache.parent.mkdir(parents=True, exist_ok=True)

    # Extract posterior group: {param_name: (chains, n_draws, *event_shape)}
    posterior = idata.posterior
    draws_dict = {
        name: np.asarray(posterior[name].values) for name in posterior.data_vars
    }

    # Save draws
    np.savez_compressed(str(draws_cache), **draws_dict)

    # Save chain_stats if available
    if hasattr(idata, "sample_stats") and idata.sample_stats is not None:
        sample_stats = idata.sample_stats
        stats_dict = {
            name: np.asarray(sample_stats[name].values)
            for name in sample_stats.data_vars
        }
        np.savez_compressed(str(stats_cache), **stats_dict)
