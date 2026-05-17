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
"""Load-or-generate reference cache for tuningfork.

This is the only module the runner and CLI call.  It owns the cache contract:
given a ``Posterior`` and a requested sample count, it either loads a
valid cached artifact or regenerates it.

Cache layout (relative to ``cache_dir``, default ``tuningfork/reference/``):

    <name>/draws.npz           gitignored; potentially large
    <name>/summary.json        committed; ~few KB
    <name>/adaptation.json     committed; long-NUTS path only
    <name>/metadata.json       committed; cache-validity stamp

Stamp fields (``metadata/<name>.json``):

    tuningfork_version              str    installed package version
    code_sha                       str    git HEAD SHA of tuningfork repo, or "untracked"
    generator                      str    "analytic" | "nuts"
    num_samples                    int    number of samples stored
    seed                           int    RNG seed used
    timestamp_utc                  str    ISO-8601 UTC timestamp
    wall_time_seconds              float|null  total regeneration wall (null on cache hit / older runs)
    divergence_rate_tolerance_used float|null  applied divergence-rate gate
                                              (null for analytic-path models;
                                               default 0.001 or per-model override
                                               from Posterior.divergence_rate_tolerance)
    certification                  dict   passed/failed + diagnostic values

Resolution order for ``get_reference_draws``::

    1. If not force_regenerate AND metadata exists AND version matches AND
       code_sha matches AND num_samples >= n AND certification.passed:
           → load draws from draws/<name>.npz; slice to first n.
    2. Else regenerate (analytic or NUTS).
    3. Always write all artifacts.

Environment variable
--------------------
``TUNINGFORK_REFERENCE_DIR`` overrides ``DEFAULT_CACHE_DIR``.
Override precedence: function-arg > env-var > default.

Concurrent-writer safety
------------------------
This module assumes no concurrent writers.  ``.npz`` files are written atomically
on POSIX (write to temp, rename).  ``json`` files use the same pattern.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

import tuningfork
from tuningfork.model._base import Posterior, ReferenceMethod

if TYPE_CHECKING:
    from tuningfork.calibration._summary import Summaries
    from tuningfork.calibration.certify_reference import (
        AdaptationParams,
        PreAdaptedWarmup,
    )

__all__ = [
    "get_reference_draws",
    "get_reference_summaries",
    "get_adaptation_params",
    "try_load_cached_draws",
    "try_load_cached_chain_stats",
    "DEFAULT_CACHE_DIR",
]

DEFAULT_CACHE_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_cache_dir(cache_dir: Path | None) -> Path:
    """Return effective cache dir: arg > env-var > default."""
    if cache_dir is not None:
        return cache_dir
    env = os.environ.get("TUNINGFORK_REFERENCE_DIR")
    if env:
        return Path(env)
    return DEFAULT_CACHE_DIR


def _get_code_sha(cache_dir: Path) -> str:
    """Return git HEAD SHA of the tuningfork repo, or 'untracked'."""
    # Walk up from the package file to find the repo root
    repo_root = Path(__file__).parent.parent.parent
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return sha.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "untracked"


def _current_version() -> str:
    return tuningfork.__version__


def _metadata_path(name: str, cache_dir: Path) -> Path:
    return cache_dir / name / "metadata.json"


def _draws_path(name: str, cache_dir: Path) -> Path:
    return cache_dir / name / "draws.npz"


def _summaries_path(name: str, cache_dir: Path) -> Path:
    return cache_dir / name / "summary.json"


def _adaptation_path(name: str, cache_dir: Path) -> Path:
    return cache_dir / name / "adaptation.json"


def _chain_stats_path(name: str, cache_dir: Path) -> Path:
    return cache_dir / name / "chain_stats.npz"


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, suffix=".tmp"
    ) as fh:
        json.dump(data, fh, indent=2)
        tmp = Path(fh.name)
    tmp.replace(path)


def _write_chain_stats(
    name: str,
    chain_stats: dict[str, np.ndarray],
    cache_dir: Path,
) -> None:
    """Write per-step chain_stats from a NUTS run to chain_stats/<name>.npz.

    chain_stats persistence is informational/diagnostic — written even on
    certification failure so the statistician can inspect the failed chain
    without re-running. The .npz is gitignored; the directory exists in
    the repo as a .gitkeep placeholder.
    """
    _atomic_write_npz(
        _chain_stats_path(name, cache_dir),
        {k: np.asarray(v) for k, v in chain_stats.items()},
    )


def _atomic_write_npz(path: Path, draws_dict: dict[str, np.ndarray]) -> None:
    """Write npz atomically via a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, delete=False, suffix=".tmp.npz"
    ) as fh:
        tmp = Path(fh.name)
    np.savez_compressed(str(tmp), **draws_dict)
    tmp.replace(path)


def _load_metadata(name: str, cache_dir: Path) -> dict | None:
    """Load metadata JSON; return None if missing or invalid."""
    path = _metadata_path(name, cache_dir)
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _cache_is_valid(meta: dict, n: int, current_version: str, current_sha: str) -> bool:
    """Return True iff the cached artifact satisfies all validity conditions.

    Per the cache-invalidation policy (decisions/2026-05-11-phase0-reference-protocol-refinements.md
    § 7: "Do not pre-emptively invalidate caches on spec changes"), the cached
    ``code_sha`` is treated as **audit trail**, not as an invalidation criterion.
    Every commit changes the SHA; gating cache validity on equality would
    invalidate every entry on every commit, which is exactly the pre-emptive
    invalidation behaviour the policy rules out. The statistician — not the
    cache validator — has authority to mark an existing groundtruth as
    "needs redo" based on chain quality.

    ``current_sha`` is still accepted as a parameter so the call site keeps its
    audit trail context, even though we no longer branch on it.
    """
    if meta.get("tuningfork_version") != current_version:
        return False
    if meta.get("num_samples", 0) < n:
        return False
    cert = meta.get("certification", {})
    if not cert.get("passed", False):
        return False
    return True


def _load_draws(name: str, n: int, cache_dir: Path) -> dict[str, jax.Array]:
    """Load draws from npz and slice to first n samples."""
    path = _draws_path(name, cache_dir)
    data = np.load(str(path))
    return {k: jnp.array(data[k][:n]) for k in data.files}


def _summaries_to_dict(summaries: Summaries) -> dict:
    """Serialise a Summaries object to a JSON-able dict."""

    def _arr_to_list(d: dict[str, jax.Array]) -> dict[str, list]:
        return {k: np.asarray(v).tolist() for k, v in d.items()}

    return {
        "mean": _arr_to_list(summaries.mean),
        "std": _arr_to_list(summaries.std),
        "q05": _arr_to_list(summaries.q05),
        "q95": _arr_to_list(summaries.q95),
        "n_samples": summaries.n_samples,
    }


def _summaries_from_dict(data: dict) -> Summaries:
    """Deserialise a Summaries object from a JSON dict."""
    from tuningfork.calibration._summary import Summaries

    def _list_to_arr(d: dict) -> dict[str, jax.Array]:
        return {k: jnp.array(v) for k, v in d.items()}

    return Summaries(
        mean=_list_to_arr(data["mean"]),
        std=_list_to_arr(data["std"]),
        q05=_list_to_arr(data["q05"]),
        q95=_list_to_arr(data["q95"]),
        n_samples=data["n_samples"],
    )


def _write_artifacts(
    entry: Posterior,
    draws: dict[str, jax.Array],
    summaries: Summaries,
    metadata: dict,
    cache_dir: Path,
    adaptation_params: AdaptationParams | None = None,
    chain_stats: dict[str, np.ndarray] | None = None,
) -> None:
    """Write all artifacts atomically."""
    # draws
    _atomic_write_npz(
        _draws_path(entry.name, cache_dir),
        {k: np.asarray(v) for k, v in draws.items()},
    )
    # summaries
    _atomic_write_json(
        _summaries_path(entry.name, cache_dir), _summaries_to_dict(summaries)
    )
    # metadata
    _atomic_write_json(_metadata_path(entry.name, cache_dir), metadata)
    # adaptation (NUTS path only)
    if adaptation_params is not None:
        adapt_dict = {
            "step_size": float(adaptation_params.step_size),
            "inverse_mass_matrix": np.asarray(
                adaptation_params.inverse_mass_matrix
            ).tolist(),
            "num_leapfrog_median": int(adaptation_params.num_leapfrog_median),
        }
        _atomic_write_json(_adaptation_path(entry.name, cache_dir), adapt_dict)
    # chain_stats (NUTS path only — diagnostic, not gate-bearing; gitignored)
    if chain_stats is not None:
        _write_chain_stats(entry.name, chain_stats, cache_dir)


def _build_metadata(
    entry: Posterior,
    draws: dict[str, jax.Array],
    seed: int,
    current_version: str,
    current_sha: str,
    certification_dict: dict,
    wall_time_seconds: float | None = None,
    divergence_rate_tolerance_used: float | None = None,
) -> dict:
    """Build the metadata stamp dict.

    ``wall_time_seconds`` (optional): total wall for the regeneration, in
    seconds. ``None`` on cache hit (no regeneration occurred).
    ``divergence_rate_tolerance_used`` (optional): the divergence-rate gate
    value applied at cert time — captures per-model overrides (e.g.,
    stoch_vol's 0.005) vs the default 0.001. ``None`` for analytic-path
    models (no divergence gate is applied) and on cache hit.

    Both fields added 2026-05-16 per the META candidate
    ``reference-metadata-schema-missing-wall-time``. Backward-compatible:
    older metadata files without these keys still load via
    ``meta.get(field, None)``; the new fields default to None when the
    cert pipeline doesn't supply them.
    """
    n = next(iter(draws.values())).shape[0]
    return {
        "name": entry.name,
        "tuningfork_version": current_version,
        "code_sha": current_sha,
        "generator": entry.reference_method.value,
        "num_samples": n,
        "seed": seed,
        "timestamp_utc": datetime.datetime.now(datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "wall_time_seconds": wall_time_seconds,
        "divergence_rate_tolerance_used": divergence_rate_tolerance_used,
        "certification": certification_dict,
    }


# ---------------------------------------------------------------------------
# Regeneration helpers
# ---------------------------------------------------------------------------


def _regenerate_analytic(
    entry: Posterior,
    n: int,
    rng_key: jax.Array,
) -> tuple[dict[str, jax.Array], Summaries, dict]:
    """Path A: analytic sampler.  Returns (draws, summaries, cert_dict)."""
    from tuningfork.calibration.certify_reference_analytic import (
        certify_reference_analytic,
    )

    draws, summaries = certify_reference_analytic(entry, n, rng_key)
    cert_dict = {
        "passed": True,
        "split_rhat_max": None,
        "min_chunk_bulk_ess": None,
        "num_divergences": None,
        "e_bfmi": None,
    }
    return draws, summaries, cert_dict


def _regenerate_nuts(
    entry: Posterior,
    n: int,
    rng_key: jax.Array,
    n_warmup: int = 5_000,
    n_chunks: int = 4,
    target_acceptance: float = 0.80,
    max_num_doublings: int = 10,
    pre_adapted: PreAdaptedWarmup | None = None,
    checkpoint_dir: Path | None = None,
    validate_warmup_fn: Callable | None = None,
) -> tuple[
    dict[str, jax.Array],
    Summaries,
    dict,
    AdaptationParams,
    dict[str, np.ndarray],
]:
    """Path B: long-NUTS certifier.  Returns (draws, summaries, cert_dict, adapt, chain_stats).

    `chain_stats` is the dict of per-step NUTS diagnostics (num_integration_steps,
    energy, is_divergent, acceptance_rate, plus other NUTSInfo._fields). On
    CertificationError from `certify_reference_nuts`, the exception carries
    chain_stats and the caller persists it via `_write_chain_stats` before
    re-raising — diagnostic data survives the failure path.

    ``pre_adapted`` (optional) lets the caller inject a previously-run warmup's
    adapted state and params; when provided, the warmup phase is skipped.
    ``checkpoint_dir`` (optional) is where the warmup checkpoint is persisted
    immediately after warmup completes (before validation, before sampling).
    ``validate_warmup_fn`` (optional) is a model-specific health-check callback.
    """
    from tuningfork.calibration.certify_reference import certify_reference_nuts

    draws, summaries, adaptation_params, cert, chain_stats = certify_reference_nuts(
        entry,
        rng_key,
        n_warmup=n_warmup,
        n_samples=n,
        n_chunks=n_chunks,
        target_acceptance=target_acceptance,
        max_num_doublings=max_num_doublings,
        pre_adapted=pre_adapted,
        checkpoint_dir=checkpoint_dir,
        validate_warmup_fn=validate_warmup_fn,
    )
    cert_dict = {
        "passed": cert.passed,
        "split_rhat_max": float(cert.split_rhat_max),
        "min_chunk_bulk_ess": float(cert.min_chunk_bulk_ess),
        "num_divergences": int(cert.num_divergences),
        "e_bfmi": float(cert.e_bfmi),
    }
    return draws, summaries, cert_dict, adaptation_params, chain_stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_reference_draws(
    entry: Posterior,
    n: int = 40_000,
    rng_key: jax.Array | None = None,
    *,
    force_regenerate: bool = False,
    cache_dir: Path | None = None,
    # NUTS-specific overrides (passed through to certify_reference_nuts)
    n_warmup: int = 5_000,
    n_chunks: int = 4,
    target_acceptance: float = 0.80,
    max_num_doublings: int = 10,
    pre_adapted: PreAdaptedWarmup | None = None,
    checkpoint_dir: Path | None = None,
    validate_warmup_fn: Callable | None = None,
) -> dict[str, jax.Array]:
    """Load reference draws from cache or regenerate.

    Resolution order:

    1. If not force_regenerate AND a valid cache stamp exists AND
       stamp.tuningfork_version == current AND stamp.code_sha == current AND
       stamp.num_samples >= n AND stamp.certification.passed:
         → load draws from cache; slice to first n along sample axis.
    2. Else regenerate:
       - entry.reference_method == ANALYTIC: call entry.analytic_sampler.
       - entry.reference_method == NUTS: invoke certify_reference_nuts.
    3. Always write all artifacts.

    Parameters
    ----------
    entry
        Registry entry describing the model.
    n
        Number of reference draws requested.
    rng_key
        JAX random key.  Required for regeneration; ignored on cache hit.
    force_regenerate
        If True, skip the cache check and always regenerate.
    cache_dir
        Override the cache directory.
    n_warmup
        NUTS warmup steps (ignored for analytic models).
    n_chunks
        Number of chunks for split-R̂ (NUTS only).
    target_acceptance
        NUTS target acceptance rate (ignored for analytic models).

    Returns
    -------
    dict mapping site name → Array of shape (n, *site_shape).
    """
    effective_dir = _resolve_cache_dir(cache_dir)
    current_version = _current_version()
    current_sha = _get_code_sha(effective_dir)

    if not force_regenerate:
        meta = _load_metadata(entry.name, effective_dir)
        if meta is not None and _cache_is_valid(meta, n, current_version, current_sha):
            return _load_draws(entry.name, n, effective_dir)

    # --- Regenerate ---
    if rng_key is None:
        raise ValueError(
            f"rng_key is required for regeneration of {entry.name!r}. "
            "Pass rng_key=jax.random.key(seed)."
        )

    seed = 0  # best-effort; actual key encodes the seed
    adaptation_params = None
    chain_stats: dict[str, np.ndarray] | None = None

    # Time the regeneration to populate metadata.wall_time_seconds. The clock
    # encloses both analytic and NUTS branches, including the on-failure
    # persistence path (so wall is captured even when cert fails).
    import time as _time

    _t0 = _time.perf_counter()

    if entry.reference_method == ReferenceMethod.ANALYTIC:
        draws, summaries, cert_dict = _regenerate_analytic(entry, n, rng_key)
    else:
        # NUTS path: wrap _regenerate_nuts so we persist chain_stats even on
        # certification failure (CertificationError carries chain_stats per
        # decision doc 2026-05-11-phase0-reference-protocol-refinements § 3).
        from tuningfork.calibration.certify_reference import CertificationError

        # Default checkpoint_dir: <cache_dir>/<model>/warmup_checkpoint/
        # The checkpoint is written immediately after warmup completes and
        # contains state.pkl, params.pkl, warmup_info.npz, health.json.
        # Gitignored (see .gitignore: reference/*/warmup_checkpoint/).
        effective_checkpoint_dir = checkpoint_dir
        if effective_checkpoint_dir is None and pre_adapted is None:
            effective_checkpoint_dir = effective_dir / entry.name / "warmup_checkpoint"

        try:
            (
                draws,
                summaries,
                cert_dict,
                adaptation_params,
                chain_stats,
            ) = _regenerate_nuts(
                entry,
                n,
                rng_key,
                n_warmup=n_warmup,
                n_chunks=n_chunks,
                target_acceptance=target_acceptance,
                max_num_doublings=max_num_doublings,
                pre_adapted=pre_adapted,
                checkpoint_dir=effective_checkpoint_dir,
                validate_warmup_fn=validate_warmup_fn,
            )
        except CertificationError as exc:
            # Failure path: persist chain_stats, draws, AND adaptation params
            # (step_size + IMM) for statistician diagnosis, then re-raise.
            # Without these, the cluster-in-parameter-space check (Lens 1 of
            # the diagnostics playbook) plus the warmup-IMM-vs-empirical
            # calibration check are blocked. Schema gap closed 2026-05-12
            # after the gp_regression cert failure's chunk-1 divergence
            # cluster could not be diagnosed because draws weren't persisted,
            # and the warmup IMM wasn't recoverable without a re-run.
            #
            # IMPORTANT: failure-path artifacts use ``.failed.<ext>``
            # filenames. A failed re-run never overwrites a previous
            # successful cert's artifacts (which could happen e.g. with
            # force_regenerate=True). All three failure filenames are
            # gitignored (see ``.gitignore``).
            if exc.chain_stats is not None:
                _write_chain_stats(entry.name, exc.chain_stats, effective_dir)
            if exc.draws is not None:
                failed_draws_path = _draws_path(entry.name, effective_dir).with_suffix(
                    ".failed.npz"
                )
                _atomic_write_npz(
                    failed_draws_path,
                    {k: np.asarray(v) for k, v in exc.draws.items()},
                )
            if exc.adaptation is not None:
                failed_adapt_path = _adaptation_path(
                    entry.name, effective_dir
                ).with_suffix(".failed.json")
                adapt_dict = {
                    "step_size": float(exc.adaptation.step_size),
                    "inverse_mass_matrix": np.asarray(
                        exc.adaptation.inverse_mass_matrix
                    ).tolist(),
                    "num_leapfrog_median": int(exc.adaptation.num_leapfrog_median),
                }
                _atomic_write_json(failed_adapt_path, adapt_dict)
            raise

    # Capture wall + applied gate AFTER regeneration completes. On NUTS path,
    # the gate is per-model (Posterior.divergence_rate_tolerance overrides the
    # module global). On analytic path no divergence gate applies; field is None.
    _wall_seconds = float(_time.perf_counter() - _t0)
    if entry.reference_method == ReferenceMethod.ANALYTIC:
        _gate_used: float | None = None
    else:
        from tuningfork.calibration.certify_reference import _DIVERGENCE_RATE_TOLERANCE

        _gate_used = (
            entry.divergence_rate_tolerance
            if entry.divergence_rate_tolerance is not None
            else _DIVERGENCE_RATE_TOLERANCE
        )

    metadata = _build_metadata(
        entry,
        draws,
        seed,
        current_version,
        current_sha,
        cert_dict,
        wall_time_seconds=_wall_seconds,
        divergence_rate_tolerance_used=_gate_used,
    )
    _write_artifacts(
        entry,
        draws,
        summaries,
        metadata,
        effective_dir,
        adaptation_params,
        chain_stats,
    )

    return draws


def try_load_cached_draws(
    entry: Posterior,
    n: int | None = None,
    *,
    cache_dir: Path | None = None,
) -> dict[str, jax.Array] | None:
    """Load reference draws from cache, or return None on miss.

    Mirrors ``get_reference_draws`` validity logic but never regenerates.
    Useful when the caller wants to short-circuit on cache hit but defer
    regeneration to a different code path (e.g., a notebook).

    Parameters
    ----------
    entry : Posterior
    n : int | None
        If None, load all available samples (no slicing). If int, load first
        ``n`` samples; cache hit requires stamp.num_samples >= n.
    cache_dir : Path | None

    Returns
    -------
    dict[str, jax.Array] or None
        Draws dict on cache hit, None on miss.
    """
    effective_dir = _resolve_cache_dir(cache_dir)
    current_version = _current_version()
    current_sha = _get_code_sha(effective_dir)

    meta = _load_metadata(entry.name, effective_dir)
    if meta is None:
        return None

    # Determine n for validity check
    check_n = n if n is not None else meta.get("num_samples", 0)
    if not _cache_is_valid(meta, check_n, current_version, current_sha):
        return None

    draws_path = _draws_path(entry.name, effective_dir)
    if not draws_path.exists():
        return None

    data = np.load(str(draws_path))
    if n is None:
        return {k: jnp.array(data[k]) for k in data.files}
    return {k: jnp.array(data[k][:n]) for k in data.files}


def try_load_cached_chain_stats(
    entry: Posterior,
    *,
    cache_dir: Path | None = None,
) -> dict[str, np.ndarray] | None:
    """Load per-step chain_stats from cache, or return None on miss.

    chain_stats are diagnostic-only (not gate-bearing); this function does NOT
    apply the metadata-validity check used by `try_load_cached_draws`. The
    file at ``chain_stats/<name>.npz`` is returned as-is if it exists. This
    is intentional: chain_stats may be written even when cert fails
    (CertificationError path persists them for statistician diagnosis), so
    the metadata stamp may not be cert-passed even though chain_stats exist.

    Parameters
    ----------
    entry : Posterior
    cache_dir : Path | None

    Returns
    -------
    dict[str, np.ndarray] or None
        Mapping per-step field name → array of shape ``(n_samples,)`` on cache
        hit; None on miss. Fields typically include `num_integration_steps`,
        `energy`, `is_divergent`, `acceptance_rate`, and other NUTSInfo
        fields exposed by BlackJAX.
    """
    effective_dir = _resolve_cache_dir(cache_dir)
    path = _chain_stats_path(entry.name, effective_dir)
    if not path.exists():
        return None
    data = np.load(str(path))
    return {k: np.asarray(data[k]) for k in data.files}


def get_reference_summaries(
    entry: Posterior,
    *,
    cache_dir: Path | None = None,
    auto_regenerate: bool = True,
) -> Summaries:
    """Load reference summaries from cache, regenerating if missing.

    Summaries are small JSON files committed to the repo, so cache hits are
    the common case even on a fresh clone.

    Parameters
    ----------
    entry
        Registry entry describing the model.
    cache_dir
        Override the cache directory.
    auto_regenerate
        If True (default) and summaries are missing, attempt to load from
        the draws file and recompute.

    Returns
    -------
    ``Summaries`` dataclass with mean/std/q05/q95 per site.

    Raises
    ------
    FileNotFoundError
        If summaries are missing and auto_regenerate=False.
    """
    from tuningfork.calibration._summary import compute_summaries

    effective_dir = _resolve_cache_dir(cache_dir)
    path = _summaries_path(entry.name, effective_dir)

    if path.exists():
        with path.open() as fh:
            return _summaries_from_dict(json.load(fh))

    if not auto_regenerate:
        raise FileNotFoundError(
            f"No summaries cached for {entry.name!r}. "
            "Call get_reference_draws() first or set auto_regenerate=True."
        )

    # Try to compute from existing draws
    draws_path = _draws_path(entry.name, effective_dir)
    if draws_path.exists():
        data = np.load(str(draws_path))
        draws = {k: jnp.array(data[k]) for k in data.files}
        summaries = compute_summaries(draws)
        _atomic_write_json(path, _summaries_to_dict(summaries))
        return summaries

    raise FileNotFoundError(
        f"No cached draws or summaries for {entry.name!r}. "
        "Call get_reference_draws() first."
    )


def get_adaptation_params(
    entry: Posterior,
    *,
    cache_dir: Path | None = None,
) -> AdaptationParams:
    """Return cached adaptation parameters for a long-NUTS reference model.

    Parameters
    ----------
    entry
        Registry entry.  Must have reference_method == NUTS.
    cache_dir
        Override the cache directory.

    Returns
    -------
    ``AdaptationParams`` with step_size, inverse_mass_matrix, num_leapfrog_median.

    Raises
    ------
    ValueError
        If ``entry.reference_method == ANALYTIC``.
    FileNotFoundError
        If no adaptation params are cached for this model.
    """
    from tuningfork.calibration.certify_reference import AdaptationParams

    if entry.reference_method == ReferenceMethod.ANALYTIC:
        raise ValueError(
            f"{entry.name!r} uses the analytic path; there are no adaptation "
            "parameters. Call get_reference_draws() to get samples."
        )

    effective_dir = _resolve_cache_dir(cache_dir)
    path = _adaptation_path(entry.name, effective_dir)

    if not path.exists():
        raise FileNotFoundError(
            f"No adaptation params cached for {entry.name!r}. "
            "Run get_reference_draws() first to populate the cache."
        )

    with path.open() as fh:
        data = json.load(fh)

    return AdaptationParams(
        step_size=float(data["step_size"]),
        inverse_mass_matrix=jnp.array(data["inverse_mass_matrix"]),
        num_leapfrog_median=int(data["num_leapfrog_median"]),
    )
