"""Load-or-generate reference cache for bjx-bench.

This is the only module the runner and CLI call.  It owns the cache contract:
given a ``Posterior`` and a requested sample count, it either loads a
valid cached artifact or regenerates it.

Cache layout (relative to ``cache_dir``, default ``bjx_bench/reference/``):

    draws/<name>.npz           gitignored; potentially large
    summaries/<name>.json      committed; ~few KB
    adaptation/<name>.json     committed; long-NUTS path only
    metadata/<name>.json       committed; cache-validity stamp

Stamp fields (``metadata/<name>.json``):

    bjx_bench_version  str   installed package version
    code_sha           str   git HEAD SHA of bjx-bench repo, or "untracked"
    generator          str   "analytic" | "long_nuts"
    num_samples        int   number of samples stored
    seed               int   RNG seed used
    timestamp_utc      str   ISO-8601 UTC timestamp
    certification      dict  passed/failed + diagnostic values

Resolution order for ``get_reference_draws``::

    1. If not force_regenerate AND metadata exists AND version matches AND
       code_sha matches AND num_samples >= n AND certification.passed:
           → load draws from draws/<name>.npz; slice to first n.
    2. Else regenerate (analytic or NUTS).
    3. Always write all artifacts.

Environment variable
--------------------
``BJX_BENCH_REFERENCE_DIR`` overrides ``DEFAULT_CACHE_DIR``.
Override precedence: function-arg > env-var > default.

Concurrent-writer safety
------------------------
Phase 1 assumes no concurrent writers.  ``.npz`` files are written atomically
on POSIX (write to temp, rename).  ``json`` files use the same pattern.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

import bjx_bench
from bjx_bench.model._base import Posterior, ReferenceMethod

if TYPE_CHECKING:
    from bjx_bench.calibration._summary import Summaries
    from bjx_bench.calibration.tier_a import AdaptationParams

__all__ = [
    "get_reference_draws",
    "get_reference_summaries",
    "get_adaptation_params",
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
    env = os.environ.get("BJX_BENCH_REFERENCE_DIR")
    if env:
        return Path(env)
    return DEFAULT_CACHE_DIR


def _get_code_sha(cache_dir: Path) -> str:
    """Return git HEAD SHA of the bjx-bench repo, or 'untracked'."""
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
    return bjx_bench.__version__


def _metadata_path(name: str, cache_dir: Path) -> Path:
    return cache_dir / "metadata" / f"{name}.json"


def _draws_path(name: str, cache_dir: Path) -> Path:
    return cache_dir / "draws" / f"{name}.npz"


def _summaries_path(name: str, cache_dir: Path) -> Path:
    return cache_dir / "summaries" / f"{name}.json"


def _adaptation_path(name: str, cache_dir: Path) -> Path:
    return cache_dir / "adaptation" / f"{name}.json"


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, suffix=".tmp"
    ) as fh:
        json.dump(data, fh, indent=2)
        tmp = Path(fh.name)
    tmp.replace(path)


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
    """Return True iff the cached artifact satisfies all validity conditions."""
    if meta.get("bjx_bench_version") != current_version:
        return False
    if meta.get("code_sha") != current_sha:
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
    from bjx_bench.calibration._summary import Summaries

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


def _build_metadata(
    entry: Posterior,
    draws: dict[str, jax.Array],
    seed: int,
    current_version: str,
    current_sha: str,
    certification_dict: dict,
) -> dict:
    """Build the metadata stamp dict."""
    n = next(iter(draws.values())).shape[0]
    return {
        "name": entry.name,
        "bjx_bench_version": current_version,
        "code_sha": current_sha,
        "generator": entry.reference_method.value,
        "num_samples": n,
        "seed": seed,
        "timestamp_utc": datetime.datetime.now(datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
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
    from bjx_bench.calibration.tier_a_analytic import certify_reference_analytic

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
    n_chunks: int = 10,
    target_acceptance: float = 0.80,
) -> tuple[dict[str, jax.Array], Summaries, dict, AdaptationParams]:
    """Path B: long-NUTS certifier.  Returns (draws, summaries, cert_dict, adapt)."""
    from bjx_bench.calibration.tier_a import certify_reference_nuts

    draws, summaries, adaptation_params, cert = certify_reference_nuts(
        entry,
        rng_key,
        n_warmup=n_warmup,
        n_samples=n,
        n_chunks=n_chunks,
        target_acceptance=target_acceptance,
    )
    cert_dict = {
        "passed": cert.passed,
        "split_rhat_max": float(cert.split_rhat_max),
        "min_chunk_bulk_ess": float(cert.min_chunk_bulk_ess),
        "num_divergences": int(cert.num_divergences),
        "e_bfmi": float(cert.e_bfmi),
    }
    return draws, summaries, cert_dict, adaptation_params


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_reference_draws(
    entry: Posterior,
    n: int = 100_000,
    rng_key: jax.Array | None = None,
    *,
    force_regenerate: bool = False,
    cache_dir: Path | None = None,
    # NUTS-specific overrides (passed through to certify_reference_nuts)
    n_warmup: int = 5_000,
    n_chunks: int = 10,
    target_acceptance: float = 0.80,
) -> dict[str, jax.Array]:
    """Load reference draws from cache or regenerate.

    Resolution order:

    1. If not force_regenerate AND a valid cache stamp exists AND
       stamp.bjx_bench_version == current AND stamp.code_sha == current AND
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

    if entry.reference_method == ReferenceMethod.ANALYTIC:
        draws, summaries, cert_dict = _regenerate_analytic(entry, n, rng_key)
    else:
        draws, summaries, cert_dict, adaptation_params = _regenerate_nuts(
            entry,
            n,
            rng_key,
            n_warmup=n_warmup,
            n_chunks=n_chunks,
            target_acceptance=target_acceptance,
        )

    metadata = _build_metadata(
        entry, draws, seed, current_version, current_sha, cert_dict
    )
    _write_artifacts(
        entry, draws, summaries, metadata, effective_dir, adaptation_params
    )

    return draws


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
    from bjx_bench.calibration._summary import compute_summaries

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
    from bjx_bench.calibration.tier_a import AdaptationParams

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
