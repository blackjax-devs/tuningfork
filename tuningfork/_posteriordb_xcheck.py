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
"""Posteriordb cross-check for shared posteriors.

For models with ``posteriordb_id != None`` (e.g., #3 8-schools, #6 radon,
#10 IRT 2PL), compare our marginal mean/std/q05/q95 to posteriordb's
Stan reference draws. Discrepancies are recorded as findings, not
failures — the gate has already passed at this point.

Tolerance
---------
- ``|Δmean| < 2 × max(SE_ours, SE_stan)``  (per-dim)
- ``|std_ratio - 1| < 0.05``                (per-dim)

where SE = std / sqrt(n_samples).

Reports persist to
``tuningfork/catalog/<model>/reference/xcheck.json`` (caller's
responsibility; call ``XCheckResult.save(path)`` after construction).
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "XCheckResult",
    "cross_check_against_posteriordb",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XCheckResult:
    """Per-dim cross-check outcome against posteriordb Stan reference draws.

    Parameters
    ----------
    model_name
        Registry name of the posterior (e.g. ``"eight_schools_ncp"``).
    posteriordb_id
        Posteriordb identifier (e.g. ``"8_schools-eight_schools_noncentered"``).
    passed
        ``True`` iff ALL per-dim tests are within tolerance.
    n_dims_compared
        Number of scalar dimensions compared (sum of sizes across sites).
    failed_dims
        Tuple of dimension labels where mean or std test failed.
        For posteriordb errors, contains a single error string.
        Empty tuple iff ``passed`` is True.
    max_abs_mean_z
        Maximum ``|Δmean| / max(SE_ours, SE_stan)`` across all dims.
        Pass criterion: < 2.  ``math.nan`` if comparison was skipped.
    max_std_ratio_dev
        Maximum ``|std_ratio - 1|`` across all dims.
        Pass criterion: < 0.05.  ``math.nan`` if comparison was skipped.
    """

    model_name: str
    posteriordb_id: str
    passed: bool
    n_dims_compared: int
    failed_dims: tuple[str, ...]
    max_abs_mean_z: float
    max_std_ratio_dev: float

    def save(self, path: Path) -> None:
        """Write this result as a JSON file.

        Parameters
        ----------
        path
            File path to write (the caller is responsible for creating
            parent directories).
        """
        d = asdict(self)
        # tuple → list for JSON serialisation
        d["failed_dims"] = list(d["failed_dims"])
        # Trailing newline keeps the file POSIX-clean and idempotent under
        # the `fix end of files` pre-commit hook (otherwise every test run
        # leaves a 1-byte cosmetic diff that has to be stashed before merge).
        path.write_text(json.dumps(d, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def _aggregate_stan_draws(
    stan_draws: Any,
) -> dict[str, np.ndarray]:
    """Aggregate posteriordb reference draws across chains.

    Posteriordb ``reference_draws()`` returns one of two formats depending on
    the client and database version:

    - ``PosteriorDatabaseGithub``: a dict keyed by ``"chain:1"``, ``"chain:2"``
      etc., each value being a dict ``{param: [draw, ...]}``.
    - ``PosteriorDatabase`` (local clone): a list of chain-dicts, each being
      ``{param: [draw, ...]}``.

    Both formats are supported.

    Returns
    -------
    dict mapping param_name → 1-D or 2-D numpy array (all chains
    concatenated).
    """
    # Normalise to an iterable of chain-value dicts
    if isinstance(stan_draws, dict):
        chain_iter = stan_draws.values()
    else:
        # list (local PosteriorDatabase format)
        chain_iter = stan_draws

    all_params: dict[str, list[Any]] = {}
    for chain_values in chain_iter:
        for param, values in chain_values.items():
            if param not in all_params:
                all_params[param] = []
            # Each element of values is one draw: scalar → float, vector → list
            all_params[param].extend(values)

    # Flatten per-dim: shape (n_draws,) for scalars, (n_draws, dim) for vectors
    result: dict[str, np.ndarray] = {}
    for param, values in all_params.items():
        arr = np.asarray(values, dtype=float)
        # arr shape: (n_draws,) for scalars or (n_draws, dim) for vectors
        result[param] = arr
    return result


def _label_dims(param: str, dim: int, ndim: int) -> str:
    """Build a human-readable dim label, e.g. ``"theta[3]"``."""
    if ndim == 0:
        return param
    return f"{param}[{dim}]"


def cross_check_against_posteriordb(
    model_name: str,
    posteriordb_id: str,
    our_summaries: dict[str, dict[str, Any]],
    n_samples_ours: int,
    posteriordb_root: Path | None = None,
    postprocess_fn: Any = None,
    our_draws: dict[str, Any] | None = None,
) -> XCheckResult:
    """Compare our summaries to posteriordb's Stan reference draws.

    Loads reference draws from the posteriordb Python client and compares
    per-dim mean and std against our computed summaries.  Designed to be
    called AFTER the reference-certification gate passes (i.e., the model is already
    certified); discrepancies are findings, not failures.

    Parameters
    ----------
    model_name
        Registry name of the model (for the result record).
    posteriordb_id
        Posteriordb posterior identifier string.
    our_summaries
        Output of ``Summaries`` converted to a plain dict via
        ``{site: {"mean": ..., "std": ..., "q05": ..., "q95": ...}}``.
        Values can be JAX arrays, numpy arrays, or Python floats/lists.
    n_samples_ours
        Number of post-warmup samples used to compute ``our_summaries``.
        Used to compute the standard error of our mean estimates.
    posteriordb_root
        Optional path to a local posteriordb checkout.  When ``None``,
        uses the ``POSTERIOR_DB_PATH`` environment variable (or the
        installed posteriordb default).
    postprocess_fn
        Optional callable that transforms unconstrained draws to the
        constrained parameter space.  When provided together with
        ``our_draws``, the transform is applied before computing our
        summary statistics, enabling apples-to-apples comparison against
        posteriordb's constrained reference.  This is the ``postprocess_fn``
        returned by ``tuningfork.model._numpyro.build_logdensity_fn``.
        When ``None`` (default), ``our_summaries`` is used directly.
    our_draws
        Optional dict ``{site: array(n_samples, *event)}`` of unconstrained
        draws (e.g. from ``_cache/draws.npz``).  Must be provided when
        ``postprocess_fn`` is set; ignored otherwise.

    Returns
    -------
    XCheckResult
        Result dataclass; persistence is the caller's responsibility —
        call ``result.save(path)`` to write the JSON report.

    Notes
    -----
    If the posteriordb client raises any exception (database unavailable,
    unknown posterior ID, etc.), returns a not-checked ``XCheckResult``
    with ``passed=False`` and the error string in ``failed_dims``.  The
    caller decides how to handle this (log, skip, raise).

    Matching of parameter names between our sites and posteriordb params
    uses a direct equality check.  If posteriordb uses a different naming
    convention (e.g. ``"theta[1]"`` vs our ``"theta"``), only matched
    params are compared; unmatched params are silently skipped.

    **Scale matching:** NumPyro stores parameters in unconstrained space
    internally; ``postprocess_fn`` maps back to constrained space.
    posteriordb reference draws are always in constrained space.  Pass
    ``postprocess_fn`` + ``our_draws`` to align scales before comparing.
    """
    # ------------------------------------------------------------------
    # 0. If postprocess_fn + our_draws provided, recompute our_summaries
    #    in constrained space so the comparison is apples-to-apples.
    # ------------------------------------------------------------------
    if postprocess_fn is not None and our_draws is not None:
        # Apply the model's constrain_fn (postprocess_fn) to each draw.
        # postprocess_fn takes a single-sample dict; vmap over the sample axis.
        import jax

        constrained_draws_jax = jax.vmap(postprocess_fn)(our_draws)
        our_summaries = {}
        n_samples_ours = 0
        for site, arr in constrained_draws_jax.items():
            arr_np = np.asarray(arr, dtype=float)
            # arr_np shape: (n_samples,) for scalars or (n_samples, *event)
            n_samples_ours = arr_np.shape[0]
            flat = arr_np.reshape(n_samples_ours, -1)  # (n_samples, d)
            our_summaries[site] = {
                "mean": np.mean(flat, axis=0),  # (d,) or scalar
                "std": np.std(flat, axis=0, ddof=1),
                "q05": np.quantile(flat, 0.05, axis=0),
                "q95": np.quantile(flat, 0.95, axis=0),
            }

    # ------------------------------------------------------------------
    # 1. Load posteriordb reference draws (lazy import — heavy dep)
    # ------------------------------------------------------------------
    # Priority: explicit posteriordb_root > POSTERIOR_DB_PATH env var >
    # PosteriorDatabaseGithub (network fallback, rate-limited).
    _db_error: str | None = None
    stan_draws_raw: Any = None
    try:
        import os

        from posteriordb import PosteriorDatabase, PosteriorDatabaseGithub

        env_path = os.environ.get("POSTERIOR_DB_PATH")
        pdb_path = (
            str(posteriordb_root)
            if posteriordb_root is not None
            else env_path if env_path else None
        )

        # Priority: local PosteriorDatabase → PosteriorDatabaseGithub fallback.
        # Don't use posterior_names() check (slow, not always supported);
        # just try pdb.posterior() directly and fall through on any exception.
        try:
            pdb = PosteriorDatabase(pdb_path)
            posterior = pdb.posterior(posteriordb_id)
            stan_draws_raw = posterior.reference_draws()
        except Exception as local_exc:  # noqa: BLE001
            _db_error = (
                f"local PosteriorDatabase ({pdb_path!r}) failed for "
                f"{posteriordb_id!r}: {type(local_exc).__name__}: {local_exc}; "
                f"falling back to PosteriorDatabaseGithub"
            )
            # Network fallback — uses the posteriordb GitHub API (rate-limited without PAT)
            pdb_gh = PosteriorDatabaseGithub()
            posterior = pdb_gh.posterior(posteriordb_id)
            stan_draws_raw = posterior.reference_draws()
            import warnings

            warnings.warn(_db_error, stacklevel=3)
    except Exception as exc:  # noqa: BLE001
        return XCheckResult(
            model_name=model_name,
            posteriordb_id=posteriordb_id,
            passed=False,
            n_dims_compared=0,
            failed_dims=(f"posteriordb-error: {type(exc).__name__}: {exc}",),
            max_abs_mean_z=math.nan,
            max_std_ratio_dev=math.nan,
        )

    # ------------------------------------------------------------------
    # 2. Aggregate stan draws across chains
    # ------------------------------------------------------------------
    stan_arrays = _aggregate_stan_draws(stan_draws_raw)
    n_stan = next(iter(stan_arrays.values())).shape[0]

    # ------------------------------------------------------------------
    # 3. Per-dim comparison
    # ------------------------------------------------------------------
    failed_dims: list[str] = []
    abs_mean_z_values: list[float] = []
    std_ratio_dev_values: list[float] = []
    n_dims = 0

    for site, site_summaries in our_summaries.items():
        # Our mean and std (convert to numpy for arithmetic)
        our_mean_arr = np.atleast_1d(np.asarray(site_summaries["mean"], dtype=float))
        our_std_arr = np.atleast_1d(np.asarray(site_summaries["std"], dtype=float))

        site_ndim = our_mean_arr.size

        # Find matching posteriordb param — handle two naming conventions:
        # (a) Direct match: our "theta_raw" == stan "theta_raw"
        # (b) Bracketed elements: stan has "theta[1]".."theta[d]" but we have
        #     "theta" as a d-dim vector.  Detect by checking for "{site}[1]".
        stan_arr: np.ndarray | None = None
        if site in stan_arrays:
            stan_arr = stan_arrays[site]
        else:
            # Try bracket convention: reconstruct from "site[1]".."site[d]" keys
            # that exist in stan_arrays.
            bracketed_keys = [k for k in stan_arrays if k.startswith(f"{site}[")]
            if bracketed_keys:
                # Sort by bracket index and concatenate column-wise
                def _bracket_idx(k: str) -> int:
                    try:
                        return int(k[len(site) + 1 : -1])
                    except ValueError:
                        return 0

                bracketed_keys.sort(key=_bracket_idx)
                cols = [stan_arrays[k].reshape(-1, 1) for k in bracketed_keys]
                stan_arr = np.hstack(cols)  # (n_draws, d)
            else:
                # Genuine mismatch — warn loudly (not silent skip)
                import warnings

                warnings.warn(
                    f"cross_check_against_posteriordb: parameter {site!r} not found "
                    f"in posteriordb {posteriordb_id!r}. "
                    f"Available params: {sorted(stan_arrays.keys())}",
                    stacklevel=2,
                )
                continue

        if stan_arr is None:
            continue

        # Handle scalar vs vector: ensure 2-D (n_draws, n_dims_at_site)
        if stan_arr.ndim == 1:
            stan_arr = stan_arr[:, np.newaxis]

        stan_mean = np.mean(stan_arr, axis=0)  # (n_dims_at_site,)
        stan_std = np.std(stan_arr, axis=0, ddof=1)  # (n_dims_at_site,)

        # Align dims (truncate to the shorter side if mismatch)
        n_compare = min(our_mean_arr.size, stan_mean.size)
        our_mean = our_mean_arr[:n_compare]
        our_std = our_std_arr[:n_compare]
        stan_m = stan_mean[:n_compare]
        stan_s = stan_std[:n_compare]

        for d in range(n_compare):
            n_dims += 1
            label = _label_dims(site, d, n_compare if site_ndim > 1 else 0)

            # Standard errors
            se_ours = our_std[d] / math.sqrt(max(n_samples_ours, 1))
            se_stan = stan_s[d] / math.sqrt(max(n_stan, 1))
            se_max = max(se_ours, se_stan, 1e-12)

            # Mean test
            abs_z = abs(float(our_mean[d]) - float(stan_m[d])) / se_max
            abs_mean_z_values.append(abs_z)
            mean_fail = abs_z >= 2.0

            # Std ratio test (guard against near-zero std)
            if stan_s[d] > 1e-12:
                std_ratio = float(our_std[d]) / float(stan_s[d])
                std_dev = abs(std_ratio - 1.0)
            else:
                std_ratio = 1.0
                std_dev = 0.0
            std_ratio_dev_values.append(std_dev)
            std_fail = std_dev >= 0.05

            if mean_fail or std_fail:
                failed_dims.append(label)

    # ------------------------------------------------------------------
    # 4. Build result
    # ------------------------------------------------------------------
    if not abs_mean_z_values:
        # No common params found — this is a name-matching failure, log loudly
        import warnings

        warnings.warn(
            f"cross_check_against_posteriordb: no parameters matched between "
            f"{model_name!r} summaries and posteriordb {posteriordb_id!r} "
            f"(our params: {sorted(our_summaries.keys())}; "
            f"stan params: {sorted(stan_arrays.keys())}). "
            f"Check the param name conventions (bracket notation vs vector).",
            stacklevel=2,
        )
        return XCheckResult(
            model_name=model_name,
            posteriordb_id=posteriordb_id,
            passed=False,
            n_dims_compared=0,
            failed_dims=("no-common-params",),
            max_abs_mean_z=math.nan,
            max_std_ratio_dev=math.nan,
        )

    max_z = max(abs_mean_z_values)
    max_std_dev = max(std_ratio_dev_values)
    passed = len(failed_dims) == 0

    return XCheckResult(
        model_name=model_name,
        posteriordb_id=posteriordb_id,
        passed=passed,
        n_dims_compared=n_dims,
        failed_dims=tuple(failed_dims),
        max_abs_mean_z=max_z,
        max_std_ratio_dev=max_std_dev,
    )
