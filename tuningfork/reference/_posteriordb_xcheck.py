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
``tuningfork/reference/<model>/xcheck.json`` (caller's
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

    Posteriordb ``reference_draws()`` returns a dict like::

        {
            "chain:1": {"mu": [v1, v2, ...], "theta": [[...], ...], ...},
            "chain:2": {...},
            ...
        }

    Returns
    -------
    dict mapping param_name → 1-D numpy array of all values (all chains
    concatenated), with multi-dim params flattened per draw.
    """
    all_params: dict[str, list[Any]] = {}
    for chain_values in stan_draws.values():
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
    """
    # ------------------------------------------------------------------
    # 1. Load posteriordb reference draws (lazy import — heavy dep)
    # ------------------------------------------------------------------
    try:
        from posteriordb import PosteriorDatabase

        pdb_path = str(posteriordb_root) if posteriordb_root is not None else None
        pdb = PosteriorDatabase(pdb_path)
        posterior = pdb.posterior(posteriordb_id)
        stan_draws_raw = posterior.reference_draws()
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

        # Find matching posteriordb param
        if site not in stan_arrays:
            # Parameter naming mismatch — skip silently
            continue

        stan_arr = stan_arrays[site]

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
        # No common params found between our summaries and posteriordb
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
