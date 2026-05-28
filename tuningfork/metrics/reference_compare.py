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
"""Sample-quality metric comparing MCMC draws against a Tier-A reference.

This module provides a single entry point, ``compute_sample_quality``, which
compares empirical summaries of MCMC draws to certified reference summaries and
returns four scalar metrics.

Design invariants
-----------------
1. **Normalisation by REFERENCE std, not empirical std.**
   Normalising by the empirical std would silently rescale a recipe that
   systematically under- or over-estimates posterior spread — the very defect
   we want to detect.  Reference-std normalisation keeps all four metrics
   unitless and comparable across parameters of arbitrarily different scale.

2. **Multi-chain flattening over (num_chains × num_samples).**
   Draws are flattened across all chains before computing empirical statistics.
   Computing per-chain stats and averaging would double-count the job of the
   R̂/ESS gate (which already guards chain-mixing quality); here we just want
   to know whether the pooled posterior approximation matches the reference.

3. **Max-over-parameters reduction.**
   Each of the four metrics is reduced to a single scalar by taking the
   maximum over all parameters.  The worst-matching parameter governs the
   overall score, analogous to the min-ESS pessimistic aggregation in the
   headline metric.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["compute_sample_quality"]

# Type aliases
_RefSummary = dict[str, float]
_DrawsDict = dict[str, np.ndarray]
_RefDict = dict[str, _RefSummary]

_REQUIRED_KEYS = frozenset({"mean", "std", "q05", "q95"})
_QUANTILE_05 = 0.05
_QUANTILE_95 = 0.95


def _validate_ref_summary(name: str, ref: _RefSummary) -> bool:
    """Return True if ref has all required keys with finite std; warn and return False otherwise.

    Parameters
    ----------
    name
        Parameter name, used only for the warning message.
    ref
        A dict that must contain ``{"mean", "std", "q05", "q95"}``.

    Returns
    -------
    bool
        ``True`` if the summary is usable; ``False`` if any value is NaN.
    """
    for key in _REQUIRED_KEYS:
        if key not in ref:
            raise ValueError(
                f"Reference summary for parameter {name!r} is missing required key "
                f"{key!r}; expected keys: {sorted(_REQUIRED_KEYS)}"
            )
    if any(np.isnan(float(ref[k])) for k in _REQUIRED_KEYS):
        warnings.warn(
            f"Reference summary for parameter {name!r} contains NaN; "
            "skipping this parameter in the max-reduction.",
            stacklevel=3,
        )
        return False
    if float(ref["std"]) <= 0.0:
        warnings.warn(
            f"Reference std for parameter {name!r} is non-positive "
            f"({ref['std']!r}); skipping this parameter.",
            stacklevel=3,
        )
        return False
    return True


def _flatten_draws(arr: np.ndarray) -> np.ndarray:
    """Flatten multi-chain draws to a 1-D or 2-D array over (chains × samples).

    Parameters
    ----------
    arr
        Array of shape ``(num_chains, num_samples, *event)`` or
        ``(num_chains, num_samples)`` for scalar parameters.

    Returns
    -------
    np.ndarray
        Shape ``(num_chains * num_samples, *event)``.
    """
    if arr.ndim < 2:
        raise ValueError(
            f"Draw array has shape {arr.shape}; expected at least 2 axes "
            "(num_chains, num_samples, ...)."
        )
    n_chains, n_samples = arr.shape[0], arr.shape[1]
    event_shape = arr.shape[2:]
    return arr.reshape(n_chains * n_samples, *event_shape)


def _param_metrics(
    flat: np.ndarray,
    ref: _RefSummary,
) -> tuple[float, float, float, float]:
    """Compute the four normalised metrics for a single parameter.

    Parameters
    ----------
    flat
        Flattened draws of shape ``(N, *event)``; statistics computed over
        all ``N`` samples after collapsing event dimensions to a scalar mean.
    ref
        Reference summary ``{"mean", "std", "q05", "q95"}``.

    Returns
    -------
    tuple
        ``(mae_norm, q05_err, q95_err, std_ratio_dev)`` — all normalised by
        ``ref["std"]``.
    """
    # Collapse event dims: compute a scalar summary per sample site.
    # For vector/matrix params we flatten and then compute the mean over
    # all event elements, giving a single "site mean" to compare.
    # This matches how reference summaries are typically produced
    # (marginal summary per scalar parameter, or mean-of-means for vectors).
    flat_scalar = (
        flat.reshape(flat.shape[0], -1).mean(axis=1) if flat.ndim > 1 else flat
    )

    ref_std = float(ref["std"])
    ref_mean = float(ref["mean"])
    ref_q05 = float(ref["q05"])
    ref_q95 = float(ref["q95"])

    emp_mean = float(np.mean(flat_scalar))
    emp_std = float(np.std(flat_scalar, ddof=1)) if flat_scalar.size > 1 else 0.0
    emp_q05 = float(np.quantile(flat_scalar, _QUANTILE_05))
    emp_q95 = float(np.quantile(flat_scalar, _QUANTILE_95))

    # Normalise by REFERENCE std — see module-level Design invariant #1.
    mae_norm = abs(emp_mean - ref_mean) / ref_std
    q05_err = abs(emp_q05 - ref_q05) / ref_std
    q95_err = abs(emp_q95 - ref_q95) / ref_std
    # std_ratio_dev measures whether the recipe produces the right spread.
    std_ratio_dev = abs(emp_std / ref_std - 1.0)

    return mae_norm, q05_err, q95_err, std_ratio_dev


def compute_sample_quality(
    draws: dict[str, np.ndarray] | np.ndarray,
    reference_summaries: dict[str, _RefSummary] | _RefSummary,
) -> dict[str, float]:
    """Compare MCMC draws to a Tier-A reference summary.

    Parameters
    ----------
    draws
        Either a dict ``{param_name: array of shape (num_chains, num_samples, *event)}``
        OR a single array of shape ``(num_chains, num_samples, *event)`` for the
        1-param case.  Multi-chain layout is flattened internally over
        ``(num_chains × num_samples)`` — see Design invariant #2.
    reference_summaries
        Per-parameter summary statistics.  Keys must match ``draws`` keys when
        ``draws`` is a dict.  Each entry is a dict with exactly four keys:

        * ``"mean"`` — reference posterior mean.
        * ``"std"`` — reference posterior standard deviation.
        * ``"q05"`` — reference 5th-percentile.
        * ``"q95"`` — reference 95th-percentile.

        For the 1-param array case, pass a single-level dict with those 4 keys
        directly (no nested per-parameter dict).

    Returns
    -------
    dict[str, float]
        Exactly four keys, each equal to the **max over all parameters** of the
        corresponding normalised error:

        * ``"mae_vs_reference"`` — ``max_i |mean(draws_i) - ref_mean_i| / ref_std_i``
        * ``"q05_error"``        — ``max_i |q05(draws_i)  - ref_q05_i|  / ref_std_i``
        * ``"q95_error"``        — ``max_i |q95(draws_i)  - ref_q95_i|  / ref_std_i``
        * ``"std_ratio_max_dev"`` — ``max_i |std(draws_i)/ref_std_i - 1.0|``

        All metrics are normalised by the REFERENCE std (Design invariant #1)
        so they are unitless and comparable across parameters of different scale.

    Raises
    ------
    ValueError
        If any draw array contains NaN values, if all parameters have NaN
        reference summaries (nothing left to reduce over), or if ``draws`` and
        ``reference_summaries`` have incompatible keys / shapes.

    Warns
    -----
    UserWarning
        If a parameter's reference summary contains NaN — that parameter is
        skipped in the max-reduction and a warning is emitted.

    Notes
    -----
    **Normalisation invariant.**  Metrics are normalised by REFERENCE std, not
    empirical std.  Using empirical std would silently rescale a recipe that
    systematically under- or over-estimates posterior spread.

    **Multi-chain flattening.**  All chains are pooled over ``(chains × samples)``
    before computing empirical statistics; per-chain averages are never taken.
    This avoids double-counting the work of the R̂ / ESS gate.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> draws = {"x": rng.standard_normal((4, 1000, 1))}
    >>> ref = {"x": {"mean": 0.0, "std": 1.0, "q05": -1.645, "q95": 1.645}}
    >>> metrics = compute_sample_quality(draws, ref)
    >>> metrics["mae_vs_reference"] < 0.1
    True
    """
    # ── Normalise inputs to dict form ────────────────────────────────────────
    if isinstance(draws, np.ndarray):
        # Single-array path: wrap in a canonical single-key dict.
        draws_dict: _DrawsDict = {"_param": draws}
        # reference_summaries must be a flat dict with the 4 required keys.
        if not isinstance(reference_summaries, dict):
            raise TypeError(
                "When draws is a numpy array, reference_summaries must be a dict "
                "with keys {'mean', 'std', 'q05', 'q95'}."
            )
        if _REQUIRED_KEYS <= reference_summaries.keys():
            # Flat 4-key dict → single-param case.
            ref_dict: _RefDict = {"_param": reference_summaries}  # type: ignore[dict-item]
        else:
            raise ValueError(
                "When draws is a numpy array, reference_summaries must be a "
                f"single-level dict with keys {sorted(_REQUIRED_KEYS)}; "
                f"got keys {sorted(reference_summaries.keys())}."
            )
    else:
        draws_dict = draws
        if not isinstance(reference_summaries, dict):
            raise TypeError("reference_summaries must be a dict when draws is a dict.")
        # Must be nested: {param: {mean, std, q05, q95}}
        ref_dict = reference_summaries  # type: ignore[assignment]

    # ── Validate draws keys align with reference keys ────────────────────────
    draw_keys = set(draws_dict.keys())
    ref_keys = set(ref_dict.keys())
    if draw_keys != ref_keys:
        raise ValueError(
            f"draws keys {sorted(draw_keys)} do not match "
            f"reference_summaries keys {sorted(ref_keys)}."
        )

    # ── Check for NaN in draws ───────────────────────────────────────────────
    for param, arr in draws_dict.items():
        # Callers must ensure jax.block_until_ready(draws) before calling this function.
        arr = np.asarray(arr)
        if np.any(np.isnan(arr)):
            raise ValueError(
                f"draws for parameter {param!r} contain NaN values. "
                "Filter or fix the chain before computing sample quality."
            )
        draws_dict[param] = arr

    # ── Compute per-parameter metrics ────────────────────────────────────────
    all_mae: list[float] = []
    all_q05: list[float] = []
    all_q95: list[float] = []
    all_std: list[float] = []

    for param in sorted(draws_dict.keys()):
        arr = draws_dict[param]
        ref = ref_dict[param]

        if not _validate_ref_summary(param, ref):
            continue  # NaN reference → skip with warning (already emitted)

        flat = _flatten_draws(arr)
        mae, q05_e, q95_e, std_e = _param_metrics(flat, ref)
        all_mae.append(mae)
        all_q05.append(q05_e)
        all_q95.append(q95_e)
        all_std.append(std_e)

    if not all_mae:
        raise ValueError(
            "All parameters have NaN reference summaries — nothing to reduce over. "
            "Provide at least one parameter with a complete, non-NaN reference summary."
        )

    return {
        "mae_vs_reference": float(np.max(all_mae)),
        "q05_error": float(np.max(all_q05)),
        "q95_error": float(np.max(all_q95)),
        "std_ratio_max_dev": float(np.max(all_std)),
    }
