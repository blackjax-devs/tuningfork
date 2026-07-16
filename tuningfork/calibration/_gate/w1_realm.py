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
"""W1/σ two-prong equivalence gate realm (SECOND-STAGE, post-#227 _gate/ package).

Implements the W1/σ distribution equivalence gate.  Runs only after R̂/ESS/div
PASS.  Compares the generated sample to the multichain GT via per-dim
Wasserstein-1 (W1) normalised by the GT pooled standard deviation (σ_d).

Two prongs
----------
**Max prong** — is any single dimension too far from GT?
    Statistic:  ``max_d W1_d / σ_d``
    Threshold:  ``floor_of_max = q_{1−α}(max_d W1_boot^(b))``, where the
                bootstrap draws per-dim INDEPENDENTLY from the GT empirical
                marginal at effective size.  Provides FWER ≤ α.

**Frac prong** — do too many dimensions deviate (even modestly)?
    Statistic:  fraction of dims with ``W1_d/σ_d > max(floor_d, τ_sci)``
    Threshold:  ``τ_frac = q_{95}(frac_null^(b))``, where null fractions are
                computed from a JOINT contiguous-block bootstrap on the GT
                rows (preserves cross-dim correlation + per-dim autocorr).

k̂ tail guard
-------------
Zhang–Stephens GPD fit on the large GT sample (N≈1e5) per tail (upper/lower
10%).  Uses ``blackjax.diagnostics._gpdfit`` on fixed upper/lower-10%
exceedances — a location-invariant estimator suitable for non-zero-centred
unconstrained posteriors.  When ``max(k̂_left, k̂_right) > 0.7``: the
dimension's W1 is replaced by trimmed-W1 (excluding top/bottom 10%).
Computed on GT, not the small generated sample — avoids noisy k̂ from
small-sample tails.

Public API
----------
``W1RealmResult`` — NamedTuple holding all outputs.
``compute_w1_realm`` — main entry point (three-layer: input validation →
    statistics → verdict assembly).
"""

from __future__ import annotations

from typing import NamedTuple

import arviz as az
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Effective-size cap for the GT side (E_s).  Above 20k, the GT-side floor
# contribution is < 0.007σ — negligible vs τ_sci.  See pinned spec.
_E_S_CAP: int = 20_000

# Scientific materiality bar: 0.05σ shift is detectable and reportable.
# Acts as lower bound for per-dim frac-prong threshold.  See pinned spec.
_TAU_SCI: float = 0.05

# k̂ threshold above which heavy-tail guard activates (trimmed-W1/PIT).
_KHAT_HEAVY_TAIL: float = 0.7

# Default bootstrap parameters.
_DEFAULT_B: int = 5000
_DEFAULT_ALPHA: float = 0.05

# Block size for joint block bootstrap (frac prong).
# ~100 gives ≈1000 independent blocks for N=100k GT rows.
_BLOCK_SIZE: int = 100

# ---------------------------------------------------------------------------
# W1 in-house implementation (no scipy.stats.wasserstein_distance)
# ---------------------------------------------------------------------------


def _w1_equal_n(a: np.ndarray, b: np.ndarray) -> float:
    """Exact 1-D W1 for equal-length samples: ``mean|sort_a − sort_b|``."""
    return float(np.mean(np.abs(np.sort(a) - np.sort(b))))


def _quantile_sorted(x_sorted: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear-interpolation quantile function on a pre-sorted array.

    Equivalent to ``np.quantile(x_sorted, t, method='linear')`` but 100–300×
    faster for large ``x_sorted`` because numpy's ``quantile`` allocates large
    intermediate arrays; this implementation avoids that allocation.

    Parameters
    ----------
    x_sorted
        1-D array, sorted ascending (will not be re-sorted).
    t
        1-D array of quantile levels in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Quantile values, same shape as ``t``.
    """
    n = len(x_sorted)
    indices = t * (n - 1)
    lo = np.floor(indices).astype(np.intp)
    hi = np.minimum(lo + 1, n - 1)
    frac = indices - lo
    return x_sorted[lo] * (1.0 - frac) + x_sorted[hi] * frac


def _w1_unequal_n(a: np.ndarray, b: np.ndarray, n_steps: int = 10_000) -> float:
    """Exact 1-D W1 via quantile-function integral for unequal-length samples.

    ``∫|Q_a(t) − Q_b(t)| dt`` approximated on a uniform grid of ``n_steps``
    points via the trapezoidal rule.  Uses ``_quantile_sorted`` for fast
    linear interpolation instead of ``np.quantile``.
    """
    t = np.linspace(0.0, 1.0, n_steps + 1)
    q_a = _quantile_sorted(np.sort(a), t)
    q_b = _quantile_sorted(np.sort(b), t)
    return float(np.trapezoid(np.abs(q_a - q_b), t))


def _w1_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Dispatch to equal-n or unequal-n 1-D W1."""
    if len(a) == len(b):
        return _w1_equal_n(a, b)
    return _w1_unequal_n(a, b)


# ---------------------------------------------------------------------------
# Zhang–Stephens GPD k̂ estimate
# ---------------------------------------------------------------------------


def _khat_gpd(x: np.ndarray, tail_frac: float = 0.10) -> tuple[float, float]:
    """Zhang–Stephens GPD shape parameter k̂ estimate in both tails of ``x``.

    Fits a Generalised Pareto Distribution (GPD) to the upper and lower
    ``tail_frac`` exceedances using ``blackjax.diagnostics._gpdfit`` (the
    Zhang–Stephens 2009 penalised MLE).  Unlike the Hill estimator, GPD
    fitting is location-invariant and unbiased on non-zero-centred
    unconstrained posteriors (e.g. the eight_schools ``tau`` log-scale
    marginal).

    Returns ``(k̂_left, k̂_right)``.

    Parameters
    ----------
    x
        1-D array of samples (the large GT marginal, N≈1e5).
    tail_frac
        Fraction of data used per tail.  Default 0.10.

    Returns
    -------
    tuple[float, float]
        ``(k̂_left, k̂_right)`` — GPD shape parameters for the lower and upper
        tails.  Negative values indicate bounded support (light tail);
        values above ``_KHAT_HEAVY_TAIL`` (0.7) trigger the trimmed-W1 guard.
        Returns ``0.0`` for a tail when ``n_tail < 5`` (insufficient data).
    """
    from blackjax.diagnostics import _gpdfit as _bj_gpdfit

    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    n_tail = max(5, int(n * tail_frac))
    if n_tail >= n:
        return 0.0, 0.0

    # Right tail: top n_tail exceedances above the (n-n_tail-1)th order stat
    tail_r = x[-n_tail:]
    cutoff_r = x[-n_tail - 1]
    k_r, _ = _bj_gpdfit(tail_r - cutoff_r)

    # Left tail: flip sign (so lower tail becomes an upper exceedance problem)
    x_flip = -x[::-1]  # sorted ascending, values ≥ 0 after flip
    tail_l = x_flip[-n_tail:]
    cutoff_l = x_flip[-n_tail - 1]
    k_l, _ = _bj_gpdfit(tail_l - cutoff_l)

    return float(k_l), float(k_r)


def _khat_max(x: np.ndarray, tail_frac: float = 0.10) -> float:
    """Return ``max(k̂_left, k̂_right)`` for 1-D array ``x``."""
    k_left, k_right = _khat_gpd(x, tail_frac)
    return max(k_left, k_right)


# ---------------------------------------------------------------------------
# Trimmed W1 (k̂ > 0.7 guard)
# ---------------------------------------------------------------------------


def _w1_trimmed(a: np.ndarray, b: np.ndarray, trim_frac: float = 0.10) -> float:
    """W1 on the interior quantile range (excluding top/bottom ``trim_frac``)."""
    lo, hi = trim_frac, 1.0 - trim_frac
    a_clipped = np.clip(a, float(np.quantile(a, lo)), float(np.quantile(a, hi)))
    b_clipped = np.clip(b, float(np.quantile(b, lo)), float(np.quantile(b, hi)))
    return _w1_1d(a_clipped, b_clipped)


# ---------------------------------------------------------------------------
# Per-dim ESS from generated sample
# ---------------------------------------------------------------------------


def _ess_gen_per_dim(gen_arr: np.ndarray) -> np.ndarray:
    """Compute ``min(bulk_ess, tail_ess)`` per dim for the generated sample.

    Parameters
    ----------
    gen_arr
        Shape ``(n_chains, n_draws, *event_shape)`` — the multichain generated
        sample (may be a single chain, i.e. ``n_chains=1``).

    Returns
    -------
    np.ndarray
        Per-dimension ``min(bulk, tail)``-ESS; shape is ``(D,)`` where
        ``D = prod(event_shape)`` (flattened).  Falls back to
        ``n_chains * n_draws`` when the ESS computation returns NaN or
        non-positive values (e.g. fewer than 4 draws).

    Notes
    -----
    Uses ``az.from_dict`` to wrap the generated sample as a DataTree before
    calling ``az.ess``.  This avoids the arviz 1.1.0
    ``_ess_tail() missing 'prob'`` error that affects the raw-array
    ``az.ess(arr, chain_axis=0, draw_axis=1, method="tail")`` call path.
    Any future regression in the ESS API will surface as an uncaught
    exception here — no bare ``except Exception`` swallows unknown failures.
    """
    n_total = gen_arr.shape[0] * gen_arr.shape[1]
    d_shape = gen_arr.shape[2:] if gen_arr.ndim > 2 else ()
    fallback_shape = d_shape if d_shape else (1,)

    # Wrap as a DataTree so that az.ess uses the DataTree code path, which
    # correctly handles both bulk and tail under arviz ≥1.1.0.  This is the
    # same pattern as ``compute_summary_stats`` in groundtruth/_emit.py.
    idata = az.from_dict(
        {"posterior": {"__gen__": gen_arr}},
        sample_dims=["chain", "draw"],
    )
    bulk_xr = az.ess(idata, method="bulk")["__gen__"]
    tail_xr = az.ess(idata, method="tail")["__gen__"]
    bulk = np.atleast_1d(np.asarray(bulk_xr).ravel())
    tail = np.atleast_1d(np.asarray(tail_xr).ravel())
    ess = np.minimum(bulk, tail)

    if not np.all(np.isfinite(ess)) or np.any(ess <= 0):
        # Fewer than 4 draws, constant chain, or other degenerate input.
        return np.full(fallback_shape, float(n_total))
    return ess


# ---------------------------------------------------------------------------
# Core W1/σ statistic computation
# ---------------------------------------------------------------------------


def _compute_w1_per_dim(
    gt_flat_by_dim: list[np.ndarray],
    gen_flat_by_dim: list[np.ndarray],
    sigma_by_dim: np.ndarray,
    khat_by_dim: np.ndarray,
) -> np.ndarray:
    """Compute per-dim ``W1_d / σ_d`` with k̂ routing.

    Parameters
    ----------
    gt_flat_by_dim
        List of length D; each element is a 1-D array of GT draws for dim d.
    gen_flat_by_dim
        List of length D; each element is a 1-D array of generated draws.
    sigma_by_dim
        Shape (D,) — GT pooled std per dim.
    khat_by_dim
        Shape (D,) — pre-computed ``max(k̂_left, k̂_right)`` per dim (from GT).

    Returns
    -------
    np.ndarray
        Shape (D,) — ``W1_d / σ_d`` per dimension.
    """
    d = len(gt_flat_by_dim)
    w1_sigma = np.zeros(d)
    for i in range(d):
        gt_i = gt_flat_by_dim[i]
        gen_i = gen_flat_by_dim[i]
        sig = float(sigma_by_dim[i])
        if sig <= 0:
            w1_sigma[i] = float("inf")
            continue
        if khat_by_dim[i] > _KHAT_HEAVY_TAIL:
            # Heavy-tail guard: trimmed W1
            w1_val = _w1_trimmed(gt_i, gen_i)
        else:
            w1_val = _w1_1d(gt_i, gen_i)
        w1_sigma[i] = w1_val / sig
    return w1_sigma


# ---------------------------------------------------------------------------
# Floor construction (max prong + frac prong)
# ---------------------------------------------------------------------------


def _build_floor(
    gt_flat_by_dim: list[np.ndarray],
    sigma_by_dim: np.ndarray,
    e_g_by_dim: np.ndarray,
    e_s_by_dim: np.ndarray,
    *,
    B: int = _DEFAULT_B,
    alpha: float = _DEFAULT_ALPHA,
    seed: int | None = None,
) -> tuple[float, np.ndarray, float]:
    """Bootstrap the W1/σ floor for both prongs.

    For b=1..B, per dim d:
    - Draw ``E_g,d`` iid samples from the GT empirical marginal (max-prong:
      per-dim independent).
    - Draw ``E_s,d`` iid samples from the GT empirical marginal (independent).
    - Compute ``W1_d/σ_d``.

    Max prong: ``floor_of_max = q_{1−α}(max_d W1^(b))``.
    Per-dim:   ``floor_d     = q_{1−α}(W1_d^(b))``.

    Returns ``(floor_of_max, floor_per_dim, tau_frac)`` where ``tau_frac`` is
    the q95 of a JOINT contiguous-block bootstrap null frac-statistic.

    Parameters
    ----------
    gt_flat_by_dim
        List of length D; element d is the full GT marginal for dim d.
    sigma_by_dim
        Shape (D,) — GT pooled std.
    e_g_by_dim
        Shape (D,) — effective size of the generated sample per dim.
    e_s_by_dim
        Shape (D,) — effective size of the GT per dim (will be capped at
        ``_E_S_CAP``).
    B
        Number of bootstrap replicates.
    alpha
        Family-wise error rate.  ``floor_of_max = q_{1−alpha}``.
    seed
        Optional integer seed for the numpy PCG64 RNG.  ``None`` → OS entropy.

    Returns
    -------
    tuple[float, np.ndarray, float]
        ``(floor_of_max, floor_per_dim, tau_frac)``
        - ``floor_of_max``  : scalar, q_{1−alpha} of bootstrap max.
        - ``floor_per_dim`` : shape (D,), per-dim q_{1−alpha}.
        - ``tau_frac``      : scalar, q_{0.95} of joint block-bootstrap null
          frac; may be ``nan`` if GT matrix cannot be assembled.
    """
    d = len(gt_flat_by_dim)
    e_s_capped = np.minimum(np.asarray(e_s_by_dim, dtype=np.float64), float(_E_S_CAP))
    e_g = np.asarray(e_g_by_dim, dtype=np.float64)

    rng = np.random.default_rng(np.random.PCG64(seed))

    # Per-dim iid bootstrap (independent across dims)
    boot_per_dim = np.zeros((d, B))
    for dim_i in range(d):
        marginal = gt_flat_by_dim[dim_i]
        n_g = max(1, int(round(float(e_g[dim_i]))))
        n_s = max(1, int(round(float(e_s_capped[dim_i]))))
        sig = float(sigma_by_dim[dim_i])
        if sig <= 0:
            boot_per_dim[dim_i, :] = float("nan")
            continue
        for b in range(B):
            sample_g = rng.choice(marginal, size=n_g, replace=True)
            sample_s = rng.choice(marginal, size=n_s, replace=True)
            boot_per_dim[dim_i, b] = _w1_1d(sample_g, sample_s) / sig

    # Max prong floor
    boot_max = np.nanmax(boot_per_dim, axis=0)  # shape (B,)
    floor_of_max = float(np.nanquantile(boot_max, 1.0 - alpha))

    # Per-dim floor
    floor_per_dim = np.nanquantile(boot_per_dim, 1.0 - alpha, axis=1)  # shape (D,)

    # Frac prong τ_frac: joint block bootstrap on GT rows
    tau_frac = _compute_tau_frac(
        gt_flat_by_dim=gt_flat_by_dim,
        sigma_by_dim=sigma_by_dim,
        floor_per_dim=floor_per_dim,
        n_gen=int(np.median(e_g)),
        B=B,
        rng=rng,
    )

    return floor_of_max, floor_per_dim, tau_frac


def _compute_tau_frac(
    gt_flat_by_dim: list[np.ndarray],
    sigma_by_dim: np.ndarray,
    floor_per_dim: np.ndarray,
    n_gen: int,
    B: int,
    rng: np.random.Generator,
) -> float:
    """Joint contiguous-block bootstrap for frac-prong null threshold (τ_frac).

    Builds a single GT matrix ``(N_gt, D)`` from all per-dim marginals, then
    draws contiguous row blocks to simulate a null generated sample of size
    ``n_gen``.  Returns the q95 of the null frac statistic.

    Returns ``nan`` when any per-dim array has a different length (can't build
    a joint matrix) or when D < 2 (frac prong is trivial with D=1).
    """
    d = len(gt_flat_by_dim)
    if d < 2:
        return float("nan")

    # All GT marginals must have the same length for the joint matrix
    n_gt = len(gt_flat_by_dim[0])
    if not all(len(m) == n_gt for m in gt_flat_by_dim):
        return float("nan")

    # Build joint GT matrix (N_gt, D)
    gt_matrix = np.column_stack(
        [np.asarray(m, dtype=np.float64) for m in gt_flat_by_dim]
    )

    block_size = _BLOCK_SIZE
    n_blocks = n_gt // block_size
    if n_blocks < 2:
        # Not enough GT rows for block bootstrap; return nan
        return float("nan")

    # Per-dim threshold: max(floor_d, τ_sci)
    tau_thresh = np.maximum(floor_per_dim, _TAU_SCI)
    sigma_arr = np.asarray(sigma_by_dim, dtype=np.float64)

    n_blocks_gen = max(1, n_gen // block_size)
    # Similarly for the "GT resample" side
    n_blocks_s = n_blocks_gen

    null_fracs = np.zeros(B)
    for b in range(B):
        # Sample contiguous row blocks for generated side
        starts_g = rng.integers(0, n_blocks, size=n_blocks_gen)
        rows_g = np.concatenate(
            [np.arange(s * block_size, (s + 1) * block_size) for s in starts_g]
        )
        gen_block = gt_matrix[rows_g[:n_gen], :]

        # Sample contiguous row blocks for GT side (independent)
        starts_s = rng.integers(0, n_blocks, size=n_blocks_s)
        rows_s = np.concatenate(
            [np.arange(s * block_size, (s + 1) * block_size) for s in starts_s]
        )
        gt_block = gt_matrix[rows_s[:n_gen], :]

        # Per-dim W1/σ
        w1_boot = np.zeros(d)
        for dim_i in range(d):
            sig = float(sigma_arr[dim_i])
            if sig <= 0:
                w1_boot[dim_i] = float("nan")
                continue
            w1_boot[dim_i] = _w1_1d(gen_block[:, dim_i], gt_block[:, dim_i]) / sig

        frac = float(np.nanmean(w1_boot > tau_thresh))
        null_fracs[b] = frac

    return float(np.quantile(null_fracs, 0.95))


# ---------------------------------------------------------------------------
# LOO conservatism guard (pin-time validation)
# ---------------------------------------------------------------------------


def _k_crit_binom(n: int, p: float = _DEFAULT_ALPHA, fwer: float = 0.10) -> int:
    """Smallest k such that P(Bin(n, p) >= k+1) <= fwer.

    Used by ``_loo_conservatism_check`` to allow the expected exceedance rate
    under a correctly-calibrated floor.  The floor is a q_{1-p} quantile,
    so ~p * n_chains exceedances are expected in the LOO null distribution;
    a "0 violations" rule would reject a correctly-calibrated floor with
    probability ~40% for n_chains=10, p=0.05.

    For n=10, p=0.05, fwer=0.10:
        P(X >= 2) = 0.0861 <= 0.10 → k_crit = 1.

    Parameters
    ----------
    n
        Number of trials (n_chains).
    p
        Expected exceedance probability per chain (default α=0.05).
    fwer
        Family-wise error rate bound.  k_crit is the largest k for which
        P(X >= k+1) <= fwer.

    Returns
    -------
    int
        Largest allowed violation count; 0 when the bound cannot be achieved
        at k=0 (i.e., n is very large or p is high).
    """
    # Compute binomial PMF incrementally: P(X=j) = C(n,j) p^j (1-p)^{n-j}
    q = 1.0 - p
    pmf = [0.0] * (n + 1)
    pmf[0] = q**n
    for j in range(1, n + 1):
        pmf[j] = pmf[j - 1] * ((n - j + 1) / j) * (p / q)
    # Accumulate CDF
    cdf = [0.0] * (n + 1)
    cdf[0] = pmf[0]
    for j in range(1, n + 1):
        cdf[j] = cdf[j - 1] + pmf[j]
    # Find smallest k s.t. P(X >= k+1) = 1 - CDF[k] <= fwer
    for k in range(n + 1):
        if 1.0 - cdf[k] <= fwer:
            return k
    return n


def _loo_conservatism_check(
    gt_flat_by_dim: list[np.ndarray],
    sigma_by_dim: np.ndarray,
    e_g_by_dim: np.ndarray,
    floor_of_max: float,
    n_chains: int,
    *,
    alpha: float = _DEFAULT_ALPHA,
    seed: int | None = None,
) -> dict:
    """Validate floor_of_max against the real leave-one-chain-out null.

    **Pin-time guard only** — call once when pinning a model's floor, not on
    every runtime invocation of ``compute_w1_realm``.

    Computes the LOO null by treating each GT chain in turn as the "generated
    sample" (each held-out chain is compared against the OTHER n_chains−1
    chains) and computing max_d W1/σ against the remaining chains.  Returns all
    LOO values and whether floor_of_max is conservative under the
    count-and-severity criterion.

    **PASS criterion:**

    ``is_conservative = (violations <= k_crit) AND (max_LOO <= floor_of_max * 1.05)``

    where

    * ``k_crit = _k_crit_binom(n_chains)`` — allows the expected exceedance rate
      (floor is a q_{1-α} quantile, so ~0.5/10 exceedances are expected; "0
      violations" rejects a correctly-calibrated floor ~40% of the time).
    * severity rule ``max_LOO <= floor * 1.05`` — catches gross overshoots (e.g.
      radon ESS-88 dim at +7.2% pre-fix fails; eight_schools at +1.9% passes).

    Parameters
    ----------
    gt_flat_by_dim
        List of length D; element d holds (n_chains * n_draws) GT samples.
    sigma_by_dim
        Shape (D,) — GT pooled std.
    e_g_by_dim
        Shape (D,) — raw draw count per dim for the held-out gen slice
        (use raw n_draws, NOT effective ESS, to match the gate's comparison).
    floor_of_max
        The bootstrap floor_of_max to validate.
    n_chains
        Number of chains in the GT (used to split gt_flat_by_dim).
    alpha
        Not used directly; kept for API symmetry.
    seed
        Not used; kept for API symmetry.

    Returns
    -------
    dict with keys:
        ``loo_max_w1_sigma``: list of float (one per held-out chain).
        ``floor_of_max``:     float (the input value, echoed).
        ``is_conservative``:  bool — True when both count and severity rules pass.
        ``violation_count``:  int (number of chains where LOO > floor_of_max).
        ``k_crit``:           int (allowed violation count for n_chains).
    """
    d = len(gt_flat_by_dim)
    # Infer n_draws per chain from total length
    n_total = len(gt_flat_by_dim[0])
    if n_total % n_chains != 0:
        raise ValueError(
            f"_loo_conservatism_check: n_total ({n_total}) is not divisible "
            f"by n_chains ({n_chains}).  GT draws must be evenly split across "
            f"chains.  Check that gt_flat_by_dim was built with n_chains×n_draws."
        )
    n_draws_per_chain = n_total // n_chains

    loo_max_values = []
    for held_out in range(n_chains):
        # Split GT marginal per dim into held-out chain vs the OTHER n_chains-1
        # (leave-one-out: held chain compared to the remaining chains only).
        gen_per_dim = []
        rest_per_dim = []
        for dim_i in range(d):
            marginal = gt_flat_by_dim[dim_i]
            reshaped = marginal.reshape(n_chains, n_draws_per_chain)
            # Limit to raw draw count E_g (not ESS — see docstring)
            n_g = max(1, int(round(float(e_g_by_dim[dim_i]))))
            n_g = min(n_g, n_draws_per_chain)
            gen_per_dim.append(reshaped[held_out, :n_g])
            # Remaining chains flattened (leave-one-out: exclude held chain)
            rest_idx = [c for c in range(n_chains) if c != held_out]
            rest_per_dim.append(reshaped[rest_idx, :].ravel())

        # Compute max_d W1/σ for this LOO split
        w1_vals = np.zeros(d)
        for dim_i in range(d):
            sig = float(sigma_by_dim[dim_i])
            if sig <= 0:
                w1_vals[dim_i] = float("inf")
                continue
            w1_vals[dim_i] = _w1_1d(gen_per_dim[dim_i], rest_per_dim[dim_i]) / sig
        loo_max_values.append(float(np.max(w1_vals)))

    k_crit = _k_crit_binom(n_chains)
    violation_count = sum(v > floor_of_max for v in loo_max_values)
    max_loo = max(loo_max_values)
    # Count-and-severity criterion: count rule + severity rule
    count_ok = violation_count <= k_crit
    severity_ok = max_loo <= floor_of_max * 1.05
    is_conservative = count_ok and severity_ok
    return {
        "loo_max_w1_sigma": loo_max_values,
        "floor_of_max": floor_of_max,
        "is_conservative": is_conservative,
        "violation_count": violation_count,
        "k_crit": k_crit,
    }


# ---------------------------------------------------------------------------
# NamedTuple result
# ---------------------------------------------------------------------------


class W1RealmResult(NamedTuple):
    """Output of the W1/σ two-prong equivalence gate realm.

    Parameters
    ----------
    max_w1_sigma
        ``max_d W1_d/σ_d`` over all dims — the max-prong statistic.
    floor_of_max
        Bootstrap ``q_{1−α}(max_d W1_boot^(b))`` — the max-prong threshold.
    frac_failing_dims
        Fraction of dims with ``W1_d/σ_d > max(floor_d, τ_sci)`` — the
        frac-prong statistic.
    tau_frac
        q_{0.95} of the joint block-bootstrap null frac — the frac-prong
        threshold.  May be ``nan`` when insufficient GT data.
    w1_sigma_per_dim
        Per-dim ``W1_d/σ_d`` values (for diagnostics).
    floor_per_dim
        Per-dim ``q_{1−α}`` floor (for diagnostics).
    khat_per_dim
        Per-dim ``max(k̂_left, k̂_right)``; ``> 0.7`` dims used trimmed-W1.
    max_prong_verdict
        ``"PASS"`` or ``"FAIL"`` from the max prong.
    frac_prong_verdict
        ``"PASS"`` or ``"FAIL"`` from the frac prong; ``"SKIP"`` when
        ``tau_frac`` is ``nan``.
    verdict
        Overall verdict: worst of ``max_prong_verdict`` and
        ``frac_prong_verdict`` (``"SKIP"`` counts as ``"PASS"`` for
        aggregation).
    alpha
        FWER level used.
    B
        Number of bootstrap replicates used for the floor.
    n_dims
        Total number of dimensions (D).
    n_heavy_tail_dims
        Number of dims with ``k̂ > 0.7`` (routing to trimmed-W1).
    loo_check
        Always ``None`` at runtime — the LOO conservatism guard is a
        pin-time validation (call ``_loo_conservatism_check`` separately
        when pinning a new model floor, not on each runtime invocation).
    """

    max_w1_sigma: float
    floor_of_max: float
    frac_failing_dims: float
    tau_frac: float
    w1_sigma_per_dim: np.ndarray
    floor_per_dim: np.ndarray
    khat_per_dim: np.ndarray
    max_prong_verdict: str  # "PASS" | "FAIL"
    frac_prong_verdict: str  # "PASS" | "FAIL" | "SKIP"
    verdict: str  # "PASS" | "FAIL"
    alpha: float
    B: int
    n_dims: int
    n_heavy_tail_dims: int
    loo_check: (
        dict | None
    )  # always None at runtime; use _loo_conservatism_check for pin-time guard


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_w1_realm(
    samples: dict[str, np.ndarray],
    ground_truth_summaries: dict[str, dict],
    gt_draws: dict[str, np.ndarray],
    *,
    B: int = _DEFAULT_B,
    alpha: float = _DEFAULT_ALPHA,
    seed: int | None = None,
    multichain: bool = True,
    n_gt_chains: int | None = None,
) -> W1RealmResult:
    """Run the W1/σ two-prong equivalence gate.

    This is the SECOND-STAGE realm — call only after R̂/ESS/div PASS.

    Parameters
    ----------
    samples
        Generated sample, dict of arrays.  Each array has shape
        ``(n_chains, n_draws, *event_shape)`` when ``multichain=True``, or
        ``(n_draws, *event_shape)`` for single-chain (auto-expanded).
    ground_truth_summaries
        Per-site summary stats from the multichain GT summary_v2 format.
        Required keys per site: ``"std"``, ``"bulk_ess"``, ``"tail_ess"``.
        Used for σ_d and E_s,d.
    gt_draws
        Per-site arrays of GT draws with shape
        ``(n_gt_chains, n_gt_draws, *event_shape)`` or
        ``(n_gt_draws, *event_shape)``.  Used for the GT empirical marginals
        and the k̂ GPD estimation.
    B
        Number of bootstrap replicates.  Default 5000 (pinned spec).
    alpha
        FWER level for the max prong.  Default 0.05.
    seed
        Integer seed for the PCG64 RNG.  ``None`` → OS entropy.
    multichain
        When ``True`` (default), treat ``samples`` arrays as already having the
        ``(n_chains, n_draws, ...)`` layout.  When ``False``, the first axis
        is ``n_draws``.
    n_gt_chains
        Unused — kept for backward API compatibility.  The LOO conservatism
        check has been moved to a pin-time helper (``_loo_conservatism_check``)
        and is no longer called at runtime.

    Returns
    -------
    W1RealmResult
        Named tuple with all statistics and verdicts.
    """
    # --- Gather per-site data into flat dim-indexed lists ---
    sites = list(samples.keys())

    gt_flat_by_dim: list[np.ndarray] = []
    gen_flat_by_dim: list[np.ndarray] = []
    sigma_list: list[float] = []
    e_s_list: list[float] = []
    e_g_list: list[float] = []

    for site in sites:
        if site not in ground_truth_summaries:
            continue
        if site not in gt_draws:
            continue

        ps = ground_truth_summaries[site]
        sigma_d = np.asarray(ps["std"], dtype=np.float64)
        bulk_ess_d = np.asarray(ps["bulk_ess"], dtype=np.float64)
        tail_ess_d = np.asarray(ps["tail_ess"], dtype=np.float64)
        e_s_d = np.minimum(bulk_ess_d, tail_ess_d)  # E_s per dim

        # Generated sample: expand to (n_chains, n_draws, *event)
        gen_arr = np.asarray(samples[site], dtype=np.float64)
        if not multichain:
            gen_arr = gen_arr[np.newaxis, ...]  # (1, n_draws, ...)
        if gen_arr.ndim == 2:
            gen_arr = gen_arr[:, :, np.newaxis]  # (n_chains, n_draws, 1)

        # E_g from generated sample: real min(bulk, tail)-ESS per dim.
        # _build_floor uses real ESS (not raw draw count) for calibration.
        e_g_arr = _ess_gen_per_dim(gen_arr)
        if e_g_arr.shape == ():
            e_g_arr = np.full(sigma_d.shape, float(e_g_arr))

        # GT draws: (n_gt_chains, n_gt_draws, *event)
        gt_arr = np.asarray(gt_draws[site], dtype=np.float64)
        if gt_arr.ndim == 2:
            gt_arr = gt_arr[:, :, np.newaxis]  # (n_gt_chains, n_gt_draws, 1)

        d = gt_arr.shape[2] if gt_arr.ndim > 2 else 1

        for dim_i in range(d):
            gt_flat_by_dim.append(gt_arr[:, :, dim_i].ravel())
            gen_flat_by_dim.append(gen_arr[:, :, dim_i].ravel())
            sigma_list.append(
                float(sigma_d.ravel()[dim_i] if sigma_d.ndim > 0 else sigma_d)
            )
            e_s_list.append(float(e_s_d.ravel()[dim_i] if e_s_d.ndim > 0 else e_s_d))
            e_g_list.append(
                float(e_g_arr.ravel()[dim_i] if e_g_arr.ndim > 0 else e_g_arr)
            )

    d_total = len(gt_flat_by_dim)
    if d_total == 0:
        # No matching sites between samples and gt_draws — return degenerate SKIP.
        empty = np.array([])
        return W1RealmResult(
            max_w1_sigma=float("nan"),
            floor_of_max=float("nan"),
            frac_failing_dims=float("nan"),
            tau_frac=float("nan"),
            w1_sigma_per_dim=empty,
            floor_per_dim=empty,
            khat_per_dim=empty,
            max_prong_verdict="SKIP",
            frac_prong_verdict="SKIP",
            verdict="SKIP",
            alpha=alpha,
            B=B,
            n_dims=0,
            n_heavy_tail_dims=0,
            loo_check=None,
        )

    sigma_arr = np.array(sigma_list)
    e_s_arr = np.array(e_s_list)
    e_g_arr = np.array(e_g_list)

    # --- k̂ guard (Zhang–Stephens GPD; computed on large GT sample) ---
    khat_arr = np.array([_khat_max(m) for m in gt_flat_by_dim])
    n_heavy_tail = int(np.sum(khat_arr > _KHAT_HEAVY_TAIL))

    # --- Per-dim W1/σ (with k̂ routing) ---
    w1_sigma = _compute_w1_per_dim(gt_flat_by_dim, gen_flat_by_dim, sigma_arr, khat_arr)

    # --- Floor construction (uses real min(bulk,tail)-ESS as E_g) ---
    floor_of_max, floor_per_dim, tau_frac = _build_floor(
        gt_flat_by_dim=gt_flat_by_dim,
        sigma_by_dim=sigma_arr,
        e_g_by_dim=e_g_arr,
        e_s_by_dim=e_s_arr,
        B=B,
        alpha=alpha,
        seed=seed,
    )

    # --- Max prong verdict ---
    max_w1_sigma = float(np.nanmax(w1_sigma))
    max_verdict = "PASS" if max_w1_sigma <= floor_of_max else "FAIL"

    # --- Frac prong verdict ---
    tau_thresh_d = np.maximum(floor_per_dim, _TAU_SCI)
    frac_failing = float(np.nanmean(w1_sigma > tau_thresh_d))

    if np.isnan(tau_frac):
        frac_verdict = "SKIP"
    else:
        frac_verdict = "PASS" if frac_failing <= tau_frac else "FAIL"

    # --- Overall verdict (worst of max + frac; SKIP ≡ PASS for aggregation) ---
    _RANK = {"PASS": 0, "SKIP": 0, "FAIL": 1}
    _RANK_STR = {0: "PASS", 1: "FAIL"}
    overall = _RANK_STR[max(_RANK[max_verdict], _RANK[frac_verdict])]

    # LOO conservatism guard is a pin-time validation (call _loo_conservatism_check
    # separately when pinning a new model floor).  Not run at runtime — it is 10×
    # LOO-W1 over all dims (expensive), and a runtime gen has no leave-one-out to
    # perform.  loo_check is always None in the runtime result.
    loo_result: dict | None = None

    return W1RealmResult(
        max_w1_sigma=max_w1_sigma,
        floor_of_max=floor_of_max,
        frac_failing_dims=frac_failing,
        tau_frac=tau_frac,
        w1_sigma_per_dim=w1_sigma,
        floor_per_dim=floor_per_dim,
        khat_per_dim=khat_arr,
        max_prong_verdict=max_verdict,
        frac_prong_verdict=frac_verdict,
        verdict=overall,
        alpha=alpha,
        B=B,
        n_dims=d_total,
        n_heavy_tail_dims=n_heavy_tail,
        loo_check=loo_result,
    )
