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
"""Headline metric for tuningfork: min bulk-ESS per gradient evaluation.

The metric an algorithm developer chases:

    headline(states_position, n_grad_evals) =
        min over dims of bulk_ESS(states_position) / n_grad_evals

where bulk_ESS is ``blackjax.diagnostics.ess_bulk`` — the rank-normalised
split-chain estimator of Vehtari et al. (2021).  This is the estimator every
published comparator quotes (Stan, ArviZ ``ess(method="bulk")``, NumPyro), so a
tuningfork headline is directly comparable to a literature number.

The min is taken across ALL dimensions of ALL sample sites — the worst-mixing
dimension determines effective run length. n_grad_evals comes from
``tuningfork.metrics.grad_counter.total_grad_evals`` and is hardware-independent.

Two estimators, one set of draws
--------------------------------
``blackjax.diagnostics`` exposes two ESS entry points that share an
autocorrelation core:

``effective_sample_size``
    Raw multi-chain autocorrelation ESS (Geyer initial monotone sequence).  No
    chain splitting, no rank normalisation.  This is what the headline used
    before the estimator switch; it is retained here as
    :func:`min_bulk_ess_classic_legacy` so a re-emitted cell can report BOTH
    values computed on the SAME draws.  Without that, a headline change cannot
    be attributed to the estimator rather than to fresh draws.

``ess_bulk``
    Splits each chain in half, rank-normalises the pooled draws with the Blom
    plotting position, then applies the same autocorrelation core.  Splitting
    surfaces within-chain drift; rank normalisation makes the estimate robust to
    heavy tails.  Both effects move the estimate relative to the raw version, and
    neither has a fixed sign — see :func:`estimator_ratio`.

Empirically verified (see tests/metrics/test_headline.py):
- ``ess_bulk((C, S, D)) → shape (D,)``: output is already aggregated across
  chains per the standard bulk-ESS definition.
- i.i.d. baseline: headline / (C*S) ≈ 0.85–1.05 for C=4, S=1000, D=5.
- AR(1) φ=0.9 suppression: headline_iid / headline_ar1 ≈ 15–22 for S=2000.
"""

from collections.abc import Callable, Mapping
from typing import Any

import jax.numpy as jnp
from blackjax.diagnostics import effective_sample_size, ess_bulk

__all__ = [
    "HEADLINE_ESS_ESTIMATOR",
    "LEGACY_ESS_ESTIMATOR",
    "build_headline_basis",
    "estimator_ratio",
    "min_bulk_ess",
    "min_bulk_ess_classic_legacy",
    "min_bulk_ess_per_grad",
]

#: Provenance stamp written into ``Recipe.headline_basis["ess_estimator"]``.
#: Any code path that fills a headline must record which estimator produced it —
#: a basis that is merely self-consistent cannot be checked for provenance.
HEADLINE_ESS_ESTIMATOR = "ess_bulk"

#: The pre-switch estimator, kept for side-by-side attribution on the same draws.
LEGACY_ESS_ESTIMATOR = "effective_sample_size"


def _min_ess_over_sites(
    states_position: Mapping[str, Any],
    ess_fn: Callable[..., Any],
) -> Any:
    """Minimum per-dimension ESS across every dimension of every sample site.

    Shared traversal + validation for both estimators so the two values a
    re-emitted cell reports differ ONLY in the estimator, never in how the sites
    were flattened or aggregated.
    """
    if not states_position:
        raise ValueError("states_position is empty; need at least one site")

    min_ess = jnp.inf
    for site, arr in states_position.items():
        if arr.ndim < 2:
            raise ValueError(
                f"site {site!r} has shape {arr.shape}; "
                "expected (n_chains, n_samples, ...) — at least 2 dimensions"
            )
        # Collapse site dims into one trailing axis: (C, S, D)
        # A scalar site (C, S) becomes (C, S, 1).
        flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
        # ess_per_dim has shape (D,) — already aggregated across chains.
        ess_per_dim = ess_fn(flat, chain_axis=0, sample_axis=1)
        min_ess = jnp.minimum(min_ess, jnp.min(ess_per_dim))
    return min_ess


def min_bulk_ess(states_position: Mapping[str, Any]) -> float:
    """Minimum rank-normalised split-chain bulk-ESS across all sites and dims.

    This is the numerator of the headline metric.  See the module docstring for
    the estimator contract.

    Parameters
    ----------
    states_position
        Dict keyed by sample-site name; each value has shape
        ``(n_chains, n_samples, *site_shape)``.

    Returns
    -------
    float
        ``min over all dims of ess_bulk``.

    Raises
    ------
    ValueError
        If ``states_position`` is empty or any site array has ``ndim < 2``.
    """
    return float(_min_ess_over_sites(states_position, ess_bulk))


def min_bulk_ess_classic_legacy(states_position: Mapping[str, Any]) -> float:
    """Minimum ESS under the pre-switch estimator, on the SAME draws.

    ``blackjax.diagnostics.effective_sample_size`` — no chain splitting, no rank
    normalisation.  Reported alongside :func:`min_bulk_ess` so that a change in a
    committed headline can be attributed to the estimator rather than to a fresh
    set of draws.  Not used for any gate or ranking decision.

    Parameters
    ----------
    states_position
        Same contract as :func:`min_bulk_ess`.

    Returns
    -------
    float
        ``min over all dims of effective_sample_size``.
    """
    return float(_min_ess_over_sites(states_position, effective_sample_size))


def estimator_ratio(
    rank_normalised: float | None, classic: float | None
) -> float | None:
    """``rank_normalised / classic`` — the estimator effect isolated on one run.

    Both arguments must come from the same draws.  Returns ``None`` when the
    ratio is undefined (non-finite or non-positive ``classic``), so callers
    record a null rather than an ``inf`` that would later read as a real effect.
    """
    if classic is None or rank_normalised is None:
        return None
    if not jnp.isfinite(classic) or classic <= 0.0:
        return None
    if not jnp.isfinite(rank_normalised):
        return None
    return float(rank_normalised / classic)


def build_headline_basis(
    states_position: Mapping[str, Any],
    *,
    denominator: float,
    total_grad_evals: int,
    grad_count_convention: str,
    is_lower_bound: bool,
) -> tuple[float, dict[str, Any]]:
    """Compute the headline metric and its accounting basis from one set of draws.

    Single source of truth for every emit path that stamps a headline, so the
    estimator provenance and the legacy side-by-side value cannot drift between
    the recipe runner, the low-rank-diagonal emit path, and any future path.

    Parameters
    ----------
    states_position
        Dict keyed by sample-site name; each value has shape
        ``(n_chains, n_samples, *site_shape)``.
    denominator
        What the headline divides by.  Normally ``total_grad_evals``; for a
        gradient-free sampler it is the total draw count, because the metric is
        then per-draw efficiency rather than per-gradient.  Must be > 0.
    total_grad_evals
        Gradient evaluations recorded for the run.  ``0`` for gradient-free
        samplers, where it documents the convention rather than the divisor.
    grad_count_convention
        Formula text from ``BaseMethod.grad_count_convention``.
    is_lower_bound
        True when the gradient count is a lower bound (Laplace family).

    Returns
    -------
    tuple of (headline_metric, headline_basis)
        ``headline_basis["min_bulk_ess"]`` is back-derived as
        ``headline * denominator`` so the recorded basis reproduces the recorded
        headline exactly, with no rounding slack for an invariant test to absorb.

    Raises
    ------
    ValueError
        If ``denominator <= 0``.
    """
    if denominator <= 0:
        raise ValueError(f"denominator must be positive, got {denominator!r}")

    rank_normalised = min_bulk_ess(states_position)
    classic = min_bulk_ess_classic_legacy(states_position)
    headline = rank_normalised / float(denominator)

    basis = {
        "total_grad_evals": int(total_grad_evals),
        # Back-derived (exact, no rounding) so headline == basis / denominator.
        "min_bulk_ess": headline * denominator,
        "ess_estimator": HEADLINE_ESS_ESTIMATOR,
        "min_bulk_ess_classic_legacy": classic,
        "estimator_ratio": estimator_ratio(rank_normalised, classic),
        "grad_count_convention": grad_count_convention,
        "is_lower_bound": is_lower_bound,
    }
    return headline, basis


def min_bulk_ess_per_grad(
    states_position: Mapping[str, Any],
    n_grad_evals: int,
) -> float:
    """Headline metric: minimum bulk-ESS per gradient evaluation.

    Parameters
    ----------
    states_position
        Dict keyed by NumPyro sample-site name. Each value is an array of shape
        ``(n_chains, n_samples, *site_shape)``. The per-site shape is flattened
        before ESS computation so a site of shape ``(C, S, 4, 4)`` is treated
        as ``(C, S, 16)`` independent dimensions.
    n_grad_evals
        Total gradient evaluations across the whole run, as returned by
        ``tuningfork.metrics.grad_counter.total_grad_evals``. Must be >= 0.

    Returns
    -------
    float
        ``min over all dims of bulk-ESS / n_grad_evals``.

        Special cases:

        - ``n_grad_evals == 0``:  returns ``inf`` if ``min_ess`` is finite,
          ``nan`` if ``min_ess`` is itself non-finite (empty / degenerate).
        - Finite ``n_grad_evals > 0``:  returns ``float(min_ess / n_grad_evals)``.

    Raises
    ------
    ValueError
        If ``n_grad_evals < 0``, ``states_position`` is empty, or any site
        array has ``ndim < 2`` (must supply at least a chain and sample axis).

    Notes
    -----
    ``blackjax.diagnostics.ess_bulk`` with ``chain_axis=0`` and ``sample_axis=1``
    returns an array of shape ``(D,)`` for input shape ``(C, S, D)`` — bulk-ESS is
    already aggregated across chains per dimension.  We therefore just take the
    per-site minimum and propagate the global min across all sites.

    The min is a pessimistic aggregation: the worst-mixing dimension in any
    sample site determines the effective run length for the whole chain.

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> from tuningfork.metrics.headline import min_bulk_ess_per_grad
    >>> key = jax.random.key(0)
    >>> samples = {"x": jax.random.normal(key, (4, 1000, 5))}
    >>> headline = min_bulk_ess_per_grad(samples, n_grad_evals=4000)
    >>> 0.5 < headline < 2.0  # i.i.d. → close to 1.0
    True
    """
    if n_grad_evals < 0:
        raise ValueError(f"n_grad_evals must be non-negative, got {n_grad_evals}")

    min_ess = _min_ess_over_sites(states_position, ess_bulk)

    if n_grad_evals == 0:
        # Caller asked for ESS-per-grad with zero grad budget.
        # If min_ess is finite (real samples exist), ESS/0 = +inf.
        # If min_ess is not finite (degenerate), propagate nan.
        return float("inf") if bool(jnp.isfinite(min_ess)) else float("nan")

    # Widen to float64 BEFORE dividing.  Dividing in the array's own float32 and
    # widening afterwards lands ~2e-8 away from build_headline_basis, which is
    # 20x the 1e-9 band the catalog exact-reproduction invariant allows.
    return float(min_ess) / float(n_grad_evals)
