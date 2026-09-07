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
"""Opt-in named-expectand reports with fully accounted costs.

A sampler comparison that quotes only the first-moment (raw-mean) ESS of the
model's own coordinates can look excellent while squared and cross functions of
the same draws mix badly.  Antithetic behaviour makes this easy to hit: a
capped raw-mean ESS can equal the total number of draws for ``theta`` while
``theta**2`` and ``theta_i * theta_j`` are far slower.

This module is **report composition only**.  It adds no ESS estimator: every
number is produced by an existing diagnostics backend
(:mod:`blackjax.diagnostics` or ArviZ), and every number carries the name of the
backend that produced it.  Nothing here changes certification, gate, or
headline-metric behaviour -- callers opt in explicitly.

What it gives you
-----------------

1. **Named function traces.**  You name the quantities you actually care about
   (``theta``, ``theta**2``, a cross product, a tail indicator).
   :func:`expectand_traces` evaluates them against a draws dict and returns
   named ``(chain, draw, *event)`` arrays, which plug straight into
   :func:`tuningfork.catalog.diagnostics.samples_to_idata`.  The input draws are
   never mutated.

2. **Raw-mean ESS reported separately from rank diagnostics.**  Each report row
   carries ``raw_mean_ess`` (no rank normalisation) *next to* ``bulk_ess``,
   ``tail_ess`` and ``rank_rhat``.  One is never silently substituted for the
   other -- that substitution is exactly what hides a slow squared function.

3. **Honest degenerate cases.**  A globally constant expectand, a
   per-chain-constant ("per-lane-constant") expectand, and a non-finite trace
   have *undefined* ESS and R-hat.  This module reports ``None`` plus a reason
   rather than passing a backend's spurious finite number through.  Repeated
   (tied) values are reported as a tie fraction, because rank-normalised
   statistics are backend-sensitive on ties.

4. **Costs that stay unknown when they are unknown.**  :class:`CostAccounting`
   carries the warmup/sampling walls and gradient counts that the repository
   already records.  A component that was not measured is ``None`` with a
   stated reason -- never ``0``.  A cost-normalised comparison is refused, with
   the blocking components named, rather than computed against a guess.  Wall
   clocks include JIT compilation; this module does not subtract an estimated
   compile time, because separating compilation is a distinct design question.

Example
-------

>>> import numpy as np
>>> from tuningfork.catalog.expectands import expectand_report
>>> draws = {"theta": np.random.default_rng(0).standard_normal((4, 500, 2))}
>>> report = expectand_report(  # doctest: +SKIP
...     draws,
...     {
...         "theta_0": lambda s: s["theta"][..., 0],
...         "theta_0_sq": lambda s: s["theta"][..., 0] ** 2,
...         "theta_0x1": lambda s: s["theta"][..., 0] * s["theta"][..., 1],
...     },
... )
>>> print(report.to_text())  # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "CostAccounting",
    "ExpectandDiagnostics",
    "ExpectandReport",
    "ComparisonRow",
    "ReportComparison",
    "expectand_traces",
    "expectand_report",
    "compare_reports",
]

#: Diagnostics backends this module can dispatch to.  ``"blackjax"`` matches the
#: estimators behind the catalog headline metric; ``"arviz"`` is the independent
#: cross-check.  They agree closely on well-behaved continuous traces and can
#: disagree by orders of magnitude on tied ones (see :func:`expectand_report`).
BACKENDS = ("blackjax", "arviz")

DEFAULT_BACKEND = "blackjax"

#: Quantile pair defining the tail-ESS statistic, matching
#: ``blackjax.diagnostics.ess_tail``.
_TAIL_PROB = (0.05, 0.95)

_TIE_CAVEAT = (
    "expectand has repeated values; rank-normalised statistics are "
    "backend-sensitive on ties (the 'blackjax' backend assigns ordinal ranks to "
    "tied values, ArviZ averages them) -- cross-check with backend='arviz'"
)

# Statistic names in report order.  ``raw_mean_ess`` is deliberately first and
# deliberately separate from the three rank-normalised statistics.
_STATISTICS = ("raw_mean_ess", "bulk_ess", "tail_ess", "rank_rhat")

# Statistics that are counts of effective draws, and so divide meaningfully by a
# cost.  ``rank_rhat`` is a convergence ratio, not a rate: neither "R-hat per
# second" nor a ratio of two R-hats carries meaning, so both are withheld.
_RATE_STATISTICS = ("raw_mean_ess", "bulk_ess", "tail_ess")

#: Default tie fraction at or above which a row carries the backend caveat.
#: A display threshold, not a correctness boundary -- ``tie_fraction`` is always
#: reported numerically, whatever this is set to.  MCMC rejections leave a few
#: repeated states in any chain; the rank-normalisation artefact this warns about
#: only becomes material when a large share of the trace is tied.
DEFAULT_TIE_CAVEAT_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Named traces
# ---------------------------------------------------------------------------


def expectand_traces(
    samples: Mapping[str, Any],
    expectands: Mapping[str, Callable[[Mapping[str, np.ndarray]], Any]],
) -> dict[str, np.ndarray]:
    """Evaluate user-named functions of the draws, without touching the draws.

    Parameters
    ----------
    samples
        Draws in multi-chain layout: ``{name: (n_chains, n_draws, *event)}``.
        Single-chain draws must be reshaped by the caller first (see
        ``samples_to_idata``'s ``n_chunks`` for the certification convention).
    expectands
        Mapping from a user-chosen name to a callable taking the whole samples
        mapping and returning an array whose leading two axes are
        ``(n_chains, n_draws)``.  The callable receives read-only views: the
        original draw arrays are never modified by this function.

    Returns
    -------
    dict[str, numpy.ndarray]
        Named traces, each ``(n_chains, n_draws, *event)``.  Suitable as the
        ``samples_dict`` argument of
        :func:`tuningfork.catalog.diagnostics.samples_to_idata`.

    Raises
    ------
    ValueError
        If ``samples`` is empty, if the draws do not share a common
        ``(n_chains, n_draws)`` topology, if ``expectands`` is empty, or if a
        trace does not carry that same leading topology.
    """
    if not samples:
        raise ValueError("samples must contain at least one draw array")
    if not expectands:
        raise ValueError("expectands must contain at least one named function")

    arrays: dict[str, np.ndarray] = {}
    for name, value in samples.items():
        arr = np.asarray(value)
        if arr.ndim < 2:
            raise ValueError(
                f"draw array {name!r} has shape {arr.shape}; multi-chain layout "
                "(n_chains, n_draws, *event) is required"
            )
        # A read-only view keeps a user expectand from mutating cached draws
        # in place.  The underlying buffer is shared, not copied.
        view = arr.view()
        view.flags.writeable = False
        arrays[name] = view

    topology = next(iter(arrays.values())).shape[:2]
    mismatched = {n: a.shape[:2] for n, a in arrays.items() if a.shape[:2] != topology}
    if mismatched:
        raise ValueError(
            "all draw arrays must share the same (n_chains, n_draws) topology; "
            f"expected {topology}, got {mismatched}"
        )

    traces: dict[str, np.ndarray] = {}
    for name, fn in expectands.items():
        trace = np.asarray(fn(arrays))
        if trace.shape[:2] != topology:
            raise ValueError(
                f"expectand {name!r} returned shape {trace.shape}; its first two "
                f"axes must be the draw topology {topology}"
            )
        traces[name] = trace
    return traces


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostAccounting:
    """Costs already recorded elsewhere in the repository, with gaps preserved.

    Every field is ``None`` when the corresponding cost was not measured.  A
    missing cost is *never* represented as ``0``: ``unknown_reasons`` states why
    each ``None`` is ``None``, and :attr:`is_fully_accounted` is ``False`` while
    any component is missing.
    """

    warmup_seconds: float | None = None
    sampling_seconds: float | None = None
    total_seconds: float | None = None
    warmup_grad_evals: int | None = None
    sampling_grad_evals: int | None = None
    compile_seconds: float | None = None
    unknown_reasons: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    source: str = "unspecified"

    #: Cost components in report order.
    COMPONENTS = (
        "warmup_seconds",
        "sampling_seconds",
        "total_seconds",
        "warmup_grad_evals",
        "sampling_grad_evals",
        "compile_seconds",
    )

    @property
    def unknown_components(self) -> tuple[str, ...]:
        """Names of the cost components that were not measured."""
        return tuple(name for name in self.COMPONENTS if getattr(self, name) is None)

    @property
    def is_fully_accounted(self) -> bool:
        """True only when every cost component carries a measured value."""
        return not self.unknown_components

    def reason_for(self, component: str) -> str:
        """Why ``component`` is unknown, or ``""`` when it is known."""
        if component not in self.COMPONENTS:
            raise KeyError(f"unknown cost component: {component!r}")
        if getattr(self, component) is not None:
            return ""
        return self.unknown_reasons.get(component, "not recorded")

    @classmethod
    def from_telemetry(cls, telemetry: Any) -> CostAccounting:
        """Read an ``ExecutionTelemetry`` without adding fields to its schema.

        Only fields the generated-run telemetry schema already defines are
        consulted: ``timing_seconds`` (``warmup``/``sampling``/``total``),
        ``warmup_grad_evals`` and ``warmup_grad_evals_reason``.  Sampling
        gradient evaluations and compile time are not in that schema, so they
        stay unknown with a stated reason.
        """
        timing = dict(getattr(telemetry, "timing_seconds", {}) or {})
        warmup_grad = getattr(telemetry, "warmup_grad_evals", None)
        warmup_reason = getattr(telemetry, "warmup_grad_evals_reason", "") or ""

        unknown: dict[str, str] = {
            "sampling_grad_evals": (
                "generated-run telemetry records warmup gradient evaluations only"
            ),
            "compile_seconds": (
                "JIT compilation is not separately measured; it is included in the "
                "recorded warmup and sampling walls and is not subtracted here"
            ),
        }
        if warmup_grad is None:
            unknown["warmup_grad_evals"] = (
                warmup_reason or "warmup gradient evaluations were not counted"
            )
        for key, component in (
            ("warmup", "warmup_seconds"),
            ("sampling", "sampling_seconds"),
            ("total", "total_seconds"),
        ):
            if timing.get(key) is None:
                unknown[component] = "telemetry did not record this wall clock"

        return cls(
            warmup_seconds=_opt_float(timing.get("warmup")),
            sampling_seconds=_opt_float(timing.get("sampling")),
            total_seconds=_opt_float(timing.get("total")),
            warmup_grad_evals=warmup_grad,
            sampling_grad_evals=None,
            compile_seconds=None,
            unknown_reasons=unknown,
            notes=(
                "wall clocks include JIT compilation",
                *(
                    (f"warmup_grad_evals basis: {warmup_reason}",)
                    if warmup_grad is not None and warmup_reason
                    else ()
                ),
            ),
            source="execution_telemetry",
        )

    @classmethod
    def from_recipe(cls, recipe: Any) -> CostAccounting:
        """Read the wall clocks a Recipe's ``calibration_budget`` already holds."""
        budget = dict(getattr(recipe, "calibration_budget", None) or {})
        warmup = _opt_float(budget.get("warmup_wall_seconds"))
        sampling = _opt_float(budget.get("sampling_wall_seconds"))
        total = (
            warmup + sampling if warmup is not None and sampling is not None else None
        )

        unknown: dict[str, str] = {
            "warmup_grad_evals": (
                "calibration_budget does not record gradient evaluations"
            ),
            "sampling_grad_evals": (
                "calibration_budget does not record gradient evaluations"
            ),
            "compile_seconds": (
                "JIT compilation is not separately measured; it is included in the "
                "recorded warmup and sampling walls and is not subtracted here"
            ),
        }
        if warmup is None:
            unknown["warmup_seconds"] = "calibration_budget has no warmup_wall_seconds"
        if sampling is None:
            unknown["sampling_seconds"] = (
                "calibration_budget has no sampling_wall_seconds"
            )
        if total is None:
            unknown["total_seconds"] = (
                "derived from warmup + sampling walls, at least one of which is "
                "not recorded"
            )

        return cls(
            warmup_seconds=warmup,
            sampling_seconds=sampling,
            total_seconds=total,
            unknown_reasons=unknown,
            notes=("wall clocks include JIT compilation",),
            source="recipe_calibration_budget",
        )


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not np.isfinite(out):
        return None
    return out


# ---------------------------------------------------------------------------
# Per-expectand diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectandDiagnostics:
    """Diagnostics for one scalar component of one named expectand.

    A vector-valued expectand contributes one instance per flattened event
    component, so that every row carries exactly one status and one set of
    numbers.
    """

    name: str
    component: int | None
    backend: str
    n_chains: int
    n_draws: int
    n_distinct: int
    tie_fraction: float
    degeneracy: str
    raw_mean_ess: float | None
    bulk_ess: float | None
    tail_ess: float | None
    rank_rhat: float | None
    undefined_reasons: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """``name`` for a scalar expectand, ``name[i]`` for a vector component."""
        return self.name if self.component is None else f"{self.name}[{self.component}]"

    @property
    def is_defined(self) -> bool:
        """True when every reported statistic has a value."""
        return not self.undefined_reasons

    def value(self, statistic: str) -> float | None:
        if statistic not in _STATISTICS:
            raise KeyError(f"unknown statistic: {statistic!r}")
        return getattr(self, statistic)  # type: ignore[no-any-return]


def _classify(trace_cs: np.ndarray) -> tuple[str, int, tuple[str, ...]]:
    """Return ``(degeneracy, n_constant_chains, warnings)`` for a ``(C, S)`` trace."""
    per_chain_constant = np.array(
        [np.all(chain == chain[0]) for chain in trace_cs], dtype=bool
    )
    n_constant_chains = int(per_chain_constant.sum())
    if bool(np.all(trace_cs == trace_cs.flat[0])):
        return "global_constant", n_constant_chains, ()
    if bool(per_chain_constant.all()):
        return "per_chain_constant", n_constant_chains, ()
    warnings: tuple[str, ...] = ()
    if n_constant_chains:
        warnings = (
            f"{n_constant_chains} of {trace_cs.shape[0]} chains are constant; "
            "reported values are dominated by the remaining chains",
        )
    return "none", n_constant_chains, warnings


def _component_diagnostics(
    name: str,
    component: int | None,
    trace_cs: np.ndarray,
    backend: str,
    tie_caveat_threshold: float = DEFAULT_TIE_CAVEAT_THRESHOLD,
) -> ExpectandDiagnostics:
    n_chains, n_draws = trace_cs.shape
    finite = bool(np.all(np.isfinite(trace_cs)))
    n_distinct = int(np.unique(trace_cs).size)
    total = trace_cs.size
    tie_fraction = 0.0 if total == 0 else 1.0 - n_distinct / total

    def undefined(reason: str, degeneracy: str) -> ExpectandDiagnostics:
        return ExpectandDiagnostics(
            name=name,
            component=component,
            backend=backend,
            n_chains=n_chains,
            n_draws=n_draws,
            n_distinct=n_distinct,
            tie_fraction=tie_fraction,
            degeneracy=degeneracy,
            raw_mean_ess=None,
            bulk_ess=None,
            tail_ess=None,
            rank_rhat=None,
            undefined_reasons={stat: reason for stat in _STATISTICS},
        )

    if not finite:
        return undefined(
            "trace contains non-finite values; ESS and R-hat are undefined",
            "non_finite",
        )

    degeneracy, _, warns = _classify(trace_cs)
    if degeneracy == "global_constant":
        return undefined(
            "expectand is constant across every chain and draw; ESS and R-hat "
            "are undefined for a zero-variance sequence",
            degeneracy,
        )
    if degeneracy == "per_chain_constant":
        return undefined(
            "every chain is constant but the chains differ; within-chain "
            "variance is zero, so autocorrelation-based ESS and rank R-hat are "
            "undefined",
            degeneracy,
        )

    stats = _backend_statistics(trace_cs, backend)
    undefined_reasons = {
        stat: (f"backend {backend!r} returned a non-finite value for this statistic")
        for stat, val in stats.items()
        if val is None
    }
    if tie_fraction >= tie_caveat_threshold:
        warns = (*warns, _TIE_CAVEAT)

    return ExpectandDiagnostics(
        name=name,
        component=component,
        backend=backend,
        n_chains=n_chains,
        n_draws=n_draws,
        n_distinct=n_distinct,
        tie_fraction=tie_fraction,
        degeneracy=degeneracy,
        raw_mean_ess=stats["raw_mean_ess"],
        bulk_ess=stats["bulk_ess"],
        tail_ess=stats["tail_ess"],
        rank_rhat=stats["rank_rhat"],
        undefined_reasons=undefined_reasons,
        warnings=warns,
    )


def _backend_statistics(trace_cs: np.ndarray, backend: str) -> dict[str, float | None]:
    """Dispatch to an existing diagnostics implementation; add no estimator."""
    if backend == "blackjax":
        raw = _blackjax_statistics(trace_cs)
    elif backend == "arviz":
        raw = _arviz_statistics(trace_cs)
    else:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    return {
        k: (v if v is not None and np.isfinite(v) else None) for k, v in raw.items()
    }


def _blackjax_statistics(trace_cs: np.ndarray) -> dict[str, float | None]:
    from blackjax.diagnostics import (  # local import: bench-group dependency
        effective_sample_size,
        ess_bulk,
        ess_tail,
        rhat,
    )

    # blackjax squeezes size-1 trailing axes, so reshape the result rather than
    # indexing it.
    arr = np.asarray(trace_cs)[:, :, None]

    def scalar(fn: Callable[..., Any]) -> float:
        out = np.asarray(fn(arr, chain_axis=0, sample_axis=1), dtype=np.float64)
        return float(out.reshape(-1)[0])

    return {
        "raw_mean_ess": scalar(effective_sample_size),
        "bulk_ess": scalar(ess_bulk),
        "tail_ess": scalar(ess_tail),
        "rank_rhat": scalar(rhat),
    }


def _arviz_statistics(trace_cs: np.ndarray) -> dict[str, float | None]:
    import arviz as az  # local import: optional [viz] dependency

    arr = np.asarray(trace_cs, dtype=np.float64)
    return {
        "raw_mean_ess": float(az.ess(arr, method="mean")),
        "bulk_ess": float(az.ess(arr, method="bulk")),
        "tail_ess": float(az.ess(arr, method="tail", prob=_TAIL_PROB)),
        "rank_rhat": float(az.rhat(arr, method="rank")),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectandReport:
    """A composed, opt-in report over named expectands.

    Carries the per-component diagnostics plus the cost accounting that makes a
    comparison against another report meaningful.  It changes no gate and no
    headline metric.
    """

    label: str
    backend: str
    entries: tuple[ExpectandDiagnostics, ...]
    cost: CostAccounting
    n_chains: int
    n_draws: int

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def by_label(self) -> dict[str, ExpectandDiagnostics]:
        """Entries keyed by :attr:`ExpectandDiagnostics.label`."""
        return {entry.label: entry for entry in self.entries}

    def to_rows(self) -> list[dict[str, Any]]:
        """Flat, JSON-friendly rows -- one per expectand component."""
        rows: list[dict[str, Any]] = []
        for entry in self.entries:
            rows.append(
                {
                    "expectand": entry.label,
                    "backend": entry.backend,
                    "degeneracy": entry.degeneracy,
                    "tie_fraction": entry.tie_fraction,
                    "raw_mean_ess": entry.raw_mean_ess,
                    "bulk_ess": entry.bulk_ess,
                    "tail_ess": entry.tail_ess,
                    "rank_rhat": entry.rank_rhat,
                    "undefined": dict(entry.undefined_reasons),
                    "warnings": list(entry.warnings),
                }
            )
        return rows

    def to_text(self) -> str:
        """Human-readable table; unknown and undefined values render as text."""
        header = (
            f"expectand report: {self.label}  "
            f"[backend={self.backend}, {self.n_chains} chains x "
            f"{self.n_draws} draws]"
        )
        columns = ("expectand", "raw-mean ESS", "bulk ESS", "tail ESS", "rank R-hat")
        rows = [columns]
        for entry in self.entries:
            rows.append(
                (
                    entry.label,
                    _fmt(entry.raw_mean_ess),
                    _fmt(entry.bulk_ess),
                    _fmt(entry.tail_ess),
                    _fmt(entry.rank_rhat, digits=4),
                )
            )
        widths = [max(len(r[i]) for r in rows) for i in range(len(columns))]
        lines = [header, ""]
        for index, row in enumerate(rows):
            lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
            if index == 0:
                lines.append("  ".join("-" * w for w in widths))

        notes: list[str] = []
        for entry in self.entries:
            for stat, reason in entry.undefined_reasons.items():
                notes.append(f"{entry.label}.{stat}: undefined -- {reason}")
            for warning in entry.warnings:
                notes.append(f"{entry.label}: {warning}")
        if notes:
            lines.extend(["", "notes:", *(f"  - {n}" for n in dict.fromkeys(notes))])

        lines.extend(["", f"costs ({self.cost.source}):"])
        for component in CostAccounting.COMPONENTS:
            value = getattr(self.cost, component)
            if value is None:
                lines.append(
                    f"  {component}: unknown -- {self.cost.reason_for(component)}"
                )
            else:
                lines.append(f"  {component}: {value}")
        for note in self.cost.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def _fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "undefined"
    return f"{value:.{digits}f}"


def expectand_report(
    samples: Mapping[str, Any],
    expectands: Mapping[str, Callable[[Mapping[str, np.ndarray]], Any]],
    *,
    cost: CostAccounting | None = None,
    backend: str = DEFAULT_BACKEND,
    label: str = "unnamed",
    tie_caveat_threshold: float = DEFAULT_TIE_CAVEAT_THRESHOLD,
) -> ExpectandReport:
    """Compose a named-expectand report from existing draws and diagnostics.

    Parameters
    ----------
    samples
        Multi-chain draws, ``{name: (n_chains, n_draws, *event)}``.  Not
        modified.
    expectands
        Named functions of the draws; see :func:`expectand_traces`.
    cost
        Costs already recorded elsewhere (see :meth:`CostAccounting.from_recipe`
        and :meth:`CostAccounting.from_telemetry`).  Defaults to an accounting
        in which every component is explicitly unknown -- never zero.
    backend
        ``"blackjax"`` (default, matching the catalog headline estimators) or
        ``"arviz"``.  Recorded on every row.  The two agree closely on
        well-behaved continuous traces; on tied traces they can differ by orders
        of magnitude, which is why ties are reported rather than smoothed over.
        The blackjax backend computes in JAX's configured precision (float32
        unless x64 is enabled); the ArviZ backend computes in float64.
    label
        Name for this report, used when comparing two of them.
    tie_caveat_threshold
        Tie fraction at or above which a row carries the backend caveat.  A
        display threshold only: ``tie_fraction`` is reported numerically on
        every row regardless.  The default keeps the caveat off the handful of
        repeated states any MCMC chain leaves behind after rejections.

    Returns
    -------
    ExpectandReport
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")

    traces = expectand_traces(samples, expectands)
    topology = next(iter(traces.values())).shape[:2]
    n_chains, n_draws = int(topology[0]), int(topology[1])

    entries: list[ExpectandDiagnostics] = []
    for name, trace in traces.items():
        arr = np.asarray(trace, dtype=np.float64)
        event_shape = arr.shape[2:]
        n_components = int(np.prod(event_shape)) if event_shape else 1
        flat = arr.reshape(n_chains, n_draws, n_components)
        for index in range(n_components):
            entries.append(
                _component_diagnostics(
                    name,
                    None if not event_shape else index,
                    flat[:, :, index],
                    backend,
                    tie_caveat_threshold=tie_caveat_threshold,
                )
            )

    return ExpectandReport(
        label=label,
        backend=backend,
        entries=tuple(entries),
        cost=cost if cost is not None else _unmeasured_cost(),
        n_chains=n_chains,
        n_draws=n_draws,
    )


def _unmeasured_cost() -> CostAccounting:
    reason = "no cost accounting was supplied to expectand_report"
    return CostAccounting(
        unknown_reasons={c: reason for c in CostAccounting.COMPONENTS},
        source="unmeasured",
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonRow:
    """One expectand compared across two reports."""

    expectand: str
    statistic: str
    baseline: float | None
    candidate: float | None
    ratio: float | None
    ratio_blocked_by: tuple[str, ...] = ()
    per_second: tuple[float | None, float | None] | None = None
    per_grad_eval: tuple[float | None, float | None] | None = None
    cost_blocked_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportComparison:
    """Side-by-side of two reports with costs carried through, never guessed."""

    baseline_label: str
    candidate_label: str
    rows: tuple[ComparisonRow, ...]
    cost_blockers: tuple[str, ...]
    only_in_baseline: tuple[str, ...] = ()
    only_in_candidate: tuple[str, ...] = ()
    backend_mismatch: tuple[str, str] | None = None

    @property
    def cost_normalised_available(self) -> bool:
        """True only when both reports measured total wall and gradient counts."""
        return not self.cost_blockers

    def to_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "expectand": row.expectand,
                "statistic": row.statistic,
                self.baseline_label: row.baseline,
                self.candidate_label: row.candidate,
                "ratio": row.ratio,
                "ratio_blocked_by": list(row.ratio_blocked_by),
                "per_second": list(row.per_second) if row.per_second else None,
                "per_grad_eval": (
                    list(row.per_grad_eval) if row.per_grad_eval else None
                ),
                "cost_blocked_by": list(row.cost_blocked_by),
            }
            for row in self.rows
        ]


def compare_reports(
    baseline: ExpectandReport,
    candidate: ExpectandReport,
    *,
    statistics: tuple[str, ...] = _STATISTICS,
) -> ReportComparison:
    """Compare two reports, refusing any comparison a missing cost would fake.

    ESS ratios are produced only where both sides are defined; a ratio blocked
    by an undefined statistic names the side that is undefined.  Cost-normalised
    figures (ESS per second, ESS per gradient evaluation) are produced only when
    both reports measured the corresponding cost; otherwise the row names the
    blocking components and reports ``None``.  A missing cost is never treated
    as zero, and no compile-time estimate is subtracted from either side.

    ``rank_rhat`` is carried side by side but is neither ratioed nor
    cost-normalised: it is a convergence ratio judged against its own threshold,
    so "R-hat per second" and "candidate R-hat over baseline R-hat" are both
    meaningless.  Only the ESS statistics in :data:`_RATE_STATISTICS` divide by a
    cost.
    """
    unknown_statistics = [s for s in statistics if s not in _STATISTICS]
    if unknown_statistics:
        raise KeyError(f"unknown statistics: {unknown_statistics}")

    base_map = baseline.by_label()
    cand_map = candidate.by_label()
    shared = [label for label in base_map if label in cand_map]

    seconds_blockers = tuple(
        f"{report.label}.total_seconds ({report.cost.reason_for('total_seconds')})"
        for report in (baseline, candidate)
        if report.cost.total_seconds is None
    )
    grads = {
        report.label: _total_grad_evals(report.cost) for report in (baseline, candidate)
    }
    grad_blockers = tuple(
        f"{report.label}.{component} ({report.cost.reason_for(component)})"
        for report in (baseline, candidate)
        for component in ("warmup_grad_evals", "sampling_grad_evals")
        if getattr(report.cost, component) is None
    )

    rows: list[ComparisonRow] = []
    for label in shared:
        for statistic in statistics:
            base_val = base_map[label].value(statistic)
            cand_val = cand_map[label].value(statistic)
            is_rate = statistic in _RATE_STATISTICS
            blocked: tuple[str, ...] = ()
            ratio: float | None = None
            if base_val is None:
                blocked += (f"{baseline.label}.{label}.{statistic} undefined",)
            if cand_val is None:
                blocked += (f"{candidate.label}.{label}.{statistic} undefined",)
            if not is_rate:
                blocked += (
                    f"{statistic} is a convergence ratio, not a rate; a ratio "
                    "of two values is not meaningful -- read each against its "
                    "own threshold",
                )
            if not blocked:
                assert base_val is not None and cand_val is not None
                if base_val == 0.0:
                    blocked += (f"{baseline.label}.{label}.{statistic} is zero",)
                else:
                    ratio = cand_val / base_val

            per_second: tuple[float | None, float | None] | None = None
            per_grad: tuple[float | None, float | None] | None = None
            cost_blocked: tuple[str, ...] = ()
            if not is_rate:
                cost_blocked += (
                    f"{statistic} is not a rate; cost normalisation does not " "apply",
                )
            elif base_val is not None and cand_val is not None:
                if seconds_blockers:
                    cost_blocked += seconds_blockers
                else:
                    per_second = (
                        _safe_div(base_val, baseline.cost.total_seconds),
                        _safe_div(cand_val, candidate.cost.total_seconds),
                    )
                if grad_blockers:
                    cost_blocked += grad_blockers
                else:
                    per_grad = (
                        _safe_div(base_val, grads[baseline.label]),
                        _safe_div(cand_val, grads[candidate.label]),
                    )
            rows.append(
                ComparisonRow(
                    expectand=label,
                    statistic=statistic,
                    baseline=base_val,
                    candidate=cand_val,
                    ratio=ratio,
                    ratio_blocked_by=blocked,
                    per_second=per_second,
                    per_grad_eval=per_grad,
                    cost_blocked_by=tuple(dict.fromkeys(cost_blocked)),
                )
            )

    return ReportComparison(
        baseline_label=baseline.label,
        candidate_label=candidate.label,
        rows=tuple(rows),
        cost_blockers=tuple(dict.fromkeys(seconds_blockers + grad_blockers)),
        only_in_baseline=tuple(key for key in base_map if key not in cand_map),
        only_in_candidate=tuple(key for key in cand_map if key not in base_map),
        backend_mismatch=(
            None
            if baseline.backend == candidate.backend
            else (baseline.backend, candidate.backend)
        ),
    )


def _total_grad_evals(cost: CostAccounting) -> int | None:
    if cost.warmup_grad_evals is None or cost.sampling_grad_evals is None:
        return None
    return cost.warmup_grad_evals + cost.sampling_grad_evals


def _safe_div(numerator: float | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)
