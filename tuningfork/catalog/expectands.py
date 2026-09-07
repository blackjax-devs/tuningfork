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
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "DEFAULT_TIE_BLOCK_THRESHOLD",
    "CostAccounting",
    "ExpectandDiagnostics",
    "ExpectandReport",
    "ComparisonRow",
    "ReportComparison",
    "expectand_traces",
    "expectand_report",
    "compare_reports",
    "sampling_grad_evals_from_chain_stats",
    "GradEvalDerivation",
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

#: Backends whose rank normalisation is order-dependent on tied values.
_ORDINAL_RANK_BACKENDS = frozenset({"blackjax"})


def _tie_disclosure(
    tie_fraction: float, mean_tie_block: float, backend: str, material: bool
) -> str:
    """Always disclose ties; raise the wording's severity when they are material.

    Disclosure is not gated on the threshold.  A backend that ranks ties
    ordinally is named whenever any tie is present, because the reader -- not a
    fixed cut-off -- decides whether a rank statistic looks anomalous.
    """
    share = (
        f"tie fraction {tie_fraction:.4g}, mean tie block {mean_tie_block:.4g} draws"
    )
    if backend in _ORDINAL_RANK_BACKENDS:
        backend_note = (
            f"the {backend!r} backend assigns ordinal ranks to tied values while "
            "ArviZ averages them, so rank-normalised statistics differ between "
            "them here -- cross-check with backend='arviz'"
        )
    else:
        backend_note = (
            f"the {backend!r} backend averages tied ranks; a backend that ranks "
            "ties ordinally would report different rank-normalised statistics"
        )
    if material:
        return (
            "expectand takes few distinct values, the regime where rank "
            f"normalisation diverges between backends ({share}); {backend_note}"
        )
    return f"expectand has repeated values ({share}); {backend_note}"


# Statistic names in report order.  ``raw_mean_ess`` is deliberately first and
# deliberately separate from the three rank-normalised statistics.
_STATISTICS = ("raw_mean_ess", "bulk_ess", "tail_ess", "rank_rhat")

# Statistics that are counts of effective draws, and so divide meaningfully by a
# cost.  ``rank_rhat`` is a convergence ratio, not a rate: neither "R-hat per
# second" nor a ratio of two R-hats carries meaning, so both are withheld.
_RATE_STATISTICS = ("raw_mean_ess", "bulk_ess", "tail_ess")

#: Mean tie-block size at or above which the tie disclosure is worded more
#: strongly.
#:
#: **An explicitly uncalibrated presentation heuristic.** It changes wording
#: only. It is not a reliability certificate, not a gate, and not a validated
#: boundary between safe and unsafe: any tie at all is disclosed together with
#: the backend's tie handling, whatever the severity says.
#:
#: It is keyed on how few distinct values a trace takes rather than on what
#: share of it is tied, on the strength of a small exploratory probe (a single
#: 4x500 fixture, one seed, one arrangement, the pinned backend versions) whose
#: numbers are recorded in ``docs/examples/trajectory-length-sensitivity.md``.
#: In that probe a continuous trace with 89% of its values repeated by rejection
#: holds still had ~2000 distinct values and the backends agreed to 0.4%, while
#: the same draws quantised to 10, 5 and 2 distinct values gave ArviZ/blackjax
#: bulk-ESS ratios of 2.6, 32 and 156. That is enough to show tie fraction alone
#: is a poor severity signal; it is not a calibration, and it does not
#: characterise how the effect varies with chain length, temporal arrangement,
#: or backend version -- a backend that changes its tie handling changes this
#: picture entirely, which is why every report records the backend and its
#: version.
DEFAULT_TIE_BLOCK_THRESHOLD = 50.0


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
    any component is missing.  A measured ``0`` is a known value, not a gap --
    a ``no_warmup`` arm really did execute zero warmup gradients.

    ``sampling_transition_grad_evals`` is deliberately narrow: it is the
    gradient work of the sampler's *transitions*, including rejected ones.  It
    is not the total gradient cost of the run.  Whatever it omits is named in
    :attr:`excluded_grad_work`, so a transition subtotal is never mistaken for
    a total.

    :attr:`view` says which accounting question this object answers:

    ``"as_measured"``
        exactly what one execution's telemetry or recipe recorded.
    ``"standalone"``
        what this alternative would cost on its own -- a shared warmup that
        several arms reuse is charged to each of them.
    ``"combined"``
        what the whole experiment actually spent -- a shared warmup is charged
        once across every arm that reused it.

    The two derived views are never inferred; build them with :meth:`combine`,
    which records its inputs in :attr:`contributors`.
    """

    warmup_seconds: float | None = None
    sampling_seconds: float | None = None
    total_seconds: float | None = None
    warmup_grad_evals: int | None = None
    sampling_transition_grad_evals: int | None = None
    compile_seconds: float | None = None
    unknown_reasons: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    source: str = "unspecified"
    view: str = "as_measured"
    contributors: tuple[str, ...] = ()
    excluded_grad_work: tuple[str, ...] = ()

    #: Cost components in report order.
    COMPONENTS = (
        "warmup_seconds",
        "sampling_seconds",
        "total_seconds",
        "warmup_grad_evals",
        "sampling_transition_grad_evals",
        "compile_seconds",
    )

    #: Components measured in seconds.  Summable only across phases the caller
    #: asserts are disjoint and sequential.
    _TIME_COMPONENTS = (
        "warmup_seconds",
        "sampling_seconds",
        "total_seconds",
        "compile_seconds",
    )

    #: Components that are counts.  Additive whether or not phases overlap.
    _COUNT_COMPONENTS = ("warmup_grad_evals", "sampling_transition_grad_evals")

    #: Accounting views :meth:`combine` can produce.
    VIEWS = ("as_measured", "standalone", "combined")

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

    def relabel(self, source: str) -> CostAccounting:
        """Same costs under a different ``source`` name.

        Used when an accounting read from telemetry needs a name that
        :meth:`combine` can report in provenance and double-charge messages.
        Copying the fields by hand instead would silently drop any field added
        later.
        """
        return replace(self, source=source)

    def provenance(self) -> tuple[str, ...]:
        """Every accounting this object was built from, innermost first."""
        return self.contributors or (self.source,)

    @classmethod
    def combine(
        cls,
        costs: tuple[CostAccounting, ...] | list[CostAccounting],
        *,
        view: str,
        phases_are_disjoint_sequential: bool,
        source: str = "combined",
    ) -> CostAccounting:
        """Add up several accountings without inventing anything they lack.

        A component is summed only when *every* contributor measured it;
        otherwise it stays ``None`` and the reason names the contributor that
        did not measure it.  A measured ``0`` contributes ``0`` and does not
        make the sum unknown.

        Wall clocks are summed only when the caller asserts, via
        ``phases_are_disjoint_sequential``, that the contributors ran one after
        another without overlap.  Gradient counts are additive regardless, so
        they are summed either way.

        The same contributor may not appear twice.  That is what stops a shared
        warmup from being charged into a total that already contains it: a
        combined accounting carries its inputs in :attr:`contributors`, so
        re-combining it with one of those inputs is rejected rather than
        silently double-counted.

        Parameters
        ----------
        costs
            Two or more accountings to add.
        view
            ``"standalone"`` or ``"combined"`` -- which question the result
            answers.  Stated explicitly because the arithmetic is identical and
            only the caller knows which one they are asking.
        phases_are_disjoint_sequential
            Whether the contributors' wall clocks may be added.  ``False``
            leaves every time component unknown with a stated reason.
        source
            Label for the resulting accounting.
        """
        costs = tuple(costs)
        if len(costs) < 2:
            raise ValueError("combine needs at least two accountings")
        if view not in ("standalone", "combined"):
            raise ValueError('view must be "standalone" or "combined"')

        seen: dict[str, int] = {}
        for cost in costs:
            for name in cost.provenance():
                seen[name] = seen.get(name, 0) + 1
        duplicated = sorted(name for name, count in seen.items() if count > 1)
        if duplicated:
            raise ValueError(
                "cannot combine: "
                + ", ".join(duplicated)
                + " appears in more than one contributor, so its cost would be "
                "charged twice; combine the parts that do not already include it"
            )

        values: dict[str, Any] = {}
        unknown: dict[str, str] = {}
        for component in cls.COMPONENTS:
            if component in cls._TIME_COMPONENTS and not phases_are_disjoint_sequential:
                unknown[component] = (
                    "contributors were not asserted to be disjoint and "
                    "sequential, so their wall clocks cannot be added"
                )
                values[component] = None
                continue
            missing = [c for c in costs if getattr(c, component) is None]
            if missing:
                unknown[component] = "; ".join(
                    f"{c.source}: {c.reason_for(component)}" for c in missing
                )
                values[component] = None
            else:
                total = sum(getattr(c, component) for c in costs)
                values[component] = (
                    int(total) if component in cls._COUNT_COMPONENTS else float(total)
                )

        notes = tuple(dict.fromkeys(note for cost in costs for note in cost.notes)) + (
            f"{view} view over {len(costs)} contributors: "
            + ", ".join(c.source for c in costs),
        )
        if not phases_are_disjoint_sequential:
            notes += ("wall clocks were not summed: phases may overlap",)

        return cls(
            **values,
            unknown_reasons=unknown,
            notes=notes,
            source=source,
            view=view,
            contributors=tuple(
                dict.fromkeys(name for cost in costs for name in cost.provenance())
            ),
            excluded_grad_work=tuple(
                dict.fromkeys(
                    item for cost in costs for item in cost.excluded_grad_work
                )
            ),
        )

    @classmethod
    def from_telemetry(
        cls,
        telemetry: Any,
        *,
        sampling_transition_grad_evals: GradEvalDerivation | None = None,
    ) -> CostAccounting:
        """Read an ``ExecutionTelemetry`` without adding fields to its schema.

        Only fields the generated-run telemetry schema already defines are
        consulted: ``timing_seconds`` (``warmup``/``sampling``/``total``),
        ``warmup_grad_evals`` and ``warmup_grad_evals_reason``.  Compile time is
        not in that schema, so it stays unknown with a stated reason.

        Sampling gradient work is not in the schema either.  Pass the result of
        :func:`sampling_grad_evals_from_chain_stats` as
        ``sampling_transition_grad_evals`` to supply the transition subtotal
        recovered from the persisted per-step statistics; its basis and
        exclusions are carried through.  Without it the component stays unknown.
        """
        timing = dict(getattr(telemetry, "timing_seconds", {}) or {})
        warmup_grad = getattr(telemetry, "warmup_grad_evals", None)
        warmup_reason = getattr(telemetry, "warmup_grad_evals_reason", "") or ""

        unknown: dict[str, str] = {
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

        derivation = sampling_transition_grad_evals
        transitions = derivation.count if derivation is not None else None
        if transitions is None:
            unknown["sampling_transition_grad_evals"] = (
                derivation.reason
                if derivation is not None
                else (
                    "generated-run telemetry records warmup gradient evaluations "
                    "only; pass sampling_grad_evals_from_chain_stats(...) to "
                    "recover the transition subtotal from the persisted per-step "
                    "statistics"
                )
            )

        notes = ["wall clocks include JIT compilation"]
        if warmup_grad is not None and warmup_reason:
            notes.append(f"warmup_grad_evals basis: {warmup_reason}")
        if derivation is not None and derivation.count is not None:
            notes.append(
                "sampling_transition_grad_evals basis: "
                f"{derivation.basis} (from {', '.join(derivation.source_fields)})"
            )

        return cls(
            warmup_seconds=_opt_float(timing.get("warmup")),
            sampling_seconds=_opt_float(timing.get("sampling")),
            total_seconds=_opt_float(timing.get("total")),
            warmup_grad_evals=warmup_grad,
            sampling_transition_grad_evals=transitions,
            compile_seconds=None,
            unknown_reasons=unknown,
            notes=tuple(notes),
            source="execution_telemetry",
            excluded_grad_work=(
                derivation.excluded
                if derivation is not None and derivation.count is not None
                else ()
            ),
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

        no_counts = "calibration_budget does not record gradient evaluations"
        unknown: dict[str, str] = {
            "warmup_grad_evals": no_counts,
            "sampling_transition_grad_evals": no_counts,
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


@dataclass(frozen=True)
class GradEvalDerivation:
    """Outcome of recovering transition gradient work from persisted statistics.

    ``count`` is ``None`` when the derivation was refused; ``reason`` then says
    why.  When ``count`` is present, ``basis`` is the sampler's own declared
    counting convention and ``source_fields`` names the per-step statistics it
    was computed from, so the number can be audited without rerunning anything.

    ``excluded`` names gradient work this subtotal does *not* contain.  It is
    never empty for a successful derivation: per-step transition statistics
    cannot see initialization or any controller/adaptation internals, so this
    is a subtotal of the sampling phase and must not be reported as a total.
    """

    count: int | None
    basis: str = ""
    source_fields: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    reason: str = ""


#: Gradient work that per-step transition statistics structurally cannot see.
_EXCLUDED_FROM_TRANSITION_COUNT = (
    "initialization (kernel.init and any per-chain state re-init)",
    "controller/adaptation internals not emitted as per-step statistics",
    "any gradient work outside the recorded sampling transitions",
)


def sampling_grad_evals_from_chain_stats(
    chain_stats: Mapping[str, Any],
    base_method_name: str,
    *,
    expected_topology: tuple[int, int] | None = None,
) -> GradEvalDerivation:
    """Recover the sampling *transition* gradient subtotal already recorded.

    Generated runs persist per-step chain statistics (``num_integration_steps``
    and friends) alongside the draws, and every ``BaseMethod`` already declares
    how one step's info becomes a gradient count.  This reuses that descriptor
    rather than hard-coding a per-sampler table, so a cost that *was* recorded
    need not be reported as unknown.  Rejected transitions are included: the
    per-step record covers every proposal the sampler paid for, not only the
    accepted ones.

    The derivation is refused, with a reason, when the recorded statistics do
    not justify it: an unknown sampler, a descriptor needing a field that was
    not persisted, ragged or empty statistics, or -- when
    ``expected_topology`` is given -- a per-step record that does not cover
    every chain and every draw, which is what a thinned or truncated record
    looks like.

    Parameters
    ----------
    chain_stats
        Per-step statistics, ``{field: (n_chains, n_draws)}`` -- e.g. the
        ``_ss_``-prefixed entries of a generated ``.npz`` artifact with the
        prefix stripped.
    base_method_name
        Key into ``tuningfork.base_method.BASE_METHODS``.
    expected_topology
        ``(n_chains, n_draws)`` the draws actually have.  When supplied, the
        statistics must cover exactly that many transitions.

    Returns
    -------
    GradEvalDerivation
    """
    from tuningfork.base_method import BASE_METHODS

    method = BASE_METHODS.get(base_method_name)
    if method is None:
        return GradEvalDerivation(
            None, reason=f"unknown base method: {base_method_name!r}"
        )
    counter = getattr(method, "grad_count_per_step", None)
    if counter is None:
        return GradEvalDerivation(
            None,
            reason=(
                f"{base_method_name} declares no grad_count_per_step contract, so "
                "its transition gradient cost cannot be derived"
            ),
        )

    arrays = {name: np.asarray(value) for name, value in chain_stats.items()}
    if not arrays:
        return GradEvalDerivation(
            None, reason="no per-step chain statistics were persisted for this run"
        )
    shapes = {arr.shape for arr in arrays.values()}
    if len(shapes) != 1:
        return GradEvalDerivation(
            None,
            reason=(
                "per-step statistics are ragged "
                f"({sorted(str(sh) for sh in shapes)}), so the number of "
                "transitions is ambiguous"
            ),
        )
    shape = shapes.pop()
    if expected_topology is not None and shape[:2] != tuple(expected_topology):
        return GradEvalDerivation(
            None,
            reason=(
                f"per-step statistics cover {shape[:2]} but the draws are "
                f"{tuple(expected_topology)}; the record does not cover every "
                "lane and transition (thinned, truncated, or partial)"
            ),
        )
    n_transitions = int(np.prod(shape[:2])) if len(shape) >= 2 else int(shape[0])
    if n_transitions == 0:
        return GradEvalDerivation(
            0,
            basis=f"{method.grad_count_convention} over zero recorded transitions",
            source_fields=tuple(sorted(arrays)),
            excluded=_EXCLUDED_FROM_TRANSITION_COUNT,
        )

    class _StepStats:
        """Attribute view over the persisted arrays, for grad_count_per_step."""

        def __init__(self, fields: Mapping[str, np.ndarray]) -> None:
            self.__dict__.update(fields)

    try:
        counts = np.asarray(counter(_StepStats(arrays)))
    except AttributeError as exc:
        return GradEvalDerivation(
            None,
            reason=(
                f"{base_method_name} counts gradients from a per-step field that "
                f"was not persisted ({exc})"
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive
        return GradEvalDerivation(
            None,
            reason=f"could not evaluate the {base_method_name} grad count: {exc}",
        )

    if counts.ndim == 0:
        # Constant-cost samplers declare one value per step; scale by the number
        # of recorded transitions, mirroring tuningfork.metrics.grad_counter.
        total = int(counts) * n_transitions
        used: tuple[str, ...] = ()
    elif counts.shape[:2] != shape[:2]:
        return GradEvalDerivation(
            None,
            reason=(
                f"{base_method_name} produced a per-step count of shape "
                f"{counts.shape}, which does not cover the {shape[:2]} recorded "
                "transitions"
            ),
        )
    else:
        total = int(counts.sum())
        used = tuple(sorted(arrays))
    if total < 0:
        return GradEvalDerivation(
            None, reason=f"{base_method_name} produced a negative gradient count"
        )
    return GradEvalDerivation(
        count=total,
        basis=f"{method.grad_count_convention} summed over {n_transitions} "
        "recorded transitions, rejected transitions included",
        source_fields=used or tuple(sorted(arrays)),
        excluded=_EXCLUDED_FROM_TRANSITION_COUNT,
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
    mean_tie_block: float
    tie_severity: str
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


def _mean_tie_block(n_total: int, n_distinct: int) -> float:
    """Average number of draws sharing a value; ``1.0`` when all are distinct."""
    if n_distinct <= 0:
        return 0.0
    return n_total / n_distinct


def _tie_severity(mean_tie_block: float, n_distinct: int, threshold: float) -> str:
    """``"none"`` / ``"minor"`` / ``"material"`` -- display severity only.

    This grades how loudly ties are reported.  It is never a validity boundary:
    any tie at all is disclosed, and both tie measures are reported numerically
    whatever the severity.
    """
    if mean_tie_block <= 1.0:
        return "none"
    return "material" if mean_tie_block >= threshold else "minor"


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
    tie_block_threshold: float = DEFAULT_TIE_BLOCK_THRESHOLD,
) -> ExpectandDiagnostics:
    n_chains, n_draws = trace_cs.shape
    finite = bool(np.all(np.isfinite(trace_cs)))
    n_distinct = int(np.unique(trace_cs).size)
    total = trace_cs.size
    tie_fraction = 0.0 if total == 0 else 1.0 - n_distinct / total
    mean_tie_block = _mean_tie_block(total, n_distinct)
    severity = _tie_severity(mean_tie_block, n_distinct, tie_block_threshold)

    def undefined(reason: str, degeneracy: str) -> ExpectandDiagnostics:
        return ExpectandDiagnostics(
            name=name,
            component=component,
            backend=backend,
            n_chains=n_chains,
            n_draws=n_draws,
            n_distinct=n_distinct,
            tie_fraction=tie_fraction,
            mean_tie_block=mean_tie_block,
            tie_severity=severity,
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
    if severity != "none":
        warns = (
            *warns,
            _tie_disclosure(
                tie_fraction, mean_tie_block, backend, severity == "material"
            ),
        )

    return ExpectandDiagnostics(
        name=name,
        component=component,
        backend=backend,
        n_chains=n_chains,
        n_draws=n_draws,
        n_distinct=n_distinct,
        tie_fraction=tie_fraction,
        mean_tie_block=mean_tie_block,
        tie_severity=severity,
        degeneracy=degeneracy,
        raw_mean_ess=stats["raw_mean_ess"],
        bulk_ess=stats["bulk_ess"],
        tail_ess=stats["tail_ess"],
        rank_rhat=stats["rank_rhat"],
        undefined_reasons=undefined_reasons,
        warnings=warns,
    )


def _backend_version(backend: str) -> str:
    """Version of the module that produced a report's numbers.

    Recorded because tie handling, rank normalisation and ESS truncation are all
    implementation details that move between releases.
    """
    try:
        if backend == "blackjax":
            import blackjax

            return str(blackjax.__version__)
        if backend == "arviz":
            import arviz

            return str(arviz.__version__)
    except Exception:  # pragma: no cover - defensive
        return "unknown"
    return "unknown"


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
    backend_version: str
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
                    "backend_version": self.backend_version,
                    "degeneracy": entry.degeneracy,
                    "tie_fraction": entry.tie_fraction,
                    "mean_tie_block": entry.mean_tie_block,
                    "tie_severity": entry.tie_severity,
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
            f"[backend={self.backend} {self.backend_version}, "
            f"{self.n_chains} chains x {self.n_draws} draws]"
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
    tie_block_threshold: float = DEFAULT_TIE_BLOCK_THRESHOLD,
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
    tie_block_threshold
        Mean tie-block size at or above which the tie disclosure is raised to
        "material".  A display threshold only: any tie is disclosed regardless,
        and both tie measures are reported numerically on every row.  See
        :data:`DEFAULT_TIE_BLOCK_THRESHOLD` for the measurement it is keyed on.

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
                    tie_block_threshold=tie_block_threshold,
                )
            )

    return ExpectandReport(
        label=label,
        backend=backend,
        backend_version=_backend_version(backend),
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
    per_transition_grad_eval: tuple[float | None, float | None] | None = None
    cost_blocked_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportComparison:
    """Side-by-side of two reports with costs carried through, never guessed."""

    baseline_label: str
    candidate_label: str
    rows: tuple[ComparisonRow, ...]
    cost_blockers: tuple[str, ...]
    cost_views: tuple[str, str] = ("as_measured", "as_measured")
    excluded_grad_work: tuple[str, ...] = ()
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
                "per_transition_grad_eval": (
                    list(row.per_transition_grad_eval)
                    if row.per_transition_grad_eval
                    else None
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

    The gradient denominator is ``warmup_grad_evals`` plus
    ``sampling_transition_grad_evals`` -- recorded transition work only.  It
    omits whatever each side's ``excluded_grad_work`` names, so ESS per
    transition gradient evaluation *overstates* efficiency; the union of both
    sides' exclusions is carried on the result.  ``cost_views`` records whether
    each side was costed as measured, standalone, or combined, since a
    standalone and a combined figure are not comparable.
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
        report.label: _transition_grad_evals(report.cost)
        for report in (baseline, candidate)
    }
    grad_blockers = tuple(
        f"{report.label}.{component} ({report.cost.reason_for(component)})"
        for report in (baseline, candidate)
        for component in ("warmup_grad_evals", "sampling_transition_grad_evals")
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
                    per_transition_grad_eval=per_grad,
                    cost_blocked_by=tuple(dict.fromkeys(cost_blocked)),
                )
            )

    return ReportComparison(
        baseline_label=baseline.label,
        candidate_label=candidate.label,
        rows=tuple(rows),
        cost_blockers=tuple(dict.fromkeys(seconds_blockers + grad_blockers)),
        cost_views=(baseline.cost.view, candidate.cost.view),
        excluded_grad_work=tuple(
            dict.fromkeys(
                baseline.cost.excluded_grad_work + candidate.cost.excluded_grad_work
            )
        ),
        only_in_baseline=tuple(key for key in base_map if key not in cand_map),
        only_in_candidate=tuple(key for key in cand_map if key not in base_map),
        backend_mismatch=(
            None
            if baseline.backend == candidate.backend
            else (baseline.backend, candidate.backend)
        ),
    )


def _transition_grad_evals(cost: CostAccounting) -> int | None:
    """Warmup plus sampling-transition gradient work, or ``None`` if incomplete.

    A subtotal, not a total: it omits whatever ``cost.excluded_grad_work``
    names.  Dividing an ESS by it therefore *overstates* efficiency, which is
    why every comparison carries the exclusions alongside the number.
    """
    if cost.warmup_grad_evals is None or cost.sampling_transition_grad_evals is None:
        return None
    return cost.warmup_grad_evals + cost.sampling_transition_grad_evals


def _safe_div(numerator: float | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)
