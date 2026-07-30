#!/usr/bin/env python3
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
"""Attribute a headline change to the ESS estimator, a config fix, or drift.

A re-emit re-runs the sampler, so a committed headline can move for several
reasons at once and a plain new-against-committed diff cannot tell them apart.

Both estimators are pure functions of the *same* draws, so the emit path records
both in ``headline_basis``.  That fixes one leg of the decomposition:

    estimator_ratio    = min_bulk_ess / min_bulk_ess_classic_legacy   (one run)
    total_change       = new_headline / committed_headline            (what ships)
    run_noise_implied  = total_change / estimator_ratio               (residual)

A third cause is reported separately rather than folded into the residual: some
committed recipes carry a gradient budget no stated protocol reproduces, so
re-measuring them corrects a denominator.  Pooling those with genuine replays is
how a 10x budget error reads as a 9.4x estimator anomaly.

The residual is **version drift**, not seed noise.  An emit is deterministic
given its seed and configuration, so a faithfully replayed cell can only move
because its dependencies moved.  This is measured, not inferred: jax 0.10.0 ->
0.10.1 shifts one model's adapted warmup step size by ~16% with the RNG stream
and a single sampler step both bit-identical — chaotic amplification through a
thousand-step warmup, not a numerics bug.  The report therefore groups drift by
the ORIGINATING version combination, so it shows which upgrades move our numbers.

Why that matters more than the estimator switch: a corpus emitted across many
dependency stacks is internally incoherent, and cross-model comparison is what
the headline metric is for.  Re-emitting puts every cell on ONE stack.  The
summary states the before/after spread of version combinations first, because
that re-baselining is the substantive result — the estimator ratio is a couple
of percent on the median cell.

The committed side is read from a git revision rather than from the working tree,
so re-running an emit twice cannot silently rebase the comparison onto an already
re-emitted value.

Usage::

    uv run python tools/estimator_delta_report.py                    # whole catalog
    uv run python tools/estimator_delta_report.py --rev HEAD~3
    uv run python tools/estimator_delta_report.py --json out.json \\
        tuningfork/catalog/mvn_10/recipes/low__nuts__window_adaptation_diag_imm.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG = REPO_ROOT / "tuningfork" / "catalog"


@dataclass
class CellDelta:
    """One cell's before/after headline decomposition.

    Three causes are kept apart on purpose.  ``estimator_ratio`` is the switch
    under study, measured on one fixed set of draws.  ``config_correction`` marks
    a cell whose committed gradient budget was itself wrong, so its movement is
    dominated by the budget being fixed rather than by anything about the metric.
    ``run_noise_implied`` is whatever remains.  It is version drift, not seed
    fragility: an emit is deterministic given its seed and configuration, so a
    faithful replay can only move because its dependencies moved.
    """

    recipe: str
    committed_headline: float | None
    new_headline: float | None
    min_bulk_ess: float | None
    min_bulk_ess_classic_legacy: float | None
    estimator_ratio: float | None
    total_change: float | None
    run_noise_implied: float | None
    ess_estimator: str | None
    config_correction: bool = False
    grad_evals_before: int | None = None
    grad_evals_after: int | None = None
    committed_blackjax: str | None = None
    committed_jax: str | None = None
    committed_tuningfork: str | None = None
    note: str = ""

    @property
    def origin_stack(self) -> str:
        """The (blackjax, jax, tuningfork) triple that produced the committed side."""
        return (
            f"blackjax {self.committed_blackjax} / jax {self.committed_jax} / "
            f"tuningfork {self.committed_tuningfork}"
        )

    @property
    def version_drift(self) -> bool:
        """Was the committed side produced under different library versions?"""
        import jax

        try:
            import blackjax
        except ImportError:  # pragma: no cover - blackjax is a hard dependency
            return False
        if self.committed_blackjax is None:
            return False
        return (
            self.committed_blackjax != blackjax.__version__
            or self.committed_jax != jax.__version__
        )


def _read_committed(rel_path: str, rev: str) -> dict | None:
    """Load a recipe as of ``rev``; ``None`` when it did not exist there."""
    proc = subprocess.run(
        ["git", "show", f"{rev}:{rel_path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def compute_delta(path: Path, rev: str) -> CellDelta:
    """Decompose one recipe's headline change against its committed counterpart."""
    rel = str(path.relative_to(REPO_ROOT))
    new = json.loads(path.read_text())
    basis = new.get("headline_basis") or {}
    committed = _read_committed(rel, rev)

    notes: list[str] = []
    committed_headline = None
    if committed is None:
        notes.append(f"no counterpart at {rev}")
    else:
        committed_headline = committed.get("headline_metric")
        if committed_headline is None:
            notes.append("committed headline is null")

    ratio = basis.get("estimator_ratio")
    if basis.get("min_bulk_ess_classic_legacy") is None:
        notes.append("not re-emitted (no legacy ESS recorded)")

    old_basis = (committed or {}).get("headline_basis") or {}
    grad_before = old_basis.get("total_grad_evals")
    grad_after = basis.get("total_grad_evals")
    cell_key = f"{Path(rel).parent.parent.name}/{Path(rel).name}"
    config_correction = cell_key in _config_correction_cells()
    if grad_before and grad_after and grad_before != grad_after:
        # The gradient budget is the headline's denominator, so a change in it
        # moves the headline for reasons unrelated to the estimator.  But an
        # adaptive sampler re-derives its own trajectory lengths every run, so a
        # few percent of movement here is ordinary — only an order-of-magnitude
        # gap indicates a budget that was wrong rather than merely re-measured.
        # Calling the ordinary case a "correction" would quarantine most of the
        # NUTS family out of the estimator statistics for no reason.
        budget_ratio = grad_after / grad_before
        if not 0.5 <= budget_ratio <= 2.0:
            config_correction = True
        if abs(budget_ratio - 1.0) > 0.05:
            notes.append(f"gradient budget {grad_before} -> {grad_after}")

    total_change = _safe_ratio(new.get("headline_metric"), committed_headline)
    return CellDelta(
        recipe=rel.replace("tuningfork/catalog/", "").replace("/recipes/", " / "),
        committed_headline=committed_headline,
        new_headline=new.get("headline_metric"),
        min_bulk_ess=basis.get("min_bulk_ess"),
        min_bulk_ess_classic_legacy=basis.get("min_bulk_ess_classic_legacy"),
        estimator_ratio=ratio,
        total_change=total_change,
        run_noise_implied=_safe_ratio(total_change, ratio),
        ess_estimator=basis.get("ess_estimator"),
        config_correction=config_correction,
        grad_evals_before=grad_before,
        grad_evals_after=grad_after,
        committed_blackjax=(committed or {}).get("blackjax_version"),
        committed_jax=(committed or {}).get("jax_version"),
        committed_tuningfork=(committed or {}).get("tuningfork_version"),
        note="; ".join(notes),
    )


def _config_correction_cells() -> dict[str, str]:
    """The re-emit driver's list of cells emitted under a corrected protocol."""
    import importlib.util
    import sys

    path = REPO_ROOT / "tools" / "reemit_sweep.py"
    if not path.exists():
        return {}
    spec = importlib.util.spec_from_file_location("reemit_sweep", path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    # Register before executing: the module's dataclasses resolve annotations by
    # looking themselves up in sys.modules.
    sys.modules["reemit_sweep"] = module
    spec.loader.exec_module(module)
    return dict(module.CONFIG_CORRECTION_CELLS)


def _fmt(value: float | None, spec: str = ".4g") -> str:
    return "-" if value is None else format(value, spec)


def print_table(deltas: list[CellDelta]) -> None:
    """Print the per-cell decomposition, widest column first."""
    width = max((len(d.recipe) for d in deltas), default=10)
    header = (
        f"{'cell':<{width}}  {'committed':>10}  {'new':>10}  {'est_ratio':>9}  "
        f"{'total_chg':>9}  {'run_noise':>9}"
    )
    print(header)
    print("-" * len(header))
    for d in deltas:
        print(
            f"{d.recipe:<{width}}  {_fmt(d.committed_headline):>10}  "
            f"{_fmt(d.new_headline):>10}  {_fmt(d.estimator_ratio, '.3f'):>9}  "
            f"{_fmt(d.total_change, '.3f'):>9}  {_fmt(d.run_noise_implied, '.3f'):>9}"
            + (f"   [{d.note}]" if d.note else "")
        )


def _quantiles(values: list[float]) -> None:
    n = len(values)
    for label, q in (
        ("min", 0.0),
        ("p10", 0.10),
        ("p25", 0.25),
        ("median", 0.50),
        ("p75", 0.75),
        ("p90", 0.90),
        ("max", 1.0),
    ):
        print(f"  {label:>6} {values[int(q * (n - 1))]:7.3f}")


def _rebaselining_summary(deltas: list[CellDelta]) -> None:
    """State the dependency-stack consolidation first — it is the substantive result.

    A corpus emitted across many (blackjax, jax, tuningfork) combinations is
    internally incoherent: cross-model comparison, which is what the headline
    metric exists for, is then comparing numbers produced by different stacks.
    """
    import blackjax
    import jax

    from tuningfork._version import __version__ as tf_version

    stacks = Counter(d.origin_stack for d in deltas if d.committed_blackjax is not None)
    on_current = sum(1 for d in deltas if not d.version_drift)

    print("=" * 78)
    print("RE-BASELINING — the corpus's dependency stack, before and after")
    print("=" * 78)
    print(f"  committed cells span {len(stacks)} distinct version combinations")
    print(f"  of {len(deltas)} cells, {on_current} were already on current versions")
    print(
        f"  after this sweep every re-emitted cell is on "
        f"blackjax {blackjax.__version__} / jax {jax.__version__} / "
        f"tuningfork {tf_version}"
    )
    print("\n  originating combinations, most common first:")
    for stack, n in stacks.most_common():
        print(f"    {n:>4}  {stack}")


def _drift_by_origin(deltas: list[CellDelta]) -> None:
    """Group drift magnitude by originating stack.

    If drift clusters on particular upgrades, that is reusable knowledge about
    which dependency transitions move our numbers.
    """
    groups: dict[str, list[float]] = {}
    for d in deltas:
        if d.run_noise_implied is None or d.committed_blackjax is None:
            continue
        groups.setdefault(d.origin_stack, []).append(d.run_noise_implied)

    if not groups:
        return
    print("\nversion drift by originating stack (residual after the estimator)")
    print(f"  {'n':>4}  {'median':>7}  {'min':>7}  {'max':>7}  origin")
    for stack, values in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        v = sorted(values)
        median = v[len(v) // 2]
        print(f"  {len(v):>4}  {median:>7.3f}  {v[0]:>7.3f}  {v[-1]:>7.3f}  {stack}")


def summarise(deltas: list[CellDelta]) -> None:
    """Report each cause separately, then the review surfaces."""
    _rebaselining_summary(deltas)
    # Config corrections are excluded from the estimator statistics: their
    # headline moved because a wrong denominator was fixed, and pooling them
    # would read as an estimator effect.
    replays = [d for d in deltas if not d.config_correction]
    corrections = [d for d in deltas if d.config_correction]

    ratios = sorted(d.estimator_ratio for d in replays if d.estimator_ratio is not None)
    if not ratios:
        print("\nNo re-emitted cells carry an estimator ratio.")
        return

    print(
        f"\nestimator_ratio — the switch, isolated on fixed draws ({len(ratios)} cells)"
    )
    _quantiles(ratios)
    print(f"  within +/-10% of 1.0 : {sum(0.9 <= r <= 1.1 for r in ratios)}")
    print(f"  within +/-25% of 1.0 : {sum(0.8 <= r <= 1.25 for r in ratios)}")
    print(f"  above 1.25           : {sum(r > 1.25 for r in ratios)}")
    print(f"  above 3.0            : {sum(r > 3.0 for r in ratios)}")

    residuals = sorted(
        d.run_noise_implied for d in replays if d.run_noise_implied is not None
    )
    if residuals:
        drifted = sum(1 for d in replays if d.version_drift)
        print(
            f"\nrun_noise_implied — everything the estimator does not explain "
            f"({len(residuals)} cells)"
        )
        _quantiles(residuals)
        print(
            f"  This is VERSION DRIFT, not seed fragility: an emit is deterministic\n"
            f"  given its seed and configuration, so a faithful replay can only move\n"
            f"  because its dependencies moved.  {drifted} of {len(replays)} replayed\n"
            f"  cells were committed under a different blackjax or jax."
        )
        _drift_by_origin(replays)

    print("\n--- review surface (a): total change beyond 25% ---")
    movers = [
        d
        for d in deltas
        if d.total_change is not None and not 0.75 <= d.total_change <= 1.25
    ]
    for d in sorted(movers, key=lambda x: -(x.total_change or 0)):
        tag = " [config correction]" if d.config_correction else ""
        print(f"  {d.total_change:8.3f}  {d.recipe}{tag}")
    if not movers:
        print("  none")

    print("\n--- review surface (b): estimator_ratio beyond 1.25 ---")
    big = [d for d in replays if d.estimator_ratio and d.estimator_ratio > 1.25]
    for d in sorted(big, key=lambda x: -(x.estimator_ratio or 0)):
        print(f"  {d.estimator_ratio:8.3f}  {d.recipe}")
    if not big:
        print("  none")

    print("\n--- review surface (c): version drift far from 1 (not the estimator) ---")
    print(
        "    a cell here is dependency-sensitive; every claim built on it inherits that"
    )
    unstable = [
        d
        for d in replays
        if d.run_noise_implied is not None
        and (d.run_noise_implied < 0.5 or d.run_noise_implied > 2.0)
    ]
    for d in sorted(unstable, key=lambda x: x.run_noise_implied or 0.0):
        drift = (
            ""
            if not d.version_drift
            else f" [committed on blackjax {d.committed_blackjax}]"
        )
        print(f"  {d.run_noise_implied:8.3f}  {d.recipe}{drift}")
    if not unstable:
        print("  none")

    if corrections:
        print(
            "\n--- config corrections (denominator fixed; NOT an estimator effect) ---"
        )
        for d in corrections:
            print(
                f"  {d.recipe}: total_change="
                f"{_fmt(d.total_change, '.3f')} "
                f"grad_evals {d.grad_evals_before} -> {d.grad_evals_after}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Recipe JSON paths (default: every recipe in the catalog)",
    )
    parser.add_argument(
        "--rev",
        default="HEAD",
        help="Git revision supplying the committed headline (default HEAD). "
        "Point this at the PRE-SWITCH commit, not HEAD, once any cell has been "
        "re-emitted on the branch — otherwise already-migrated cells compare "
        "against themselves and report a total change of exactly 1.000.",
    )
    parser.add_argument("--json", help="Also write the rows to this JSON file")
    parser.add_argument(
        "--only-reemitted",
        action="store_true",
        help="Skip cells that carry no legacy ESS (i.e. were not re-emitted)",
    )
    args = parser.parse_args()

    paths = (
        [Path(p).resolve() for p in args.paths]
        if args.paths
        else sorted(CATALOG.glob("*/recipes/*.json"))
    )
    deltas = [compute_delta(p, args.rev) for p in paths]
    if args.only_reemitted:
        deltas = [d for d in deltas if d.min_bulk_ess_classic_legacy is not None]

    if not deltas:
        print("No recipes matched.")
        return 1

    print_table(deltas)
    summarise(deltas)

    if args.json:
        Path(args.json).write_text(json.dumps([asdict(d) for d in deltas], indent=2))
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
