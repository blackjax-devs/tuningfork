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
"""Attribute a headline change to the ESS estimator vs. run-to-run noise.

A re-emit re-runs the sampler, so a committed headline moves for two reasons at
once: the estimator changed, and the draws are new.  Diffing new-against-committed
alone cannot separate a real estimator effect from a seed-fragile cell.

Both estimators are pure functions of the *same* draws, so the emit path records
both in ``headline_basis``.  That fixes one leg of the decomposition:

    estimator_ratio    = min_bulk_ess / min_bulk_ess_classic_legacy   (one run)
    total_change       = new_headline / committed_headline            (what ships)
    run_noise_implied  = total_change / estimator_ratio               (residual)

``run_noise_implied`` far from 1 is a seed-stability finding about that cell, not
a measurement artefact — worth reporting separately from the estimator effect.

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
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG = REPO_ROOT / "tuningfork" / "catalog"


@dataclass
class CellDelta:
    """One cell's before/after headline decomposition."""

    recipe: str
    committed_headline: float | None
    new_headline: float | None
    min_bulk_ess: float | None
    min_bulk_ess_classic_legacy: float | None
    estimator_ratio: float | None
    total_change: float | None
    run_noise_implied: float | None
    ess_estimator: str | None
    note: str = ""


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
        note="; ".join(notes),
    )


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


def summarise(deltas: list[CellDelta]) -> None:
    """Report the estimator-ratio distribution and flag unstable cells."""
    ratios = sorted(d.estimator_ratio for d in deltas if d.estimator_ratio is not None)
    if not ratios:
        print("\nNo re-emitted cells carry an estimator ratio.")
        return

    n = len(ratios)
    print(f"\nestimator_ratio over {n} re-emitted cells")
    print(f"  min    {ratios[0]:.3f}")
    print(f"  median {ratios[n // 2]:.3f}")
    print(f"  max    {ratios[-1]:.3f}")
    print(f"  cells below 0.5 : {sum(r < 0.5 for r in ratios)}")
    print(f"  cells above 2.0 : {sum(r > 2.0 for r in ratios)}")

    # A cell whose residual is far from 1 changed for reasons the estimator does
    # not explain — that is a seed-stability finding, not a migration artefact.
    unstable = [
        d
        for d in deltas
        if d.run_noise_implied is not None
        and (d.run_noise_implied < 0.5 or d.run_noise_implied > 2.0)
    ]
    if unstable:
        print("\nrun_noise_implied far from 1 (seed stability, not the estimator):")
        for d in sorted(unstable, key=lambda x: x.run_noise_implied or 0.0):
            print(f"  {d.recipe:<50} {d.run_noise_implied:.3f}")


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
        help="Git revision supplying the committed headline (default HEAD)",
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
