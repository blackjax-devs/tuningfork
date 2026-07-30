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
"""Check that re-emitted recipes were RUN with the configuration they recorded.

The plan-time gate in ``reemit_sweep.py`` proves a reconstruction is faithful to
the committed artifact.  It cannot prove the executor passed that reconstruction
on — a driver that builds the right plan and then forgets an argument produces a
plausible recipe under the wrong settings, and every plan-time check stays green.
That happened: the plan carried each variational warmup's recorded optimisation
budget and the emit call dropped it, so five cells re-ran at the registry default.

This closes the loop from the other end, comparing what is ON DISK now against
what was committed at the baseline revision.

Every skip path counts, and most of them fail
---------------------------------------------
An earlier version of this module was a one-shot script whose skips were silent,
which made it pass vacuously in two ways a reviewer demonstrated: delete
``headline_basis.ess_estimator`` from an artifact and its mismatch disappears
along with it; make one ``base_method_name`` unregistered and ``reconstruct()``
declines the cell, again taking its mismatch with it.  Both are the shape of an
ordinary future schema change.

So no path here shrinks the denominator quietly:

- a stamped cell the driver cannot reconstruct is a **failure** — the driver
  emitted it, so the driver must be able to rebuild its call;
- a stamped cell with no counterpart at the baseline is a **failure** — there is
  nothing to have reproduced;
- a cell whose bytes CHANGED since the baseline but carries no stamp is a
  **failure** — that is precisely deleting the stamp to make a mismatch go away,
  and it is the one route a floor alone cannot see, because dropping a single
  stamp keeps the count comfortably above any floor;
- cells with no estimator stamp are counted, and the stamped total is held above
  a floor, so removing stamps en masse fails instead of passing on a smaller set;
- pinned precision flips are reported, and a pin entry that no longer describes
  the artifacts is a **failure**, so the acceptance list cannot rot.

``tests/recipes/test_verify_emitted_configs.py`` runs this against the pinned
baseline as part of the suite; it is a standing gate rather than an audit tool.

Usage::

    uv run python tools/verify_emitted_configs.py --baseline b09c247
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG = REPO_ROOT / "tuningfork" / "catalog"

#: The revision every re-emitted cell claims to reproduce.  Pinned rather than
#: defaulted to a moving ref: comparing against HEAD would compare each
#: re-emitted artifact with itself and report a clean bill for any corruption
#: committed alongside it.
BASELINE_REVISION = "b09c2476"

#: Below this many stamped cells the corpus has gone dark rather than clean.  138
#: cells carry the stamp today; the margin is headroom for in-flight per-cell
#: recert work, not licence for a third of the corpus to stop being checked.
MIN_STAMPED_CELLS = 130


def _driver():
    spec = importlib.util.spec_from_file_location(
        "reemit_sweep", REPO_ROOT / "tools" / "reemit_sweep.py"
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["reemit_sweep"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@dataclass
class VerifyReport:
    """What the artifact-side comparison found, including what it did not check.

    The counts are part of the result, not diagnostics: a mismatch count of zero
    means nothing without the number of cells it was computed over.
    """

    baseline: str
    stamped: int = 0
    checked: int = 0
    unstamped: int = 0
    config_mismatches: list[str] = field(default_factory=list)
    unreconstructable: list[str] = field(default_factory=list)
    missing_baseline: list[str] = field(default_factory=list)
    modified_without_stamp: list[str] = field(default_factory=list)
    precision_flips: list[str] = field(default_factory=list)
    stale_precision_pins: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[str]:
        """Every finding that should fail the gate, labelled by kind."""
        return (
            [f"config mismatch: {m}" for m in self.config_mismatches]
            + [f"stamped but unreconstructable: {m}" for m in self.unreconstructable]
            + [
                f"stamped with no baseline counterpart: {m}"
                for m in self.missing_baseline
            ]
            + [
                f"changed since baseline but carries no stamp: {m}"
                for m in self.modified_without_stamp
            ]
            + [f"stale precision-flip pin: {m}" for m in self.stale_precision_pins]
        )

    @property
    def vacuous(self) -> str | None:
        """Why this run proves nothing, or ``None`` if it proves something."""
        if self.stamped < MIN_STAMPED_CELLS:
            return (
                f"only {self.stamped} cells carry an estimator stamp "
                f"(floor {MIN_STAMPED_CELLS}) — the corpus has gone dark, not clean"
            )
        if self.checked < MIN_STAMPED_CELLS:
            return (
                f"only {self.checked} of {self.stamped} stamped cells were compared "
                f"(floor {MIN_STAMPED_CELLS})"
            )
        return None

    def render(self) -> str:
        lines = [
            f"baseline revision        : {self.baseline}",
            f"cells carrying a stamp   : {self.stamped}",
            f"re-emitted cells checked : {self.checked}",
            f"cells with no stamp      : {self.unstamped} (not re-emitted; not checked)",
            f"config mismatches        : {len(self.config_mismatches)}",
            f"changed without a stamp  : {len(self.modified_without_stamp)}",
            f"precision flips (pinned) : {len(self.precision_flips)}",
        ]
        if self.vacuous:
            lines.append(f"VACUOUS                  : {self.vacuous}")
        for f in self.failures:
            lines.append(f"  {f}")
        for f in self.precision_flips:
            lines.append(f"  precision flip: {f}")
        return "\n".join(lines)


def _changed_since(baseline: str) -> set[Path]:
    """Recipe paths whose bytes differ from ``baseline``, working tree included."""
    proc = subprocess.run(
        ["git", "diff", "--name-only", baseline, "--", "tuningfork/catalog"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return {
        REPO_ROOT / rel
        for rel in proc.stdout.split()
        if "/recipes/" in rel and rel.endswith(".json")
    }


def verify(baseline: str = BASELINE_REVISION) -> VerifyReport:
    """Compare every stamped artifact on disk against its counterpart at ``baseline``."""
    driver = _driver()
    report = VerifyReport(baseline=baseline)
    seen_flips: set[str] = set()
    changed = _changed_since(baseline)

    for path in sorted(CATALOG.glob("*/recipes/*.json")):
        on_disk = json.loads(path.read_text())
        key = f"{path.parent.parent.name}/{path.name}"
        # Only re-emitted cells carry the estimator stamp; the rest are untouched.
        if not (on_disk.get("headline_basis") or {}).get("ess_estimator"):
            report.unstamped += 1
            if path in changed:
                report.modified_without_stamp.append(key)
            continue
        report.stamped += 1

        rel = path.relative_to(REPO_ROOT)
        proc = subprocess.run(
            ["git", "show", f"{baseline}:{rel}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            report.missing_baseline.append(key)
            continue
        committed = json.loads(proc.stdout)

        cfg = driver.reconstruct(path)
        if isinstance(cfg, driver.Skip):
            report.unreconstructable.append(f"{key}: {cfg.reason}")
            continue
        report.checked += 1

        # Observed independently of the pin, so a flip is reported whether or not
        # anyone recorded it; config_fidelity_violations suppresses the pinned ones.
        if driver.recorded_x64(committed) != cfg.recorded_x64:
            seen_flips.add(key)
            report.precision_flips.append(
                f"{key}: {driver.recorded_x64(committed)!r} -> {cfg.recorded_x64!r}"
            )

        # Compare the config the emitted artifact RECORDS against the committed one.
        for bad in driver.config_fidelity_violations(cfg, committed):
            report.config_mismatches.append(f"{key}: {bad}")

    report.stale_precision_pins = [
        f"{key} no longer flips precision — drop it from PRECISION_FLIP_CELLS"
        for key in driver.PRECISION_FLIP_CELLS
        if key not in seen_flips
    ]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE_REVISION)
    args = parser.parse_args()

    report = verify(args.baseline)
    print(report.render())
    return 1 if (report.failures or report.vacuous) else 0


if __name__ == "__main__":
    sys.exit(main())
