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

Usage::

    uv run python tools/verify_emitted_configs.py --baseline b09c247
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG = REPO_ROOT / "tuningfork" / "catalog"


def _driver():
    spec = importlib.util.spec_from_file_location(
        "reemit_sweep", REPO_ROOT / "tools" / "reemit_sweep.py"
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["reemit_sweep"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="b09c247")
    args = parser.parse_args()

    driver = _driver()
    checked = 0
    offenders: list[str] = []

    for path in sorted(CATALOG.glob("*/recipes/*.json")):
        on_disk = json.loads(path.read_text())
        # Only re-emitted cells carry the estimator stamp; the rest are untouched.
        if not (on_disk.get("headline_basis") or {}).get("ess_estimator"):
            continue
        rel = path.relative_to(REPO_ROOT)
        proc = subprocess.run(
            ["git", "show", f"{args.baseline}:{rel}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            continue
        committed = json.loads(proc.stdout)

        cfg = driver.reconstruct(path)
        if isinstance(cfg, driver.Skip):
            continue
        checked += 1
        # Compare the config the emitted artifact RECORDS against the committed one.
        bad = driver.config_fidelity_violations(cfg, committed)
        if bad:
            offenders.append(
                f"{path.parent.parent.name}/{path.name}: " + "; ".join(bad)
            )

    print(f"re-emitted cells checked : {checked}")
    print(f"config mismatches        : {len(offenders)}")
    for o in offenders:
        print(f"  {o}")
    return 1 if offenders else 0


if __name__ == "__main__":
    sys.exit(main())
