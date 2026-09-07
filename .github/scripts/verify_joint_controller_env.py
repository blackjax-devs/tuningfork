#!/usr/bin/env python
"""Verify the pinned-upstream BlackJAX overlay actually took effect.

Version strings are not evidence: an editable overlay and a released wheel can
report confusingly similar values, and a stray ``uv`` re-sync restores the
released package without changing anything a version check would notice.  So
this asserts three independent things before any test runs:

1. the imported module's ``__file__`` resolves INSIDE the pinned checkout;
2. that checkout's git HEAD equals the pinned SHA exactly; and
3. ``staged_adaptation`` actually accepts ``n_chains`` -- the capability the
   joint job exists to exercise -- probed on the live signature.

Exits non-zero with the observed values on any mismatch.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    expected_sha = os.environ["BLACKJAX_PINNED_SHA"]
    checkout = Path(os.environ["BLACKJAX_PINNED_PATH"]).resolve()

    import blackjax

    module_file = Path(blackjax.__file__).resolve()
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    has_n_chains = (
        "n_chains" in inspect.signature(blackjax.staged_adaptation).parameters
    )

    print(f"blackjax.__version__ : {blackjax.__version__}")
    print(f"blackjax.__file__    : {module_file}")
    print(f"pinned checkout      : {checkout}")
    print(f"checkout HEAD        : {head}")
    print(f"expected SHA         : {expected_sha}")
    print(f"n_chains capability  : {has_n_chains}")

    failures: list[str] = []
    if checkout not in module_file.parents:
        failures.append(
            f"imported blackjax is not the pinned checkout: {module_file} is not "
            f"under {checkout}. A re-sync most likely restored the released package."
        )
    if head != expected_sha:
        failures.append(f"pinned checkout HEAD {head} != expected {expected_sha}")
    if not has_n_chains:
        failures.append(
            "staged_adaptation does not accept n_chains; the joint controller "
            "capability this job gates on is absent"
        )

    for failure in failures:
        print(f"::error::{failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
