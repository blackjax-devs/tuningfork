#!/usr/bin/env python
"""Assert every required case EXECUTED -- skipped or uncollected fails.

In the pinned-upstream job a skip is not an acceptable outcome: it would mean
the capability was silently absent and the gate passed without testing
anything.  pytest already exits non-zero when a node id does not resolve, which
covers "uncollected"; this covers "collected but not run".

This counts cases; it does not know what they cover.  The workflow header
records which of them actually exercise the generated joint program (three of
the five) so the count is not mistaken for five joint executions.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    report, expected = Path(sys.argv[1]), int(sys.argv[2])
    suite = ET.parse(report).getroot().find("testsuite")
    if suite is None:
        print("::error::no <testsuite> in the junit report")
        return 1

    counts = {
        k: int(suite.get(k, 0)) for k in ("tests", "skipped", "failures", "errors")
    }
    print(f"junit counts: {counts} (expected tests={expected}, skipped=0)")

    failures = []
    if counts["tests"] != expected:
        failures.append(
            f"expected exactly {expected} required joint cases, ran {counts['tests']}"
        )
    if counts["skipped"]:
        failures.append(
            f"{counts['skipped']} required joint case(s) SKIPPED; a skip is a "
            "failing gate in this job"
        )
    if counts["failures"] or counts["errors"]:
        failures.append(f"{counts['failures']} failure(s), {counts['errors']} error(s)")

    for failure in failures:
        print(f"::error::{failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
