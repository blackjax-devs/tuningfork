#!/usr/bin/env python
"""Verify the pinned-upstream BlackJAX overlay actually took effect.

Version strings are not evidence: an editable overlay and a released wheel can
report confusingly similar values, and a stray ``uv`` re-sync restores the
released package without changing anything a version check would notice.  So
this asserts three independent things before any test runs:

1. the imported module's ``__file__`` resolves INSIDE the pinned checkout;
2. that checkout's git HEAD equals the pinned SHA exactly;
3. ``staged_adaptation`` actually accepts ``n_chains`` -- the capability the
   joint job exists to exercise -- probed on the live signature; and
4. the same holds under a CHILD-LIKE invocation: a fresh process, same
   interpreter, run from a foreign working directory the way the launcher runs
   generated programs. This is an actual ``import blackjax``, so it is evidence
   about what a real import resolves to, not merely what a resolver reports.

Together with the receipt assertion in the joint tests -- that the generated
program ran under this same interpreter -- that covers the ``sys.path[0]``
shadowing case a same-process check alone would miss.

Exits non-zero with the observed values on any mismatch.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Actually imports blackjax, in a fresh process, so the answer is what a real
# import resolves to rather than what a resolver predicts.
_CHILD_LIKE_IMPORT = (
    "import blackjax, json, sys; "
    "print(json.dumps({'file': blackjax.__file__, "
    "'version': blackjax.__version__, "
    "'n_chains': 'n_chains' in __import__('inspect')"
    ".signature(blackjax.staged_adaptation).parameters, "
    "'sys_path_0': sys.path[0]}))"
)


def _child_like_import(checkout: Path) -> tuple[dict, str | None]:
    """Import blackjax the way a launched generated program would.

    Same interpreter, fresh process, and a foreign working directory -- the
    launcher runs generated programs from their own work directory, so cwd is
    the one input a same-process check cannot speak for.
    """
    import json

    with tempfile.TemporaryDirectory() as work:
        completed = subprocess.run(
            [sys.executable, "-c", _CHILD_LIKE_IMPORT],
            capture_output=True,
            text=True,
            cwd=work,
            check=False,
        )
    if completed.returncode != 0:
        return (
            {},
            f"child-like import exited {completed.returncode}: {completed.stderr[-800:]}",
        )
    try:
        return json.loads(completed.stdout), None
    except ValueError as exc:
        return {}, f"child-like import output was not JSON: {exc}"


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

    child, child_error = _child_like_import(checkout)
    if child_error is None:
        print(f"child-like import    : {child.get('file')}")
        print(f"child-like version   : {child.get('version')}")
        print(f"child-like n_chains  : {child.get('n_chains')}")
        print(f"child-like sys.path0 : {child.get('sys_path_0')!r}")

    failures: list[str] = []
    if child_error is not None:
        failures.append(child_error)
    else:
        child_file = child.get("file")
        if child_file is None or checkout not in Path(child_file).resolve().parents:
            failures.append(
                f"a child-like invocation imported blackjax from {child_file}, "
                f"not from the pinned checkout {checkout}"
            )
        if not child.get("n_chains"):
            failures.append(
                "a child-like invocation imported a blackjax without n_chains"
            )
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
