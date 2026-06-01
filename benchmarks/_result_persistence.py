"""Result persistence for the nightly benchmark cross-date regression check.

Results are stored on a dedicated ``benchmark-results`` branch so they survive
between nightly runs without cluttering the main-branch history.

Schema per seed (``benchmark-results/<seed>.json``)::

    {
      "date": "2026-06-01",
      "seed": 20260601,
      "env": {
        "blackjax_sha": "abc123...",
        "jax_version": "0.10.1",
        "numpy_version": "2.2.1",
        "python_version": "3.13.2",
        "runner_image": "ubuntu-24.04"
      },
      "cells": {
        "<cell_id>": {
          "n_divergences": 0,
          "min_bulk_ess": 1842.3,
          "max_abs_mean_z": 0.423,
          "runtime_warmup_s": 3.2,
          "runtime_sample_s": 0.8,
          "correctness_passed": true
        }
      }
    }
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

RESULTS_BRANCH = "benchmark-results"
_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Environment fingerprint
# ---------------------------------------------------------------------------


def get_env_fingerprint() -> dict[str, str]:
    """Collect environment metadata for triage (distinguishes env drift from regression)."""
    try:
        import jax

        jax_version = jax.__version__
    except Exception:  # noqa: BLE001
        jax_version = "unknown"

    try:
        import numpy as np

        numpy_version = np.__version__
    except Exception:  # noqa: BLE001
        numpy_version = "unknown"

    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    # blackjax SHA: try to get it from the installed package
    try:
        import blackjax

        blackjax_sha = getattr(blackjax, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        blackjax_sha = "unknown"

    runner_image = os.environ.get("ImageOS", os.environ.get("RUNNER_OS", "unknown"))

    return {
        "blackjax_sha": blackjax_sha,
        "jax_version": jax_version,
        "numpy_version": numpy_version,
        "python_version": python_version,
        "runner_image": runner_image,
    }


# ---------------------------------------------------------------------------
# Read / write benchmark-results branch
# ---------------------------------------------------------------------------


def load_prior_result(seed: int, repo_root: Path = _REPO_ROOT) -> dict[str, Any] | None:
    """Load <seed>.json from the benchmark-results branch.  Returns None on miss."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{RESULTS_BRANCH}:{seed}.json"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except Exception:  # noqa: BLE001
        return None


def load_recent_results(
    n: int = 3, repo_root: Path = _REPO_ROOT
) -> list[dict[str, Any]]:
    """Load the most recent N seed results from the benchmark-results branch.

    Used for ESS-trend baseline (3-night median).
    """
    try:
        # List all json files on the branch sorted by seed (= YYYYMMDD = chronological)
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "ls-tree",
                "--name-only",
                RESULTS_BRANCH,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        files = sorted(
            [f for f in result.stdout.strip().splitlines() if f.endswith(".json")],
            reverse=True,  # most recent first (YYYYMMDD desc)
        )[:n]

        results = []
        for fname in files:
            seed_str = fname.replace(".json", "")
            try:
                seed = int(seed_str)
            except ValueError:
                continue
            r = load_prior_result(seed, repo_root)
            if r is not None:
                results.append(r)
        return results
    except Exception:  # noqa: BLE001
        return []


def store_result(
    seed: int,
    cells: dict[str, dict[str, Any]],
    run_date: date | None = None,
    repo_root: Path = _REPO_ROOT,
) -> bool:
    """Commit <seed>.json to the benchmark-results branch.

    Creates the branch if it doesn't exist.  Returns True on success.
    """
    result_json: dict[str, Any] = {
        "date": (run_date or date.today()).isoformat(),
        "seed": seed,
        "env": get_env_fingerprint(),
        "cells": cells,
    }

    try:
        # Ensure the branch exists; create an orphan branch if not
        check = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "ls-remote",
                "--heads",
                "origin",
                RESULTS_BRANCH,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        branch_exists = bool(check.stdout.strip())

        if not branch_exists:
            # Create empty orphan branch on origin via a temp worktree approach
            # Simpler: use git commit-tree directly
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "checkout",
                    "--orphan",
                    RESULTS_BRANCH,
                ],
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["git", "-C", str(repo_root), "rm", "-rf", "--quiet", "."],
                capture_output=True,
                check=False,
            )

        # Write the JSON to the branch directly (no worktree needed)
        blob = subprocess.run(
            ["git", "-C", str(repo_root), "hash-object", "-w", "--stdin"],
            input=json.dumps(result_json, indent=2) + "\n",
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        # Build a minimal tree with the new file
        old_tree = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "ls-tree",
                RESULTS_BRANCH,
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

        # Filter out existing entry for this seed (if any) and add new
        entries = [
            line
            for line in old_tree.splitlines()
            if not line.endswith(f"\t{seed}.json")
        ]
        entries.append(f"100644 blob {blob}\t{seed}.json")

        new_tree = subprocess.run(
            ["git", "-C", str(repo_root), "mktree"],
            input="\n".join(entries) + "\n",
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        # Get parent commit (if branch exists)
        parent = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "rev-parse",
                "--verify",
                f"refs/heads/{RESULTS_BRANCH}",
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

        commit_cmd = ["git", "-C", str(repo_root), "commit-tree", new_tree]
        if parent:
            commit_cmd += ["-p", parent]
        commit_cmd += ["-m", f"bench: seed={seed} date={result_json['date']}"]

        new_commit = subprocess.run(
            commit_cmd, capture_output=True, text=True, check=True
        ).stdout.strip()

        # Update the branch ref
        subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "update-ref",
                f"refs/heads/{RESULTS_BRANCH}",
                new_commit,
            ],
            capture_output=True,
            check=True,
        )

        # Push the branch
        subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "push",
                "origin",
                f"{RESULTS_BRANCH}:{RESULTS_BRANCH}",
            ],
            capture_output=True,
            check=False,  # non-fatal: nightly continues even if push fails
        )
        return True

    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not store benchmark result: {exc}", file=sys.stderr)
        return False
