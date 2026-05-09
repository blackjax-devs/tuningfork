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
"""Phase 1 end-to-end tests — CLI subprocess integration.

Spawns `bjx-bench tier-a ...` as a subprocess with an isolated
BJX_BENCH_REFERENCE_DIR pointing at a tmp_path so tests do not pollute
the committed reference cache.

Cache validity note: the CLI's _get_code_sha() reads git HEAD, so within a
single git state all cache hits are deterministic.

Eight-schools note: the CLI hard-codes n_warmup=500, n_chunks=4 for NUTS
models (v1). With n=4000 and seed=42 the run certifies in ~40s.
"""

import json
import os
import subprocess
import time
from pathlib import Path


def _run_cli(
    args: list[str],
    tmp_path: Path,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """Run `uv run bjx-bench <args>` with an isolated reference dir."""
    env = {**os.environ, "BJX_BENCH_REFERENCE_DIR": str(tmp_path)}
    return subprocess.run(
        ["uv", "run", "bjx-bench", *args],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
        cwd=str(Path(__file__).parent.parent),
    )


class TestTierACLI:
    """Integration tests for `bjx-bench tier-a`."""

    def test_mvn_populates_cache(self, tmp_path: Path) -> None:
        """First run: cache should be populated."""
        _run_cli(["tier-a", "mvn_10", "--n", "1000"], tmp_path)
        assert (tmp_path / "draws" / "mvn_10.npz").exists()
        assert (tmp_path / "metadata" / "mvn_10.json").exists()
        assert (tmp_path / "summaries" / "mvn_10.json").exists()

    def test_mvn_cache_hit_second_run(self, tmp_path: Path) -> None:
        """Second run must be a cache hit — metadata timestamp must not change."""
        _run_cli(["tier-a", "mvn_10", "--n", "1000"], tmp_path)
        meta_path = tmp_path / "metadata" / "mvn_10.json"
        with meta_path.open() as fh:
            first_meta = json.load(fh)
        first_ts = first_meta["timestamp_utc"]

        # Small sleep to ensure timestamp would differ on regeneration
        time.sleep(1)

        _run_cli(["tier-a", "mvn_10", "--n", "1000"], tmp_path)
        with meta_path.open() as fh:
            second_meta = json.load(fh)
        second_ts = second_meta["timestamp_utc"]

        assert (
            first_ts == second_ts
        ), f"Cache hit expected but timestamp changed: {first_ts!r} → {second_ts!r}"

    def test_mvn_force_regenerates(self, tmp_path: Path) -> None:
        """--force must update the timestamp (regeneration happened)."""
        _run_cli(["tier-a", "mvn_10", "--n", "1000"], tmp_path)
        meta_path = tmp_path / "metadata" / "mvn_10.json"
        with meta_path.open() as fh:
            first_meta = json.load(fh)
        first_ts = first_meta["timestamp_utc"]

        time.sleep(1)

        _run_cli(["tier-a", "mvn_10", "--n", "1000", "--force"], tmp_path)
        with meta_path.open() as fh:
            second_meta = json.load(fh)
        second_ts = second_meta["timestamp_utc"]

        assert (
            first_ts != second_ts
        ), f"--force expected regeneration but timestamp did not change: {first_ts!r}"

    def test_mvn_output_contains_summary(self, tmp_path: Path) -> None:
        """CLI must print a summary table with expected fields."""
        result = _run_cli(["tier-a", "mvn_10", "--n", "1000"], tmp_path)
        stdout = result.stdout
        assert "mvn_10" in stdout
        assert "analytic" in stdout
        assert "1,000" in stdout or "1000" in stdout

    def test_unknown_model_exits_nonzero(self, tmp_path: Path) -> None:
        """Unknown model name must exit with code 1."""
        env = {**os.environ, "BJX_BENCH_REFERENCE_DIR": str(tmp_path)}
        proc = subprocess.run(
            ["uv", "run", "bjx-bench", "tier-a", "no_such_model"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(__file__).parent.parent),
        )
        assert proc.returncode != 0
        assert "unknown model" in proc.stderr.lower() or "no_such_model" in proc.stderr

    def test_eight_schools_certifies(self, tmp_path: Path) -> None:
        """8-Schools NCP must pass certification and exit 0.

        Uses n=4000, seed=42 (matched to the unit test in test_tier_a_nuts.py).
        The CLI hard-codes n_warmup=500, n_chunks=4 for NUTS models (v1).
        """
        result = _run_cli(
            ["tier-a", "eight_schools_ncp", "--n", "4000", "--seed", "42"],
            tmp_path,
            timeout=120,
        )
        assert (
            "PASSED" in result.stdout
        ), f"Expected certification PASSED in CLI output.\nstdout:\n{result.stdout}"
        # Cache artifacts must exist
        assert (tmp_path / "draws" / "eight_schools_ncp.npz").exists()
        assert (tmp_path / "adaptation" / "eight_schools_ncp.json").exists()
        meta_path = tmp_path / "metadata" / "eight_schools_ncp.json"
        assert meta_path.exists()
        with meta_path.open() as fh:
            meta = json.load(fh)
        assert meta["certification"]["passed"] is True
