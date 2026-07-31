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
"""End-to-end tests for the ``tuningfork leaderboard`` CLI."""

import json
import os
import subprocess
from pathlib import Path

import pytest


def _run_leaderboard(
    args: list[str],
    tmp_path: Path,
    timeout: int = 60,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run ``uv run tuningfork leaderboard <args>``."""
    env = {**os.environ, "TUNINGFORK_REFERENCE_DIR": str(tmp_path)}
    return subprocess.run(
        ["uv", "run", "tuningfork", "leaderboard", *args],
        env=env,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
        cwd=str(Path(__file__).parent.parent),
    )


@pytest.mark.e2e
class TestLeaderboardCLI:
    """Integration tests for ``tuningfork leaderboard``."""

    def test_leaderboard_mvn_10_markdown(self, tmp_path: Path) -> None:
        """tuningfork leaderboard mvn_10 must exit 0 and print markdown table."""
        result = _run_leaderboard(["mvn_10"], tmp_path)
        assert (
            result.returncode == 0
        ), f"Expected exit 0.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        stdout = result.stdout
        # Table header and title must be present
        assert (
            "Leaderboard for mvn_10" in stdout
        ), f"'Leaderboard for mvn_10' missing from stdout:\n{stdout}"
        assert (
            "| effort |" in stdout
        ), f"Markdown table header missing from stdout:\n{stdout}"
        assert "|" in stdout, f"Table pipes missing from stdout:\n{stdout}"
        # At least a few rows expected
        lines = stdout.strip().split("\n")
        table_lines = [line for line in lines if line.startswith("|")]
        # header + separator + at least 2 data rows
        assert (
            len(table_lines) >= 4
        ), f"Expected at least 2 data rows in markdown table, got {len(table_lines)} table lines:\n{stdout}"

    def test_leaderboard_mvn_10_effort_medium(self, tmp_path: Path) -> None:
        """tuningfork leaderboard mvn_10 --effort medium filters to MEDIUM rows."""
        result = _run_leaderboard(["mvn_10", "--effort", "medium"], tmp_path)
        assert (
            result.returncode == 0
        ), f"Expected exit 0.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        stdout = result.stdout
        assert "Leaderboard for mvn_10" in stdout
        # Count MEDIUM-effort rows (ignoring header/separator)
        lines = stdout.strip().split("\n")
        data_lines = [
            line
            for line in lines
            if line.startswith("|") and "effort" not in line and "-----" not in line
        ]
        assert (
            len(data_lines) >= 1
        ), f"Expected at least 1 MEDIUM-effort row, got {len(data_lines)} rows:\n{stdout}"
        # All data rows should have "medium" in the effort column.
        for line in data_lines:
            parts = line.split("|")
            effort_col = parts[1].strip()
            assert (
                effort_col == "medium"
            ), f"Expected 'medium' in effort column, got '{effort_col}' in line:\n{line}"

    def test_leaderboard_mvn_10_json_format(self, tmp_path: Path) -> None:
        """tuningfork leaderboard mvn_10 --format json must output JSON list."""
        result = _run_leaderboard(["mvn_10", "--format", "json"], tmp_path)
        assert (
            result.returncode == 0
        ), f"Expected exit 0.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        # Parse JSON output
        data = json.loads(result.stdout)
        assert isinstance(data, list), f"Expected JSON list, got {type(data)}"
        assert (
            len(data) >= 1
        ), f"Expected at least 1 element in JSON list, got {len(data)}"
        # Verify schema
        for item in data:
            assert "effort" in item
            assert "model_name" in item
            assert "base_method_name" in item
            assert "warmup_name" in item
            assert "headline_metric" in item

    def test_leaderboard_bad_model_exits_2(self, tmp_path: Path) -> None:
        """tuningfork leaderboard does_not_exist must exit 2 and mention model in stderr."""
        result = _run_leaderboard(["does_not_exist"], tmp_path)
        assert (
            result.returncode == 2
        ), f"Expected exit 2 for bad model.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        # Stderr should mention model or list valid models
        stderr_lower = result.stderr.lower()
        assert (
            "does_not_exist" in result.stderr or "model" in stderr_lower
        ), f"Expected bad-model name or 'model' in stderr:\n{result.stderr}"
