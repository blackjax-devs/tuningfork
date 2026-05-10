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
"""Phase 3 end-to-end tests — `bjx-bench warmup` CLI subprocess integration.

Spawns ``bjx-bench warmup <model> <algo> ...`` as a subprocess with an isolated
``BJX_BENCH_REFERENCE_DIR`` pointing at a ``tmp_path`` so tests do not pollute
the committed reference cache or each other.

Four tests:
1. Success smoke: NUTS on MVN-10, n_warmup=200, exit 0, summary on stdout.
2. Save flag: NUTS on MVN-10 with ``--save PATH`` produces valid JSON with
   expected keys (effort, warmup_name, base_method_params with step_size).
3. Bad model: unknown model name → exit 2, stderr lists valid models.
4. Bad warmup-algo compatibility: NUTS with mclmc_tuning → exit 2, stderr
   mentions incompatibility.

Runtime target: <60 s total (n_warmup=200 for fast smoke runs). With JAX
import overhead (~3–5 s per subprocess) the total budget is comfortably
under 1 min on CPU.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest


def _run_warmup(
    args: list[str],
    tmp_path: Path,
    timeout: int = 120,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run ``uv run bjx-bench warmup <args>`` with an isolated reference dir."""
    env = {**os.environ, "BJX_BENCH_REFERENCE_DIR": str(tmp_path)}
    return subprocess.run(
        ["uv", "run", "bjx-bench", "warmup", *args],
        env=env,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
        cwd=str(Path(__file__).parent.parent),
    )


@pytest.mark.e2e
class TestWarmupCLI:
    """Integration tests for ``bjx-bench warmup``."""

    def test_nuts_mvn_smoke(self, tmp_path: Path) -> None:
        """NUTS on MVN-10 with n_warmup=200 must exit 0 and print summary keys."""
        result = _run_warmup(
            [
                "mvn_10",
                "nuts",
                "--n-warmup",
                "200",
                "--seed",
                "0",
            ],
            tmp_path,
        )
        assert (
            result.returncode == 0
        ), f"Expected exit 0.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        stdout = result.stdout
        # Summary table keys must appear
        assert "warmup" in stdout, f"'warmup' missing from stdout:\n{stdout}"
        assert "step_size" in stdout, f"'step_size' missing from stdout:\n{stdout}"
        assert (
            "wall_seconds" in stdout
        ), f"'wall_seconds' missing from stdout:\n{stdout}"
        # Model and algo must appear in the banner
        assert "mvn_10" in stdout, f"'mvn_10' missing from banner:\n{stdout}"
        assert "nuts" in stdout, f"'nuts' missing from banner:\n{stdout}"

    def test_save_produces_valid_json(self, tmp_path: Path) -> None:
        """``--save PATH`` must write a JSON file with the correct schema."""
        save_path = tmp_path / "recipe.json"
        result = _run_warmup(
            [
                "mvn_10",
                "nuts",
                "--n-warmup",
                "200",
                "--seed",
                "0",
                "--save",
                str(save_path),
            ],
            tmp_path,
        )
        assert (
            result.returncode == 0
        ), f"Expected exit 0.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert (
            save_path.exists()
        ), f"Expected JSON file at {save_path} but it was not created."
        with save_path.open() as fh:
            data = json.load(fh)

        # Top-level schema keys required by the Recipe spec
        required_keys = {
            "effort",
            "warmup_name",
            "base_method_params",
            "model_name",
            "base_method_name",
        }
        missing = required_keys - set(data.keys())
        assert (
            not missing
        ), f"Missing top-level keys in saved JSON: {missing}\nKeys present: {set(data.keys())}"

        # Verify effort is "medium"
        assert (
            data["effort"] == "medium"
        ), f"Expected effort='medium', got {data['effort']!r}"

        # Verify warmup_name is stan_window (default)
        assert (
            data["warmup_name"] == "stan_window"
        ), f"Expected warmup_name='stan_window', got {data['warmup_name']!r}"

        # Verify base_method_params contains step_size and inverse_mass_matrix
        bmp = data["base_method_params"]
        assert (
            "step_size" in bmp
        ), f"'step_size' missing from base_method_params. Keys: {set(bmp.keys())}"
        assert (
            "inverse_mass_matrix" in bmp
        ), f"'inverse_mass_matrix' missing from base_method_params. Keys: {set(bmp.keys())}"

    def test_bad_model_exits_2(self, tmp_path: Path) -> None:
        """Unknown model name must exit 2 and mention valid model names in stderr."""
        result = _run_warmup(
            ["does_not_exist", "nuts"],
            tmp_path,
        )
        assert result.returncode == 2, (
            f"Expected exit 2 for bad model.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        # Stderr should mention the bad name or list valid models
        stderr_lower = result.stderr.lower()
        assert (
            "does_not_exist" in result.stderr or "model" in stderr_lower
        ), f"Expected bad-model name or 'model' in stderr:\n{result.stderr}"
        # At least one valid model name should appear
        assert any(
            name in result.stderr
            for name in ("mvn_10", "neals_funnel", "eight_schools")
        ), f"Expected at least one valid model name in stderr:\n{result.stderr}"

    def test_bad_warmup_compatibility_exits_2(self, tmp_path: Path) -> None:
        """NUTS with mclmc_tuning warmup must exit 2 (incompatible)."""
        result = _run_warmup(
            [
                "mvn_10",
                "nuts",
                "--warmup",
                "mclmc_tuning",
                "--seed",
                "0",
            ],
            tmp_path,
        )
        assert result.returncode == 2, (
            f"Expected exit 2 for incompatible warmup.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        # Stderr should mention incompatibility or compatibility
        stderr_lower = result.stderr.lower()
        assert (
            "compatible" in stderr_lower
        ), f"Expected 'compatible' in stderr:\n{result.stderr}"


def _run_leaderboard(
    args: list[str],
    tmp_path: Path,
    timeout: int = 60,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run ``uv run bjx-bench leaderboard <args>``."""
    env = {**os.environ, "BJX_BENCH_REFERENCE_DIR": str(tmp_path)}
    return subprocess.run(
        ["uv", "run", "bjx-bench", "leaderboard", *args],
        env=env,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
        cwd=str(Path(__file__).parent.parent),
    )


@pytest.mark.e2e
class TestLeaderboardCLI:
    """Integration tests for ``bjx-bench leaderboard``."""

    def test_leaderboard_mvn_10_markdown(self, tmp_path: Path) -> None:
        """bjx-bench leaderboard mvn_10 must exit 0 and print markdown table."""
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

    def test_leaderboard_mvn_10_effort_high(self, tmp_path: Path) -> None:
        """bjx-bench leaderboard mvn_10 --effort high must filter to HIGH-only rows."""
        result = _run_leaderboard(["mvn_10", "--effort", "high"], tmp_path)
        assert (
            result.returncode == 0
        ), f"Expected exit 0.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        stdout = result.stdout
        assert "Leaderboard for mvn_10" in stdout
        # Count HIGH-effort rows (ignoring header/separator)
        lines = stdout.strip().split("\n")
        data_lines = [
            line
            for line in lines
            if line.startswith("|") and "effort" not in line and "-----" not in line
        ]
        # Should have exactly 2 HIGH rows: NUTS and HMC
        assert (
            len(data_lines) >= 1
        ), f"Expected at least 1 HIGH-effort row, got {len(data_lines)} rows:\n{stdout}"
        # All data rows should have "high" in effort column
        for line in data_lines:
            parts = line.split("|")
            effort_col = parts[1].strip()
            assert (
                effort_col == "high"
            ), f"Expected 'high' in effort column, got '{effort_col}' in line:\n{line}"

    def test_leaderboard_mvn_10_json_format(self, tmp_path: Path) -> None:
        """bjx-bench leaderboard mvn_10 --format json must output JSON list."""
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
        """bjx-bench leaderboard does_not_exist must exit 2 and mention model in stderr."""
        result = _run_leaderboard(["does_not_exist"], tmp_path)
        assert (
            result.returncode == 2
        ), f"Expected exit 2 for bad model.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        # Stderr should mention model or list valid models
        stderr_lower = result.stderr.lower()
        assert (
            "does_not_exist" in result.stderr or "model" in stderr_lower
        ), f"Expected bad-model name or 'model' in stderr:\n{result.stderr}"
