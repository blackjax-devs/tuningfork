"""Phase 2 end-to-end tests — `bjx-bench tune` CLI subprocess integration.

Spawns ``bjx-bench tune <model> <algo> ...`` as a subprocess with an isolated
``BJX_BENCH_REFERENCE_DIR`` pointing at a ``tmp_path`` so tests do not pollute
the committed reference cache or each other.

Five tests:
1. Success smoke: NUTS on MVN-10, n_trials=3, exit 0, summary table on stdout.
2. Sampler flag: HMC on MVN-10 with ``--sampler random``, exit 0.
3. Save flag: NUTS on MVN-10 with ``--save PATH`` produces valid JSON schema.
4. Bad model: unknown model name → exit 2, stderr lists valid models.
5. Bad algo: unknown algo name → exit 2, stderr lists valid algorithms.

Runtime target: <120 s total (n_trials=3, n_samples=100, n_warmup=100 for
fast smoke runs). With JAX + Optuna import overhead (~5–10 s per subprocess)
and 3 MCMC trials each the total budget is comfortably under 2 min on CPU.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _run_tune(
    args: list[str],
    tmp_path: Path,
    timeout: int = 180,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run ``uv run bjx-bench tune <args>`` with an isolated reference dir."""
    env = {**os.environ, "BJX_BENCH_REFERENCE_DIR": str(tmp_path)}
    return subprocess.run(
        ["uv", "run", "bjx-bench", "tune", *args],
        env=env,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
        cwd=str(Path(__file__).parent.parent),
    )


class TestTuneCLI:
    """Integration tests for ``bjx-bench tune``."""

    def test_nuts_mvn_smoke(self, tmp_path: Path) -> None:
        """NUTS on MVN-10 with n_trials=3 must exit 0 and print summary keys."""
        result = _run_tune(
            [
                "mvn_10",
                "nuts",
                "--n-trials",
                "3",
                "--n-warmup",
                "100",
                "--n-samples",
                "100",
            ],
            tmp_path,
        )
        assert (
            result.returncode == 0
        ), f"Expected exit 0.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        stdout = result.stdout
        # Summary table keys must appear
        assert "best_score" in stdout, f"'best_score' missing from stdout:\n{stdout}"
        assert (
            "default_score" in stdout
        ), f"'default_score' missing from stdout:\n{stdout}"
        assert (
            "default_works" in stdout
        ), f"'default_works' missing from stdout:\n{stdout}"
        assert (
            "n_trials_to_threshold" in stdout
        ), f"'n_trials_to_threshold' missing from stdout:\n{stdout}"
        assert (
            "n_trials_to_best" in stdout
        ), f"'n_trials_to_best' missing from stdout:\n{stdout}"
        assert (
            "wall_seconds_total" in stdout
        ), f"'wall_seconds_total' missing from stdout:\n{stdout}"
        assert "best_params" in stdout, f"'best_params' missing from stdout:\n{stdout}"
        # Model and algo must appear in the banner
        assert "mvn_10" in stdout, f"'mvn_10' missing from banner:\n{stdout}"
        assert "nuts" in stdout, f"'nuts' missing from banner:\n{stdout}"

    def test_hmc_random_sampler(self, tmp_path: Path) -> None:
        """HMC on MVN-10 with ``--sampler random`` must exit 0."""
        result = _run_tune(
            [
                "mvn_10",
                "hmc",
                "--n-trials",
                "3",
                "--n-warmup",
                "100",
                "--n-samples",
                "100",
                "--sampler",
                "random",
            ],
            tmp_path,
        )
        assert result.returncode == 0, (
            f"Expected exit 0 with --sampler random.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        # Sampler name should appear in the banner line
        assert (
            "random" in result.stdout
        ), f"Expected 'random' in stdout banner:\n{result.stdout}"

    def test_save_produces_valid_json(self, tmp_path: Path) -> None:
        """``--save PATH`` must write a JSON file with the correct schema."""
        save_path = tmp_path / "recipe.json"
        result = _run_tune(
            [
                "mvn_10",
                "nuts",
                "--n-trials",
                "3",
                "--n-warmup",
                "100",
                "--n-samples",
                "100",
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

        # Top-level schema keys required by the spec
        required_keys = {
            "algorithm_name",
            "posterior_name",
            "best_params",
            "best_score",
            "n_trials_completed",
            "n_seeds",
            "history",
            "difficulty",
        }
        missing = required_keys - set(data.keys())
        assert (
            not missing
        ), f"Missing top-level keys in saved JSON: {missing}\nKeys present: {set(data.keys())}"

        # Basic type checks
        assert isinstance(data["algorithm_name"], str)
        assert isinstance(data["posterior_name"], str)
        assert isinstance(data["best_params"], dict)
        assert isinstance(data["best_score"], (int, float))
        assert isinstance(data["n_trials_completed"], int)
        assert isinstance(data["history"], list)
        assert isinstance(data["difficulty"], dict)

        # Difficulty sub-keys
        diff = data["difficulty"]
        for key in (
            "default_score",
            "best_score",
            "default_works",
            "n_trials_to_threshold",
            "n_trials_to_best",
        ):
            assert (
                key in diff
            ), f"difficulty.{key!r} missing from saved JSON difficulty dict."

    def test_bad_model_exits_2(self, tmp_path: Path) -> None:
        """Unknown model name must exit 2 and mention valid model names in stderr."""
        result = _run_tune(
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
            "does_not_exist" in result.stderr or "known" in stderr_lower
        ), f"Expected bad-model name or 'known' in stderr:\n{result.stderr}"
        # At least one valid model name should appear to be helpful
        assert any(
            name in result.stderr
            for name in ("mvn_10", "neals_funnel", "eight_schools")
        ), f"Expected at least one valid model name in stderr:\n{result.stderr}"

    def test_bad_algo_exits_2(self, tmp_path: Path) -> None:
        """Unknown algorithm name must exit 2 and mention valid algo names in stderr."""
        result = _run_tune(
            ["mvn_10", "fake_algo"],
            tmp_path,
        )
        assert result.returncode == 2, (
            f"Expected exit 2 for bad algo.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        stderr_lower = result.stderr.lower()
        assert (
            "fake_algo" in result.stderr or "known" in stderr_lower
        ), f"Expected bad-algo name or 'known' in stderr:\n{result.stderr}"
        # At least one valid algo name should appear
        assert any(
            name in result.stderr
            for name in ("nuts", "hmc", "mala", "barker", "rwm", "mclmc")
        ), f"Expected at least one valid algo name in stderr:\n{result.stderr}"
