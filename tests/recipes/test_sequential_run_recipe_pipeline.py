"""Tests for sequential_run_recipe_pipeline and the exact-wge runner change.

All tests are @fast (pure logic / mocking, no JAX trace).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tuningfork.recipes.sequential_run_recipe_pipeline import _skip_reason, run_pipeline

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _recipe_dict(
    model: str = "mvn_10",
    bm: str = "nuts",
    warmup: str = "window_adaptation_diag_imm",
    verdict: str = "PASS",
    step_size: float | None = 0.3,
    imm: object = None,
    imm_path: str | None = None,
) -> dict:
    bmp: dict = {}
    if step_size is not None:
        bmp["step_size"] = step_size
    if imm is not None:
        bmp["inverse_mass_matrix"] = imm
    return {
        "model_name": model,
        "base_method_name": bm,
        "warmup_name": warmup,
        "warmups": [{"name": warmup, "params": {"n_warmup": 1000, "num_chains": 4}}],
        "base_method_params": bmp,
        "calibration_budget": {"n_warmup": 1000, "n_samples": 1000, "num_chains": 4},
        "gate_evidence": {"auto": {"verdict": verdict}},
        "inverse_mass_matrix_path": imm_path,
        "tuning_seed": 20260517,
        "effort": "LOW",
    }


# ---------------------------------------------------------------------------
# _skip_reason
# ---------------------------------------------------------------------------


def test_skip_failed_filename(tmp_path):
    rp = tmp_path / "failed__nuts__window_adaptation_diag_imm.json"
    assert _skip_reason(rp, _recipe_dict(verdict="PASS")) == "recipe_failed"


def test_skip_fail_verdict(tmp_path):
    rp = tmp_path / "low__nuts__window_adaptation_diag_imm.json"
    assert _skip_reason(rp, _recipe_dict(verdict="FAIL")) == "recipe_failed"


def test_skip_mclmc(tmp_path):
    rp = tmp_path / "low__mclmc__mclmc_tuning.json"
    r = _recipe_dict(bm="mclmc", verdict="PASS")
    assert _skip_reason(rp, r) == "mclmc_family:mclmc"


def test_skip_adjusted_mclmc(tmp_path):
    rp = tmp_path / "low__adjusted_mclmc__adjusted_mclmc_tuning.json"
    r = _recipe_dict(bm="adjusted_mclmc", verdict="PASS")
    assert _skip_reason(rp, r) is not None
    assert "mclmc_family" in _skip_reason(rp, r)


def test_skip_no_warmup(tmp_path):
    rp = tmp_path / "low__ghmc__no_warmup.json"
    r = _recipe_dict(warmup="no_warmup", verdict="PASS")
    assert _skip_reason(rp, r) == "no_warmup"


def test_skip_sidecar_no_path(tmp_path):
    rp = tmp_path / "low__nuts__window_adaptation_low_rank_imm.json"
    r = _recipe_dict(imm="sidecar", imm_path=None, verdict="PASS")
    assert _skip_reason(rp, r) == "sidecar_imm_no_path"


def test_skip_step_size_none(tmp_path):
    rp = tmp_path / "low__hmc__inner_nuts.json"
    r = _recipe_dict(step_size=None, verdict="PASS")
    assert _skip_reason(rp, r) == "step_size_none"


def test_no_skip_valid(tmp_path):
    rp = tmp_path / "low__nuts__window_adaptation_diag_imm.json"
    r = _recipe_dict(verdict="PASS")
    assert _skip_reason(rp, r) is None


def test_no_skip_sidecar_with_path(tmp_path):
    rp = tmp_path / "low__dmhmc__window_adaptation_dense_imm.json"
    r = _recipe_dict(imm="sidecar", imm_path="some/path.npz", verdict="PASS")
    assert _skip_reason(rp, r) is None


# ---------------------------------------------------------------------------
# run_pipeline — smoke behavior (mocked subprocess)
# ---------------------------------------------------------------------------


def _write_recipe(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2) + "\n")


def _mock_run_recipe_pass(*args, **kwargs):
    return "pass", "PASS"


def _mock_run_recipe_fail(*args, **kwargs):
    return "fail_stochastic", "FAIL"


def _mock_run_recipe_timeout(*args, **kwargs):
    return "timeout", ">300s"


def _mock_run_recipe_error(*args, **kwargs):
    return "error", "SomeError: something went wrong"


class _DevNull:
    def write(self, s):
        pass

    def flush(self):
        pass


def test_run_pipeline_skip_failed(tmp_path):
    """failed__ recipe → skipped, no subprocess call."""
    catalog_root = tmp_path / "catalog"
    rp = (
        catalog_root
        / "mvn_10"
        / "recipes"
        / "failed__nuts__window_adaptation_diag_imm.json"
    )
    _write_recipe(rp, _recipe_dict(verdict="FAIL"))

    with patch(
        "tuningfork.recipes.sequential_run_recipe_pipeline._run_recipe_with_timeout",
        side_effect=_mock_run_recipe_pass,
    ) as mock_run:
        report = run_pipeline(
            catalog_root=catalog_root,
            repo_root=tmp_path,
            log_file=_DevNull(),
            smoke_paths=[rp],
        )

    mock_run.assert_not_called()
    assert report.skipped == 1
    assert report.passed == 0


def test_run_pipeline_pass_overwrites(tmp_path):
    """PASS re-run increments pass counter."""
    catalog_root = tmp_path / "catalog"
    rp = (
        catalog_root
        / "mvn_10"
        / "recipes"
        / "low__nuts__window_adaptation_diag_imm.json"
    )
    _write_recipe(rp, _recipe_dict(verdict="PASS"))

    with patch(
        "tuningfork.recipes.sequential_run_recipe_pipeline._run_recipe_with_timeout",
        side_effect=_mock_run_recipe_pass,
    ):
        report = run_pipeline(
            catalog_root=catalog_root,
            repo_root=tmp_path,
            log_file=_DevNull(),
            smoke_paths=[rp],
        )

    assert report.passed == 1
    assert report.skipped == 0


def test_run_pipeline_timeout_skip_continue(tmp_path):
    """Timeout → skip + continue (loop doesn't die)."""
    catalog_root = tmp_path / "catalog"
    rp1 = (
        catalog_root
        / "mvn_10"
        / "recipes"
        / "low__nuts__window_adaptation_diag_imm.json"
    )
    rp2 = (
        catalog_root
        / "mvn_10"
        / "recipes"
        / "low__hmc__window_adaptation_diag_imm.json"
    )
    r = _recipe_dict(verdict="PASS")
    _write_recipe(rp1, r)
    _write_recipe(
        rp2, dict(r, base_method_params={"step_size": 0.4, "num_integration_steps": 64})
    )

    call_count = [0]

    def mock_run_timeout_then_pass(*a, **k):
        call_count[0] += 1
        if call_count[0] == 1:
            return "timeout", ">300s"
        return "pass", "PASS"

    with patch(
        "tuningfork.recipes.sequential_run_recipe_pipeline._run_recipe_with_timeout",
        side_effect=mock_run_timeout_then_pass,
    ):
        report = run_pipeline(
            catalog_root=catalog_root,
            repo_root=tmp_path,
            log_file=_DevNull(),
            smoke_paths=[rp1, rp2],
        )

    # Both recipes processed; first timed out, second passed
    assert report.timeouts == 1
    assert report.passed == 1
    assert report.total == 2


def test_run_pipeline_broad_error_skip_continue(tmp_path):
    """Unhandled error in recipe → skip + continue (loop doesn't die)."""
    catalog_root = tmp_path / "catalog"
    rp1 = (
        catalog_root
        / "mvn_10"
        / "recipes"
        / "low__nuts__window_adaptation_diag_imm.json"
    )
    rp2 = (
        catalog_root
        / "mvn_10"
        / "recipes"
        / "low__hmc__window_adaptation_diag_imm.json"
    )
    r = _recipe_dict(verdict="PASS")
    _write_recipe(rp1, r)
    _write_recipe(
        rp2, dict(r, base_method_params={"step_size": 0.4, "num_integration_steps": 64})
    )

    call_count = [0]

    def mock_run_error_then_pass(*a, **k):
        call_count[0] += 1
        if call_count[0] == 1:
            return "error", "RuntimeError: kaboom"
        return "pass", "PASS"

    with patch(
        "tuningfork.recipes.sequential_run_recipe_pipeline._run_recipe_with_timeout",
        side_effect=mock_run_error_then_pass,
    ):
        report = run_pipeline(
            catalog_root=catalog_root,
            repo_root=tmp_path,
            log_file=_DevNull(),
            smoke_paths=[rp1, rp2],
        )

    assert report.errors == 1
    assert report.passed == 1


def test_run_pipeline_smoke_mode_no_commits(tmp_path):
    """Smoke mode (smoke_paths set) → no git commits even on pass."""
    catalog_root = tmp_path / "catalog"
    rp = (
        catalog_root
        / "mvn_10"
        / "recipes"
        / "low__nuts__window_adaptation_diag_imm.json"
    )
    _write_recipe(rp, _recipe_dict(verdict="PASS"))

    with (
        patch(
            "tuningfork.recipes.sequential_run_recipe_pipeline._run_recipe_with_timeout",
            side_effect=_mock_run_recipe_pass,
        ),
        patch(
            "tuningfork.recipes.sequential_run_recipe_pipeline._git_commit_model"
        ) as mock_commit,
    ):
        run_pipeline(
            catalog_root=catalog_root,
            repo_root=tmp_path,
            log_file=_DevNull(),
            smoke_paths=[rp],
        )

    mock_commit.assert_not_called()


def test_run_pipeline_chunk_commit_on_pass(tmp_path):
    """Full mode (no smoke_paths) → git commit when passes > 0."""
    catalog_root = tmp_path / "catalog"
    rp = (
        catalog_root
        / "mvn_10"
        / "recipes"
        / "low__nuts__window_adaptation_diag_imm.json"
    )
    _write_recipe(rp, _recipe_dict(verdict="PASS"))

    with (
        patch(
            "tuningfork.recipes.sequential_run_recipe_pipeline._run_recipe_with_timeout",
            side_effect=_mock_run_recipe_pass,
        ),
        patch(
            "tuningfork.recipes.sequential_run_recipe_pipeline._git_commit_model",
            return_value="abc123",
        ) as mock_commit,
    ):
        run_pipeline(
            catalog_root=catalog_root,
            repo_root=tmp_path,
            log_file=_DevNull(),
        )

    mock_commit.assert_called_once()


# ---------------------------------------------------------------------------
# Exact-wge: _compute_warmup_grad_evals CUMSUM path
# ---------------------------------------------------------------------------


def test_compute_warmup_grad_evals_cumsum():
    """CUMSUM path: batched_warmup_info with num_integration_steps → exact sum."""
    import numpy as np

    from tuningfork.recipes._recipe_runner import _compute_warmup_grad_evals

    # Simulate (num_chains=4, n_warmup=10) per-step NIS
    nis = np.ones((4, 10), dtype=int) * 7  # NUTS with L=7 each step
    mock_info = MagicMock()
    mock_info.num_integration_steps = nis

    wge = _compute_warmup_grad_evals(
        batched_params={},
        batched_warmup_info=mock_info,
        base_method=None,
        n_warmup=10,
        num_chains=4,
    )
    assert wge == 4 * 10 * 7  # 280


def test_compute_warmup_grad_evals_cumsum_dynamic():
    """Variable-L NUTS: CUMSUM gives exact total, not n×median."""
    import numpy as np

    from tuningfork.recipes._recipe_runner import _compute_warmup_grad_evals

    # Variable NIS: some steps use 1, some use 15 (NUTS doubling)
    rng = np.random.default_rng(0)
    nis = rng.choice([1, 3, 7, 15], size=(4, 100))
    mock_info = MagicMock()
    mock_info.num_integration_steps = nis

    wge = _compute_warmup_grad_evals(
        batched_params={},
        batched_warmup_info=mock_info,
        base_method=None,
        n_warmup=100,
        num_chains=4,
    )
    assert wge == int(np.sum(nis))  # exact, not 4×100×median


def test_compute_warmup_grad_evals_mclmc():
    """mclmc path: _total_tuning_steps in batched_params → exact."""
    from tuningfork.recipes._recipe_runner import _compute_warmup_grad_evals

    wge = _compute_warmup_grad_evals(
        batched_params={"_total_tuning_steps": 3333},
        batched_warmup_info=None,
        base_method=None,
        n_warmup=10000,
        num_chains=4,
    )
    assert wge == 3333


def test_compute_warmup_grad_evals_null():
    """No info available → None."""
    from tuningfork.recipes._recipe_runner import _compute_warmup_grad_evals

    wge = _compute_warmup_grad_evals(
        batched_params={},
        batched_warmup_info=None,
        base_method=None,
        n_warmup=1000,
        num_chains=4,
    )
    assert wge is None
