"""Tests for sequential_run_recipe_pipeline (model A: targeted-patch).

All tests are @fast (pure logic / mocking, no JAX trace).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tuningfork.recipes.sequential_run_recipe_pipeline import (
    _skip_reason,
    _wge_from_config,
    run_pipeline,
)

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
    nis: int | None = None,
    imm: object = None,
    imm_path: str | None = None,
    wge: int | None = None,
    wge_is_est: bool | None = None,
    q05: float | None = None,
) -> dict:
    bmp: dict = {}
    if step_size is not None:
        bmp["step_size"] = step_size
    if nis is not None:
        bmp["num_integration_steps"] = nis
    if imm is not None:
        bmp["inverse_mass_matrix"] = imm
    budget: dict = {"n_warmup": 1000, "n_samples": 1000, "num_chains": 4}
    if wge is not None:
        budget["warmup_grad_evals"] = wge
    if wge_is_est is not None:
        budget["warmup_grad_evals_is_estimate"] = wge_is_est
    return {
        "model_name": model,
        "base_method_name": bm,
        "warmup_name": warmup,
        "warmups": [{"name": warmup, "params": {"n_warmup": 1000, "num_chains": 4}}],
        "base_method_params": bmp,
        "calibration_budget": budget,
        "sample_quality": {
            "mae_vs_reference": 0.05,
            "q05_error": q05,
            "q95_error": None,
            "std_ratio_max_dev": None,
        },
        "gate_evidence": {"auto": {"verdict": verdict}},
        "headline_metric": 0.00352,
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


def test_skip_no_warmup(tmp_path):
    rp = tmp_path / "low__ghmc__no_warmup.json"
    assert _skip_reason(rp, _recipe_dict(warmup="no_warmup")) == "no_warmup"


def test_skip_step_size_none(tmp_path):
    rp = tmp_path / "low__hmc__inner_nuts.json"
    assert _skip_reason(rp, _recipe_dict(step_size=None)) == "step_size_none"


def test_no_skip_valid(tmp_path):
    rp = tmp_path / "low__nuts__window_adaptation_diag_imm.json"
    assert _skip_reason(rp, _recipe_dict(verdict="PASS")) is None


# ---------------------------------------------------------------------------
# _wge_from_config
# ---------------------------------------------------------------------------


def test_wge_from_config_hmc():
    r = _recipe_dict(bm="hmc", nis=64)
    wge, is_est = _wge_from_config("hmc", r)
    assert wge == 1000 * 4 * 64
    assert is_est is False


def test_wge_from_config_laplace_hmc_is_estimate():
    r = _recipe_dict(bm="laplace_hmc", nis=10)
    wge, is_est = _wge_from_config("laplace_hmc", r)
    assert wge == 1000 * 4 * 10
    assert is_est is True


def test_wge_from_config_nuts_returns_none():
    r = _recipe_dict(bm="nuts")
    wge, is_est = _wge_from_config("nuts", r)
    assert wge is None


def test_wge_from_config_hmc_no_nis():
    r = _recipe_dict(bm="hmc")  # no nis
    wge, is_est = _wge_from_config("hmc", r)
    assert wge is None


# ---------------------------------------------------------------------------
# run_pipeline — never touches headline / verdict
# ---------------------------------------------------------------------------


def _write_recipe(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2) + "\n")


class _DevNull:
    def write(self, s):
        pass

    def flush(self):
        pass


def test_headline_and_verdict_byte_identical(tmp_path):
    """headline_metric and verdict MUST be byte-identical after patch."""
    catalog_root = tmp_path / "catalog"
    rp = (
        catalog_root
        / "mvn_10"
        / "recipes"
        / "low__hmc__window_adaptation_diag_imm.json"
    )
    r = _recipe_dict(bm="hmc", nis=64, verdict="REVIEW", q05=None)
    r["headline_metric"] = 0.00352
    _write_recipe(rp, r)

    run_pipeline(
        catalog_root=catalog_root,
        repo_root=tmp_path,
        log_file=_DevNull(),
        smoke_paths=[rp],
    )

    updated = json.loads(rp.read_text())
    assert updated["headline_metric"] == 0.00352  # byte-identical
    assert updated["gate_evidence"]["auto"]["verdict"] == "REVIEW"  # byte-identical


def test_wge_patched_from_config(tmp_path):
    """Fixed-HMC gets exact wge from config."""
    catalog_root = tmp_path / "catalog"
    rp = (
        catalog_root
        / "mvn_10"
        / "recipes"
        / "low__hmc__window_adaptation_diag_imm.json"
    )
    _write_recipe(rp, _recipe_dict(bm="hmc", nis=64))

    run_pipeline(
        catalog_root=catalog_root,
        repo_root=tmp_path,
        log_file=_DevNull(),
        smoke_paths=[rp],
    )

    updated = json.loads(rp.read_text())
    assert updated["calibration_budget"]["warmup_grad_evals"] == 1000 * 4 * 64
    assert "warmup_grad_evals_is_estimate" not in updated["calibration_budget"]


def test_wge_already_exact_skip(tmp_path):
    """Already-exact wge → no change."""
    catalog_root = tmp_path / "catalog"
    rp = (
        catalog_root
        / "mvn_10"
        / "recipes"
        / "low__hmc__window_adaptation_diag_imm.json"
    )
    r = _recipe_dict(bm="hmc", nis=64, wge=256000)
    _write_recipe(rp, r)

    run_pipeline(
        catalog_root=catalog_root,
        repo_root=tmp_path,
        log_file=_DevNull(),
        smoke_paths=[rp],
    )

    updated = json.loads(rp.read_text())
    assert updated["calibration_budget"]["warmup_grad_evals"] == 256000


def test_sq_from_cache(tmp_path):
    """sq is loaded from cached draws and recomputed."""
    catalog_root = tmp_path / "catalog"
    model_dir = catalog_root / "mvn_10"
    rp = model_dir / "recipes" / "low__nuts__window_adaptation_diag_imm.json"
    _write_recipe(rp, _recipe_dict())

    # Write fake cached draws and GT summary
    cache_dir = model_dir / "_cache"
    cache_dir.mkdir(parents=True)
    np.savez(
        str(cache_dir / "low__nuts__window_adaptation_diag_imm.draws.npz"),
        x=np.random.default_rng(0).standard_normal((4, 1000)),
    )
    ref_dir = model_dir / "reference"
    ref_dir.mkdir(parents=True)
    (ref_dir / "summary.json").write_text(
        json.dumps(
            {
                "mean": {"x": 0.0},
                "std": {"x": 1.0},
                "q05": {"x": -1.64},
                "q95": {"x": 1.64},
            }
        )
    )

    run_pipeline(
        catalog_root=catalog_root,
        repo_root=tmp_path,
        log_file=_DevNull(),
        smoke_paths=[rp],
    )

    updated = json.loads(rp.read_text())
    sq = updated["sample_quality"]
    assert sq["q05_error"] is not None
    assert sq["q95_error"] is not None
    assert sq["std_ratio_max_dev"] is not None
    assert sq["mae_vs_reference"] == 0.05  # unchanged


def test_sq_null_when_no_cache(tmp_path):
    """When no cache file → sq fields stay null."""
    catalog_root = tmp_path / "catalog"
    rp = (
        catalog_root
        / "mvn_10"
        / "recipes"
        / "low__nuts__window_adaptation_diag_imm.json"
    )
    _write_recipe(rp, _recipe_dict())

    run_pipeline(
        catalog_root=catalog_root,
        repo_root=tmp_path,
        log_file=_DevNull(),
        smoke_paths=[rp],
    )

    updated = json.loads(rp.read_text())
    sq = updated["sample_quality"]
    assert sq["q05_error"] is None  # no cache → leave null


def test_smoke_mode_no_commits(tmp_path):
    """Smoke mode never commits."""
    catalog_root = tmp_path / "catalog"
    rp = (
        catalog_root
        / "mvn_10"
        / "recipes"
        / "low__hmc__window_adaptation_diag_imm.json"
    )
    _write_recipe(rp, _recipe_dict(bm="hmc", nis=64))

    with patch(
        "tuningfork.recipes.sequential_run_recipe_pipeline._git_commit_model"
    ) as mock_commit:
        run_pipeline(
            catalog_root=catalog_root,
            repo_root=tmp_path,
            log_file=_DevNull(),
            smoke_paths=[rp],
        )

    mock_commit.assert_not_called()


def test_large_diff_recorded(tmp_path):
    """Large wge change (>2×) is recorded in large_diffs."""
    catalog_root = tmp_path / "catalog"
    rp = (
        catalog_root
        / "mvn_10"
        / "recipes"
        / "low__hmc__window_adaptation_diag_imm.json"
    )
    # Old wge was 28000 (wrong estimate), new = 1000×4×64=256000 (>2×)
    r = _recipe_dict(bm="hmc", nis=64, wge=28000, wge_is_est=True)
    _write_recipe(rp, r)

    report = run_pipeline(
        catalog_root=catalog_root,
        repo_root=tmp_path,
        log_file=_DevNull(),
        smoke_paths=[rp],
    )

    assert len(report.large_diffs) >= 1
    wge_diffs = [d for d in report.large_diffs if d.field == "wge"]
    assert len(wge_diffs) == 1
    assert wge_diffs[0].new_val == 256000
    assert wge_diffs[0].old_val == 28000


# ---------------------------------------------------------------------------
# _compute_warmup_grad_evals CUMSUM path (unchanged)
# ---------------------------------------------------------------------------


def test_compute_warmup_grad_evals_cumsum():
    from unittest.mock import MagicMock

    from tuningfork.recipes._recipe_runner import _compute_warmup_grad_evals

    nis = np.ones((4, 10), dtype=int) * 7
    mock_info = MagicMock()
    mock_info.num_integration_steps = nis

    wge = _compute_warmup_grad_evals(
        batched_params={},
        batched_warmup_info=mock_info,
        base_method=None,
        n_warmup=10,
        num_chains=4,
    )
    assert wge == 4 * 10 * 7


def test_compute_warmup_grad_evals_mclmc():
    from tuningfork.recipes._recipe_runner import _compute_warmup_grad_evals

    wge = _compute_warmup_grad_evals(
        batched_params={"_total_tuning_steps": 3333},
        batched_warmup_info=None,
        base_method=None,
        n_warmup=10000,
        num_chains=4,
    )
    assert wge == 3333 * 2 * 4


# ---------------------------------------------------------------------------
# Regression: laplace_dhmc typo (P0.T0.1)
# ---------------------------------------------------------------------------


def test_laplace_dhmc_is_dynamic_not_null():
    """Regression test: laplace_dhmc must be classified as _DYNAMIC.

    Bug: typo "laplace_dhdc" in _DYNAMIC frozenset caused laplace_dhmc recipes
    to fall through to null wge path instead of warmup-rerun dynamic path.
    This test ensures laplace_dhmc is correctly routed to the dynamic path
    which runs warmup subprocess to get exact warmup_grad_evals from CUMSUM NIS.
    """
    from tuningfork.recipes.sequential_run_recipe_pipeline import _DYNAMIC

    assert "laplace_dhmc" in _DYNAMIC, (
        "laplace_dhmc must be in _DYNAMIC frozenset to trigger warmup-rerun "
        "gradient-evaluation accounting (CUMSUM num_integration_steps). "
        "The typo 'laplace_dhdc' prevented this routing."
    )
    assert (
        "laplace_dhdc" not in _DYNAMIC
    ), "laplace_dhdc is a typo and must not appear in _DYNAMIC"
