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
"""Tests for tuningfork.calibration.revalidation.

Coverage:
- compute_stage1_verdict: PASS / REVIEW / FAIL paths including the regression
  case where min-bulk-ESS ≈ 4.3 → stage-1 FAIL and W1 is never consulted.
- compute_stage1_verdict: thresholds imported from DEFAULT_THRESHOLDS (no
  hardcoded constants; drift is not possible).
- classify_recipe_path: path-A, path-B, path-C, and SK classification.
- _cell_regen_seed: determinism across calls.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tuningfork.calibration._gate.constants import DEFAULT_THRESHOLDS
from tuningfork.calibration.revalidation import (
    _cell_regen_seed,
    classify_recipe_path,
    compute_stage1_verdict,
)

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# compute_stage1_verdict
# ---------------------------------------------------------------------------


class TestComputeStage1Verdict:
    """Unit tests for compute_stage1_verdict."""

    def test_pass_clean_iid_draws(self):
        """Clean IID draws with zero divergences → PASS."""
        rng = np.random.default_rng(42)
        # 4 chains × 500 draws × 1 dim — well-mixed, ESS ≈ 2000 >> 400 threshold
        draws = {"x": rng.standard_normal((4, 500, 1))}
        result = compute_stage1_verdict(draws, n_divergences=0)

        assert result["stage1_verdict"] == "PASS"
        assert result["rhat_max"] is not None
        assert result["rhat_max"] < 1.01  # within PASS band
        assert result["min_bulk_ess"] is not None
        assert result["min_bulk_ess"] >= 400.0  # within PASS band
        assert result["n_divergences"] == 0

    def test_fail_from_high_divergences(self):
        """n_divergences ≥ NDIV_FAIL threshold → FAIL even with clean chains."""
        rng = np.random.default_rng(0)
        draws = {"x": rng.standard_normal((4, 500, 1))}

        # NDIV_FAIL boundary from DEFAULT_THRESHOLDS (review hi = fail lo)
        ndiv_fail = int(DEFAULT_THRESHOLDS["n_divergences"]["review"][1])
        result = compute_stage1_verdict(draws, n_divergences=ndiv_fail)

        assert result["stage1_verdict"] == "FAIL"
        assert result["n_divergences"] == ndiv_fail

    def test_review_from_moderate_divergences(self):
        """n_divergences in the REVIEW band → REVIEW."""
        rng = np.random.default_rng(0)
        draws = {"x": rng.standard_normal((4, 500, 1))}

        ndiv_review_lo = int(DEFAULT_THRESHOLDS["n_divergences"]["review"][0])
        result = compute_stage1_verdict(draws, n_divergences=ndiv_review_lo)

        assert result["stage1_verdict"] == "REVIEW"

    def test_regression_degenerate_ess_below_100(self):
        """Regression: degenerate re-gen with min-bulk-ESS ≈ 4.3 → FAIL.

        This is the key confound-fix regression: a collapsed re-gen (e.g.
        fullrank_vi converging to a different mode, producing near-iid but
        massively misspecified draws) must be caught by stage-1 and NEVER
        reach W1, where it would produce a spurious FAIL verdict and a false
        PASS→FAIL flip.

        We use constant-value chains (zero within-chain variance) to reproduce
        the degenerate case.  ``ess_bulk`` on zero-variance chains returns NaN.
        ``_classify_metric(NaN, ...)`` returns "FAIL" because NaN satisfies no
        ``lo ≤ x < hi`` band condition — stricter than the old hardcoded
        ``NaN < 100`` comparison which was False (not FAIL!).

        Note: blackjax.diagnostics.rhat internally splits chains in half
        (Vehtari 2021 split-chain), so each chain needs at least 4 draws for
        the split halves to satisfy the 2-sample minimum.  We use 10 draws.
        """
        constant_draws = {"x": np.zeros((2, 10, 1))}  # 2 chains × 10 draws
        result = compute_stage1_verdict(constant_draws, n_divergences=0)

        assert result["stage1_verdict"] == "FAIL", (
            "Degenerate draws (NaN ESS from zero-variance chains) must trigger "
            "stage-1 FAIL so W1 is never consulted."
        )
        # w1_verdict is not returned by compute_stage1_verdict — the caller
        # (process_catalog_cell) gates on stage1_verdict != "PASS" and sets
        # w1_verdict=None.  This test verifies the gating predicate fires.

    def test_regression_tiny_ess_triggers_fail(self):
        """Tiny sample budget (2×5) guarantees ESS < ESS_FAIL=100 → FAIL."""
        rng = np.random.default_rng(7)
        # 2 chains × 5 draws — max possible ESS ≈ 10, well below ESS_FAIL=100
        draws = {"x": rng.standard_normal((2, 5, 1))}
        result = compute_stage1_verdict(draws, n_divergences=0)

        ess_fail = DEFAULT_THRESHOLDS["min_bulk_ess"]["review"][0]
        # ESS must be below the FAIL threshold (< review lo = ESS_FAIL)
        if result["min_bulk_ess"] is not None:
            assert result["min_bulk_ess"] < ess_fail, (
                f"min_bulk_ess={result['min_bulk_ess']} should be < {ess_fail} "
                "for 2×5 draws"
            )
        assert result["stage1_verdict"] == "FAIL"

    def test_none_n_divergences_skipped(self):
        """n_divergences=None (MCLMC / info-less paths) → divergence band skipped."""
        rng = np.random.default_rng(99)
        draws = {"x": rng.standard_normal((4, 500, 1))}
        result = compute_stage1_verdict(draws, n_divergences=None)

        # Result is well-formed; divergence branch did not contribute
        assert result["n_divergences"] is None
        assert result["stage1_verdict"] in ("PASS", "REVIEW", "FAIL")

    def test_thresholds_imported_not_hardcoded(self):
        """Stage-1 FAIL boundary comes from DEFAULT_THRESHOLDS, not local constants.

        Verifies that the boundary value used by compute_stage1_verdict matches
        DEFAULT_THRESHOLDS exactly.  If the gate thresholds are updated,
        compute_stage1_verdict automatically inherits the change.
        """
        rng = np.random.default_rng(0)
        draws = {"x": rng.standard_normal((4, 500, 1))}

        # n_divergences exactly at the FAIL boundary (= review hi)
        ndiv_fail = int(DEFAULT_THRESHOLDS["n_divergences"]["review"][1])
        result_fail = compute_stage1_verdict(draws, n_divergences=ndiv_fail)

        # One below the FAIL boundary (= review hi - 1) must be REVIEW or PASS
        result_not_fail = compute_stage1_verdict(draws, n_divergences=ndiv_fail - 1)

        assert (
            result_fail["stage1_verdict"] == "FAIL"
        ), f"n_div={ndiv_fail} (= DEFAULT_THRESHOLDS FAIL boundary) must → FAIL"
        assert result_not_fail["stage1_verdict"] in (
            "PASS",
            "REVIEW",
        ), f"n_div={ndiv_fail - 1} (< FAIL boundary) must not → FAIL"

    def test_multisite_worst_verdict_wins(self):
        """Worst verdict across multiple sites wins (one bad site → global FAIL)."""
        rng = np.random.default_rng(42)
        # Site "good": 4 chains × 500 draws → ESS ≈ 2000
        # Site "bad": 2 chains × 10 draws, constant → NaN ESS → FAIL
        # (10 draws so split-chain has ≥ 2 per half; constant → zero variance → NaN)
        draws = {
            "good": rng.standard_normal((4, 500, 1)),
            "bad": np.zeros((2, 10, 1)),  # constant → NaN ESS → FAIL
        }
        result = compute_stage1_verdict(draws, n_divergences=0)
        assert result["stage1_verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# classify_recipe_path
# ---------------------------------------------------------------------------


class TestClassifyRecipePath:
    """Unit tests for classify_recipe_path."""

    def _make_recipe(self, tmp_path, model: str, stem: str, content: dict) -> object:
        """Create a minimal recipe JSON at the expected catalog structure."""
        # Mimic catalog/<model>/recipes/<stem>.json layout
        model_dir = tmp_path / model
        recipes_dir = model_dir / "recipes"
        recipes_dir.mkdir(parents=True)
        recipe_path = recipes_dir / f"{stem}.json"
        recipe_path.write_text(json.dumps(content))
        return recipe_path

    def _add_gt_files(self, tmp_path, model: str) -> None:
        """Create stub GT files so the classifier sees a valid GT."""
        gt_dir = tmp_path / model / "groundtruth_samples" / "blackjax"
        gt_dir.mkdir(parents=True)
        (gt_dir / "draws.npz").write_bytes(b"")  # stub presence-only check
        (gt_dir / "summary_v2.json").write_text("{}")

    def _add_cache(self, tmp_path, model: str, stem: str) -> None:
        """Create a stub cached-draws file for path-A classification."""
        cache_dir = tmp_path / model / "_cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / f"{stem}.draws.npz").write_bytes(b"")

    def test_path_a_when_cache_exists(self, tmp_path):
        """Recipe with cached draws → path A."""
        model = "mvn_10"
        stem = "low__nuts__window_adaptation_diag_imm"
        recipe = {
            "gate_evidence": {"auto": {"verdict": "PASS"}},
            "base_method_name": "nuts",
            "calibration_budget": {"num_chains": 4},
        }
        path = self._make_recipe(tmp_path, model, stem, recipe)
        self._add_gt_files(tmp_path, model)
        self._add_cache(tmp_path, model, stem)

        assert classify_recipe_path(path) == "A"

    def test_path_b_no_cache_skip_warmup(self, tmp_path):
        """nuts recipe without cached draws → path B."""
        model = "mvn_10"
        stem = "low__nuts__window_adaptation_diag_imm"
        recipe = {
            "gate_evidence": {"auto": {"verdict": "PASS"}},
            "base_method_name": "nuts",
            "calibration_budget": {"num_chains": 4},
        }
        path = self._make_recipe(tmp_path, model, stem, recipe)
        self._add_gt_files(tmp_path, model)
        # No cache → path B

        assert classify_recipe_path(path) == "B"

    def test_path_c_for_mclmc(self, tmp_path):
        """mclmc recipe without cached draws → path C (full warmup required)."""
        model = "mvn_10"
        stem = "low__mclmc__mclmc_tuning"
        recipe = {
            "gate_evidence": {"auto": {"verdict": "PASS"}},
            "base_method_name": "mclmc",
            "calibration_budget": {"num_chains": 4},
        }
        path = self._make_recipe(tmp_path, model, stem, recipe)
        self._add_gt_files(tmp_path, model)

        assert classify_recipe_path(path) == "C"

    def test_path_c_for_sidecar_imm(self, tmp_path):
        """Recipe with sidecar IMM and no cache → path C."""
        model = "mvn_10"
        stem = "low__nuts__low_rank_window_adaptation"
        recipe = {
            "gate_evidence": {"auto": {"verdict": "PASS"}},
            "base_method_name": "nuts",
            "base_method_params": {"inverse_mass_matrix": "sidecar"},
            "calibration_budget": {"num_chains": 4},
        }
        path = self._make_recipe(tmp_path, model, stem, recipe)
        self._add_gt_files(tmp_path, model)

        assert classify_recipe_path(path) == "C"

    def test_sk_for_vi_base_method(self, tmp_path):
        """VI base method → SK (W1 N/A)."""
        model = "mvn_10"
        stem = "low__meanfield_vi__meanfield_vi"
        recipe = {
            "gate_evidence": {"auto": {"verdict": "PASS"}},
            "base_method_name": "meanfield_vi",
        }
        path = self._make_recipe(tmp_path, model, stem, recipe)
        self._add_gt_files(tmp_path, model)

        assert classify_recipe_path(path) == "SK"

    def test_sk_for_failed_recipe(self, tmp_path):
        """``failed__*`` file → SK without reading content."""
        model = "mvn_10"
        model_dir = tmp_path / model / "recipes"
        model_dir.mkdir(parents=True)
        path = model_dir / "failed__nuts__window_adaptation_diag_imm.json"
        path.write_text(json.dumps({"gate_evidence": {"auto": {"verdict": "FAIL"}}}))

        assert classify_recipe_path(path) == "SK"

    def test_sk_for_non_pass_verdict(self, tmp_path):
        """Recipe with gate verdict != PASS → SK."""
        model = "mvn_10"
        stem = "low__nuts__window_adaptation_diag_imm"
        recipe = {
            "gate_evidence": {"auto": {"verdict": "REVIEW"}},
            "base_method_name": "nuts",
        }
        path = self._make_recipe(tmp_path, model, stem, recipe)
        self._add_gt_files(tmp_path, model)

        assert classify_recipe_path(path) == "SK"

    def test_sk_when_no_gt(self, tmp_path):
        """Recipe without GT files → SK."""
        model = "mvn_10"
        stem = "low__nuts__window_adaptation_diag_imm"
        recipe = {
            "gate_evidence": {"auto": {"verdict": "PASS"}},
            "base_method_name": "nuts",
            "calibration_budget": {"num_chains": 4},
        }
        path = self._make_recipe(tmp_path, model, stem, recipe)
        # No _add_gt_files → missing GT

        assert classify_recipe_path(path) == "SK"

    def test_sk_for_large_nc_chees(self, tmp_path):
        """CHEES recipe with nc > CPU_NC_LIMIT → SK (GPU-scale, CPU-infeasible)."""
        model = "irt_2pl"
        stem = "medium__dynamic_hmc__chees"
        recipe = {
            "gate_evidence": {"auto": {"verdict": "PASS"}},
            "base_method_name": "dynamic_hmc",
            "calibration_budget": {"num_chains": 128},  # > 32 limit
        }
        path = self._make_recipe(tmp_path, model, stem, recipe)
        self._add_gt_files(tmp_path, model)

        assert classify_recipe_path(path) == "SK"

    def test_path_c_for_small_nc_chees(self, tmp_path):
        """CHEES recipe with nc ≤ CPU_NC_LIMIT → path C."""
        model = "mvn_10"
        stem = "low__dynamic_hmc__chees"
        recipe = {
            "gate_evidence": {"auto": {"verdict": "PASS"}},
            "base_method_name": "dynamic_hmc",
            "calibration_budget": {"num_chains": 16},  # ≤ 32
        }
        path = self._make_recipe(tmp_path, model, stem, recipe)
        self._add_gt_files(tmp_path, model)

        assert classify_recipe_path(path) == "C"

    def test_sk_for_laplace_without_cache(self, tmp_path):
        """Laplace methods without cached draws → SK (MAP-init sensitive)."""
        model = "mvn_10"
        stem = "low__laplace_hmc__window_adaptation_diag_imm"
        recipe = {
            "gate_evidence": {"auto": {"verdict": "PASS"}},
            "base_method_name": "laplace_hmc",
            "calibration_budget": {"num_chains": 4},
        }
        path = self._make_recipe(tmp_path, model, stem, recipe)
        self._add_gt_files(tmp_path, model)

        assert classify_recipe_path(path) == "SK"

    def test_path_a_for_laplace_with_cache(self, tmp_path):
        """Laplace method with cached draws → path A (cache exists, skip the skip)."""
        model = "mvn_10"
        stem = "low__laplace_hmc__window_adaptation_diag_imm"
        recipe = {
            "gate_evidence": {"auto": {"verdict": "PASS"}},
            "base_method_name": "laplace_hmc",
            "calibration_budget": {"num_chains": 4},
        }
        path = self._make_recipe(tmp_path, model, stem, recipe)
        self._add_gt_files(tmp_path, model)
        self._add_cache(tmp_path, model, stem)

        assert classify_recipe_path(path) == "A"


# ---------------------------------------------------------------------------
# _cell_regen_seed
# ---------------------------------------------------------------------------


class TestCellRegenSeed:
    """Tests for the deterministic per-cell seed derivation."""

    def test_deterministic_across_calls(self):
        """Same key always produces the same seed."""
        key = "mvn_10/low__nuts__window_adaptation_diag_imm"
        s1 = _cell_regen_seed(key)
        s2 = _cell_regen_seed(key)
        assert s1 == s2

    def test_different_keys_give_different_seeds(self):
        """Different recipe keys produce different seeds (collision would be a bug)."""
        s1 = _cell_regen_seed("mvn_10/low__nuts__window_adaptation_diag_imm")
        s2 = _cell_regen_seed("banana/low__nuts__window_adaptation_diag_imm")
        assert s1 != s2

    def test_seed_is_non_negative_int(self):
        """Seed is a non-negative 31-bit integer."""
        seed = _cell_regen_seed("any/key")
        assert isinstance(seed, int)
        assert 0 <= seed <= 0x7FFFFFFF

    def test_stable_against_base_seed_change(self):
        """Different base seeds give different per-cell seeds."""
        key = "mvn_10/low__nuts__window_adaptation_diag_imm"
        s1 = _cell_regen_seed(key, base_seed=42)
        s2 = _cell_regen_seed(key, base_seed=0)
        assert s1 != s2
