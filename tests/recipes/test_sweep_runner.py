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
"""Tests for Phase G: V4/V5/V6 step policies + sweep_runner + verdict guard.

All tests marked @pytest.mark.fast — purely structural/unit tests, no JAX
trace or sampler execution.
"""
import math

import pytest

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# V4 (poisson) round-trip
# ---------------------------------------------------------------------------


class TestV4Poisson:
    """V4 poisson step_policy — build_step_policy round-trip."""

    def test_basic_uncapped(self) -> None:
        """Uncapped Poisson(lam=20) with floor 1 builds without error."""
        import jax

        from tuningfork.base_method._step_policy_registry import build_step_policy

        spec = {"kind": "poisson", "lam": 20, "low": 1, "high": None}
        fn = build_step_policy(spec)
        key = jax.random.key(42)
        result = fn(key)
        assert hasattr(result, "shape"), "poisson fn should return jax.Array"
        assert int(result) >= 1, f"L must be ≥ 1 (floor); got {int(result)}"

    def test_truncated_poisson(self) -> None:
        """Truncated Poisson(lam=20, high=30) always returns L < 30."""
        import jax

        from tuningfork.base_method._step_policy_registry import build_step_policy

        spec = {"kind": "poisson", "lam": 20, "low": 1, "high": 30}
        fn = build_step_policy(spec)
        for seed in range(10):
            key = jax.random.key(seed)
            result = int(fn(key))
            assert 1 <= result < 30, f"seed={seed}: got {result}, expected [1, 30)"

    def test_floor_applied(self) -> None:
        """With floor=5, output is always ≥ 5."""
        import jax

        from tuningfork.base_method._step_policy_registry import build_step_policy

        spec = {"kind": "poisson", "lam": 10, "low": 5, "high": None}
        fn = build_step_policy(spec)
        for seed in range(20):
            key = jax.random.key(seed)
            result = int(fn(key))
            assert result >= 5, f"seed={seed}: got {result}, expected ≥ 5"

    def test_missing_lam_raises(self) -> None:
        """Missing 'lam' raises ValueError."""
        from tuningfork.base_method._step_policy_registry import build_step_policy

        with pytest.raises(ValueError, match="lam"):
            build_step_policy({"kind": "poisson", "low": 1})

    def test_json_roundtrip(self) -> None:
        """poisson spec survives JSON serialisation."""
        import json

        from tuningfork.base_method._step_policy_registry import build_step_policy

        spec = {"kind": "poisson", "lam": 15, "low": 1, "high": None}
        spec_rt = json.loads(json.dumps(spec))
        fn = build_step_policy(spec_rt)
        assert callable(fn)


# ---------------------------------------------------------------------------
# V5 (log_uniform_int) round-trip
# ---------------------------------------------------------------------------


class TestV5LogUniformInt:
    """V5 log_uniform_int step_policy — build_step_policy round-trip."""

    def test_basic(self) -> None:
        """log_uniform_int [1, 1024] builds and returns integer in [1, 1024]."""
        import jax

        from tuningfork.base_method._step_policy_registry import build_step_policy

        spec = {"kind": "log_uniform_int", "low": 1, "high": 1024}
        fn = build_step_policy(spec)
        for seed in range(20):
            key = jax.random.key(seed)
            result = int(fn(key))
            assert 1 <= result <= 1024, f"seed={seed}: got {result}, expected [1, 1024]"

    def test_mean_is_approx_log_uniform(self) -> None:
        """Mean of log_uniform_int [1, 1024] is ~148 (spec §5)."""
        import jax
        import numpy as np

        from tuningfork.base_method._step_policy_registry import build_step_policy

        spec = {"kind": "log_uniform_int", "low": 1, "high": 1024}
        fn = build_step_policy(spec)
        keys = jax.vmap(fn)(jax.random.split(jax.random.key(0), 4000))
        mean = float(np.mean(np.asarray(keys)))
        # Theoretical E[L] = (H - L) / log(H/L) ≈ (1024-1)/log(1024) ≈ 148
        expected_mean = (1024 - 1) / math.log(1024)
        assert (
            abs(mean - expected_mean) < 20
        ), f"V5 mean={mean:.1f} far from theoretical {expected_mean:.1f}"

    def test_low_must_be_positive(self) -> None:
        """low=0 raises ValueError (log(0) is undefined)."""
        from tuningfork.base_method._step_policy_registry import build_step_policy

        with pytest.raises(ValueError, match="low.*≥.*1"):
            build_step_policy({"kind": "log_uniform_int", "low": 0, "high": 1024})

    def test_low_lt_high_required(self) -> None:
        """low ≥ high raises ValueError."""
        from tuningfork.base_method._step_policy_registry import build_step_policy

        with pytest.raises(ValueError, match="low < high"):
            build_step_policy({"kind": "log_uniform_int", "low": 100, "high": 50})

    def test_missing_fields_raise(self) -> None:
        from tuningfork.base_method._step_policy_registry import build_step_policy

        with pytest.raises(ValueError, match="low.*high"):
            build_step_policy({"kind": "log_uniform_int"})


# ---------------------------------------------------------------------------
# V6 (pow2_choice) round-trip
# ---------------------------------------------------------------------------


class TestV6Pow2Choice:
    """V6 pow2_choice step_policy — build_step_policy round-trip."""

    def test_basic(self) -> None:
        """pow2_choice {2,4,8,16,32,64} returns a value from the set."""
        import jax

        from tuningfork.base_method._step_policy_registry import build_step_policy

        options = [2, 4, 8, 16, 32, 64]
        spec = {"kind": "pow2_choice", "options": options}
        fn = build_step_policy(spec)
        for seed in range(20):
            key = jax.random.key(seed)
            result = int(fn(key))
            assert result in options, f"seed={seed}: got {result}, not in {options}"

    def test_single_option(self) -> None:
        """Single-element options list returns that element always."""
        import jax

        from tuningfork.base_method._step_policy_registry import build_step_policy

        spec = {"kind": "pow2_choice", "options": [32]}
        fn = build_step_policy(spec)
        key = jax.random.key(0)
        assert int(fn(key)) == 32

    def test_empty_options_raises(self) -> None:
        from tuningfork.base_method._step_policy_registry import build_step_policy

        with pytest.raises(ValueError, match="non-empty"):
            build_step_policy({"kind": "pow2_choice", "options": []})

    def test_nonpositive_option_raises(self) -> None:
        from tuningfork.base_method._step_policy_registry import build_step_policy

        with pytest.raises(ValueError, match="positive"):
            build_step_policy({"kind": "pow2_choice", "options": [0, 4, 8]})

    def test_missing_options_raises(self) -> None:
        from tuningfork.base_method._step_policy_registry import build_step_policy

        with pytest.raises(ValueError, match="options"):
            build_step_policy({"kind": "pow2_choice"})

    def test_json_roundtrip(self) -> None:
        import json

        from tuningfork.base_method._step_policy_registry import build_step_policy

        spec = {"kind": "pow2_choice", "options": [2, 4, 8, 16, 32, 64]}
        spec_rt = json.loads(json.dumps(spec))
        fn = build_step_policy(spec_rt)
        assert callable(fn)


# ---------------------------------------------------------------------------
# sweep_runner structural tests
# ---------------------------------------------------------------------------


class TestSweepRunner:
    """Unit tests for _sweep_runner scaffolding logic (no sampler execution)."""

    def test_build_default_candidates_no_nis(self) -> None:
        """Without NIS median or chain_stats, candidates are [V1, V6, V2, V5]."""
        from tuningfork.recipes._sweep_runner import (
            V1_SPEC,
            V2_SPEC,
            V5_SPEC,
            V6_SPEC,
            build_default_candidates,
        )

        candidates = build_default_candidates()
        assert candidates[0] == V1_SPEC, "first must be V1"
        # V6 included because nis_median is None (NIS unknown → include V6 as fallback)
        assert V6_SPEC in candidates, "V6 must be included when nis_median is None"
        assert V2_SPEC in candidates, "V2 (fallback) must always be included"
        assert V5_SPEC in candidates, "V5 (last resort) must always be included"
        # V4 omitted — no nis_median
        assert not any(
            c.get("kind") == "poisson" for c in candidates
        ), "V4 should be omitted when nis_median is None"

    def test_build_default_candidates_with_nis(self) -> None:
        """With NIS median=15, V4 is included and V6 is included (15 ≤ 64)."""
        from tuningfork.recipes._sweep_runner import build_default_candidates

        candidates = build_default_candidates(nis_median=15)
        kinds = [c.get("kind") for c in candidates]
        assert "poisson" in kinds, "V4 must be included when nis_median is given"
        assert "pow2_choice" in kinds, "V6 must be included when nis_median=15 ≤ 64"
        # V4 lam should be NIS median
        v4 = next(c for c in candidates if c.get("kind") == "poisson")
        assert v4["lam"] == 15

    def test_build_default_candidates_high_nis_skips_v6(self) -> None:
        """With NIS median=100 > 64, V6 pow2 grid is skipped (max option=64 too short)."""
        from tuningfork.recipes._sweep_runner import build_default_candidates

        candidates = build_default_candidates(nis_median=100)
        assert not any(
            c.get("kind") == "pow2_choice" for c in candidates
        ), "V6 should be omitted when nis_median > 64"

    def test_candidate_ordering_cheapest_first(self) -> None:
        """Default candidate ordering: V1 → V4 → V6 → V2 → V5 (V7 omitted without cache)."""
        from tuningfork.recipes._sweep_runner import build_default_candidates

        candidates = build_default_candidates(nis_median=20)
        kinds = [c.get("kind") for c in candidates]
        # V1 first
        assert kinds[0] == "uniform_int" and candidates[0]["high"] == 50
        # V4 poisson second
        assert kinds[1] == "poisson"
        # V6 pow2 third
        assert kinds[2] == "pow2_choice"
        # V2 before V5
        v2_idx = next(
            i
            for i, c in enumerate(candidates)
            if c.get("kind") == "uniform_int" and c.get("low") == 50
        )
        v5_idx = next(
            i for i, c in enumerate(candidates) if c.get("kind") == "log_uniform_int"
        )
        assert v2_idx < v5_idx, "V2 must come before V5"

    def test_candidate_result_passes_gate(self) -> None:
        """CandidateResult.passes_gate uses the correct gate thresholds."""
        from tuningfork.recipes._sweep_runner import (
            GATE_DIV_RATE_PASS,
            GATE_ESS_PASS,
            GATE_RHAT_PASS,
            CandidateResult,
        )

        spec = {"kind": "uniform_int", "low": 5, "high": 50}
        # Clean PASS
        c = CandidateResult(spec, "PASS", 1.005, 650.0, 0, 4000, 10.0)
        assert c.passes_gate()
        # rhat just over threshold
        c_rhat = CandidateResult(spec, "REVIEW", GATE_RHAT_PASS, 650.0, 0, 4000, 10.0)
        assert not c_rhat.passes_gate()
        # ESS just under threshold
        c_ess = CandidateResult(spec, "REVIEW", 1.005, GATE_ESS_PASS - 1, 0, 4000, 10.0)
        assert not c_ess.passes_gate()
        # div rate just over threshold (GATE_DIV_RATE_PASS = 5%; 210/4000 = 5.25%)
        n_div_over = int(4000 * (GATE_DIV_RATE_PASS + 0.0025))
        c_div = CandidateResult(spec, "REVIEW", 1.005, 650.0, n_div_over, 4000, 10.0)
        assert not c_div.passes_gate()

    def test_spec_to_slug(self) -> None:
        from tuningfork.recipes._sweep_runner import _spec_to_slug

        assert (
            _spec_to_slug({"kind": "uniform_int", "low": 5, "high": 50})
            == "v1-uniform5-50"
        )
        assert _spec_to_slug({"kind": "poisson", "lam": 15}) == "v4-poisson15"
        assert (
            _spec_to_slug({"kind": "log_uniform_int", "low": 1, "high": 1024})
            == "v5-logunif"
        )
        assert _spec_to_slug({"kind": "pow2_choice", "options": [2, 4]}) == "v6-pow2"
        assert (
            _spec_to_slug({"kind": "empirical", "values": [], "weights": []})
            == "v7-empirical"
        )

    def test_sweep_and_pick_rejects_non_dynamic_samplers(self) -> None:
        """sweep_and_pick raises ValueError for non-dynamic_hmc/dmhmc samplers."""
        from tuningfork.recipes._sweep_runner import sweep_and_pick

        with pytest.raises(ValueError, match="dynamic_hmc.*dmhmc"):
            sweep_and_pick("mvn_10", "window_adaptation_diag_imm", "nuts", [])


# ---------------------------------------------------------------------------
# Verdict guard on stamp_headline_from_chain_stats
# ---------------------------------------------------------------------------


class TestVerdictGuard:
    """stamp_headline_from_chain_stats must refuse non-PASS recipes."""

    def _make_recipe(self, verdict: str):
        """Build a minimal fake recipe with the given gate verdict."""
        from tuningfork.recipes._base import Effort, Recipe

        return Recipe(
            model_name="mvn_10",
            base_method_name="dynamic_hmc",
            warmup_name="window_adaptation_diag_imm",
            effort=Effort.MEDIUM,
            base_method_params={"step_size": 0.5},
            warmup_params={"n_warmup": 1000},
            headline_metric=None,
            sample_quality=None,
            calibration_budget={"trials": 0, "wall_seconds_estimate": 0.0},
            difficulty=None,
            instructions="",
            gate_evidence={
                "auto": {
                    "verdict": verdict,
                    "min_bulk_ess": 450.0 if verdict == "PASS" else 350.0,
                    "rhat_max": 1.005,
                    "n_divergences": 0,
                }
            },
            tuning_seed=0,
            tuningfork_version="0.0.0.dev0",
            blackjax_version="1.0.0",
            jax_version="0.4.0",
            timestamp_utc="2026-01-01T00:00:00Z",
        )

    def test_review_no_longer_raises(self, tmp_path) -> None:
        """stamp_headline_from_chain_stats allows REVIEW recipes (convention update 2026-05-30).

        User decision: REVIEW recipes now expose headline for easier review.
        REVIEW is borderline GT-agreement; ESS is real and useful.
        """
        import warnings

        from tuningfork.base_method import BASE_METHODS
        from tuningfork.recipes._recipe_runner import stamp_headline_from_chain_stats

        recipe = self._make_recipe("REVIEW")
        bm = BASE_METHODS["dynamic_hmc"]
        # REVIEW is now allowed — no chain_stats in tmp_path → warning, returns unchanged
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = stamp_headline_from_chain_stats(recipe, bm, catalog_root=tmp_path)
        assert result.headline_metric is None  # no chain_stats → unchanged (not raised)

    def test_fail_raises(self) -> None:
        """stamp_headline_from_chain_stats raises on FAIL recipe (no meaningful ESS)."""
        from tuningfork.base_method import BASE_METHODS
        from tuningfork.recipes._recipe_runner import stamp_headline_from_chain_stats

        recipe = self._make_recipe("FAIL")
        bm = BASE_METHODS["dynamic_hmc"]
        with pytest.raises(ValueError, match="refusing.*FAIL"):
            stamp_headline_from_chain_stats(recipe, bm)

    def test_pass_proceeds(self, tmp_path) -> None:
        """PASS recipe with no chain_stats gets a warning but no error."""
        import warnings

        from tuningfork.base_method import BASE_METHODS
        from tuningfork.recipes._recipe_runner import stamp_headline_from_chain_stats

        recipe = self._make_recipe("PASS")
        bm = BASE_METHODS["dynamic_hmc"]
        # No chain_stats.npz in tmp_path → warning, returns unchanged recipe
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = stamp_headline_from_chain_stats(recipe, bm, catalog_root=tmp_path)
        # Should return the recipe unchanged (no crash)
        assert result.headline_metric is None  # no chain_stats → unchanged
        assert any(
            "chain_stats" in str(warning.message) for warning in w
        ), "Expected a warning about missing chain_stats"

    def test_already_stamped_noop(self) -> None:
        """Recipe with non-null headline_metric is returned unchanged (no-op)."""
        import dataclasses

        from tuningfork.base_method import BASE_METHODS
        from tuningfork.recipes._recipe_runner import stamp_headline_from_chain_stats

        recipe = self._make_recipe("PASS")
        recipe = dataclasses.replace(recipe, headline_metric=0.01)
        bm = BASE_METHODS["dynamic_hmc"]
        result = stamp_headline_from_chain_stats(recipe, bm)
        assert result is recipe or result.headline_metric == 0.01

    def test_unknown_verdict_proceeds(self, tmp_path) -> None:
        """Recipe with unknown or None verdict does not raise (no guard fires)."""
        from tuningfork.base_method import BASE_METHODS
        from tuningfork.recipes._recipe_runner import stamp_headline_from_chain_stats

        recipe = self._make_recipe("PASS")
        # Patch verdict to None (no gate run yet)
        import dataclasses

        ge = dict(recipe.gate_evidence)
        ge["auto"] = dict(ge["auto"])
        ge["auto"]["verdict"] = None
        recipe = dataclasses.replace(recipe, gate_evidence=ge)
        bm = BASE_METHODS["dynamic_hmc"]
        # Should not raise — None verdict means gate hasn't run, OK to attempt
        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            stamp_headline_from_chain_stats(recipe, bm, catalog_root=tmp_path)
