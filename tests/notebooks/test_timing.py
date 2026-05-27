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
"""Tests for catalog timing context helpers."""

import pytest

from tuningfork.catalog._timing import compute_total_warmup_steps, format_timing_context

pytestmark = pytest.mark.fast


class MockRecipe:
    """Minimal mock Recipe for testing timing context."""

    def __init__(self, calibration_budget=None, warmups=None):
        self.calibration_budget = calibration_budget or {}
        self.warmups = warmups or []


class TestComputeTotalWarmupSteps:
    """Tests for compute_total_warmup_steps function."""

    def test_single_phase(self):
        """Single warmup phase returns its n_warmup."""
        warmups = [{"name": "window_adaptation_diag_imm", "params": {"n_warmup": 500}}]
        assert compute_total_warmup_steps(warmups) == 500

    def test_two_phases(self):
        """Multiple phases are summed."""
        warmups = [
            {"name": "pathfinder", "params": {"n_warmup": 100}},
            {"name": "window_adaptation_diag_imm", "params": {"n_warmup": 400}},
        ]
        assert compute_total_warmup_steps(warmups) == 500

    def test_three_phases(self):
        """Three or more phases are summed correctly."""
        warmups = [
            {"name": "pathfinder", "params": {"n_warmup": 100}},
            {"name": "window_adaptation_diag_imm", "params": {"n_warmup": 200}},
            {"name": "window_adaptation_diag_imm", "params": {"n_warmup": 300}},
        ]
        assert compute_total_warmup_steps(warmups) == 600

    def test_none_warmups(self):
        """None returns None."""
        assert compute_total_warmup_steps(None) is None

    def test_empty_warmups(self):
        """Empty list returns None."""
        assert compute_total_warmup_steps([]) is None

    def test_missing_n_warmup(self):
        """Missing n_warmup in any phase returns None."""
        warmups = [{"name": "window_adaptation_diag_imm", "params": {}}]
        assert compute_total_warmup_steps(warmups) is None

    def test_missing_params_key(self):
        """Missing params dict returns None."""
        warmups = [{"name": "window_adaptation_diag_imm"}]
        assert compute_total_warmup_steps(warmups) is None

    def test_mixed_missing_n_warmup(self):
        """If any phase lacks n_warmup, return None."""
        warmups = [
            {"name": "pathfinder", "params": {"n_warmup": 100}},
            {"name": "window_adaptation_diag_imm", "params": {}},  # missing n_warmup
        ]
        assert compute_total_warmup_steps(warmups) is None


class TestFormatTimingContext:
    """Tests for format_timing_context function."""

    def test_complete_data(self):
        """Complete recipe produces all context strings."""
        recipe = MockRecipe(
            calibration_budget={
                "n_warmup": 1000,
                "n_samples": 1000,
                "num_chains": 4,
                "warmup_wall_seconds": 6.977,
                "sampling_wall_seconds": 16.045,
            },
            warmups=[
                {"name": "window_adaptation_diag_imm", "params": {"n_warmup": 1000}}
            ],
        )

        ctx = format_timing_context(recipe)

        assert ctx["warmup_wall"] == "1000 steps × 4 chains"
        assert ctx["sampling_wall"] == "1000/chain × 4 chains = 4000 draws"
        assert ctx["per_draw"] == "per chain·draw (4000 draws)"
        assert ctx["total_wall"] == "warmup + sampling"
        assert ctx["machine"] == ""

    def test_two_phase_warmup(self):
        """Multi-phase warmup sums correctly in context."""
        recipe = MockRecipe(
            calibration_budget={
                "n_samples": 2000,
                "num_chains": 2,
                "warmup_wall_seconds": 5.0,
                "sampling_wall_seconds": 10.0,
            },
            warmups=[
                {"name": "pathfinder", "params": {"n_warmup": 200}},
                {"name": "window_adaptation_diag_imm", "params": {"n_warmup": 800}},
            ],
        )

        ctx = format_timing_context(recipe)

        # Total warmup: 200 + 800 = 1000 steps
        assert ctx["warmup_wall"] == "1000 steps × 2 chains"
        # Sampling: 2000/chain × 2 chains = 4000 draws
        assert ctx["sampling_wall"] == "2000/chain × 2 chains = 4000 draws"
        assert ctx["per_draw"] == "per chain·draw (4000 draws)"

    def test_no_warmups_list(self):
        """Legacy recipe with no warmups list shows dashes."""
        recipe = MockRecipe(
            calibration_budget={
                "n_samples": 1000,
                "num_chains": 4,
                "warmup_wall_seconds": 6.977,
                "sampling_wall_seconds": 16.045,
            },
            warmups=[],  # empty warmups list
        )

        ctx = format_timing_context(recipe)

        assert ctx["warmup_wall"] == "—"
        # Sampling still works
        assert ctx["sampling_wall"] == "1000/chain × 4 chains = 4000 draws"

    def test_missing_num_chains(self):
        """Missing num_chains shows dashes for dependent fields."""
        recipe = MockRecipe(
            calibration_budget={
                "n_samples": 1000,
                "warmup_wall_seconds": 6.977,
                "sampling_wall_seconds": 16.045,
            },
            warmups=[
                {"name": "window_adaptation_diag_imm", "params": {"n_warmup": 1000}}
            ],
        )

        ctx = format_timing_context(recipe)

        assert ctx["warmup_wall"] == "—"  # needs num_chains
        assert ctx["sampling_wall"] == "—"  # needs num_chains
        assert ctx["per_draw"] == "—"  # needs num_chains

    def test_missing_n_samples(self):
        """Missing n_samples shows dashes for dependent fields."""
        recipe = MockRecipe(
            calibration_budget={
                "num_chains": 4,
                "warmup_wall_seconds": 6.977,
                "sampling_wall_seconds": 16.045,
            },
            warmups=[
                {"name": "window_adaptation_diag_imm", "params": {"n_warmup": 1000}}
            ],
        )

        ctx = format_timing_context(recipe)

        assert ctx["warmup_wall"] == "1000 steps × 4 chains"  # has what it needs
        assert ctx["sampling_wall"] == "—"  # needs n_samples
        assert ctx["per_draw"] == "—"  # needs n_samples

    def test_no_wall_times(self):
        """Missing wall times show blank (not dashes) for total_wall."""
        recipe = MockRecipe(
            calibration_budget={
                "n_samples": 1000,
                "num_chains": 4,
                # no warmup_wall_seconds or sampling_wall_seconds
            },
            warmups=[
                {"name": "window_adaptation_diag_imm", "params": {"n_warmup": 1000}}
            ],
        )

        ctx = format_timing_context(recipe)

        assert ctx["total_wall"] == ""  # blank, not dashes

    def test_empty_budget(self):
        """Empty calibration_budget shows all dashes."""
        recipe = MockRecipe(
            calibration_budget={},
            warmups=[],
        )

        ctx = format_timing_context(recipe)

        assert ctx["warmup_wall"] == "—"
        assert ctx["sampling_wall"] == "—"
        assert ctx["per_draw"] == "—"
        assert ctx["total_wall"] == ""
        assert ctx["machine"] == ""

    def test_none_budget(self):
        """None calibration_budget is treated as empty dict."""
        recipe = MockRecipe(
            calibration_budget=None,
            warmups=[],
        )

        ctx = format_timing_context(recipe)

        assert ctx["warmup_wall"] == "—"
        assert ctx["sampling_wall"] == "—"
