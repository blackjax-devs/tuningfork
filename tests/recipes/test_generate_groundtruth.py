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
"""Tests for the ground-truth orchestrator (_generate_groundtruth.py).

Fast tests: analytic-path smoke (no JAX trace).
Slow tests: analytic-path multi-model sweep (still no NUTS chain).

The NUTS-path PASS-verdict test was removed 2026-05-17 (PR #5). Asserting a
real NUTS cert PASS from a unit test is fundamentally incompatible with both
(a) "tests should be fast" — the gate threshold is absolute so the cert needs
≥ ~2800 samples + adaptation to clear it (~60 s on CI), and (b) "tests should
be robust" — CI run 25983264151 demonstrated that even at n=4000/seed=42 the
PRNG path on GH-Actions runners can fall just under threshold without any
chain pathology. The gate-logic correctness is covered by the synthetic-input
unit tests in ``tests/reference/test_nuts.py::TestCertifyNutsGateLogic``;
the structural plumbing of ``certify_reference_nuts`` is covered by
``TestCertifyNutsInterface``; the end-to-end PASS verdict on real data is
exercised by the production recipe-generation pipeline, not by unit tests.
"""

from pathlib import Path

import pytest

from tuningfork.model import MODELS
from tuningfork.recipes._generate_groundtruth import (
    generate_groundtruth_recipe,
    sweep_all,
)


@pytest.mark.fast
def test_generate_groundtruth_analytic_returns_none_and_populates_cache(
    tmp_path: Path,
) -> None:
    """generate_groundtruth_recipe for an analytic model (mvn_10) returns None
    and populates draws/summaries/metadata cache files."""
    entry = MODELS["mvn_10"]

    result = generate_groundtruth_recipe(
        entry,
        seed=42,
        cache_dir=tmp_path,
        # n_samples doesn't drive a NUTS run for analytic models, but we use
        # a small value to keep the test fast (fewer i.i.d. draws to write)
        n_samples=500,
    )

    # Analytic path: no recipe emitted
    assert result is None

    # Cache files must be populated
    assert (tmp_path / "draws" / "mvn_10.npz").exists(), "draws npz missing"
    assert (tmp_path / "summaries" / "mvn_10.json").exists(), "summaries json missing"
    assert (tmp_path / "metadata" / "mvn_10.json").exists(), "metadata json missing"

    # No adaptation file for analytic models
    assert not (tmp_path / "adaptation" / "mvn_10.json").exists()


@pytest.mark.slow
def test_sweep_all_analytic_models_pass(tmp_path: Path) -> None:
    """sweep_all on two analytic models returns 2-entry summary with both passed=True."""
    # mvn_10 and neals_funnel are both analytic (no NUTS chain)
    results = sweep_all(
        models=["mvn_10", "neals_funnel"],
        seed=42,
        n_samples=500,
        cache_dir=tmp_path,
    )

    assert set(results.keys()) == {"mvn_10", "neals_funnel"}

    for name, summary in results.items():
        assert summary["passed"] is True, f"{name} failed: {summary}"
        assert summary["generator"] == "analytic"
        assert summary["wall_seconds"] >= 0.0
        assert summary["recipe_path"] is None  # no recipe for analytic
        assert summary["cert_diagnostics"] is None  # no cert for analytic
