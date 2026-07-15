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
"""Tests for verify_groundtruth.

All tests are fast (no JAX traces, no chain runs) — they exercise the gate
and coherence logic using committed GT artifacts as synthetic "generated"
summaries or by injecting crafted dicts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tuningfork.groundtruth._dispatch import committed_gt_dir, load_committed_summary
from tuningfork.groundtruth._verify import (
    _check_coherence,
    _check_gate,
    verify_groundtruth,
)

pytestmark = pytest.mark.fast

# --------------------------------------------------------------------------- #
# _check_gate unit tests
# --------------------------------------------------------------------------- #


def _gate_summary(
    *,
    max_rhat: float = 1.005,
    min_bulk_ess: float = 500.0,
    total_divergences: int = 0,
    min_e_bfmi: float | None = 0.5,
    n_total: int = 100_000,
    generator: str = "nuts_perchain",
) -> dict[str, Any]:
    """Build a minimal summary_v2 dict for gate testing."""
    return {
        "generator": generator,
        "n_total": n_total,
        "quality_gate": {
            "max_rhat": max_rhat,
            "min_bulk_ess": min_bulk_ess,
            "total_divergences": total_divergences,
            "min_e_bfmi": min_e_bfmi,
        },
    }


def test_gate_pass_nuts() -> None:
    """Clean NUTS summary passes gate."""
    passed, details = _check_gate(_gate_summary())
    assert passed
    assert details["rhat_ok"]
    assert details["ess_ok"]
    assert details["div_ok"]
    assert details["ebfmi_ok"]


def test_gate_fail_rhat() -> None:
    """max_rhat > 1.01 fails gate."""
    passed, details = _check_gate(_gate_summary(max_rhat=1.015))
    assert not passed
    assert not details["rhat_ok"]


def test_gate_fail_ess() -> None:
    """min_bulk_ess < 400 fails gate for NUTS."""
    passed, details = _check_gate(_gate_summary(min_bulk_ess=350.0))
    assert not passed
    assert not details["ess_ok"]


def test_gate_fail_divergences() -> None:
    """High divergence rate fails gate."""
    passed, details = _check_gate(_gate_summary(total_divergences=200, n_total=100_000))
    assert not passed
    assert not details["div_ok"]


def test_gate_fail_ebfmi() -> None:
    """min_e_bfmi < 0.3 fails gate for NUTS."""
    passed, details = _check_gate(_gate_summary(min_e_bfmi=0.25))
    assert not passed
    assert not details["ebfmi_ok"]


def test_gate_analytic_skips_nuts_checks() -> None:
    """Analytic models skip ESS / div / E-BFMI checks."""
    summary = _gate_summary(
        generator="analytic_iid",
        min_bulk_ess=10.0,
        total_divergences=100,
        min_e_bfmi=0.1,
        max_rhat=1.005,
    )
    passed, details = _check_gate(summary)
    assert passed, "analytic model with bad ESS/div/ebfmi should still pass"
    assert details["ess_ok"]
    assert details["div_ok"]
    assert details["ebfmi_ok"]


def test_gate_no_ebfmi_field() -> None:
    """Missing min_e_bfmi field is treated as no constraint."""
    passed, _ = _check_gate(_gate_summary(min_e_bfmi=None))
    assert passed


# --------------------------------------------------------------------------- #
# _check_coherence unit tests
# --------------------------------------------------------------------------- #


def _per_site_summary(
    mean: list[float],
    se: list[float],
) -> dict[str, Any]:
    """Build a minimal per_site dict for one site."""
    return {
        "mean": mean,
        "between_chain_se": se,
        "std": [1.0] * len(mean),
        "q05": [0.0] * len(mean),
        "q95": [0.0] * len(mean),
        "bulk_ess": [500.0] * len(mean),
        "tail_ess": [500.0] * len(mean),
        "rhat": [1.005] * len(mean),
    }


def test_coherence_pass_identical_summaries() -> None:
    """Identical generated and committed summaries → z=0 → pass."""
    per_site = {"mu": _per_site_summary([1.0, 2.0], [0.01, 0.01])}
    summary = {"per_site": per_site}
    passed, results = _check_coherence(summary, summary, z_threshold=3.0)
    assert passed
    assert all(r["max_z"] == pytest.approx(0.0) for r in results)


def test_coherence_fail_large_deviation() -> None:
    """Generated mean deviates by 10σ → fail."""
    gen = {"per_site": {"mu": _per_site_summary([11.0], [0.1])}}
    com = {"per_site": {"mu": _per_site_summary([1.0], [0.1])}}
    passed, results = _check_coherence(gen, com, z_threshold=3.0)
    assert not passed
    assert results[0]["max_z"] == pytest.approx(10.0 / 0.1)


def test_coherence_se_floor() -> None:
    """Sites with near-zero SE use _SE_FLOOR to avoid division by zero."""
    gen = {"per_site": {"mu": _per_site_summary([1.0], [0.0])}}
    com = {"per_site": {"mu": _per_site_summary([1.0], [0.0])}}
    passed, results = _check_coherence(gen, com, z_threshold=3.0)
    assert passed
    assert np.isfinite(results[0]["max_z"])


def test_coherence_skip_missing_site() -> None:
    """Sites in generated but not committed are skipped (no error)."""
    gen = {
        "per_site": {
            "mu": _per_site_summary([1.0], [0.01]),
            "sigma": _per_site_summary([0.5], [0.005]),
        }
    }
    com = {"per_site": {"mu": _per_site_summary([1.0], [0.01])}}
    passed, results = _check_coherence(gen, com, z_threshold=3.0)
    assert passed
    assert len(results) == 1  # sigma was skipped


# --------------------------------------------------------------------------- #
# verify_groundtruth integration: committed GT verifies against itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_name", ["mvn_10", "radon"])
def test_verify_committed_gt_passes_against_itself(model_name: str) -> None:
    """Committed GT summary passes gate + coherence when compared to itself."""
    committed = load_committed_summary(model_name)
    gt_draws = committed_gt_dir(model_name) / "draws.npz"

    result = verify_groundtruth(
        model_name,
        committed,
        gt_draws,
        print_results=False,
    )
    assert result, (
        f"{model_name}: committed GT failed verify against itself "
        "(gate or coherence check)"
    )


def test_verify_returns_false_on_gate_fail() -> None:
    """verify_groundtruth returns False when gate fails."""
    committed = load_committed_summary("mvn_10")
    # Inject a bad rhat into the committed summary
    bad_summary = dict(committed)
    bad_summary["quality_gate"] = dict(committed["quality_gate"])
    bad_summary["quality_gate"]["max_rhat"] = 1.5  # clearly fails
    gt_draws = committed_gt_dir("mvn_10") / "draws.npz"

    result = verify_groundtruth(
        "mvn_10",
        bad_summary,
        gt_draws,
        print_results=False,
    )
    assert not result


def test_verify_returns_false_on_coherence_fail() -> None:
    """verify_groundtruth returns False when coherence fails."""
    committed = load_committed_summary("mvn_10")
    # Shift all per_site means by 100× posterior std → z >> 3
    bad_summary = dict(committed)
    bad_per_site = {}
    for site, stats in committed["per_site"].items():
        bad_stats = dict(stats)
        mean_arr = np.asarray(stats["mean"])
        std_arr = np.asarray(stats["std"])
        bad_stats["mean"] = (mean_arr + 100.0 * std_arr).tolist()
        bad_per_site[site] = bad_stats
    bad_summary["per_site"] = bad_per_site
    gt_draws = committed_gt_dir("mvn_10") / "draws.npz"

    result = verify_groundtruth(
        "mvn_10",
        bad_summary,
        gt_draws,
        print_results=False,
    )
    assert not result
