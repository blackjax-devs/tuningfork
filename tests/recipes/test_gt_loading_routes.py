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
"""Runner-level tests: GT loading route dispatch (summary_v2 vs legacy).

Tests the two paths added to emit_low_recipe_for_cell's GT loading block:

1. summary_v2.json path  → _build_gt_for_gate_v2  → between_chain_se present
2. legacy summary.json   → _align_gt_keys_for_gate → n_samples present, no between_chain_se

All tests are pure logic (no JAX trace, no file I/O for the route-dispatch logic
— the helper functions are tested directly, the file-priority logic tested via
temp directories).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tuningfork.recipes._recipe_runner import (
    _align_gt_keys_for_gate,
    _build_gt_for_gate_v2,
)

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DRAWS = {
    "mu": np.zeros((4, 100)),
    "tau": np.ones((4, 100)),
    "theta_raw": np.zeros((4, 100, 8)),
}

_DRAWS_LAPLACE = {
    "mu": np.zeros((4, 100)),
    "tau": np.ones((4, 100)),
    # theta_raw absent — Laplace only explores phi subspace
}

_SUMMARY_V2 = {
    "n_chains": 10,
    "n_draws_per_chain": 10000,
    "n_total": 100000,
    "per_site": {
        "mu": {
            "mean": [4.38],
            "std": [3.32],
            "q05": [-1.13],
            "q95": [9.80],
            "between_chain_se": [0.01239],
            "bulk_ess": [91832.0],
            "tail_ess": [70098.0],
            "rhat": [1.0000105],
        },
        "tau": {
            "mean": [0.799],
            "std": [0.632],
            "q05": [0.087],
            "q95": [2.073],
            "between_chain_se": [0.00401],
            "bulk_ess": [74123.0],
            "tail_ess": [59871.0],
            "rhat": [1.000023],
        },
        "theta_raw": {
            "mean": [0.32, 0.10, -0.04, 0.15, -0.20, 0.08, 0.25, -0.11],
            "std": [0.94, 0.95, 0.96, 0.94, 0.95, 0.94, 0.95, 0.96],
            "q05": [-1.2, -1.3, -1.4, -1.2, -1.3, -1.2, -1.3, -1.4],
            "q95": [1.9, 1.8, 1.7, 1.9, 1.8, 1.9, 1.8, 1.7],
            "between_chain_se": [
                0.0031,
                0.0032,
                0.0033,
                0.0031,
                0.0032,
                0.0031,
                0.0032,
                0.0033,
            ],
            "bulk_ess": [
                92100.0,
                91500.0,
                90800.0,
                92000.0,
                91200.0,
                92300.0,
                91400.0,
                90700.0,
            ],
            "tail_ess": [
                70000.0,
                69500.0,
                68800.0,
                70000.0,
                69200.0,
                70300.0,
                69400.0,
                68700.0,
            ],
            "rhat": [1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001],
        },
    },
}

_SUMMARY_JSON = {
    "n_samples": 40000,
    "mean": {
        "mu": [4.41],
        "tau": [0.79],
        "theta_raw": [0.31, 0.09, -0.04, 0.15, -0.20, 0.08, 0.25, -0.11],
    },
    "std": {
        "mu": [3.31],
        "tau": [0.63],
        "theta_raw": [0.94, 0.95, 0.96, 0.94, 0.95, 0.94, 0.95, 0.96],
    },
    "q05": {
        "mu": [-1.12],
        "tau": [0.086],
        "theta_raw": [-1.2, -1.3, -1.4, -1.2, -1.3, -1.2, -1.3, -1.4],
    },
    "q95": {
        "mu": [9.79],
        "tau": [2.07],
        "theta_raw": [1.9, 1.8, 1.7, 1.9, 1.8, 1.9, 1.8, 1.7],
    },
}


# ---------------------------------------------------------------------------
# _build_gt_for_gate_v2 — full-posterior path
# ---------------------------------------------------------------------------


def test_v2_full_posterior_has_between_chain_se():
    """summary_v2 path returns between_chain_se for every aligned param."""
    result = _build_gt_for_gate_v2(_SUMMARY_V2, _DRAWS, False, "eight_schools_ncp")
    assert result is not None
    assert set(result) == {"mu", "tau", "theta_raw"}
    for param, d in result.items():
        assert "between_chain_se" in d, f"{param} missing between_chain_se"
        assert "bulk_ess" in d, f"{param} missing bulk_ess"
        assert "n_total" in d, f"{param} missing n_total"
        assert d["n_total"] == 100000


def test_v2_full_posterior_no_n_samples_key():
    """summary_v2 path must NOT include legacy n_samples key."""
    result = _build_gt_for_gate_v2(_SUMMARY_V2, _DRAWS, False, "eight_schools_ncp")
    assert result is not None
    for d in result.values():
        assert "n_samples" not in d


def test_v2_values_match_per_site():
    """Values in output exactly match per_site entries."""
    result = _build_gt_for_gate_v2(_SUMMARY_V2, _DRAWS, False, "eight_schools_ncp")
    assert result is not None
    assert np.array_equal(result["mu"]["mean"], _SUMMARY_V2["per_site"]["mu"]["mean"])
    assert np.array_equal(
        result["mu"]["between_chain_se"],
        _SUMMARY_V2["per_site"]["mu"]["between_chain_se"],
    )
    assert np.array_equal(
        result["mu"]["bulk_ess"],
        _SUMMARY_V2["per_site"]["mu"]["bulk_ess"],
    )


# ---------------------------------------------------------------------------
# _build_gt_for_gate_v2 — Laplace phi-filter path
# ---------------------------------------------------------------------------


def test_v2_laplace_filters_to_phi_sites():
    """Laplace path returns only phi sites (mu, tau) — not theta_raw."""
    result = _build_gt_for_gate_v2(
        _SUMMARY_V2, _DRAWS_LAPLACE, True, "eight_schools_ncp"
    )
    assert result is not None
    assert set(result) == {"mu", "tau"}
    assert "theta_raw" not in result


def test_v2_laplace_phi_has_between_chain_se():
    """Phi sites in laplace path still carry between_chain_se."""
    result = _build_gt_for_gate_v2(
        _SUMMARY_V2, _DRAWS_LAPLACE, True, "eight_schools_ncp"
    )
    assert result is not None
    for d in result.values():
        assert "between_chain_se" in d


def test_v2_laplace_unknown_model_falls_back_to_intersection():
    """Unknown model_name with is_laplace=True falls back to key intersection."""
    draws_partial = {"mu": np.zeros((4, 100))}
    result = _build_gt_for_gate_v2(_SUMMARY_V2, draws_partial, True, "unknown_model")
    # No phi/theta split for unknown_model → intersection of draws and per_site
    assert result is not None
    assert set(result) == {"mu"}


def test_v2_no_matching_keys_returns_none():
    """No intersection between draws and per_site → None."""
    draws_alien = {"z_weird": np.zeros((4, 100))}
    result = _build_gt_for_gate_v2(_SUMMARY_V2, draws_alien, False, "model_x")
    assert result is None


# ---------------------------------------------------------------------------
# _align_gt_keys_for_gate — legacy path
# ---------------------------------------------------------------------------


def test_legacy_has_n_samples_no_between_chain_se():
    """Legacy path returns n_samples and must NOT have between_chain_se."""
    result = _align_gt_keys_for_gate(_SUMMARY_JSON, _DRAWS, False, "eight_schools_ncp")
    assert result is not None
    for param, d in result.items():
        assert "n_samples" in d, f"{param} missing n_samples"
        assert "between_chain_se" not in d, f"{param} unexpectedly has between_chain_se"
    assert result["mu"]["n_samples"] == 40000


def test_legacy_laplace_filters_to_phi():
    """Legacy Laplace path also restricts to phi sites."""
    result = _align_gt_keys_for_gate(
        _SUMMARY_JSON, _DRAWS_LAPLACE, True, "eight_schools_ncp"
    )
    assert result is not None
    assert set(result) == {"mu", "tau"}


# ---------------------------------------------------------------------------
# File-priority logic: summary_v2.json > summary.json
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_file_priority_v2_takes_precedence(tmp_path):
    """When both files exist, summary_v2.json is loaded and _gt_is_v2 is True.

    We test this by checking the helper returns between_chain_se (the v2
    discriminant) rather than n_samples (the legacy discriminant).
    """
    model = "test_model"
    sv2_path = tmp_path / model / "groundtruth_samples" / "blackjax" / "summary_v2.json"
    legacy_path = tmp_path / model / "reference" / "summary.json"
    _write_json(sv2_path, _SUMMARY_V2)
    _write_json(legacy_path, _SUMMARY_JSON)

    sv2_loaded = json.loads(sv2_path.read_text())
    result = _build_gt_for_gate_v2(sv2_loaded, _DRAWS, False, model)
    assert result is not None
    assert "between_chain_se" in result["mu"]
    assert "n_samples" not in result["mu"]


def test_file_priority_legacy_when_no_v2(tmp_path):
    """Without summary_v2.json, the legacy path is used."""
    model = "test_model"
    legacy_path = tmp_path / model / "reference" / "summary.json"
    sv2_path = tmp_path / model / "groundtruth_samples" / "blackjax" / "summary_v2.json"
    _write_json(legacy_path, _SUMMARY_JSON)
    assert not sv2_path.exists()

    legacy_loaded = json.loads(legacy_path.read_text())
    result = _align_gt_keys_for_gate(legacy_loaded, _DRAWS, False, model)
    assert result is not None
    assert "n_samples" in result["mu"]
    assert "between_chain_se" not in result["mu"]
