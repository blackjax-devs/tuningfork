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
"""Tests for the declarative Laplace phi/theta split configuration."""

import pytest

from tuningfork.recipes._laplace_config import LAPLACE_PHI_THETA_SPLITS

# ---------------------------------------------------------------------------
# 1. Fast: schema/table correctness
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_gp_regression_in_splits_table() -> None:
    """gp_regression is registered in _LAPLACE_PHI_THETA_SPLITS."""
    assert (
        "gp_regression" in LAPLACE_PHI_THETA_SPLITS
    ), "_LAPLACE_PHI_THETA_SPLITS missing 'gp_regression' entry"


@pytest.mark.fast
def test_gp_regression_phi_sites() -> None:
    """gp_regression phi sites are the 3 log-scale hyperparameters."""
    phi_sites, _ = LAPLACE_PHI_THETA_SPLITS["gp_regression"]
    expected = {"log_lengthscale", "log_kernel_scale", "log_noise_scale"}
    assert (
        set(phi_sites) == expected
    ), f"phi sites mismatch: expected {expected}, got {set(phi_sites)}"


@pytest.mark.fast
def test_gp_regression_theta_sites() -> None:
    """gp_regression theta sites is ('f_raw',) — the NCP base variable."""
    _, theta_sites = LAPLACE_PHI_THETA_SPLITS["gp_regression"]
    assert theta_sites == (
        "f_raw",
    ), f"theta sites mismatch: expected ('f_raw',), got {theta_sites}"
