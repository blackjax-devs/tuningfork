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
"""Tests for the _LAPLACE_PHI_THETA_SPLITS table in _recipe_runner.py.

Covers:
  1. (fast) gp_regression is in the table with correct phi/theta site names.
  2. (slow) _build_laplace_components correctly constructs the Laplace
     pipeline from the gp_regression model: phi_init, log_joint_fn,
     theta_init, and marginal_logdensity_fn are all finite and callable.
"""

import jax
import jax.numpy as jnp
import pytest

from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.recipes._recipe_runner import (
    _LAPLACE_PHI_THETA_SPLITS,
    _build_laplace_components,
)

# ---------------------------------------------------------------------------
# 1. Fast: schema/table correctness
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_gp_regression_in_splits_table() -> None:
    """gp_regression is registered in _LAPLACE_PHI_THETA_SPLITS."""
    assert (
        "gp_regression" in _LAPLACE_PHI_THETA_SPLITS
    ), "_LAPLACE_PHI_THETA_SPLITS missing 'gp_regression' entry"


@pytest.mark.fast
def test_gp_regression_phi_sites() -> None:
    """gp_regression phi sites are the 3 log-scale hyperparameters."""
    phi_sites, _ = _LAPLACE_PHI_THETA_SPLITS["gp_regression"]
    expected = {"log_lengthscale", "log_kernel_scale", "log_noise_scale"}
    assert (
        set(phi_sites) == expected
    ), f"phi sites mismatch: expected {expected}, got {set(phi_sites)}"


@pytest.mark.fast
def test_gp_regression_theta_sites() -> None:
    """gp_regression theta sites is ('f_raw',) — the NCP base variable."""
    _, theta_sites = _LAPLACE_PHI_THETA_SPLITS["gp_regression"]
    assert theta_sites == (
        "f_raw",
    ), f"theta sites mismatch: expected ('f_raw',), got {theta_sites}"


# ---------------------------------------------------------------------------
# 2. Slow: _build_laplace_components round-trip on gp_regression
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_build_laplace_components_gp_regression() -> None:
    """_build_laplace_components returns finite phi_init, callable log_joint_fn,
    finite theta_init, and callable marginal_logdensity_fn for gp_regression.
    """
    model = MODELS["gp_regression"]
    key = jax.random.key(42)
    full_position, joint_logdensity_fn, _ = build_logdensity_fn(key, model)

    result = _build_laplace_components(
        "gp_regression", full_position, joint_logdensity_fn
    )
    assert result is not None, (
        "_build_laplace_components returned None for gp_regression "
        "(model not in _LAPLACE_PHI_THETA_SPLITS?)"
    )

    phi_init, log_joint_fn, theta_init, marginal_logdensity_fn = result

    # phi_init has the right keys and finite values
    phi_sites, _ = _LAPLACE_PHI_THETA_SPLITS["gp_regression"]
    for site in phi_sites:
        assert site in phi_init, f"phi_init missing site {site!r}"
        assert jnp.isfinite(
            phi_init[site]
        ), f"phi_init[{site!r}] not finite: {phi_init[site]}"

    # theta_init has f_raw and finite values
    assert "f_raw" in theta_init, "theta_init missing 'f_raw' site"
    assert jnp.all(
        jnp.isfinite(theta_init["f_raw"])
    ), "theta_init['f_raw'] has non-finite values"
    assert theta_init["f_raw"].shape == (200,), (
        f"theta_init['f_raw'] shape mismatch: expected (200,), got "
        f"{theta_init['f_raw'].shape}"
    )

    # log_joint_fn is callable and returns a finite scalar
    lp = log_joint_fn(theta_init, phi_init)
    assert jnp.isfinite(lp), f"log_joint_fn at init returns non-finite: {lp}"

    # marginal_logdensity_fn is callable and returns a finite scalar
    lp_marginal = marginal_logdensity_fn(phi_init)
    assert jnp.isfinite(
        lp_marginal
    ), f"marginal_logdensity_fn at phi_init returns non-finite: {lp_marginal}"
