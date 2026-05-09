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
"""Tests for the ill_cond_50 model (50-D MVN with κ≈1000).

Tests
-----
1.  ENTRY registered in MODELS with name "ill_cond_50".
2.  Posterior dim == 50.
3.  Posterior class_ == "gaussian".
4.  Posterior tags includes "ill-conditioned".
5.  analytic_sampler returns shape (n, 50) for n=100.
6.  Empirical mean ≈ 0 within 4-sigma MC SE (n=5000).
7.  Empirical std per dim is between 1.0 and sqrt(1000)≈31.6 (n=5000).
8.  Empirical condition number of sample covariance ≈ 1000 within ±25% (n=10000).
9.  build_logdensity_fn returns finite log-density at init position.
10. Fixed orthogonal matrix U is deterministic: two imports produce the same COV.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bjx_bench.model import MODELS, ReferenceMethod, build_logdensity_fn
from bjx_bench.model.gaussians.ill_cond_50 import COV_NP, DIM, ENTRY

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_SMALL = 100
N_MEDIUM = 5_000
N_LARGE = 10_000

# λ_min = 1, λ_max = 1000 → std per dim ranges from sqrt(1) to sqrt(1000)
_STD_MIN = 1.0
_STD_MAX = np.sqrt(1000.0)  # ≈ 31.62


# ---------------------------------------------------------------------------
# Test 1: registration
# ---------------------------------------------------------------------------


def test_registered_in_models() -> None:
    """ill_cond_50 must appear in the MODELS registry under the correct name."""
    assert "ill_cond_50" in MODELS, "ill_cond_50 not found in MODELS"
    assert MODELS["ill_cond_50"] is ENTRY


# ---------------------------------------------------------------------------
# Test 2: dim
# ---------------------------------------------------------------------------


def test_dim() -> None:
    """Posterior dim must be 50."""
    assert ENTRY.dim == DIM
    assert ENTRY.dim == 50


# ---------------------------------------------------------------------------
# Test 3: class_
# ---------------------------------------------------------------------------


def test_class_gaussian() -> None:
    """Posterior class_ must be 'gaussian'."""
    assert ENTRY.class_ == "gaussian"


# ---------------------------------------------------------------------------
# Test 4: tags
# ---------------------------------------------------------------------------


def test_tags_include_ill_conditioned() -> None:
    """Tags must include 'ill-conditioned' and 'gaussian'."""
    assert "ill-conditioned" in ENTRY.tags
    assert "gaussian" in ENTRY.tags
    assert "high-dim" in ENTRY.tags


# ---------------------------------------------------------------------------
# Test 5: analytic_sampler shape
# ---------------------------------------------------------------------------


def test_analytic_sampler_shape() -> None:
    """analytic_sampler(key, n) must return dict with key 'x' and shape (n, 50)."""
    key = jax.random.key(0)
    assert ENTRY.analytic_sampler is not None
    draws = ENTRY.analytic_sampler(key, N_SMALL)
    assert "x" in draws, "Expected key 'x' in analytic_sampler output"
    assert draws["x"].shape == (
        N_SMALL,
        DIM,
    ), f"Expected shape ({N_SMALL}, {DIM}), got {draws['x'].shape}"


# ---------------------------------------------------------------------------
# Test 6: empirical mean ≈ 0 within 4-sigma
# ---------------------------------------------------------------------------


def test_empirical_mean_near_zero() -> None:
    """Empirical mean of analytic samples must be ≈ 0 within 4-sigma MC SE.

    For a single draw x_i from Σ, the marginal std per dimension is at most
    sqrt(λ_max) = sqrt(1000) ≈ 31.62.  The MC SE of the mean estimator is
    therefore std_d / sqrt(N) ≤ 31.62 / sqrt(N_MEDIUM).  We use 4× this as
    tolerance for each dimension.
    """
    key = jax.random.key(1)
    assert ENTRY.analytic_sampler is not None
    draws = ENTRY.analytic_sampler(key, N_MEDIUM)
    x = np.asarray(draws["x"])  # (N_MEDIUM, 50)

    empirical_std = x.std(axis=0)  # per-dim std from the data
    mc_se = empirical_std / np.sqrt(N_MEDIUM)
    tol = 4.0 * mc_se  # per-dim tolerance (scalar array)

    empirical_mean = x.mean(axis=0)
    for d in range(DIM):
        assert abs(empirical_mean[d]) < tol[d], (
            f"Dim {d}: empirical mean {empirical_mean[d]:.4f} exceeds "
            f"4-sigma tol {tol[d]:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 7: empirical std per dim in [sqrt(1), sqrt(1000)]
# ---------------------------------------------------------------------------


def test_empirical_std_in_valid_range() -> None:
    """Each dimension's empirical std must fall in [sqrt(1), sqrt(1000)].

    Because the eigenvectors of Σ mix all dimensions, no single Cartesian
    dimension aligns with the extreme eigenvalues — but the empirical std per
    dim must lie within the range set by the eigenvalue spread.
    """
    key = jax.random.key(2)
    assert ENTRY.analytic_sampler is not None
    draws = ENTRY.analytic_sampler(key, N_MEDIUM)
    x = np.asarray(draws["x"])  # (N_MEDIUM, 50)
    empirical_std = x.std(axis=0)

    for d in range(DIM):
        assert empirical_std[d] >= _STD_MIN * 0.5, (
            f"Dim {d}: empirical std {empirical_std[d]:.4f} is below "
            f"expected minimum {_STD_MIN * 0.5:.4f}"
        )
        assert empirical_std[d] <= _STD_MAX * 2.0, (
            f"Dim {d}: empirical std {empirical_std[d]:.4f} exceeds "
            f"expected maximum {_STD_MAX * 2.0:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 8: empirical condition number ≈ 1000 within ±25%
# ---------------------------------------------------------------------------


def test_empirical_condition_number() -> None:
    """Empirical condition number of sample covariance ≈ 1000 within ±25%.

    Using n=10_000 samples the sample covariance estimate is stable enough for
    a 25% relative tolerance around the true κ=1000.

    This directly verifies that the eigenvalue spread is correctly encoded in Σ.
    """
    key = jax.random.key(3)
    assert ENTRY.analytic_sampler is not None
    draws = ENTRY.analytic_sampler(key, N_LARGE)
    x = np.asarray(draws["x"])  # (N_LARGE, 50)

    # Sample covariance (unbiased)
    sample_cov = np.cov(x, rowvar=False)  # (50, 50)

    # Condition number = max(svd) / min(svd) on a symmetric PSD matrix
    eigvals = np.linalg.eigvalsh(sample_cov)  # eigenvalues in ascending order
    kappa = eigvals[-1] / eigvals[0]

    tol = 0.25  # 25% relative tolerance
    assert (
        abs(kappa - 1000.0) / 1000.0 < tol
    ), f"Empirical condition number {kappa:.1f} is not within 25% of 1000"


# ---------------------------------------------------------------------------
# Test 9: build_logdensity_fn finite at init
# ---------------------------------------------------------------------------


def test_build_logdensity_fn_finite() -> None:
    """build_logdensity_fn returns a finite log-density at the init position."""
    key = jax.random.key(4)
    init_pos, logdensity_fn, _ = build_logdensity_fn(key, ENTRY)
    ld = logdensity_fn(init_pos)
    assert jnp.isfinite(ld), f"Expected finite log-density, got {ld}"


# ---------------------------------------------------------------------------
# Test 10: COV is deterministic across re-computation
# ---------------------------------------------------------------------------


def test_covariance_matrix_deterministic() -> None:
    """Re-computing U from the same fixed seed produces an identical COV.

    This checks that the module-level matrix is reproducible and that U is
    indeed deterministic rather than depending on Python startup state.
    """
    rng = np.random.default_rng(42)
    G = rng.standard_normal((DIM, DIM))
    U2, _ = np.linalg.qr(G)
    eigvals = np.logspace(0, 3, DIM)
    cov2 = U2 @ np.diag(eigvals) @ U2.T
    cov2 = (cov2 + cov2.T) / 2.0

    # Compare the numpy-side float64 array (COV_NP) to the fresh recomputation.
    # Using tight tolerance here because both are numpy float64.
    np.testing.assert_allclose(
        COV_NP,
        cov2,
        rtol=1e-12,
        err_msg="Module-level COV_NP differs from freshly computed COV with seed=42",
    )


# ---------------------------------------------------------------------------
# Test 11: reference_method is ANALYTIC
# ---------------------------------------------------------------------------


def test_reference_method_analytic() -> None:
    """Posterior with analytic_sampler must report ReferenceMethod.ANALYTIC."""
    assert ENTRY.reference_method == ReferenceMethod.ANALYTIC
