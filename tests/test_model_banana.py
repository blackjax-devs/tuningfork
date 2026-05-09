"""Tests for the banana model (2-D banana-shaped / Rosenbrock-style distribution).

Tests
-----
1.  ENTRY registered in MODELS with name "banana".
2.  Posterior dim == 2, class_ == "pathological", tags contain "curved-manifold".
3.  analytic_sampler returns dict with keys {"x1", "x2"}, each shape (n,).
4.  Empirical mean of x1 ≈ 0 within 4σ MC SE at n=10_000.
5.  Empirical std of x1 ≈ 2√2 ≈ 2.828 within 4σ MC SE at n=10_000.
6.  Empirical mean(x2 - x1²/4) ≈ 0 within 4σ — verifies the conditional mean.
7.  Empirical std(x2 - x1²/4) ≈ 1 within 4σ — verifies the conditional std.
8.  build_logdensity_fn returns finite log-density at init position.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bjx_bench.model import MODELS, ReferenceMethod, build_logdensity_fn
from bjx_bench.model.pathological.banana import DIM, ENTRY, SIGMA_X1

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_SHAPE = 100
N_MOMENTS = 10_000

# Expected values derived from the analytic distribution.
_X1_MEAN_EXPECTED = 0.0
_X1_STD_EXPECTED = SIGMA_X1  # = 2√2 ≈ 2.8284
_RESIDUAL_MEAN_EXPECTED = 0.0  # E[x2 - x1²/4] = 0
_RESIDUAL_STD_EXPECTED = 1.0  # Std[x2 - x1²/4] = 1

# Number of standard deviations for acceptance (4-sigma is very loose at n=10k).
_NSIGMA = 4.0


# ---------------------------------------------------------------------------
# Test 1: registration
# ---------------------------------------------------------------------------


def test_registered_in_models() -> None:
    """banana must appear in MODELS under the correct name."""
    assert "banana" in MODELS, "banana not found in MODELS"
    assert MODELS["banana"] is ENTRY


# ---------------------------------------------------------------------------
# Test 2: schema — dim, class_, tags
# ---------------------------------------------------------------------------


def test_dim_class_tags() -> None:
    """Posterior dim must be 2, class_ 'pathological', tags include 'curved-manifold'."""
    assert ENTRY.dim == DIM
    assert ENTRY.dim == 2
    assert ENTRY.class_ == "pathological"
    assert "curved-manifold" in ENTRY.tags
    assert "pathological" in ENTRY.tags
    assert "low-dim" in ENTRY.tags


# ---------------------------------------------------------------------------
# Test 3: analytic_sampler shape
# ---------------------------------------------------------------------------


def test_analytic_sampler_shape() -> None:
    """analytic_sampler(key, n) must return dict with keys 'x1', 'x2', each (n,)."""
    key = jax.random.key(0)
    assert ENTRY.analytic_sampler is not None
    draws = ENTRY.analytic_sampler(key, N_SHAPE)
    assert set(draws.keys()) == {
        "x1",
        "x2",
    }, f"Expected keys {{'x1', 'x2'}}, got {set(draws.keys())}"
    assert draws["x1"].shape == (
        N_SHAPE,
    ), f"Expected x1 shape ({N_SHAPE},), got {draws['x1'].shape}"
    assert draws["x2"].shape == (
        N_SHAPE,
    ), f"Expected x2 shape ({N_SHAPE},), got {draws['x2'].shape}"


# ---------------------------------------------------------------------------
# Test 4: empirical mean of x1 ≈ 0 within 4σ MC SE
# ---------------------------------------------------------------------------


def test_x1_mean_near_zero() -> None:
    """Empirical mean of x1 must be ≈ 0 within 4-sigma MC SE at n=10_000."""
    key = jax.random.key(1)
    assert ENTRY.analytic_sampler is not None
    draws = ENTRY.analytic_sampler(key, N_MOMENTS)
    x1 = np.asarray(draws["x1"])

    empirical_mean = float(x1.mean())
    mc_se = float(x1.std()) / math.sqrt(N_MOMENTS)
    tol = _NSIGMA * mc_se

    assert (
        abs(empirical_mean - _X1_MEAN_EXPECTED) < tol
    ), f"x1 empirical mean {empirical_mean:.4f} exceeds {_NSIGMA}σ tol {tol:.4f}"


# ---------------------------------------------------------------------------
# Test 5: empirical std of x1 ≈ 2√2 within 4σ MC SE
# ---------------------------------------------------------------------------


def test_x1_std_near_sigma() -> None:
    """Empirical std of x1 must be ≈ 2√2 ≈ 2.828 within 4-sigma MC SE at n=10_000."""
    key = jax.random.key(2)
    assert ENTRY.analytic_sampler is not None
    draws = ENTRY.analytic_sampler(key, N_MOMENTS)
    x1 = np.asarray(draws["x1"])

    empirical_std = float(x1.std())
    # MC SE of the sample std estimator ≈ true_std / sqrt(2 * (n - 1))
    mc_se = _X1_STD_EXPECTED / math.sqrt(2 * (N_MOMENTS - 1))
    tol = _NSIGMA * mc_se

    assert abs(empirical_std - _X1_STD_EXPECTED) < tol, (
        f"x1 empirical std {empirical_std:.4f} is not within {_NSIGMA}σ of "
        f"expected {_X1_STD_EXPECTED:.4f} (tol={tol:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 6: empirical mean(x2 - x1²/4) ≈ 0 within 4σ
# ---------------------------------------------------------------------------


def test_conditional_mean_near_zero() -> None:
    """mean(x2 - x1²/4) must be ≈ 0 within 4-sigma MC SE — verifies conditional mean."""
    key = jax.random.key(3)
    assert ENTRY.analytic_sampler is not None
    draws = ENTRY.analytic_sampler(key, N_MOMENTS)
    x1 = np.asarray(draws["x1"])
    x2 = np.asarray(draws["x2"])

    residual = x2 - x1**2 / 4.0
    empirical_mean = float(residual.mean())
    mc_se = float(residual.std()) / math.sqrt(N_MOMENTS)
    tol = _NSIGMA * mc_se

    assert (
        abs(empirical_mean - _RESIDUAL_MEAN_EXPECTED) < tol
    ), f"Conditional mean residual {empirical_mean:.4f} exceeds {_NSIGMA}σ tol {tol:.4f}"


# ---------------------------------------------------------------------------
# Test 7: empirical std(x2 - x1²/4) ≈ 1 within 4σ
# ---------------------------------------------------------------------------


def test_conditional_std_near_one() -> None:
    """std(x2 - x1²/4) must be ≈ 1 within 4-sigma MC SE — verifies conditional std."""
    key = jax.random.key(4)
    assert ENTRY.analytic_sampler is not None
    draws = ENTRY.analytic_sampler(key, N_MOMENTS)
    x1 = np.asarray(draws["x1"])
    x2 = np.asarray(draws["x2"])

    residual = x2 - x1**2 / 4.0
    empirical_std = float(residual.std())
    # MC SE of sample std estimator ≈ true_std / sqrt(2(n-1)) = 1 / sqrt(2(n-1))
    mc_se = _RESIDUAL_STD_EXPECTED / math.sqrt(2 * (N_MOMENTS - 1))
    tol = _NSIGMA * mc_se

    assert abs(empirical_std - _RESIDUAL_STD_EXPECTED) < tol, (
        f"Conditional std residual {empirical_std:.4f} is not within {_NSIGMA}σ of "
        f"expected {_RESIDUAL_STD_EXPECTED:.4f} (tol={tol:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 8: build_logdensity_fn returns finite log-density at init position
# ---------------------------------------------------------------------------


def test_build_logdensity_fn_finite() -> None:
    """build_logdensity_fn must return a finite log-density at the init position."""
    key = jax.random.key(5)
    init_pos, logdensity_fn, _ = build_logdensity_fn(key, ENTRY)
    ld = logdensity_fn(init_pos)
    assert jnp.isfinite(ld), f"Expected finite log-density, got {ld}"


# ---------------------------------------------------------------------------
# Bonus: reference_method is ANALYTIC (follows from having analytic_sampler)
# ---------------------------------------------------------------------------


def test_reference_method_analytic() -> None:
    """Posterior with analytic_sampler must report ReferenceMethod.ANALYTIC."""
    assert ENTRY.reference_method == ReferenceMethod.ANALYTIC
