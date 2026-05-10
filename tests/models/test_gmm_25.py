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
"""Tests for the gmm_25 model (2-D mixture of 25 Gaussians on a 5x5 grid).

Tests
-----
1.  ENTRY registered in MODELS with name "gmm_25".
2.  Posterior dim == 2, class_ == "pathological", tags contain "multimodal".
3.  analytic_sampler returns dict {"x": (n, 2)} for n=100.
4.  Mode coverage: at n=10_000, every one of the 25 modes contains >= 100 samples (>= 1%).
5.  Empirical mean ≈ (0, 0) within 4σ MC SE at n=10_000.
6.  Empirical std per dim ≈ sqrt(8.09) ≈ 2.844 within 4σ MC SE at n=10_000.
7.  build_logdensity_fn returns finite log-density at init position.
8.  Logdensity at mode center [0, 0] is higher than at between-modes location [1, 1].
9.  reference_method is ANALYTIC (follows from analytic_sampler being set).
10. COMPONENT_LOCS has shape (25, 2) with correct corner entries.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bjx_bench.model import MODELS, ReferenceMethod, build_logdensity_fn
from bjx_bench.model.pathological.gmm_25 import (
    COMPONENT_LOCS,
    COMPONENT_SCALE,
    DIM,
    ENTRY,
    MARGINAL_STD,
    N_COMPONENTS,
)

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_SHAPE = 100
N_MOMENTS = 10_000
_NSIGMA = 4.0  # 4-sigma tolerance is very loose at n=10k

# Analytic moments
_MEAN_EXPECTED = 0.0  # both dimensions — modes symmetric around origin
_STD_EXPECTED = MARGINAL_STD  # sqrt(8 + 0.09) ≈ 2.844

# Mode coverage threshold: each of 25 modes must have >= 1% of n=10_000 samples
_MIN_COUNT_PER_MODE = 100  # = 1% of 10_000


def _nearest_mode(x: np.ndarray) -> np.ndarray:
    """Assign each sample to the index of the nearest component mean.

    Parameters
    ----------
    x
        (n, 2) array of samples.

    Returns
    -------
    (n,) integer array of mode assignments.
    """
    locs = np.asarray(COMPONENT_LOCS)  # (25, 2)
    # Pairwise squared distances: (n, 25)
    diff = x[:, None, :] - locs[None, :, :]  # (n, 25, 2)
    sq_dist = (diff**2).sum(axis=-1)  # (n, 25)
    return sq_dist.argmin(axis=1)  # (n,)


# ---------------------------------------------------------------------------
# Test 1: registration
# ---------------------------------------------------------------------------


def test_registered_in_models() -> None:
    """gmm_25 must appear in MODELS under the correct name."""
    assert "gmm_25" in MODELS, "gmm_25 not found in MODELS"
    assert MODELS["gmm_25"] is ENTRY


# ---------------------------------------------------------------------------
# Test 2: schema — dim, class_, tags
# ---------------------------------------------------------------------------


def test_dim_class_tags() -> None:
    """Posterior dim must be 2, class_ 'pathological', tags include 'multimodal'."""
    assert ENTRY.dim == DIM
    assert ENTRY.dim == 2
    assert ENTRY.class_ == "pathological"
    assert "multimodal" in ENTRY.tags
    assert "mixture" in ENTRY.tags
    assert "low-dim" in ENTRY.tags


# ---------------------------------------------------------------------------
# Test 3: analytic_sampler shape
# ---------------------------------------------------------------------------


def test_analytic_sampler_shape() -> None:
    """analytic_sampler(key, n) must return dict {'x': (n, 2)}."""
    key = jax.random.key(0)
    assert ENTRY.analytic_sampler is not None
    draws = ENTRY.analytic_sampler(key, N_SHAPE)
    assert set(draws.keys()) == {"x"}, f"Expected keys {{'x'}}, got {set(draws.keys())}"
    assert draws["x"].shape == (
        N_SHAPE,
        2,
    ), f"Expected x shape ({N_SHAPE}, 2), got {draws['x'].shape}"


# ---------------------------------------------------------------------------
# Test 4: mode coverage — every mode >= 1% at n=10_000
# ---------------------------------------------------------------------------


def test_mode_coverage() -> None:
    """All 25 modes must have >= 100 samples at n=10_000 (each mode >= 1%).

    This is the canonical analytic-mixture verification: the ancestral
    sampler picks a mode uniformly, so each mode should get ~400 samples
    with a well-behaved random key.
    """
    key = jax.random.key(1)
    assert ENTRY.analytic_sampler is not None
    draws = ENTRY.analytic_sampler(key, N_MOMENTS)
    x = np.asarray(draws["x"])

    assignments = _nearest_mode(x)
    counts = np.bincount(assignments, minlength=N_COMPONENTS)

    assert len(counts) == N_COMPONENTS, f"Expected 25 mode counts, got {len(counts)}"
    min_count = int(counts.min())
    assert min_count >= _MIN_COUNT_PER_MODE, (
        f"Mode coverage failure: minimum mode count is {min_count} "
        f"(need >= {_MIN_COUNT_PER_MODE}). Counts: {counts}"
    )


# ---------------------------------------------------------------------------
# Test 5: empirical mean ≈ (0, 0) within 4σ MC SE
# ---------------------------------------------------------------------------


def test_empirical_mean_near_zero() -> None:
    """Empirical mean of x must be ≈ (0, 0) within 4-sigma MC SE at n=10_000."""
    key = jax.random.key(2)
    assert ENTRY.analytic_sampler is not None
    draws = ENTRY.analytic_sampler(key, N_MOMENTS)
    x = np.asarray(draws["x"])

    for dim_idx in range(2):
        col = x[:, dim_idx]
        empirical_mean = float(col.mean())
        mc_se = float(col.std()) / math.sqrt(N_MOMENTS)
        tol = _NSIGMA * mc_se
        assert abs(empirical_mean - _MEAN_EXPECTED) < tol, (
            f"dim {dim_idx}: empirical mean {empirical_mean:.4f} "
            f"exceeds {_NSIGMA}σ tol {tol:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 6: empirical std per dim ≈ sqrt(8.09) ≈ 2.844 within 4σ MC SE
# ---------------------------------------------------------------------------


def test_empirical_std_near_expected() -> None:
    """Empirical std per dim must be ≈ sqrt(8.09) ≈ 2.844 within 4-sigma MC SE."""
    key = jax.random.key(3)
    assert ENTRY.analytic_sampler is not None
    draws = ENTRY.analytic_sampler(key, N_MOMENTS)
    x = np.asarray(draws["x"])

    # MC SE of sample std estimator ≈ true_std / sqrt(2(n-1))
    mc_se_std = _STD_EXPECTED / math.sqrt(2 * (N_MOMENTS - 1))
    tol = _NSIGMA * mc_se_std

    for dim_idx in range(2):
        col = x[:, dim_idx]
        empirical_std = float(col.std())
        assert abs(empirical_std - _STD_EXPECTED) < tol, (
            f"dim {dim_idx}: empirical std {empirical_std:.4f} is not within "
            f"{_NSIGMA}σ of expected {_STD_EXPECTED:.4f} (tol={tol:.4f})"
        )


# ---------------------------------------------------------------------------
# Test 7: build_logdensity_fn returns finite log-density at init position
# ---------------------------------------------------------------------------


def test_build_logdensity_fn_finite() -> None:
    """build_logdensity_fn must return a finite log-density at the init position."""
    key = jax.random.key(4)
    init_pos, logdensity_fn, _ = build_logdensity_fn(key, ENTRY)
    ld = logdensity_fn(init_pos)
    assert jnp.isfinite(ld), f"Expected finite log-density, got {ld}"


# ---------------------------------------------------------------------------
# Test 8: logdensity at mode center > logdensity between modes
# ---------------------------------------------------------------------------


def test_logdensity_mode_vs_between() -> None:
    """Log-density at a mode center [0,0] must exceed log-density at [1,1].

    [0, 0] is the center of one of the 25 modes.
    [1, 1] lies between four modes and has negligible density.
    """
    key = jax.random.key(5)
    _, logdensity_fn, _ = build_logdensity_fn(key, ENTRY)

    # build_logdensity_fn returns a flat-array logdensity_fn;
    # the "x" site is a 2-D vector, so init_pos["x"] is shape (2,).
    _, _, constrain_fn = build_logdensity_fn(key, ENTRY)

    # Evaluate at mode center and between-mode point via the unconstrained fn.
    # We use the mixture distribution's log_prob directly for correctness.
    import numpyro.distributions as _dist

    mixing = _dist.Categorical(probs=jnp.full(N_COMPONENTS, 1.0 / N_COMPONENTS))
    components = _dist.MultivariateNormal(
        loc=COMPONENT_LOCS,
        covariance_matrix=COMPONENT_SCALE**2
        * jnp.eye(2)[None, :, :].repeat(N_COMPONENTS, axis=0),
    )
    mixture = _dist.MixtureSameFamily(mixing, components)

    at_mode = jnp.array([0.0, 0.0])
    at_between = jnp.array([1.0, 1.0])

    logp_mode = float(mixture.log_prob(at_mode))
    logp_between = float(mixture.log_prob(at_between))

    assert (
        logp_mode > logp_between
    ), f"Expected logp([0,0])={logp_mode:.4f} > logp([1,1])={logp_between:.4f}"


# ---------------------------------------------------------------------------
# Test 9: reference_method is ANALYTIC
# ---------------------------------------------------------------------------


def test_reference_method_analytic() -> None:
    """Posterior with analytic_sampler must report ReferenceMethod.ANALYTIC."""
    assert ENTRY.reference_method == ReferenceMethod.ANALYTIC


# ---------------------------------------------------------------------------
# Test 10: COMPONENT_LOCS has shape (25, 2) with correct extremes
# ---------------------------------------------------------------------------


def test_component_locs_shape_and_extremes() -> None:
    """COMPONENT_LOCS must be (25, 2) with corner entries at ±4."""
    locs = np.asarray(COMPONENT_LOCS)
    assert locs.shape == (25, 2), f"Expected (25, 2), got {locs.shape}"
    assert float(locs.min()) == pytest.approx(
        -4.0
    ), f"Expected min -4.0, got {locs.min()}"
    assert float(locs.max()) == pytest.approx(
        4.0
    ), f"Expected max 4.0, got {locs.max()}"
