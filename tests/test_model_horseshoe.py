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
"""Tests for the horseshoe model (204-D Finnish horseshoe sparse linear regression).

Tests
-----
1. test_dim             : MODELS['horseshoe'].dim == 204
2. test_data_shape      : X_DATA.shape == (200, 100), Y_DATA.shape == (200,)
3. test_sparsity        : roughly 5% active in BETA_TRUE (allow 2-10 active)
4. test_logdensity_finite: logdensity_fn returns finite float at zeros init
5. test_logdensity_matches_blackjax_reference: cross-check vs inlined
   make_horseshoe_logdensity from blackjax/tests/test_benchmarks.py @ 2eb62abb.

Notes
-----
All tests are @pytest.mark.fast (no MCMC).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import pytest

from bjx_bench.model import MODELS, build_logdensity_fn
from bjx_bench.model.glm.horseshoe import (
    _X_JAX,
    _Y_JAX,
    BETA_TRUE,
    DIM,
    ENTRY,
    X_DATA,
    Y_DATA,
    M,
    N,
    horseshoe_regression,
)

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Inlined reference: make_horseshoe_logdensity from blackjax/tests/test_benchmarks.py
# @ SHA 2eb62abb (imported here because blackjax.tests is not on the dep path).
# ---------------------------------------------------------------------------


def _make_horseshoe_logdensity_reference(
    N: int = 100,
    M: int = 200,
    m0: int = 10,
    slab_scale: float = 3.0,
    slab_df: float = 25.0,
    seed: int = 42,
):
    """Finnish (regularised) horseshoe sparse linear regression.

    Inlined from blackjax/tests/test_benchmarks.py @ 2eb62abb.
    Pure JAX implementation; positive params use log transform.

    Returns (logdensity_flat, init_flat, logdensity_dict, init_dict).
    """
    rng = np.random.default_rng(seed)
    X = jnp.array(rng.standard_normal((N, M)), dtype=jnp.float32)
    beta0 = np.zeros(M, dtype=np.float32)
    active = rng.binomial(1, 0.05, M).astype(bool)
    beta0[active] = (rng.standard_normal(active.sum()) + 10).astype(np.float32)
    y = jnp.array(rng.normal(np.array(X) @ beta0, 1.0), dtype=jnp.float32)

    half_slab_df = float(0.5 * slab_df)
    slab_scale2 = float(slab_scale**2)
    tau0_coef = float(m0 / (M - m0)) / float(N) ** 0.5

    def logdensity_dict(params):
        alpha = params["alpha"]
        log_sigma = params["sigma"]
        log_tau = params["tau_tilde"]
        log_c2 = params["c2_tilde"]
        log_lam = params["lambda_"]
        beta_tilde = params["beta_tilde"]

        sigma = jnp.exp(log_sigma)
        tau_tilde = jnp.exp(log_tau)
        c2_tilde = jnp.exp(log_c2)
        lambda_ = jnp.exp(log_lam)

        tau = tau0_coef * sigma * tau_tilde
        c2 = slab_scale2 * c2_tilde
        lam_tilde = jnp.sqrt(
            c2 * jnp.square(lambda_) / (c2 + jnp.square(tau) * jnp.square(lambda_))
        )
        beta = tau * lam_tilde * beta_tilde
        mu = X @ beta + alpha

        lp = stats.norm.logpdf(alpha, 0.0, 2.0)
        lp += jnp.log(2.0) + stats.norm.logpdf(sigma, 0.0, 2.0)
        lp += jnp.log(2.0) - jnp.log(jnp.pi) - jnp.log1p(tau_tilde**2)
        lp += (
            half_slab_df * jnp.log(half_slab_df)
            - jax.scipy.special.gammaln(half_slab_df)
            - (half_slab_df + 1.0) * jnp.log(c2_tilde)
            - half_slab_df / c2_tilde
        )
        lp += jnp.sum(jnp.log(2.0) - jnp.log(jnp.pi) - jnp.log1p(lambda_**2))
        lp += jnp.sum(stats.norm.logpdf(beta_tilde, 0.0, 1.0))
        lp += jnp.sum(stats.norm.logpdf(y, mu, sigma))
        lp += log_sigma + log_tau + log_c2 + jnp.sum(log_lam)

        return lp

    init_dict = {
        "alpha": jnp.array(0.0),
        "sigma": jnp.array(0.0),
        "tau_tilde": jnp.array(0.0),
        "c2_tilde": jnp.array(0.0),
        "lambda_": jnp.zeros(M),
        "beta_tilde": jnp.zeros(M),
    }

    init_flat, unflatten = jax.flatten_util.ravel_pytree(init_dict)

    def logdensity_flat(flat):
        return logdensity_dict(unflatten(flat))

    return logdensity_flat, init_flat, logdensity_dict, init_dict


# ---------------------------------------------------------------------------
# Test 1: dim
# ---------------------------------------------------------------------------


def test_dim() -> None:
    """MODELS['horseshoe'].dim must equal 204."""
    assert MODELS["horseshoe"].dim == 204
    assert ENTRY.dim == DIM == 204


# ---------------------------------------------------------------------------
# Test 2: data shape
# ---------------------------------------------------------------------------


def test_data_shape() -> None:
    """X_DATA must be (200, 100); Y_DATA must be (200,)."""
    assert X_DATA.shape == (N, M), f"Expected ({N}, {M}), got {X_DATA.shape}"
    assert Y_DATA.shape == (N,), f"Expected ({N},), got {Y_DATA.shape}"


# ---------------------------------------------------------------------------
# Test 3: sparsity of BETA_TRUE
# ---------------------------------------------------------------------------


def test_sparsity() -> None:
    """BETA_TRUE should have roughly 5% active features (2-10 nonzero allowed)."""
    n_active = int(np.count_nonzero(BETA_TRUE))
    assert 2 <= n_active <= 15, (
        f"Expected 2-15 active features (5% Bernoulli of 100), got {n_active}. "
        "If Bernoulli noise produced an outlier, widen the bound or re-check seed."
    )


# ---------------------------------------------------------------------------
# Test 4: logdensity finite at zeros init
# ---------------------------------------------------------------------------


def test_logdensity_finite() -> None:
    """build_logdensity_fn must return finite log-density at the zeros init."""
    key = jax.random.key(0)
    init_pos, logdensity_fn, _ = build_logdensity_fn(key, ENTRY)
    ld = logdensity_fn(init_pos)
    assert jnp.isfinite(ld), f"Expected finite log-density at init, got {ld}"


# ---------------------------------------------------------------------------
# Test 5: cross-check against inlined BlackJAX reference
# ---------------------------------------------------------------------------


def test_logdensity_matches_blackjax_reference() -> None:
    """NumPyro log-joint must match inlined BlackJAX reference at atol=1e-4.

    The reference (make_horseshoe_logdensity from blackjax/tests/test_benchmarks.py
    @ 2eb62abb) operates in log-transformed unconstrained space with explicit Jacobians.

    The NumPyro model computes the same joint in constrained space. To compare
    fairly, we evaluate both at the SAME constrained point:
        alpha=0, sigma=1, tau_tilde=1, c2_tilde=1, lambda_=ones(M), beta_tilde=zeros(M).
    This corresponds to the reference's all-zeros unconstrained init (where each
    positive param is stored as log(1)=0).

    For NumPyro we use ``numpyro.infer.util.log_density``, which accepts constrained
    params and returns the log-joint WITHOUT Jacobian corrections (pure prior + lik).

    For the reference we use ``logdensity_dict(init_dict)`` which evaluates the
    log-joint in log-transformed space (positive params = exp(log_params)) and
    includes explicit Jacobian corrections for the log transforms. We then subtract
    the Jacobian contribution to get the pure log-joint for comparison.

    Both implementations use the same synthetic data (seed=42, N=200, M=100).
    """
    from numpyro.infer.util import log_density as numpyro_log_density

    # Reference implementation with our hyperparams and seed
    _logdensity_flat, _init_flat, logdensity_dict_ref, init_dict_ref = (
        _make_horseshoe_logdensity_reference(
            N=200,
            M=100,
            m0=10,
            slab_scale=3.0,
            slab_df=25.0,
            seed=42,
        )
    )

    # The constrained init point: alpha=0, sigma=1, tau_tilde=1, c2_tilde=1,
    # lambda_=ones(M), beta_tilde=zeros(M).
    # In the reference's unconstrained space: log(1)=0 for positive params.
    # init_dict_ref is already this point: {"alpha": 0, "sigma": 0, "tau_tilde": 0,
    # "c2_tilde": 0, "lambda_": zeros(M), "beta_tilde": zeros(M)}.
    ld_ref_unconstrained = float(logdensity_dict_ref(init_dict_ref))

    # The Jacobian correction in the reference is:
    #   log_sigma + log_tau + log_c2 + sum(log_lam)
    # At init_dict_ref: all log params = 0 → Jacobian = 0 + 0 + 0 + 0 = 0.
    # So ld_ref_unconstrained == pure log-joint at sigma=1, tau_tilde=1, c2_tilde=1, lambda=ones.
    # (No Jacobian subtraction needed at the zeros unconstrained point.)

    # NumPyro: evaluate log-joint at the same constrained point.
    constrained_point = {
        "alpha": jnp.array(0.0),
        "sigma": jnp.array(1.0),
        "tau_tilde": jnp.array(1.0),
        "c2_tilde": jnp.array(1.0),
        "lambda_": jnp.ones(M),
        "beta_tilde": jnp.zeros(M),
    }
    ld_numpyro_joint, _ = numpyro_log_density(
        horseshoe_regression,
        model_args=(_X_JAX, _Y_JAX),
        model_kwargs={},
        params=constrained_point,
    )
    ld_numpyro = float(ld_numpyro_joint)

    assert np.isfinite(ld_numpyro), f"NumPyro log-joint not finite: {ld_numpyro}"
    assert np.isfinite(
        ld_ref_unconstrained
    ), f"Reference log-joint not finite: {ld_ref_unconstrained}"

    assert np.isclose(ld_numpyro, ld_ref_unconstrained, atol=1e-4), (
        f"NumPyro log-joint ({ld_numpyro:.6f}) differs from BlackJAX reference "
        f"({ld_ref_unconstrained:.6f}) by "
        f"{abs(ld_numpyro - ld_ref_unconstrained):.2e} (atol=1e-4). "
        "Check prior parameterization, likelihood, or data mismatch."
    )
