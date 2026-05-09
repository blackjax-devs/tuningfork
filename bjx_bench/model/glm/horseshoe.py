"""204-D Finnish horseshoe sparse linear regression — Phase 4 Block-B model #9 (P4.6).

Model class: Finnish (regularised) horseshoe sparse linear regression.
Reference: Piironen & Vehtari (2017) "Sparsity information and regularization in
the horseshoe and other shrinkage priors", Electronic Journal of Statistics.

Statistician verdict (TL-orchestrated, 2026-05-08):
    Approve-with-modifications. Adopt Finnish (regularized) horseshoe per
    Piironen & Vehtari 2017. NCP on beta is mandatory; slab c² reduces (not
    complicates) the tau-funnel pathology empirically. This implementation
    matches `blackjax/tests/test_benchmarks.py::make_horseshoe_logdensity`
    (SHA 2eb62abb), which is a vetted upstream reference.

c2_tilde prior convention:
    c2 = slab_scale² * c2_tilde where c2_tilde ~ InvGamma(slab_df/2, slab_df/2).
    With slab_df=25 → InvGamma(12.5, 12.5). The slab variance caps the effective
    shrinkage factor lam_tilde, preventing total shrinkage of high-signal predictors.

Parameterization (NCP — non-centered on beta):
    alpha ~ Normal(0, 2)
    sigma ~ HalfNormal(2)
    tau_tilde ~ HalfCauchy(1)
    c2_tilde ~ InvGamma(slab_df/2, slab_df/2)
    lambda_ ~ HalfCauchy(1)^M
    beta_tilde ~ Normal(0, 1)^M

    tau0 = m0 / (M - m0) / sqrt(N), with m0=10
    tau = tau0 * sigma * tau_tilde
    c2 = slab_scale^2 * c2_tilde, with slab_scale=3.0
    lam_tilde = sqrt(c2 * lambda^2 / (c2 + tau^2 * lambda^2))
    beta = tau * lam_tilde * beta_tilde   (NCP)

Unconstrained dimensionality:
    alpha (1) + sigma (1) + tau_tilde (1) + c2_tilde (1) + lambda_ (100) + beta_tilde (100) = 204.

Synthetic data (seed=42, matches upstream BlackJAX reference):
    N=200 observations, M=100 features.
    Sparsity: 5% Bernoulli (~5 active features).
    Nonzero coefficients ~ N(10, 1) — large signal pierces regularised shrinkage.
    sigma_obs = 1.0.

Tier-A budget:
    In-spawn verification: n_warmup=2000, n_samples=2000.
    Production cache: n_warmup=2000, n_samples=20000.

Discrimination claim:
    Extends the GLM discrimination ladder:
    logistic_synthetic (3-D baseline) → german_credit (26-D, real data) →
    horseshoe (204-D, sparse). Tests sampler performance on the funnel-like
    tau geometry and high-dimensional NCP structure.

References:
    Piironen, J. & Vehtari, A. (2017). Sparsity information and regularization
    in the horseshoe and other shrinkage priors.
    Electronic Journal of Statistics, 11(2), 5018-5051.

    BlackJAX upstream reference: blackjax/tests/test_benchmarks.py @ 2eb62abb
    (function make_horseshoe_logdensity).

    PLAN_bjx_bench_phase4.md § "Block B", row P4.6 (horseshoe).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from bjx_bench.model._base import Posterior

__all__ = ["ENTRY", "X_DATA", "Y_DATA", "BETA_TRUE"]

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

#: Number of observations
N: int = 200

#: Number of features
M: int = 100

#: Sparsity prior expected number of active features
M0: float = 10.0

#: Slab scale (Finnish horseshoe)
SLAB_SCALE: float = 3.0

#: Slab degrees of freedom
SLAB_DF: float = 25.0

#: Unconstrained dimensionality:
#: alpha(1) + sigma(1) + tau_tilde(1) + c2_tilde(1) + lambda_(M) + beta_tilde(M) = 2*M + 4
DIM: int = 2 * M + 4  # = 204

# ---------------------------------------------------------------------------
# Generate synthetic dataset at import time (seed=42, matches upstream)
# ---------------------------------------------------------------------------

_rng = np.random.default_rng(42)

X_DATA: np.ndarray = _rng.standard_normal((N, M)).astype(np.float32)

BETA_TRUE: np.ndarray = np.zeros(M, dtype=np.float32)
_active: np.ndarray = _rng.binomial(1, 0.05, M).astype(bool)
BETA_TRUE[_active] = (_rng.standard_normal(_active.sum()) + 10.0).astype(np.float32)

Y_DATA: np.ndarray = _rng.normal(X_DATA @ BETA_TRUE, 1.0).astype(np.float32)

# Convert to JAX arrays for model use
_X_JAX = jnp.array(X_DATA)
_Y_JAX = jnp.array(Y_DATA)

# ---------------------------------------------------------------------------
# NumPyro model (Finnish horseshoe, NCP on beta)
# ---------------------------------------------------------------------------


def horseshoe_regression(
    X: jnp.ndarray,
    y: jnp.ndarray,
    m0: float = M0,
    slab_scale: float = SLAB_SCALE,
    slab_df: float = SLAB_DF,
) -> None:
    """NumPyro model: Finnish horseshoe sparse linear regression.

    Parameters
    ----------
    X
        Feature matrix of shape (N, M).
    y
        Response vector of shape (N,).
    m0
        Prior expected number of active features.
    slab_scale
        Slab scale parameter (c = slab_scale, c2 = slab_scale^2 * c2_tilde).
    slab_df
        Slab degrees of freedom; c2_tilde ~ InvGamma(slab_df/2, slab_df/2).
    """
    n_obs, n_features = X.shape

    # Global parameters
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 2.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))

    # Global shrinkage (Finnish horseshoe)
    tau_tilde = numpyro.sample("tau_tilde", dist.HalfCauchy(1.0))
    tau0 = (m0 / (n_features - m0)) / jnp.sqrt(n_obs)
    tau = tau0 * sigma * tau_tilde

    # Slab regularisation
    half_slab_df = 0.5 * slab_df
    c2_tilde = numpyro.sample(
        "c2_tilde",
        dist.InverseGamma(half_slab_df, half_slab_df),
    )
    c2 = slab_scale**2 * c2_tilde

    # Local shrinkage (NCP on beta)
    lambda_ = numpyro.sample("lambda_", dist.HalfCauchy(jnp.ones(n_features)))
    lam_tilde = jnp.sqrt(
        c2 * jnp.square(lambda_) / (c2 + jnp.square(tau) * jnp.square(lambda_))
    )

    # Non-centred beta
    beta_tilde = numpyro.sample("beta_tilde", dist.Normal(jnp.zeros(n_features), 1.0))
    beta = tau * lam_tilde * beta_tilde

    # Likelihood
    mu = X @ beta + alpha
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

ENTRY = Posterior(
    name="horseshoe",
    dim=DIM,
    class_="glm",
    tags=(
        "glm",
        "regression",
        "sparse",
        "horseshoe",
        "real_dim_high",
        "ncp",
        "funnel",
    ),
    numpyro_model=horseshoe_regression,
    model_args=(_X_JAX, _Y_JAX),
    description=(
        "204-D Finnish (regularised) horseshoe sparse linear regression "
        "(N=200, M=100, 5% sparsity). "
        "Priors: alpha~N(0,2), sigma~HN(2), tau_tilde~HC(1), "
        "c2_tilde~InvGamma(12.5,12.5), lambda_~HC(1)^M, beta_tilde~N(0,1)^M. "
        "NCP on beta. Matches BlackJAX upstream test_benchmarks @ 2eb62abb. "
        "Discriminates samplers on tau-funnel geometry at high dim."
    ),
)
