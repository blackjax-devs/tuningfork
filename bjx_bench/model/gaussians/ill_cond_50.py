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
"""50-D ill-conditioned Gaussian — Block-A model with condition number κ(Σ) = 1000."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from bjx_bench.model._base import Posterior

__all__ = ["ENTRY"]

# The covariance is constructed as:
#     Σ = U Λ Uᵀ
# where:
#     Λ = diag(λ₁, …, λ₅₀)  with λᵢ logarithmically spaced from 1 to 1000
#     U = fixed deterministic orthogonal matrix from QR(Gaussian(seed=42))
# This gives condition number κ(Σ) = λ_max / λ_min = 1000.
# The model discriminates sampler families via metric sensitivity:
#   - HMC / NUTS with well-adapted inverse mass matrix (from stan_window) should
#     achieve near-isotropic effective step sizes and high ESS / grad_eval.
#   - RWM / MALA struggle unless the proposal covariance is preconditioned.
#   - MCLMC uses a global L and step_size; the ill-conditioning stress-tests its
#     L-tuning heuristic.
# Analytic preflight:
#   No statistician preflight needed — the distribution is well-studied multivariate
#   normal and the encoding is verified deterministically by the test suite.
DIM = 50

# ---------------------------------------------------------------------------
# Build the fixed covariance matrix at module import time.
# The key insight: U is computed from a fixed seed so the matrix is identical
# across Python sessions and machines (numpy's Random(seed) is deterministic
# and seed-portable since numpy 1.17 with the new Generator interface).
# ---------------------------------------------------------------------------

# Eigenvalues: logarithmically spaced from λ_min=1 to λ_max=1000
# → κ(Σ) = 1000 exactly.
_EIGVALS = np.logspace(0, 3, DIM)  # 50 values from 10^0=1 to 10^3=1000

# Fixed orthogonal matrix U from QR decomposition of a seeded random matrix.
# We use numpy.random.Generator with a fixed seed for reproducibility.
_rng_np = np.random.default_rng(42)
_G = _rng_np.standard_normal((DIM, DIM))
_U, _ = np.linalg.qr(_G)  # _U is DIM×DIM orthogonal

# Covariance: Σ = U Λ Uᵀ
_COV_np = _U @ np.diag(_EIGVALS) @ _U.T
# Symmetrise to eliminate floating-point asymmetry.
_COV_np = (_COV_np + _COV_np.T) / 2.0

# JAX-side constants (converted once; immutable for the lifetime of the module).
COV: jax.Array = jnp.array(_COV_np)

# Expose the numpy-side covariance for determinism tests (float64, no JAX truncation).
COV_NP: np.ndarray = _COV_np

# Cholesky factor for sampling: x = L z, z ~ N(0, I)
# Equivalent to x = U @ sqrt(Λ) @ z since Σ = U Λ Uᵀ.
# We use jnp.linalg.cholesky so sampling is identical to NumPyro's internal path.
CHOL: jax.Array = jnp.linalg.cholesky(COV)


def _model() -> None:
    """NumPyro model: 50-D MVN with ill-conditioned covariance."""
    numpyro.sample(
        "x",
        dist.MultivariateNormal(loc=jnp.zeros(DIM), covariance_matrix=COV),
    )


def _analytic_sampler(rng_key: jax.Array, n: int) -> dict[str, jax.Array]:
    """Return n i.i.d. draws from the ill-conditioned 50-D Gaussian.

    Uses the Cholesky decomposition x = L z, z ~ N(0, I_50) which is
    numerically identical to MultivariateNormal.sample and avoids recomputing
    the decomposition on every call.

    Parameters
    ----------
    rng_key
        JAX random key.
    n
        Number of samples.

    Returns
    -------
    dict with key ``"x"`` and value of shape ``(n, 50)``.
    """
    z = jax.random.normal(rng_key, (n, DIM))
    # x = (CHOL @ z.T).T  — each column of z.T multiplied by L
    x = z @ CHOL.T
    return {"x": x}


ENTRY = Posterior(
    name="ill_cond_50",
    dim=DIM,
    class_="gaussian",
    tags=("gaussian", "ill-conditioned", "high-dim"),
    numpyro_model=_model,
    analytic_sampler=_analytic_sampler,
    description=(
        "50-D MVN with covariance Σ = U diag(λ) Uᵀ where λ is log-spaced "
        "from 1 to 1000 (κ≈1000) and U is a fixed deterministic orthogonal "
        "matrix. Discriminates HMC vs MCLMC vs RWM via metric sensitivity."
    ),
)
