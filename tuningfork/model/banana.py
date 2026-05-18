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
"""2-D banana-shaped (Rosenbrock-style) distribution — Block-A pathological model."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from tuningfork.model._base import Posterior

__all__ = ["ENTRY"]

# The joint density is:
#     x₁ ~ N(0, σ²)            with σ² = 8  (std = 2√2 ≈ 2.83)
#     x₂ | x₁ ~ N(x₁² / 4, 1)
# The joint log-density (dropping constants) is:
#     log p(x₁, x₂) = -x₁² / (2 σ²) - (x₂ - x₁²/4)² / 2
# The level sets are banana-shaped — a curved manifold in 2-D.  This
# distribution is a classic test for samplers' ability to follow nonlinear
# geometry: the posterior mean of x₂ is non-linear in x₁, so naïve
# Gaussian-metric samplers (RWM, unadapted MALA) typically produce highly
# correlated chains.  NUTS / HMC with a diagonal mass matrix adapted from
# warmup handles this better, but the curved manifold still reveals
# step-size sensitivity.
# Analytic preflight:
#   No statistician preflight needed — the conditional structure is
#   analytically tractable and the test suite verifies marginal + conditional
#   moments exactly.
# Reference:
#   The parameterisation follows the standard non-convex Banana test case.
#   The shape is closely related to the
#   Rosenbrock function used as a non-convex benchmark in optimisation.
# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

# Variance (and std) of x1.  Var = 8 → std = 2√2 ≈ 2.8284.
SIGMA_X1: float = 2.0 * float(jnp.sqrt(2.0))  # ≈ 2.8284

DIM = 2


# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------


def _model() -> None:
    """NumPyro model: 2-D banana distribution.

    x1 ~ N(0, 8)
    x2 | x1 ~ N(x1**2 / 4, 1)
    """
    x1 = numpyro.sample("x1", dist.Normal(0.0, SIGMA_X1))
    numpyro.sample("x2", dist.Normal(x1**2 / 4.0, 1.0))


# ---------------------------------------------------------------------------
# Analytic sampler
# ---------------------------------------------------------------------------


def _analytic_sampler(rng_key: jax.Array, n: int) -> dict[str, jax.Array]:
    """Return n i.i.d. draws from the banana distribution.

    Uses ancestral sampling — the joint factorises as p(x1) p(x2|x1):

      1. Sample x1 ~ N(0, σ²) with σ = 2√2.
      2. Sample x2 ~ N(x1² / 4, 1).

    Parameters
    ----------
    rng_key
        JAX random key.
    n
        Number of samples.

    Returns
    -------
    dict with keys ``"x1"`` and ``"x2"``, each of shape ``(n,)``.
    """
    k1, k2 = jax.random.split(rng_key)
    x1 = jax.random.normal(k1, (n,)) * SIGMA_X1
    x2 = x1**2 / 4.0 + jax.random.normal(k2, (n,))
    return {"x1": x1, "x2": x2}


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

ENTRY = Posterior(
    name="banana",
    dim=DIM,
    class_="pathological",
    tags=("pathological", "curved-manifold", "low-dim"),
    numpyro_model=_model,
    analytic_sampler=_analytic_sampler,
    description=(
        "2-D banana-shaped distribution: x1 ~ N(0, 8), x2 ~ N(x1**2/4, 1). "
        "Curved manifold; discriminates MALA vs Barker vs HMC by metric sensitivity."
    ),
    headline_params=None,
    headline_coords=None,
)
