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
"""2-D mixture of 25 isotropic Gaussians on a 5×5 grid — Block-A multimodal pathological model."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from bjx_bench.model._base import Posterior

__all__ = ["ENTRY"]

# The joint density is a mixture of 25 components with equal weights (1/25).
# The component means are arranged on a regular 5×5 grid spanning [-4, 4]^2:
#     μ_k ∈ {−4, −2, 0, 2, 4}² for k = 1, …, 25
# Each component is an isotropic Gaussian:
#     x | k ~ N(μ_k, σ²I)   with σ = 0.3
# The marginal log-density (dropping constants) is:
#     log p(x) = log Σ_k (1/25) N(x; μ_k, σ²I)
# Analytic moments:
#     E[x] = 0 (modes symmetric around origin)
#     Var[x_i] = E[Var[x_i | k]] + Var[E[x_i | k]]
#              = σ² + E[μ_k,i²]
#              = 0.09 + (16 + 4 + 0 + 4 + 16) / 5
#              = 0.09 + 8.0 = 8.09
#     Std[x_i] = √8.09 ≈ 2.844
# With σ = 0.3 the components are well-separated (inter-mode distance = 2,
# within-component 2σ = 0.6 ≪ 2): there is no between-mode overlap, so
# gradient-based MCMC (NUTS/HMC) without tempering will be trapped in a
# single mode.  This discriminates SMC and parallel-tempered methods
# vs vanilla NUTS/HMC.
# Analytic preflight (Block A exception, per PLAN_bjx_bench_phase4.md):
#   No statistician preflight needed — the analytic sampler is
#   exact by construction and the test suite verifies shape, mode coverage,
#   marginal moments, and logdensity ordering.
# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

# Grid axis: 5 evenly spaced points in [-4, 4].
GRID_AXIS: jax.Array = jnp.array([-4.0, -2.0, 0.0, 2.0, 4.0])

# All 25 component means arranged as (25, 2) array.
# meshgrid with indexing="ij" followed by reshape gives row-major enumeration:
# (-4,-4), (-4,-2), ..., (4,4).
COMPONENT_LOCS: jax.Array = jnp.stack(
    jnp.meshgrid(GRID_AXIS, GRID_AXIS, indexing="ij"), axis=-1
).reshape(25, 2)

# Per-component isotropic std (well-separated: inter-mode gap = 2, 2σ = 0.6).
COMPONENT_SCALE: float = 0.3

# Number of mixture components.
N_COMPONENTS: int = 25

# Output dimension.
DIM: int = 2

# Analytic marginal std per dimension: sqrt(grid_var + component_var)
# grid_var = E[μ_k^2] = (16+4+0+4+16)/5 = 8.0
# component_var = σ² = 0.09
MARGINAL_STD: float = float(jnp.sqrt(8.0 + COMPONENT_SCALE**2))  # ≈ 2.844


# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------


def _model() -> None:
    """NumPyro model: 2-D 25-mode Gaussian mixture on a 5×5 grid.

    mixing ~ Categorical(probs=[1/25, …, 1/25])
    x | mixing ~ MixtureSameFamily(mixing, N(μ_k, σ²I))
    """
    mixing = dist.Categorical(probs=jnp.full(N_COMPONENTS, 1.0 / N_COMPONENTS))
    components = dist.MultivariateNormal(
        loc=COMPONENT_LOCS,
        covariance_matrix=COMPONENT_SCALE**2
        * jnp.eye(2)[None, :, :].repeat(N_COMPONENTS, axis=0),
    )
    numpyro.sample("x", dist.MixtureSameFamily(mixing, components))


# ---------------------------------------------------------------------------
# Analytic sampler
# ---------------------------------------------------------------------------


def _analytic_sampler(rng_key: jax.Array, n: int) -> dict[str, jax.Array]:
    """Return n i.i.d. draws from the 25-mode Gaussian mixture.

    Uses sequential ancestral sampling:
      1. Draw mode index k ~ Categorical(1/25, …, 1/25).
      2. Draw x ~ N(μ_k, σ²I).

    Parameters
    ----------
    rng_key
        JAX random key.
    n
        Number of samples.

    Returns
    -------
    dict with key ``"x"`` of shape ``(n, 2)``.
    """
    k_mode, k_comp = jax.random.split(rng_key)
    mode_idx = jax.random.randint(k_mode, (n,), 0, N_COMPONENTS)
    centers = COMPONENT_LOCS[mode_idx]  # (n, 2)
    noise = jax.random.normal(k_comp, (n, 2)) * COMPONENT_SCALE  # (n, 2)
    return {"x": centers + noise}


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

ENTRY = Posterior(
    name="gmm_25",
    dim=DIM,
    class_="pathological",
    tags=("multimodal", "mixture", "low-dim"),
    numpyro_model=_model,
    analytic_sampler=_analytic_sampler,
    description=(
        "2-D mixture of 25 isotropic Gaussians on a 5x5 grid in [-4,4]^2 with sigma=0.3. "
        "Discriminates SMC vs tempered NUTS via mode coverage."
    ),
)
