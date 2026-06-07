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
"""Log-Gaussian Cox Process (LGCP) — 1600-D NCP separable GP model (40x40 grid)."""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from tuningfork.model._base import Posterior

__all__ = [
    "ENTRY",
    "GRID_SIZE",
    "DIM",
    "Y_DATA",
]

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

GRID_SIZE: int = 40
DIM: int = GRID_SIZE * GRID_SIZE  # = 1600

# ---------------------------------------------------------------------------
# Deterministic synthetic data generation (at import time)
# ---------------------------------------------------------------------------

_RNG_SEED: int = 42
_sigma: float = 1.0
_lengthscale: float = 0.1
_area: float = 1.0 / DIM

_rng = np.random.default_rng(_RNG_SEED)
_coords = np.linspace(0.0, 1.0, GRID_SIZE)
_dist_matrix = _coords[:, None] - _coords[None, :]
_K_1D = _sigma**2 * np.exp(-0.5 * (_dist_matrix / _lengthscale) ** 2) + 1e-6 * np.eye(
    GRID_SIZE
)
_L_1D = np.linalg.cholesky(_K_1D)

_z = _rng.normal(size=(GRID_SIZE, GRID_SIZE))
_gp_latent = _L_1D @ _z @ _L_1D.T
_intensity = np.exp(_gp_latent) * _area
_y_matrix = _rng.poisson(_intensity)

Y_DATA: jnp.ndarray = jnp.array(_y_matrix, dtype=jnp.int32)
L_1D_GLOBAL: jnp.ndarray = jnp.array(_L_1D, dtype=jnp.float32)

# ---------------------------------------------------------------------------
# NumPyro model (Separable GP prior LGCP)
# ---------------------------------------------------------------------------


def log_gaussian_cox_process(y: jnp.ndarray) -> None:
    """NumPyro model: 1600-D NCP separable GP prior LGCP on a 40x40 grid.

    Separable covariance is formulated via Kronecker products, which avoids
    the O(D^3) covariance factorization of size 1600x1600. Instead, it only
    requires 1D Cholesky factorizations of size 40x40.
    """
    area = 1.0 / DIM

    # Latent standard normal parameters
    z = numpyro.sample("z", dist.Normal(jnp.zeros((GRID_SIZE, GRID_SIZE)), 1.0))

    # Reconstruct 2D GP: gp_latent = L_1D_GLOBAL @ z @ L_1D_GLOBAL^T
    gp_latent = L_1D_GLOBAL @ z @ L_1D_GLOBAL.T

    # Poisson observation model
    intensity = jnp.exp(gp_latent) * area
    numpyro.sample("y", dist.Poisson(intensity), obs=y)


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

ENTRY = Posterior(
    name="lgcp",
    dim=DIM,
    class_="latent_gaussian",
    tags=(
        "latent_gaussian",
        "separable_gp",
        "ncp",
        "synthetic_data",
    ),
    numpyro_model=log_gaussian_cox_process,
    model_args=(Y_DATA,),
    posteriordb_id=None,
    description=(
        "1600-D NCP separable GP prior Log-Gaussian Cox Process on a 40×40 grid with "
        "synthetic Poisson data (seed=42). Not tied to any real-data benchmark. Natural "
        "MCLMC showcase target due to high dimensionality and isotropic NCP "
        "z-parameterization."
    ),
    headline_params=None,
    headline_coords=None,
)
