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
"""Gaussian process regression — 203-D NCP Cholesky parameterization (RBF kernel, n=200 points)."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from tuningfork.model._base import Posterior

__all__ = [
    "ENTRY",
    "X_DATA",
    "Y_DATA",
    "F_TRUE",
    "N_OBS",
]

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

#: Number of observations (n = 200 per statistician verdict)
N_OBS: int = 200

#: Unconstrained dimensionality:
#:   log_lengthscale(1) + log_kernel_scale(1) + log_noise_scale(1) + f_raw(200)
DIM: int = 3 + N_OBS  # = 203

#: Jitter added to kernel diagonal for numerical stability
JITTER: float = 1e-6

#: Path to committed .npz data file
_NPZ_PATH: Path = Path(__file__).parent.parent.parent / "data" / "gp_regression.npz"

# ---------------------------------------------------------------------------
# Load synthetic data (shape: (200,), float32)
# ---------------------------------------------------------------------------

_data = np.load(_NPZ_PATH)

X_DATA: jnp.ndarray = jnp.array(_data["X"], dtype=jnp.float32)
Y_DATA: jnp.ndarray = jnp.array(_data["y"], dtype=jnp.float32)
F_TRUE: jnp.ndarray = jnp.array(_data["f_true"], dtype=jnp.float32)

# Validate shapes
assert X_DATA.shape == (N_OBS,), f"Expected X_DATA shape ({N_OBS},), got {X_DATA.shape}"
assert Y_DATA.shape == (N_OBS,), f"Expected Y_DATA shape ({N_OBS},), got {Y_DATA.shape}"

# ---------------------------------------------------------------------------
# NumPyro model (NCP Cholesky GP regression)
# ---------------------------------------------------------------------------


def gp_regression(X: jnp.ndarray, y: jnp.ndarray, n: int = N_OBS) -> None:
    """NumPyro model: 203-D NCP Cholesky RBF GP regression.

    1D Gaussian Process regression with RBF kernel and non-centered
    parameterization via Cholesky decomposition. Posterior dim = 203 for n=200.

    Parameters
    ----------
    X
        Input locations of shape (n,).
    y
        Observed outputs of shape (n,).
    n
        Number of data points (200 for the synthetic gp_regression dataset).

    Notes
    -----
    Hyperpriors are specified on the log-scale to ensure positivity without
    hard constraints:

    - ``log_lengthscale ~ Normal(0, 1)``  → lengthscale ~ LogNormal(0, 1)
    - ``log_kernel_scale ~ Normal(0, 1)`` → kernel_scale ~ LogNormal(0, 1)
    - ``log_noise_scale ~ Normal(-2, 1)`` → noise_scale ~ LogNormal(-2, 1)

    The GP latent function is parameterized via NCP: ``f = L @ z`` where
    ``L = cholesky(K + 1e-6 * I, lower=True)`` and ``z ~ Normal(0, I)``.
    This decouples z from the hyperparameter-dependent Cholesky factor,
    avoiding the funnel geometry that arises in the centered parameterization.

    Jitter = 1e-6 ensures positive-definiteness for nominal hyperparams.
    Very-small-lengthscale configurations may need larger jitter (1e-5).
    """
    log_lengthscale = numpyro.sample("log_lengthscale", dist.Normal(0.0, 1.0))
    log_kernel_scale = numpyro.sample("log_kernel_scale", dist.Normal(0.0, 1.0))
    log_noise_scale = numpyro.sample("log_noise_scale", dist.Normal(-2.0, 1.0))

    lengthscale = jnp.exp(log_lengthscale)
    kernel_scale = jnp.exp(log_kernel_scale)
    noise_scale = jnp.exp(log_noise_scale)

    # RBF kernel matrix: K[i,j] = kernel_scale^2 * exp(-0.5 * ||x_i - x_j||^2 / ls^2)
    sqdist = (X[:, None] - X[None, :]) ** 2
    K = kernel_scale**2 * jnp.exp(-0.5 * sqdist / lengthscale**2)
    K = K + JITTER * jnp.eye(n)
    L = jax.scipy.linalg.cholesky(K, lower=True)

    # NCP base variable z ~ Normal(0, I); f = L @ z
    z = numpyro.sample("f_raw", dist.Normal(jnp.zeros(n), 1.0))
    f = numpyro.deterministic("f", L @ z)

    numpyro.sample("y", dist.Normal(f, noise_scale), obs=y)


# Statistician verdict (TL-orchestrated, 2026-05-08):
#     Approve-with-modifications. Joint posterior over (3 hyperparams, 200 latents)
#     = 203-D. NCP via Cholesky for f. posteriordb_id=None.
#     NUTS/HMC HIGH BO defensible; Pathfinder expected to underperform on
#     log-scale hyperparam marginals due to non-Gaussian geometry.
#     Parameterization (NCP Cholesky):
#         log_lengthscale  ~ Normal(0, 1)           # RBF kernel length scale (log)
#         log_kernel_scale ~ Normal(0, 1)           # RBF kernel output scale (log)
#         log_noise_scale  ~ Normal(-2, 1)          # observation noise std (log)
#         lengthscale  = exp(log_lengthscale)
#         kernel_scale = exp(log_kernel_scale)
#         noise_scale  = exp(log_noise_scale)
#         K   = kernel_scale^2 * exp(-0.5 * sqdist / lengthscale^2) + 1e-6 * I
#         L   = cholesky(K, lower=True)             # jitter = 1e-6
#         z   ~ Normal(0, 1)^200                    # NCP base variable
#         f   = L @ z                               # deterministic GP values
#         y   ~ Normal(f, noise_scale)
#     Unconstrained dimensionality = 203:
#         log_lengthscale(1) + log_kernel_scale(1) + log_noise_scale(1) + f_raw(200)
#     Jitter note:
#         Jitter = 1e-6 ensures positive-definiteness for nominal hyperparams.
#         Very-small-lengthscale configurations (lengthscale < ~0.01) may still
#         cause numerical issues; increase jitter to 1e-5 if that occurs.
#     CP vs NCP rationale:
#         Centered parameterization couples f tightly to the lengthscale-dependent
#         Cholesky L, creating funnel geometry. NCP (f = L @ z) decouples z from
#         hyperparameters in the prior, allowing NUTS/MCLMC to explore hyperparams
#         independently of the 200 latent values.
#     posteriordb_id = None:
#         No exact posteriordb match for 1D RBF GP regression at n=200.
#         reference-certification uses Long-NUTS self-check (split-R̂ < 1.01) only.
#     Data: synthetic, generated by tools/generate_gp_regression.py.
#         Parameters: f(x) = sin(2*pi*x), noise_scale=0.1, n=200,
#         X ~ Uniform(0,1), seed=jax.random.PRNGKey(42).
#         Committed as tuningfork/data/gp_regression.npz
#         with arrays (X, y, f_true) of shape (200,), dtype float32.
# References:
#     Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for
#         Machine Learning. MIT Press. (RW06, Chapter 2.)
#     Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo.
#         arXiv:1701.02434. (NCP geometry motivation.)
# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

ENTRY = Posterior(
    name="gp_regression",
    dim=DIM,
    class_="latent_gaussian",
    tags=(
        "latent_gaussian",
        "gp",
        "rbf_kernel",
        "ncp",
        "synthetic",
    ),
    numpyro_model=gp_regression,
    model_args=(X_DATA, Y_DATA),
    model_kwargs={"n": N_OBS},
    posteriordb_id=None,  # no exact posteriordb match for 1D RBF GP at n=200
    citations=(
        "Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for "
        "Machine Learning. MIT Press. (Chapter 2.)",
        "Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. "
        "arXiv:1701.02434.",
    ),
    description=(
        "203-D NCP Cholesky RBF GP regression (Rasmussen & Williams 2006). "
        "n=200 synthetic observations, f(x)=sin(2*pi*x), noise_scale=0.1, "
        "X~Uniform(0,1), seed=PRNGKey(42). "
        "Priors: log_lengthscale~N(0,1), log_kernel_scale~N(0,1), "
        "log_noise_scale~N(-2,1), f_raw~N(0,I)^200 (NCP Cholesky). "
        "posteriordb_id=None (no upstream reference draws; Long-NUTS self-check). "
        "Dim=203: log_lengthscale(1)+log_kernel_scale(1)+log_noise_scale(1)+f_raw(200)."
    ),
)
