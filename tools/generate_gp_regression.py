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
"""Generate synthetic GP regression data and persist as .npz.

Provenance:
    Synthetic data — NOT from an external source.
    Model: 1D Gaussian Process regression with RBF kernel, NCP Cholesky
           parameterization.
    Parameters:
        true_lengthscale  = 0.2   (RBF kernel length scale)
        true_kernel_scale = 1.0   (RBF kernel output scale)
        true_noise_scale  = 0.1   (observation noise std)
        f(x)              = sin(2 * pi * x)  (ground-truth function)
    n = 200 inputs X ~ Uniform(0, 1).
    Seed: jax.random.PRNGKey(42).

Output:
    bjx_bench/data/gp_regression.npz  — dict with:
        X       : float32 array of shape (200,)  — inputs in [0, 1]
        y       : float32 array of shape (200,)  — noisy observations
        f_true  : float32 array of shape (200,)  — noiseless ground-truth

Usage:
    cd bjx-bench
    uv run python tools/generate_gp_regression.py
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Ground-truth parameters
TRUE_NOISE_SCALE: float = 0.1
N: int = 200

_OUT = Path(__file__).parent.parent / "bjx_bench" / "data" / "gp_regression.npz"


def simulate(
    n: int = N,
    noise_scale: float = TRUE_NOISE_SCALE,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate GP regression data from the ground-truth function f(x) = sin(2*pi*x).

    Inputs X are drawn Uniform(0, 1); observations y = f(X) + eps where
    eps ~ Normal(0, noise_scale).

    Parameters
    ----------
    n
        Number of data points.
    noise_scale
        Standard deviation of observation noise.
    seed
        Integer seed for ``jax.random.PRNGKey``.

    Returns
    -------
    X : np.ndarray of shape (n,) dtype float32 — inputs in [0, 1]
    y : np.ndarray of shape (n,) dtype float32 — noisy observations
    f_true : np.ndarray of shape (n,) dtype float32 — noiseless function values
    """
    key = jax.random.PRNGKey(seed)
    key_x, key_eps = jax.random.split(key)

    X = jax.random.uniform(key_x, shape=(n,), minval=0.0, maxval=1.0)
    X = jnp.sort(X)  # sort for nicer plots, does not affect inference

    f_true = jnp.sin(2.0 * jnp.pi * X)
    eps = jax.random.normal(key_eps, shape=(n,)) * noise_scale
    y = f_true + eps

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(f_true, dtype=np.float32),
    )


def main() -> None:
    X, y, f_true = simulate()

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_OUT, X=X, y=y, f_true=f_true)

    print(f"Saved to {_OUT}")
    print(f"  X: shape={X.shape}, min={X.min():.4f}, max={X.max():.4f}")
    print(f"  y: shape={y.shape}, min={y.min():.4f}, max={y.max():.4f}")
    print(f"  f_true: shape={f_true.shape}, mean={f_true.mean():.4f}")
    print(f"  noise ~ N(0, {TRUE_NOISE_SCALE}) applied to f=sin(2*pi*X)")


if __name__ == "__main__":
    main()
