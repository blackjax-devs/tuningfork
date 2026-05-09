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
"""Generate synthetic stochastic volatility returns and persist as .npy.

Provenance:
    Synthetic data — NOT from an external source.
    Model: Kim-Shephard-Chib (1998) stochastic volatility, NCP recursive AR(1).
    Parameters:
        mu_true    = -10.0   (mean log-volatility)
        phi_true   =  0.95   (AR(1) persistence)
        sigma_true =  0.25   (innovation scale)
    T = 500 time steps.
    Seed: jax.random.key(20260508).

Output:
    bjx_bench/data/stoch_vol_returns.npy  — float32 array of shape (500,)

Usage:
    cd bjx-bench
    uv run python tools/generate_stoch_vol.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Ground-truth parameters (KSC 1998 daily financial vol scale)
MU_TRUE: float = -10.0
PHI_TRUE: float = 0.95
SIGMA_TRUE: float = 0.25
T: int = 500

_OUT = Path(__file__).parent.parent / "bjx_bench" / "data" / "stoch_vol_returns.npy"


def simulate(
    mu: float,
    phi: float,
    sigma: float,
    T: int,
    seed: int = 20260508,
) -> np.ndarray:
    """Simulate returns from the KSC stochastic volatility model.

    Follows the same NCP recursive AR(1) used in stoch_vol.py so synthetic
    data exactly matches the model's generative story.

    Parameters
    ----------
    mu
        Mean log-volatility.
    phi
        AR(1) persistence coefficient.
    sigma
        Innovation scale.
    T
        Number of time steps.
    seed
        Integer seed for ``jax.random.key``.

    Returns
    -------
    np.ndarray of shape (T,) dtype float32.
    """
    key = jax.random.key(seed)
    key_h, key_r = jax.random.split(key)

    # Draw latent log-volatility process
    h_raw = jax.random.normal(key_h, shape=(T,))

    # Stationary initialisation
    h0 = mu + (sigma / jnp.sqrt(1.0 - phi**2)) * h_raw[0]

    # Recursive AR(1): h[t] = mu + phi*(h[t-1] - mu) + sigma*h_raw[t]
    def step(h_prev, h_raw_t):
        h_t = mu + phi * (h_prev - mu) + sigma * h_raw_t
        return h_t, h_t

    _, h_rest = jax.lax.scan(step, h0, h_raw[1:])
    h = jnp.concatenate([h0[None], h_rest])  # shape (T,)

    # Observation model: returns[t] ~ Normal(0, exp(h[t] / 2))
    eps = jax.random.normal(key_r, shape=(T,))
    returns = jnp.exp(h / 2.0) * eps  # shape (T,)

    return np.array(returns, dtype=np.float32)


def main() -> None:
    returns = simulate(MU_TRUE, PHI_TRUE, SIGMA_TRUE, T)

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(_OUT, returns)

    print(f"Saved {returns.shape} float32 array to {_OUT}")
    print(
        f"  min={returns.min():.4f}  max={returns.max():.4f}  "
        f"mean_abs={np.abs(returns).mean():.4f}  std={returns.std():.4f}"
    )
    print(
        f"  (expected mean_abs ~exp(mu/2) = exp({MU_TRUE}/2) "
        f"= {float(jnp.exp(MU_TRUE / 2)):.4f})"
    )


if __name__ == "__main__":
    main()
