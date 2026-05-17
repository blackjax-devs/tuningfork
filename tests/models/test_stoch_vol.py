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
"""Tests for the stoch_vol model (503-D NCP AR(1) state-space).

All tests are marked ``fast`` — no chain sampling, only structural checks.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tuningfork.model import MODELS, build_logdensity_fn
from tuningfork.model.stoch_vol import ENTRY, RETURNS, T_LENGTH

pytestmark = pytest.mark.fast


def test_dim() -> None:
    """Registry entry has dim == 503 (mu+phi+log_sigma+h_raw[500])."""
    assert MODELS["stoch_vol"].dim == 503
    assert ENTRY.dim == 503


def test_data_shape() -> None:
    """RETURNS has shape (500,) and T_LENGTH == 500."""
    assert RETURNS.shape == (500,)
    assert T_LENGTH == 500


def test_returns_realistic_scale() -> None:
    """SP500 daily returns (mean-centered) have std ~1 and mean_abs ~0.5-1.

    Real-data scale: SP500 daily returns are typically ±5% range with most
    samples within ±2%. Mean-centered: mean ≈ 0, std ≈ 1-1.5, mean_abs ~0.8.
    Replaced 2026-05-12 from synthetic (mu_true=-10 → mean_abs ≈ 0.007).
    """
    arr = np.array(RETURNS)
    mean_abs = float(np.abs(arr).mean())
    # Should be O(1) for SP500-scale daily returns. Sanity: < 5% probability
    # of |r| > 5 (extreme single-day move); typical mean_abs is 0.5-1.
    assert (
        0.3 < mean_abs < 3.0
    ), f"mean_abs={mean_abs:.4f} outside [0.3, 3.0] (SP500 daily returns range)"
    # Mean-centered → close to zero
    assert (
        abs(float(arr.mean())) < 1e-4
    ), f"mean={arr.mean():.6e} should be ≈ 0 for mean-centered returns"


def test_logdensity_finite() -> None:
    """Log-density is finite at the NumPyro-initialised position."""
    key = jax.random.key(0)
    init_pos, logdensity_fn, _ = build_logdensity_fn(key, ENTRY)
    ld = logdensity_fn(init_pos)
    assert jnp.isfinite(ld), f"Expected finite log-density at init, got {ld}"


def test_posteriordb_id_none() -> None:
    """posteriordb_id is None — no upstream reference draws available."""
    assert MODELS["stoch_vol"].posteriordb_id is None
