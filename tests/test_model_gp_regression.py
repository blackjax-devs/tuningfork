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
"""Tests for the gp_regression model (P4.11, Block D: 203-D NCP Cholesky RBF GP).

All tests are marked ``fast`` — no chain sampling, only structural checks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bjx_bench.model import MODELS, build_logdensity_fn
from bjx_bench.model.latent_gaussian.gp_regression import ENTRY, N_OBS, X_DATA, Y_DATA

pytestmark = pytest.mark.fast


def test_dim() -> None:
    """Registry entry has dim == 203 (log_ls+log_ks+log_ns+f_raw[200])."""
    assert MODELS["gp_regression"].dim == 203
    assert ENTRY.dim == 203


def test_data_shape() -> None:
    """X_DATA and Y_DATA have shape (200,) and N_OBS == 200."""
    assert X_DATA.shape == (200,)
    assert Y_DATA.shape == (200,)
    assert N_OBS == 200


def test_x_range() -> None:
    """X_DATA lies in [0, 1] — drawn from Uniform(0, 1)."""
    x_np = np.array(X_DATA)
    assert float(x_np.min()) >= 0.0, f"X_DATA.min()={x_np.min():.6f} < 0"
    assert float(x_np.max()) <= 1.0, f"X_DATA.max()={x_np.max():.6f} > 1"


def test_logdensity_finite() -> None:
    """Log-density is finite at the NumPyro-initialised position (exercises Cholesky)."""
    key = jax.random.key(0)
    init_pos, logdensity_fn, _ = build_logdensity_fn(key, ENTRY)
    ld = logdensity_fn(init_pos)
    assert jnp.isfinite(ld), f"Expected finite log-density at init, got {ld}"


def test_posteriordb_id_none() -> None:
    """posteriordb_id is None — no exact posteriordb match for 1D RBF GP at n=200."""
    assert MODELS["gp_regression"].posteriordb_id is None
