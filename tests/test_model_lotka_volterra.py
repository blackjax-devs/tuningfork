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
"""Tests for the lotka_volterra ODE inverse model (P4.10, Block D).

All tests are marked ``fast`` — they run the ProbDiffEq solver exactly once
(test_logdensity_finite) or perform pure data/shape assertions. The solver
call is the slowest test in this file.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bjx_bench.model import MODELS, build_logdensity_fn
from bjx_bench.model.ode.lotka_volterra import (
    MU_TRUE,
    OBSERVATION_TIMES,
    OBSERVATIONS,
    T_OBS,
)

pytestmark = pytest.mark.fast


def test_dim() -> None:
    """Dim must be 7 per statistician verdict (plan said 4 — correction applied)."""
    assert MODELS["lotka_volterra"].dim == 7


def test_data_shape() -> None:
    """Observations must be (40, 2) and T_OBS == 40."""
    assert OBSERVATIONS.shape == (40, 2)
    assert OBSERVATION_TIMES.shape == (40,)
    assert T_OBS == 40


def test_observations_finite() -> None:
    """All observations and time points must be finite (no NaN/Inf)."""
    assert bool(jnp.all(jnp.isfinite(OBSERVATIONS)))
    assert bool(jnp.all(jnp.isfinite(OBSERVATION_TIMES)))


def test_logdensity_finite() -> None:
    """Log-density at the init position must be finite.

    This test exercises the full ProbDiffEq ODE solve path. It is the
    slowest test in this file (~1–3 s on CPU due to solver compilation).
    """
    entry = MODELS["lotka_volterra"]
    key = jax.random.key(0)
    init_position, logdensity_fn, _ = build_logdensity_fn(key, entry)
    lp = logdensity_fn(init_position)
    assert np.isfinite(float(lp)), f"logdensity not finite at init position: {lp}"


def test_posteriordb_id_none() -> None:
    """posteriordb_id must be None (no upstream reference draws available)."""
    assert MODELS["lotka_volterra"].posteriordb_id is None


def test_mu_true_keys() -> None:
    """MU_TRUE dict must contain all 7 ground-truth parameters."""
    expected_keys = {"alpha", "beta", "gamma", "delta", "u0", "v0", "sigma_obs"}
    assert set(MU_TRUE.keys()) == expected_keys


def test_mu_true_values() -> None:
    """Ground-truth values must match the synthetic data spec."""
    assert abs(MU_TRUE["alpha"] - 0.5) < 1e-5
    assert abs(MU_TRUE["beta"] - 0.05) < 1e-5
    assert abs(MU_TRUE["gamma"] - 0.5) < 1e-5
    assert abs(MU_TRUE["delta"] - 0.05) < 1e-5
    assert abs(MU_TRUE["u0"] - 10.0) < 1e-4
    assert abs(MU_TRUE["v0"] - 5.0) < 1e-4
    assert abs(MU_TRUE["sigma_obs"] - 0.5) < 1e-5
