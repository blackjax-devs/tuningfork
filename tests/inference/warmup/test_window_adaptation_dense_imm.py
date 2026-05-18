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
"""Tests for window_adaptation_dense_imm warmup."""

import jax
import jax.numpy as jnp
import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.warmup import WARMUPS

pytestmark = pytest.mark.slow


def test_window_adaptation_dense_imm_returns_correct_imm_shape():
    """Verify that window_adaptation_dense_imm returns dense IMM with shape (num_chains, d, d)."""
    mvn_10 = MODELS["mvn_10"]
    nuts = BASE_METHODS["nuts"]
    warmup = WARMUPS["window_adaptation_dense_imm"]

    # Build logdensity and init position
    key = jax.random.key(42)
    init_position, logdensity_fn, _ = build_logdensity_fn(key, mvn_10)

    # Run warmup with num_chains=2
    num_chains = 2
    n_warmup = 200
    states, adapted_params = warmup.runner(
        jax.random.key(0),
        init_position,
        n_warmup,
        nuts,
        logdensity_fn=logdensity_fn,
        num_chains=num_chains,
    )

    # Check that IMM shape is (num_chains, d, d) — dense, not diagonal
    imm = adapted_params["inverse_mass_matrix"]
    assert imm.ndim == 3, f"Dense IMM should be 3-D, got {imm.ndim}-D"
    assert (
        imm.shape[0] == num_chains
    ), f"IMM batch dim should be {num_chains}, got {imm.shape[0]}"

    d = imm.shape[1]
    assert (
        imm.shape[2] == d
    ), f"Dense IMM should be square in last two dims, got {imm.shape}"

    # Check step_size shape
    step_size = adapted_params["step_size"]
    assert step_size.shape == (
        num_chains,
    ), f"step_size should have shape ({num_chains},), got {step_size.shape}"

    # Check that IMM values are reasonable (no NaN, finite)
    assert jnp.all(jnp.isfinite(imm)), "IMM contains NaN or Inf"
    assert jnp.all(imm > 0), "IMM should be positive (inverse of SPD matrix)"
