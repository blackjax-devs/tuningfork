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
"""Tests for low_rank_window_adaptation warmup."""

import jax
import jax.numpy as jnp
import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.warmup import WARMUPS

pytestmark = pytest.mark.slow


def test_low_rank_window_adaptation_returns_metric():
    """Verify that low_rank_window_adaptation returns a Metric object."""
    ill_cond_50 = MODELS["ill_cond_50"]
    nuts = BASE_METHODS["nuts"]
    warmup = WARMUPS["low_rank_window_adaptation"]

    # Build logdensity and init position
    key = jax.random.key(42)
    init_position, logdensity_fn, _ = build_logdensity_fn(key, ill_cond_50)

    # Run warmup with num_chains=2
    num_chains = 2
    n_warmup = 200
    states, adapted_params = warmup.runner(
        jax.random.key(0),
        init_position,
        n_warmup,
        nuts,
        logdensity_fn=logdensity_fn,
        max_rank=8,
        num_chains=num_chains,
    )

    # Check that we got a Metric object (it's a NamedTuple)
    imm = adapted_params["inverse_mass_matrix"]
    # The metric should be a NamedTuple with a specific structure for low-rank
    # We can check it's callable (has momentum generation) by checking for expected attrs
    assert hasattr(
        imm, "generate_momentum"
    ), "Metric should have 'generate_momentum' method"
    assert hasattr(imm, "metric_fn"), "Metric should have 'metric_fn' method"

    # Check step_size shape
    step_size = adapted_params["step_size"]
    assert step_size.shape == (
        num_chains,
    ), f"step_size should have shape ({num_chains},), got {step_size.shape}"

    # Check that step_size values are reasonable (no NaN, finite, positive)
    assert jnp.all(jnp.isfinite(step_size)), "step_size contains NaN or Inf"
    assert jnp.all(step_size > 0), "step_size should be positive"
