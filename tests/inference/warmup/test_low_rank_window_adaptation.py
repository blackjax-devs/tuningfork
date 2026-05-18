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
    """Verify that low_rank_window_adaptation returns a LowRankInverseMassMatrix.

    Since blackjax#917 (b094083c, 2026-05-18) the inverse_mass_matrix returned
    by ``low_rank_window_adaptation`` is a ``LowRankInverseMassMatrix`` NamedTuple
    with public (sigma, U, lam) array fields — pytree-flat and vmap-compatible.
    """
    ill_cond_50 = MODELS["ill_cond_50"]
    nuts = BASE_METHODS["nuts"]
    warmup = WARMUPS["low_rank_window_adaptation"]

    # Build logdensity and init position
    key = jax.random.key(42)
    init_position, logdensity_fn, _ = build_logdensity_fn(key, ill_cond_50)

    # Run warmup with num_chains=2
    num_chains = 2
    n_warmup = 200
    d = 50  # ill_cond_50 dimension
    max_rank = 8
    states, adapted_params = warmup.runner(
        jax.random.key(0),
        init_position,
        n_warmup,
        nuts,
        logdensity_fn=logdensity_fn,
        max_rank=max_rank,
        num_chains=num_chains,
    )

    # Check structured IMM: LowRankInverseMassMatrix NamedTuple with sigma/U/lam.
    imm = adapted_params["inverse_mass_matrix"]
    assert hasattr(imm, "sigma"), "IMM should have 'sigma' field"
    assert hasattr(imm, "U"), "IMM should have 'U' field"
    assert hasattr(imm, "lam"), "IMM should have 'lam' field"

    # Shapes are batched on the leading num_chains axis:
    # sigma: (num_chains, d), U: (num_chains, d, max_rank), lam: (num_chains, max_rank)
    assert imm.sigma.shape == (
        num_chains,
        d,
    ), f"sigma shape {imm.sigma.shape} != ({num_chains}, {d})"
    assert imm.U.shape == (
        num_chains,
        d,
        max_rank,
    ), f"U shape {imm.U.shape} != ({num_chains}, {d}, {max_rank})"
    assert imm.lam.shape == (
        num_chains,
        max_rank,
    ), f"lam shape {imm.lam.shape} != ({num_chains}, {max_rank})"

    # Finiteness across all IMM leaves
    for leaf_name in ("sigma", "U", "lam"):
        leaf = getattr(imm, leaf_name)
        assert jnp.all(jnp.isfinite(leaf)), f"IMM.{leaf_name} contains NaN/Inf"

    # Check step_size shape + values
    step_size = adapted_params["step_size"]
    assert step_size.shape == (
        num_chains,
    ), f"step_size should have shape ({num_chains},), got {step_size.shape}"
    assert jnp.all(jnp.isfinite(step_size)), "step_size contains NaN or Inf"
    assert jnp.all(step_size > 0), "step_size should be positive"
