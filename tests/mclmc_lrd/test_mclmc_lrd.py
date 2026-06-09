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
"""Tests for the LRD-preconditioned MCLMC sampler (tuningfork integration).

Covers the production path: isokinetic_mclachlan + LowRankInverseMassMatrix
dispatched via blackjax.mclmc (PR #936).  Geometry: ill_cond_50 — the only
independently reproduced PASS target (statistician multi-seed, 2026-06-09).
"""

import jax
import jax.numpy as jnp
import pytest
from blackjax.diagnostics import effective_sample_size, potential_scale_reduction
from blackjax.mcmc.metrics import LowRankInverseMassMatrix

from tuningfork.base_method.mclmc import (
    decompose_covariance_low_rank,
    run_internal_lrd_mclmc,
)
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model.ill_cond_50 import COV

pytestmark = pytest.mark.slow


def test_lrd_decompose_covariance_low_rank():
    """decompose_covariance_low_rank returns correct shapes and satisfies L L^T ≈ M^{-1}."""
    d, k = 50, 10
    sigma, U, lam = decompose_covariance_low_rank(COV, k)

    assert sigma.shape == (d,), f"sigma shape {sigma.shape} != ({d},)"
    assert U.shape == (d, k), f"U shape {U.shape} != ({d}, {k})"
    assert lam.shape == (k,), f"lam shape {lam.shape} != ({k},)"
    assert jnp.all(sigma > 0), "sigma must be strictly positive"
    assert jnp.all(lam > 0), "lam must be strictly positive"

    # Reconstruct M^{-1} via L_LR L_LR^T and compare to the full inverse.
    # L_LR = diag(sigma) @ (I + U @ (sqrt(lam) - 1) @ U^T)
    sqrt_lam = jnp.sqrt(lam)
    L = jnp.diag(sigma) @ (jnp.eye(d) + U @ (jnp.diag(sqrt_lam - 1.0) @ U.T))
    M_inv_reconstructed = L @ L.T
    M_inv_exact = jnp.linalg.inv(COV)
    rel_err = jnp.linalg.norm(M_inv_reconstructed - M_inv_exact) / jnp.linalg.norm(
        M_inv_exact
    )
    assert rel_err < 1e-4, f"L L^T ≈ M^{{-1}} rel err {rel_err:.2e} exceeds 1e-4"


def test_lrd_mclmc_ill_cond_50():
    """Internal LRD MCLMC converges on ill_cond_50 (d=50, κ=100, k=40)."""
    entry = MODELS["ill_cond_50"]
    key = jax.random.key(98765)
    init_key, run_key = jax.random.split(key)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)

    sigma, U, lam = decompose_covariance_low_rank(COV, k=40)
    lrd_imm = LowRankInverseMassMatrix(sigma=sigma, U=U, lam=lam)

    samples, _ = run_internal_lrd_mclmc(
        logdensity_fn, init_position, lrd_imm, run_key, n_warmup=1000, n_samples=1000
    )

    # samples shape: (num_chains, n_samples, d) = (4, 1000, 50)
    # Use pure-JAX diagnostics — no arviz dependency.
    rhat = potential_scale_reduction(samples, chain_axis=0, sample_axis=1)
    ess = effective_sample_size(samples, chain_axis=0, sample_axis=1)
    rhat_max = float(jnp.max(rhat))
    ess_min = float(jnp.min(ess))

    assert rhat_max < 1.05, f"R-hat {rhat_max:.4f} >= 1.05"
    assert ess_min >= 100.0, f"min ESS {ess_min:.1f} < 100"
