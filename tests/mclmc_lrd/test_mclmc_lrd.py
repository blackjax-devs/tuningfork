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
    """decompose_covariance_low_rank returns correct shapes and satisfies L L^T ≈ COV.

    decompose_covariance_low_rank reconstructs Σ (the target covariance), not its
    inverse.  In MCLMC the inverse-mass-matrix M^{-1} = Σ = COV, so L L^T ≈ COV is
    the correct algebraic identity.  We use k=d=50 (full rank) so truncation error
    is zero and the identity holds to machine precision (~2.8e-7 per statistician).
    """
    d, k = 50, 50  # k=d → exact reconstruction, no truncation error
    sigma, U, lam = decompose_covariance_low_rank(COV, k)

    assert sigma.shape == (d,), f"sigma shape {sigma.shape} != ({d},)"
    assert U.shape == (d, k), f"U shape {U.shape} != ({d}, {k})"
    assert lam.shape == (k,), f"lam shape {lam.shape} != ({k},)"
    assert jnp.all(sigma > 0), "sigma must be strictly positive"
    assert jnp.all(lam > 0), "lam must be strictly positive"

    # Reconstruct COV via L_LR L_LR^T and compare directly to COV.
    # decompose_covariance_low_rank gives sigma=sqrt(diag(COV)), U, lam s.t.
    #   L_LR = diag(sigma) @ (I + U @ diag(sqrt(lam) - 1) @ U^T)
    #   L_LR L_LR^T ≈ COV   (exact at k=d)
    sqrt_lam = jnp.sqrt(lam)
    L = jnp.diag(sigma) @ (jnp.eye(d) + U @ (jnp.diag(sqrt_lam - 1.0) @ U.T))
    cov_reconstructed = L @ L.T
    rel_err = jnp.linalg.norm(cov_reconstructed - COV) / jnp.linalg.norm(COV)
    assert rel_err < 1e-4, f"L L^T ≈ COV rel err {rel_err:.2e} exceeds 1e-4"


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

    # samples is a PyTree (dict of arrays) with leading dims (num_chains, n_samples).
    # Apply pure-JAX diagnostics per leaf, then reduce across leaves.
    rhat_tree = jax.tree.map(
        lambda x: potential_scale_reduction(x, chain_axis=0, sample_axis=1), samples
    )
    ess_tree = jax.tree.map(
        lambda x: effective_sample_size(x, chain_axis=0, sample_axis=1), samples
    )
    rhat_max = float(
        jnp.max(jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(rhat_tree)]))
    )
    ess_min = float(
        jnp.min(jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(ess_tree)]))
    )

    assert rhat_max < 1.05, f"R-hat {rhat_max:.4f} >= 1.05"
    assert ess_min >= 100.0, f"min ESS {ess_min:.1f} < 100"


def test_mclmc_lrd_tuning_warmup_returns_lrd_imm():
    """mclmc_lrd_tuning warmup returns LowRankInverseMassMatrix in adapted_params.

    Checks the mclmc_lrd_tuning Warmup runner returns adapted_params with:
    - "step_size" shape (1,) for num_chains=1
    - "L" shape (1,)
    - "inverse_mass_matrix" is a LowRankInverseMassMatrix namedtuple
    """
    from tuningfork.base_method import BASE_METHODS
    from tuningfork.model import MODELS
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.warmup import WARMUPS

    entry = MODELS["ill_cond_50"]
    key = jax.random.key(42)
    init_key, warmup_key = jax.random.split(key)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)

    warmup = WARMUPS["mclmc_lrd_tuning"]
    base_method = BASE_METHODS["mclmc"]

    states, adapted_params = warmup.runner(
        warmup_key,
        init_position,
        n_warmup=500,
        base_method=base_method,
        logdensity_fn=logdensity_fn,
        num_chains=1,
        k_rank=10,
    )

    assert "step_size" in adapted_params
    assert "L" in adapted_params
    assert "inverse_mass_matrix" in adapted_params

    imm = adapted_params["inverse_mass_matrix"]
    assert isinstance(imm, LowRankInverseMassMatrix), type(imm)
    # Certified runner broadcasts the shared LRD IMM to a leading num_chains axis
    # (mclmc_lrd_tuning.py:43-47) — with num_chains=1 the shapes are (1,d)/(1,d,k)/(1,k).
    assert imm.sigma.shape == (1, 50), imm.sigma.shape  # (num_chains=1, d=50)
    assert imm.U.shape == (1, 50, 10), imm.U.shape  # (num_chains=1, d=50, k=10)
    assert imm.lam.shape == (1, 10), imm.lam.shape  # (num_chains=1, k=10)

    # step_size and L should have leading dim num_chains=1.
    assert adapted_params["step_size"].shape == (1,)
    assert adapted_params["L"].shape == (1,)


def test_from_warmup_only_mclmc_lrd_tuning_squeeze():
    """Recipe.from_warmup_only with mclmc_lrd_tuning squeezes step_size/L, preserves LRD.

    After squeeze_single_chain, step_size and L become scalars while the
    LowRankInverseMassMatrix passes through verbatim (per-leaf fix).
    """
    from tuningfork.base_method import BASE_METHODS
    from tuningfork.model import MODELS
    from tuningfork.recipes import Effort, Recipe
    from tuningfork.warmup import WARMUPS

    entry = MODELS["ill_cond_50"]
    warmup = WARMUPS["mclmc_lrd_tuning"]
    base_method = BASE_METHODS["mclmc"]

    recipe = Recipe.from_warmup_only(
        entry,
        base_method,
        warmup,
        n_warmup=300,
        rng_key=jax.random.key(99),
        k_rank=10,
    )

    assert recipe.effort == Effort.MEDIUM
    # LRD IMM must be in base_method_params as a LowRankInverseMassMatrix.
    imm = recipe.base_method_params.get("inverse_mass_matrix")
    assert isinstance(
        imm, LowRankInverseMassMatrix
    ), f"Expected LowRankInverseMassMatrix, got {type(imm)}"
    assert "step_size" in recipe.base_method_params
    assert "L" in recipe.base_method_params
    # step_size and L should be Python floats (squeezed scalars).
    assert isinstance(
        recipe.base_method_params["step_size"], float
    ), f"step_size should be float, got {type(recipe.base_method_params['step_size'])}"
