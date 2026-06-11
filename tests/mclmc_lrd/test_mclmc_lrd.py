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

    Note: with a short pilot (pilot_n_samples=1000 on a 50-d ill_cond_50 model),
    the upstream rank guard may clamp k_rank=10 → k_used=1 due to low pilot
    n_eff.  The test accepts the clamping warning and verifies the batched shapes
    are consistent with whatever k_used the rank guard selects.
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

    # The rank guard may fire for short pilots (k_rank=10 can exceed floor(n_eff/2)
    # when the pilot has not yet mixed).  Accept the UserWarning explicitly.
    with pytest.warns(UserWarning, match="rank-safety bound|Clamping"):
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

    # The runner broadcasts the shared LRD IMM to a leading num_chains axis.
    # With num_chains=1: sigma (1, d), U (1, d, k_used), lam (1, k_used).
    # k_used may be < k_rank=10 if the rank guard clamped it — verify the
    # broadcast contract holds for whatever k_used the guard selected.
    k_used = imm.lam.shape[1]
    assert imm.sigma.shape == (1, 50), imm.sigma.shape  # (num_chains=1, d=50)
    assert imm.U.shape == (1, 50, k_used), imm.U.shape  # (num_chains=1, d=50, k_used)
    assert imm.lam.shape == (1, k_used), imm.lam.shape  # (num_chains=1, k_used)
    assert k_used >= 1, f"k_used={k_used} must be at least 1 (rank guard floor)"

    # step_size and L should have leading dim num_chains=1.
    assert adapted_params["step_size"].shape == (1,)
    assert adapted_params["L"].shape == (1,)


def test_target_acceptance_rate_forwarded_without_raise():
    """target_acceptance_rate kwarg must not raise on the unadjusted path.

    _recipe_runner.py passes target_acceptance_rate unconditionally at both
    call sites.  The runner must accept it silently (document-and-ignore) for
    the unadjusted (inner_kernel="mclmc") path so recipe/cert/rerun invocations
    do not crash with a TypeError.
    """
    from tuningfork.base_method import BASE_METHODS
    from tuningfork.model import MODELS
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.warmup import WARMUPS

    entry = MODELS["ill_cond_50"]
    key = jax.random.key(7)
    init_key, warmup_key = jax.random.split(key)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)

    warmup = WARMUPS["mclmc_lrd_tuning"]
    base_method = BASE_METHODS["mclmc"]

    # Accept rank-guard UserWarning from short pilot.
    with pytest.warns(UserWarning, match="rank-safety bound|Clamping"):
        # Must NOT raise TypeError — target_acceptance_rate is accepted silently
        # on the unadjusted path.
        _, adapted_params = warmup.runner(
            warmup_key,
            init_position,
            n_warmup=300,
            base_method=base_method,
            logdensity_fn=logdensity_fn,
            num_chains=1,
            k_rank=5,
            inner_kernel="mclmc",
            target_acceptance_rate=0.8,  # recipe runner passes this unconditionally
        )

    # Contract: _settle_steps key must be present and equal _SETTLE_STEPS.
    assert "_settle_steps" in adapted_params, "adapted_params missing '_settle_steps'"
    from tuningfork.warmup.mclmc_lrd_tuning import _SETTLE_STEPS

    assert adapted_params["_settle_steps"] == _SETTLE_STEPS


def test_settle_produces_non_init_states():
    """After warmup, returned states must NOT be at the original init_position.

    mclmc has reinit_state=False, so the sampling loop consumes returned states
    directly.  This test asserts the settle pass actually advanced positions
    beyond the fresh init — warm-started states, not cold-start init stubs.
    """
    from tuningfork.base_method import BASE_METHODS
    from tuningfork.model import MODELS
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.warmup import WARMUPS

    entry = MODELS["ill_cond_50"]
    key = jax.random.key(13)
    init_key, warmup_key = jax.random.split(key)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)

    warmup = WARMUPS["mclmc_lrd_tuning"]
    base_method = BASE_METHODS["mclmc"]

    with pytest.warns(UserWarning, match="rank-safety bound|Clamping"):
        states, _ = warmup.runner(
            warmup_key,
            init_position,
            n_warmup=300,
            base_method=base_method,
            logdensity_fn=logdensity_fn,
            num_chains=1,
            k_rank=5,
        )

    # states.position is (num_chains=1, d=50).  The settle pass ran _SETTLE_STEPS
    # MCLMC steps — positions must differ from the original init_position.
    settled_pos = jax.tree.map(lambda x: x[0], states.position)  # squeeze chain dim
    for leaf_name, leaf_val in (
        init_position.items()
        if isinstance(init_position, dict)
        else [("x", init_position)]
    ):
        init_leaf = (
            init_position[leaf_name]
            if isinstance(init_position, dict)
            else init_position
        )
        settled_leaf = (
            settled_pos[leaf_name] if isinstance(settled_pos, dict) else settled_pos
        )
        # At least one element should differ (MCLMC always moves in 200 steps).
        assert not jnp.allclose(
            jnp.asarray(init_leaf), jnp.asarray(settled_leaf), atol=1e-6
        ), f"Settled position identical to init for leaf '{leaf_name}' — settle did not run"
        break  # one leaf is sufficient to prove settle executed


def test_unexpected_kwarg_raises_type_error():
    """Runner raises TypeError immediately for unrecognised kwargs.

    The TypeError guard runs before any JAX execution, so dummy None inputs
    are sufficient (the error is raised before jax.random.split is reached).
    """
    from tuningfork.warmup.mclmc_lrd_tuning import _runner

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _runner(
            None,
            None,
            100,
            None,
            logdensity_fn=None,
            totally_unknown_kwarg=42,
        )


def test_frac_tune1_non_certified_raises_value_error():
    """Runner raises ValueError when frac_tune1 deviates from the certified 0.5.

    This is a regression guard for C2 (upstream hardcodes frac_tune1=0.5 on the
    adjusted path; passing any other value would silently bake a misconfigured
    recipe without this guard).
    """
    from tuningfork.warmup.mclmc_lrd_tuning import _runner

    with pytest.raises(ValueError, match="frac_tune1"):
        _runner(
            None,
            None,
            100,
            None,
            logdensity_fn=None,
            frac_tune1=0.3,
        )


def test_kwargs_forwarded_to_upstream(monkeypatch):
    """All expected kwargs arrive at blackjax.mclmc_lrd_warmup under correct names.

    Monkeypatches blackjax.mclmc_lrd_warmup (via the module's `blackjax` reference)
    to intercept the call and record kwargs, then delegates to the real function.
    Asserts every tuningfork-side param name maps to the correct upstream name.
    """
    import tuningfork.warmup.mclmc_lrd_tuning as _lrd_mod
    from tuningfork.base_method import BASE_METHODS
    from tuningfork.model import MODELS
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.warmup import WARMUPS

    entry = MODELS["ill_cond_50"]
    key = jax.random.key(55)
    init_key, warmup_key = jax.random.split(key)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, entry)

    captured: dict = {}
    _original = _lrd_mod.blackjax.mclmc_lrd_warmup

    def _capturing(*args, **kwargs):
        captured.update(kwargs)
        return _original(*args, **kwargs)

    monkeypatch.setattr(_lrd_mod.blackjax, "mclmc_lrd_warmup", _capturing)

    warmup = WARMUPS["mclmc_lrd_tuning"]
    base_method = BASE_METHODS["mclmc"]

    with pytest.warns(UserWarning, match="rank-safety bound|Clamping"):
        warmup.runner(
            warmup_key,
            init_position,
            n_warmup=300,
            base_method=base_method,
            logdensity_fn=logdensity_fn,
            num_chains=1,
            k_rank=5,
            pilot_n_warmup=200,
            pilot_n_samples=200,
            inner_kernel="mclmc",
            l_init_floor_factor=1.2,
            adjusted_num_steps=1500,
            target_acceptance_rate=0.85,
        )

    # Verify every tuningfork-side name maps to the correct upstream param name.
    assert captured.get("k") == 5, f"k_rank → k: got {captured.get('k')!r}"
    assert (
        captured.get("pilot_num_warmup") == 200
    ), f"pilot_n_warmup → pilot_num_warmup: got {captured.get('pilot_num_warmup')!r}"
    assert (
        captured.get("pilot_num_samples") == 200
    ), f"pilot_n_samples → pilot_num_samples: got {captured.get('pilot_num_samples')!r}"
    assert (
        captured.get("lrd_num_steps") == 300
    ), f"n_warmup → lrd_num_steps: got {captured.get('lrd_num_steps')!r}"
    assert (
        captured.get("num_chains") == 1
    ), f"num_chains → num_chains: got {captured.get('num_chains')!r}"
    assert (
        captured.get("inner_kernel") == "mclmc"
    ), f"inner_kernel → inner_kernel: got {captured.get('inner_kernel')!r}"
    assert (
        captured.get("floor_factor") == 1.2
    ), f"l_init_floor_factor → floor_factor: got {captured.get('floor_factor')!r}"
    assert (
        captured.get("adjusted_num_steps") == 1500
    ), f"adjusted_num_steps → adjusted_num_steps: got {captured.get('adjusted_num_steps')!r}"
    # target_acceptance_rate on unadjusted path must NOT reach upstream as adjusted_target.
    assert "adjusted_target" not in captured, (
        "target_acceptance_rate should be ignored on the unadjusted path, "
        f"but 'adjusted_target' found in captured kwargs: {captured.get('adjusted_target')!r}"
    )


def test_from_warmup_only_mclmc_lrd_tuning_squeeze():
    """Recipe.from_warmup_only with mclmc_lrd_tuning squeezes step_size/L, preserves LRD.

    After squeeze_single_chain, step_size and L become scalars while the
    LowRankInverseMassMatrix passes through verbatim (per-leaf fix).

    Note: with a short pilot the rank guard may clamp k_rank; the test accepts
    that UserWarning and verifies the squeezed contract only.
    """
    from tuningfork.base_method import BASE_METHODS
    from tuningfork.model import MODELS
    from tuningfork.recipes import Effort, Recipe
    from tuningfork.warmup import WARMUPS

    entry = MODELS["ill_cond_50"]
    warmup = WARMUPS["mclmc_lrd_tuning"]
    base_method = BASE_METHODS["mclmc"]

    # Accept rank-guard UserWarning for short-pilot tests.
    with pytest.warns(UserWarning, match="rank-safety bound|Clamping"):
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


# ---------------------------------------------------------------------------
# B-c hardening: dtype-agnostic settle-key reshape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_key",
    [
        pytest.param(lambda: jax.random.key(0), id="typed-key"),
        pytest.param(lambda: jax.random.PRNGKey(0), id="legacy-PRNGKey"),
    ],
)
def test_settle_accepts_typed_and_legacy_keys(make_key):
    """Settle pass must work for both jax.random.key (typed) and PRNGKey (legacy).

    Regression guard for the settle-key reshape at mclmc_lrd_tuning.py — typed
    keys produce ``jax.random.split`` output shape ``(N,)`` while legacy
    ``PRNGKey`` produces ``(N, 2)``; the wrapper must not hard-code either shape.

    Uses the same k_rank=10 / short-pilot config as
    ``test_mclmc_lrd_tuning_warmup_returns_lrd_imm``, which reliably fires the
    rank-guard UserWarning (n_eff < 20 → k_safe < k=10 for ill_cond_50).
    The test's contract is "no crash + settle diagnostics key present", not the
    warning per se.
    """
    from tuningfork.base_method import BASE_METHODS
    from tuningfork.model import MODELS
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.warmup import WARMUPS

    entry = MODELS["ill_cond_50"]
    warmup = WARMUPS["mclmc_lrd_tuning"]
    base_method = BASE_METHODS["mclmc"]

    rng_key = make_key()
    # Build init position using a typed key (build_logdensity_fn requires it).
    # Use a fixed key independent of rng_key so both parametrize variants start
    # from the same init_position.
    init_position, logdensity_fn, _ = build_logdensity_fn(jax.random.key(99), entry)

    # The rank guard fires for k_rank=10 on a short pilot (ill_cond_50 n_eff < 20).
    with pytest.warns(UserWarning, match="rank-safety bound|Clamping"):
        states, adapted_params = warmup.runner(
            rng_key,
            init_position,
            n_warmup=500,
            base_method=base_method,
            logdensity_fn=logdensity_fn,
            num_chains=1,
            k_rank=10,
        )

    # Minimal contract: adapted_params has the expected keys, no crash.
    assert "step_size" in adapted_params
    assert "L" in adapted_params
    assert "inverse_mass_matrix" in adapted_params
    assert "_settle_steps" in adapted_params
