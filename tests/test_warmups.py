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
"""Tests for the Phase 3 (P3.1) warmup registry and Phase 5 (P5.0c) multi-chain contract.

Covers:
  1. WARMUPS dict has exactly the three expected entries.
  2. is_compatible() for stan_window: hmc/nuts → True; mclmc → False.
  3. is_compatible() for mclmc_tuning: mclmc → True; nuts → False.
  4. is_compatible() for no_warmup: any name → True (sentinel "*").
  5. stan_window smoke: NUTS on 10-D MVN at n_warmup=200, num_chains=1 (single-chain shim).
  6. mclmc_tuning smoke: MCLMC on 10-D MVN at n_warmup=200, num_chains=1 (single-chain shim).
  7. no_warmup smoke: RWM (gradient-free) and NUTS, num_chains=1.
  8. Compatibility error via _run_warmup (wrong warmup for algorithm).
  9. Auto-dispatch in tune_algorithm: mclmc → mclmc_tuning, nuts → stan_window,
     rwm → no_warmup (verified via result structure).
 10. tune_algorithm regression: existing calls with warmup_name=None still pass.
 11. Multi-chain contract tests (P5.0c): shape checks for num_chains=1/4/8,
     pre-batched init_position, dense mass matrix, MCLMC multi-chain, no_warmup multi-chain.
"""

import math

import jax
import jax.numpy as jnp
import pytest

from bjx_bench.calibration.tier_b import _run_warmup, tune_algorithm
from bjx_bench.inference.base_method import BASE_METHODS
from bjx_bench.inference.warmup import WARMUPS, Warmup
from bjx_bench.model import MODELS

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_MVN = MODELS["mvn_10"]
_NUTS = BASE_METHODS["nuts"]
_HMC = BASE_METHODS["hmc"]
_MCLMC = BASE_METHODS["mclmc"]
_RWM = BASE_METHODS["rwm"]
_MALA = BASE_METHODS["mala"]
_GHMC = BASE_METHODS["ghmc"]

# Build a shared 10-D MVN logdensity_fn + init_position for smoke tests.
# We do this once at module level to avoid per-test model compilation.
_SEED = 42
_RNG_KEY = jax.random.key(_SEED)

# ---------------------------------------------------------------------------
# Helper: build logdensity_fn + position from the model registry
# ---------------------------------------------------------------------------


def _build_logdensity(posterior_entry, key):
    from bjx_bench.model._numpyro import build_logdensity_fn

    init_position, logdensity_fn, _ = build_logdensity_fn(key, posterior_entry)
    return init_position, logdensity_fn


# ---------------------------------------------------------------------------
# 1. Registry contains exactly 3 entries
# ---------------------------------------------------------------------------


class TestWarmupRegistry:
    """WARMUPS registry structure tests (fast; no chain runs)."""

    def test_warmups_has_expected_entries(self) -> None:
        """Subset assertion: all known warmups must be present (META-011 pattern)."""
        expected = {
            "stan_window",
            "mclmc_tuning",
            "adjusted_mclmc_tuning",
            "no_warmup",
            "pathfinder",
            "multipathfinder",
            "meads",
            "chees",
        }
        assert expected <= set(WARMUPS.keys()), (
            f"Missing warmup entries: {expected - set(WARMUPS.keys())}. "
            f"Registered: {sorted(WARMUPS)}"
        )

    def test_warmups_has_stan_window(self) -> None:
        assert "stan_window" in WARMUPS

    def test_warmups_has_mclmc_tuning(self) -> None:
        assert "mclmc_tuning" in WARMUPS

    def test_warmups_has_no_warmup(self) -> None:
        assert "no_warmup" in WARMUPS

    def test_warmups_has_pathfinder(self) -> None:
        """P5.4: pathfinder warmup is registered."""
        assert "pathfinder" in WARMUPS

    def test_warmups_has_multipathfinder(self) -> None:
        """P5.4: multipathfinder warmup is registered."""
        assert "multipathfinder" in WARMUPS

    def test_warmups_has_meads(self) -> None:
        """P5.5: meads warmup is registered."""
        assert "meads" in WARMUPS

    def test_all_entries_are_warmup_instances(self) -> None:
        for name, entry in WARMUPS.items():
            assert isinstance(
                entry, Warmup
            ), f"WARMUPS[{name!r}] is not a Warmup instance"

    def test_warmup_names_match_keys(self) -> None:
        for key, entry in WARMUPS.items():
            assert (
                entry.name == key
            ), f"WARMUPS[{key!r}].name = {entry.name!r} doesn't match key"


# ---------------------------------------------------------------------------
# 2–4. is_compatible()
# ---------------------------------------------------------------------------


class TestIsCompatible:
    """is_compatible() for each warmup (fast; pure logic)."""

    # -- stan_window --
    def test_stan_window_compatible_with_hmc(self) -> None:
        assert WARMUPS["stan_window"].is_compatible("hmc")

    def test_stan_window_compatible_with_nuts(self) -> None:
        assert WARMUPS["stan_window"].is_compatible("nuts")

    def test_stan_window_compatible_with_barker(self) -> None:
        assert WARMUPS["stan_window"].is_compatible("barker")

    def test_stan_window_compatible_with_mala(self) -> None:
        assert WARMUPS["stan_window"].is_compatible("mala")

    def test_stan_window_not_compatible_with_mclmc(self) -> None:
        assert not WARMUPS["stan_window"].is_compatible("mclmc")

    def test_stan_window_not_compatible_with_rwm(self) -> None:
        assert not WARMUPS["stan_window"].is_compatible("rwm")

    # -- mclmc_tuning --
    def test_mclmc_tuning_compatible_with_mclmc(self) -> None:
        assert WARMUPS["mclmc_tuning"].is_compatible("mclmc")

    def test_mclmc_tuning_not_compatible_with_nuts(self) -> None:
        assert not WARMUPS["mclmc_tuning"].is_compatible("nuts")

    def test_mclmc_tuning_not_compatible_with_hmc(self) -> None:
        assert not WARMUPS["mclmc_tuning"].is_compatible("hmc")

    def test_mclmc_tuning_not_compatible_with_rwm(self) -> None:
        assert not WARMUPS["mclmc_tuning"].is_compatible("rwm")

    # -- pathfinder (P5.4) --
    def test_pathfinder_compatible_with_nuts(self) -> None:
        assert WARMUPS["pathfinder"].is_compatible("nuts")

    def test_pathfinder_compatible_with_hmc(self) -> None:
        assert WARMUPS["pathfinder"].is_compatible("hmc")

    def test_pathfinder_compatible_with_barker(self) -> None:
        assert WARMUPS["pathfinder"].is_compatible("barker")

    def test_pathfinder_not_compatible_with_mclmc(self) -> None:
        assert not WARMUPS["pathfinder"].is_compatible("mclmc")

    # -- multipathfinder (P5.4) --
    def test_multipathfinder_compatible_with_nuts(self) -> None:
        assert WARMUPS["multipathfinder"].is_compatible("nuts")

    def test_multipathfinder_compatible_with_hmc(self) -> None:
        assert WARMUPS["multipathfinder"].is_compatible("hmc")

    def test_multipathfinder_compatible_with_barker(self) -> None:
        assert WARMUPS["multipathfinder"].is_compatible("barker")

    def test_multipathfinder_not_compatible_with_mclmc(self) -> None:
        assert not WARMUPS["multipathfinder"].is_compatible("mclmc")

    # -- no_warmup (sentinel "*") --
    def test_no_warmup_compatible_with_nuts(self) -> None:
        assert WARMUPS["no_warmup"].is_compatible("nuts")

    def test_no_warmup_compatible_with_mclmc(self) -> None:
        assert WARMUPS["no_warmup"].is_compatible("mclmc")

    def test_no_warmup_compatible_with_rwm(self) -> None:
        assert WARMUPS["no_warmup"].is_compatible("rwm")

    def test_no_warmup_compatible_with_anything(self) -> None:
        assert WARMUPS["no_warmup"].is_compatible("hypothetical_future_sampler")

    def test_no_warmup_has_star_sentinel(self) -> None:
        assert "*" in WARMUPS["no_warmup"].compatible_methods


# ---------------------------------------------------------------------------
# 5. stan_window smoke: NUTS on MVN-10 at n_warmup=200, num_chains=1 (single-chain shim)
# ---------------------------------------------------------------------------


class TestStanWindowSmoke:
    """stan_window smoke test on NUTS + MVN-10.

    Uses num_chains=1 to preserve backward-compatible shim semantics.
    Output shapes have a leading dim of 1.
    """

    def _run(self, seed: int, **kw):
        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        warmup_key = jax.random.fold_in(key, 1)
        return WARMUPS["stan_window"].runner(
            warmup_key, init_pos, 200, _NUTS, logdensity_fn=logdensity_fn, **kw
        )

    def test_returns_state_and_adapted_params(self) -> None:
        state, params = self._run(101, num_chains=1)
        assert state is not None
        assert isinstance(params, dict)

    def test_adapted_params_has_step_size(self) -> None:
        _, params = self._run(102, num_chains=1)
        assert "step_size" in params, f"params keys: {list(params.keys())}"

    def test_adapted_params_has_inverse_mass_matrix(self) -> None:
        _, params = self._run(103, num_chains=1)
        assert "inverse_mass_matrix" in params, f"params keys: {list(params.keys())}"

    def test_step_size_positive(self) -> None:
        _, params = self._run(104, num_chains=1)
        step_sizes = jnp.asarray(params["step_size"])
        # shape (1,) — all positive
        assert bool(jnp.all(step_sizes > 0)), f"step_size={step_sizes} not all > 0"

    def test_inverse_mass_matrix_shape(self) -> None:
        _, params = self._run(105, num_chains=1)
        imm = params["inverse_mass_matrix"]
        # num_chains=1 → shape (1, 10)
        assert imm.shape == (
            1,
            10,
        ), f"inverse_mass_matrix.shape={imm.shape}, expected (1, 10) for num_chains=1 diagonal"

    def test_dense_mass_matrix_shape(self) -> None:
        """P5.0b: is_mass_matrix_diagonal=False produces (num_chains, d, d) IMM."""
        _, params = self._run(106, num_chains=1, is_mass_matrix_diagonal=False)
        imm = params["inverse_mass_matrix"]
        # num_chains=1 → shape (1, 10, 10)
        assert imm.shape == (
            1,
            10,
            10,
        ), f"inverse_mass_matrix.shape={imm.shape}, expected (1, 10, 10)"

    def test_dense_mass_matrix_is_symmetric_positive_definite(self) -> None:
        """Sanity check: dense IMM should be symmetric and PD (per chain)."""
        _, params = self._run(107, num_chains=1, is_mass_matrix_diagonal=False)
        imm = params["inverse_mass_matrix"]
        # imm has shape (1, 10, 10); check chain 0
        imm_chain0 = imm[0]
        assert jnp.allclose(
            imm_chain0, imm_chain0.T, atol=1e-6
        ), "dense IMM must be symmetric"
        try:
            jnp.linalg.cholesky(imm_chain0)
        except Exception as exc:  # pragma: no cover
            raise AssertionError(f"dense IMM not positive definite: {exc}") from exc


# ---------------------------------------------------------------------------
# 6. mclmc_tuning smoke: MCLMC on MVN-10 at n_warmup=200, num_chains=1
# ---------------------------------------------------------------------------


class TestMclmcTuningSmoke:
    """mclmc_tuning smoke test on MCLMC + MVN-10.

    Uses num_chains=1 to preserve backward-compatible shim semantics.
    Output shapes have a leading dim of 1.
    """

    def _run(self, seed: int) -> tuple[object, dict]:
        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        warmup_key = jax.random.fold_in(key, 1)
        return WARMUPS["mclmc_tuning"].runner(
            warmup_key, init_pos, 200, _MCLMC, logdensity_fn=logdensity_fn, num_chains=1
        )

    def test_returns_state_and_adapted_params(self) -> None:
        state, params = self._run(201)
        assert state is not None
        assert isinstance(params, dict)

    def test_adapted_params_has_L(self) -> None:
        _, params = self._run(202)
        assert "L" in params, f"params keys: {list(params.keys())}"

    def test_adapted_params_has_step_size(self) -> None:
        _, params = self._run(203)
        assert "step_size" in params, f"params keys: {list(params.keys())}"

    def test_adapted_params_has_inverse_mass_matrix(self) -> None:
        _, params = self._run(204)
        assert "inverse_mass_matrix" in params, f"params keys: {list(params.keys())}"

    def test_adapted_params_has_total_tuning_steps(self) -> None:
        _, params = self._run(205)
        assert "_total_tuning_steps" in params, f"params keys: {list(params.keys())}"

    def test_L_positive(self) -> None:
        _, params = self._run(206)
        # shape (1,) for num_chains=1
        assert bool(jnp.all(jnp.asarray(params["L"]) > 0)), f"L={params['L']} not > 0"

    def test_step_size_positive(self) -> None:
        _, params = self._run(207)
        assert bool(
            jnp.all(jnp.asarray(params["step_size"]) > 0)
        ), f"step_size={params['step_size']} not > 0"

    def test_inverse_mass_matrix_shape(self) -> None:
        _, params = self._run(208)
        imm = params["inverse_mass_matrix"]
        # num_chains=1 → shape (1, 10)
        assert imm.shape == (
            1,
            10,
        ), f"inverse_mass_matrix.shape={imm.shape}, expected (1, 10)"

    def test_total_tuning_steps_positive(self) -> None:
        _, params = self._run(209)
        steps = int(params["_total_tuning_steps"])
        assert steps > 0, f"_total_tuning_steps={steps} <= 0"


# ---------------------------------------------------------------------------
# 7. no_warmup smoke (num_chains=1 shim)
# ---------------------------------------------------------------------------


class TestNoWarmupSmoke:
    """no_warmup smoke tests on RWM and NUTS (num_chains=1 shim)."""

    def test_rwm_returns_state_and_empty_params(self) -> None:
        key = jax.random.key(301)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        state, params = WARMUPS["no_warmup"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _RWM,
            logdensity_fn=logdensity_fn,
            num_chains=1,
        )
        assert state is not None
        assert params == {}, f"Expected empty dict, got {params}"

    def test_nuts_returns_state_and_empty_params(self) -> None:
        key = jax.random.key(302)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        state, params = WARMUPS["no_warmup"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
            num_chains=1,
        )
        assert state is not None
        assert params == {}, f"Expected empty dict, got {params}"

    def test_mclmc_rng_key_threading_works(self) -> None:
        """MCLMC requires kernel.init(position, rng_key); no_warmup handles this."""
        key = jax.random.key(303)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        # Should NOT raise TypeError about unexpected rng_key argument.
        state, params = WARMUPS["no_warmup"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _MCLMC,
            logdensity_fn=logdensity_fn,
            num_chains=1,
        )
        assert state is not None
        assert params == {}


# ---------------------------------------------------------------------------
# 8. Compatibility error via _run_warmup
# ---------------------------------------------------------------------------


class TestCompatibilityError:
    """_run_warmup raises ValueError when warmup is incompatible with algorithm."""

    def test_mclmc_tuning_on_nuts_raises(self) -> None:
        key = jax.random.key(401)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        with pytest.raises(ValueError, match="not compatible with"):
            _run_warmup(
                logdensity_fn=logdensity_fn,
                init_position=init_pos,
                algorithm_entry=_NUTS,
                n_warmup=50,
                rng_key=jax.random.fold_in(key, 1),
                warmup_name="mclmc_tuning",
            )

    def test_stan_window_on_mclmc_raises(self) -> None:
        key = jax.random.key(402)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        with pytest.raises(ValueError, match="not compatible with"):
            _run_warmup(
                logdensity_fn=logdensity_fn,
                init_position=init_pos,
                algorithm_entry=_MCLMC,
                n_warmup=50,
                rng_key=jax.random.fold_in(key, 1),
                warmup_name="stan_window",
            )

    def test_unknown_warmup_name_raises(self) -> None:
        key = jax.random.key(403)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        with pytest.raises(ValueError, match="unknown warmup"):
            _run_warmup(
                logdensity_fn=logdensity_fn,
                init_position=init_pos,
                algorithm_entry=_NUTS,
                n_warmup=50,
                rng_key=jax.random.fold_in(key, 1),
                warmup_name="nonexistent_warmup",
            )


# ---------------------------------------------------------------------------
# 9. Auto-dispatch in tune_algorithm (smoke, 1 trial each)
# ---------------------------------------------------------------------------


_AUTO_N_TRIALS = 1
_AUTO_N_SEEDS = 1
_AUTO_N_CHAINS = 1
_AUTO_N_SAMPLES = 50
_AUTO_N_WARMUP = 100


class TestAutoDispatch:
    """tune_algorithm auto-dispatch resolves warmup_name=None correctly.

    We only run n_trials=1 to keep runtime low; the goal is to confirm
    the dispatch doesn't raise and produces a valid TuningResult.
    """

    def test_mclmc_auto_dispatches_to_mclmc_tuning(self) -> None:
        """MCLMC with warmup_name=None should use mclmc_tuning (not stan_window)."""
        result = tune_algorithm(
            _MVN,
            _MCLMC,
            rng_key=jax.random.key(501),
            n_trials=_AUTO_N_TRIALS,
            n_seeds=_AUTO_N_SEEDS,
            n_chains=_AUTO_N_CHAINS,
            n_samples=_AUTO_N_SAMPLES,
            n_warmup=_AUTO_N_WARMUP,
        )
        # If auto-dispatch went to stan_window instead, it would raise
        # ValueError("not compatible with").  So if we reach this assertion,
        # dispatch is correct.
        assert result.base_method_name == "mclmc"

    def test_nuts_auto_dispatches_to_stan_window(self) -> None:
        """NUTS with warmup_name=None should use stan_window."""
        result = tune_algorithm(
            _MVN,
            _NUTS,
            rng_key=jax.random.key(502),
            n_trials=_AUTO_N_TRIALS,
            n_seeds=_AUTO_N_SEEDS,
            n_chains=_AUTO_N_CHAINS,
            n_samples=_AUTO_N_SAMPLES,
            n_warmup=_AUTO_N_WARMUP,
        )
        assert result.base_method_name == "nuts"
        # Verify: best_score is finite (stan_window warmup worked).
        assert math.isfinite(result.best_score), f"best_score={result.best_score}"

    def test_rwm_auto_dispatches_to_no_warmup(self) -> None:
        """RWM with warmup_name=None should use no_warmup."""
        result = tune_algorithm(
            _MVN,
            _RWM,
            rng_key=jax.random.key(503),
            n_trials=_AUTO_N_TRIALS,
            n_seeds=_AUTO_N_SEEDS,
            n_chains=_AUTO_N_CHAINS,
            n_samples=_AUTO_N_SAMPLES,
            n_warmup=_AUTO_N_WARMUP,
        )
        assert result.base_method_name == "rwm"

    def test_explicit_warmup_name_overrides_auto(self) -> None:
        """Passing warmup_name='no_warmup' for NUTS should skip stan_window."""
        result = tune_algorithm(
            _MVN,
            _NUTS,
            warmup_name="no_warmup",
            rng_key=jax.random.key(504),
            n_trials=_AUTO_N_TRIALS,
            n_seeds=_AUTO_N_SEEDS,
            n_chains=_AUTO_N_CHAINS,
            n_samples=_AUTO_N_SAMPLES,
            n_warmup=_AUTO_N_WARMUP,
        )
        # no_warmup is compatible with NUTS via "*"; should not raise.
        assert result.base_method_name == "nuts"


# ---------------------------------------------------------------------------
# 10. Regression: existing tune_algorithm calls still pass unchanged
# ---------------------------------------------------------------------------


class TestTuneAlgorithmRegression:
    """Confirm that the refactored _run_warmup produces the same structural
    outcomes as the old inline dispatch.  Only structural tests (not numeric)
    because the warmup key path is identical to old code.
    """

    _N_WARMUP = 200
    _N_SAMPLES = 200
    _N_TRIALS = 3
    _N_SEEDS = 1
    _N_CHAINS = 1

    def _run(self, algo, seed):
        return tune_algorithm(
            _MVN,
            algo,
            n_trials=self._N_TRIALS,
            n_seeds=self._N_SEEDS,
            n_chains=self._N_CHAINS,
            n_samples=self._N_SAMPLES,
            n_warmup=self._N_WARMUP,
            rng_key=jax.random.key(seed),
        )

    def test_nuts_result_structure_unchanged(self) -> None:
        result = self._run(_NUTS, 600)
        assert result.base_method_name == "nuts"
        assert result.n_trials_completed == self._N_TRIALS
        assert len(result.history) == self._N_TRIALS
        assert math.isfinite(result.best_score)

    def test_hmc_result_structure_unchanged(self) -> None:
        result = self._run(_HMC, 601)
        assert result.base_method_name == "hmc"
        assert result.n_trials_completed == self._N_TRIALS
        assert "step_size" in result.best_params
        assert "num_integration_steps" in result.best_params

    def test_mclmc_result_structure_unchanged(self) -> None:
        result = self._run(_MCLMC, 602)
        assert result.base_method_name == "mclmc"
        assert result.n_trials_completed == self._N_TRIALS
        assert "step_size" in result.best_params
        assert "L" in result.best_params

    def test_rwm_result_structure_unchanged(self) -> None:
        result = self._run(_RWM, 603)
        assert result.base_method_name == "rwm"
        assert result.n_trials_completed == self._N_TRIALS

    def test_mala_result_structure_unchanged(self) -> None:
        result = self._run(_MALA, 604)
        assert result.base_method_name == "mala"
        assert result.n_trials_completed == self._N_TRIALS


# ---------------------------------------------------------------------------
# 11. Multi-chain contract tests (P5.0c)
# ---------------------------------------------------------------------------


def _state_leading_dim(states) -> int:
    """Return the leading dimension of the vmapped state pytree."""
    leaves = jax.tree.leaves(states)
    return leaves[0].shape[0]


def _position_shape(states) -> tuple:
    """Return (num_chains, *param_shape) from the position pytree."""
    pos_leaves = jax.tree.leaves(states.position)
    return pos_leaves[0].shape


class TestStanWindowMultiChain:
    """P5.0c: stan_window multi-chain shape contract tests."""

    def _run(self, seed: int, num_chains: int, **kw):
        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        return WARMUPS["stan_window"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
            **kw,
        )

    def test_default_num_chains_is_4(self) -> None:
        """No num_chains kwarg → default 4; position leading dim == 4."""
        key = jax.random.key(1001)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        states, params = WARMUPS["stan_window"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
        )
        leading = _state_leading_dim(states)
        assert leading == 4, f"Default num_chains should be 4, got {leading}"

    def test_explicit_num_chains_4_state_shape(self) -> None:
        """num_chains=4 → position shape is (4, 10)."""
        states, params = self._run(1002, num_chains=4)
        pos_shape = _position_shape(states)
        assert pos_shape == (4, 10), f"Expected (4, 10), got {pos_shape}"

    def test_explicit_num_chains_4_step_size_shape(self) -> None:
        """num_chains=4 → step_size has shape (4,)."""
        _, params = self._run(1003, num_chains=4)
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (4,), f"Expected (4,), got {ss.shape}"

    def test_explicit_num_chains_4_imm_shape(self) -> None:
        """num_chains=4 → inverse_mass_matrix has shape (4, 10)."""
        _, params = self._run(1004, num_chains=4)
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (4, 10), f"Expected (4, 10), got {imm.shape}"

    def test_explicit_num_chains_8(self) -> None:
        """num_chains=8 → leading dim 8."""
        states, params = self._run(1005, num_chains=8)
        leading = _state_leading_dim(states)
        assert leading == 8, f"Expected leading dim 8, got {leading}"
        assert params["step_size"].shape == (
            8,
        ), f"Expected (8,), got {params['step_size'].shape}"
        assert params["inverse_mass_matrix"].shape == (
            8,
            10,
        ), f"Expected (8, 10), got {params['inverse_mass_matrix'].shape}"

    def test_num_chains_1_not_squeezed(self) -> None:
        """num_chains=1 → leading dim 1 (NOT squeezed)."""
        states, params = self._run(1006, num_chains=1)
        leading = _state_leading_dim(states)
        assert leading == 1, f"num_chains=1 should give leading dim 1, got {leading}"
        assert params["step_size"].shape == (
            1,
        ), f"Expected (1,), got {params['step_size'].shape}"

    def test_pre_batched_init_position(self) -> None:
        """init_position with leading dim == num_chains passes through verbatim."""
        num_chains = 4
        key = jax.random.key(1007)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        # Pre-batch: replicate 4 times
        batched_pos = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_chains,) + x.shape), init_pos
        )
        states, params = WARMUPS["stan_window"].runner(
            jax.random.fold_in(key, 1),
            batched_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
        )
        pos_shape = _position_shape(states)
        assert pos_shape == (4, 10), f"Pre-batched: expected (4, 10), got {pos_shape}"

    def test_dense_mm_multi_chain_shape(self) -> None:
        """num_chains=2, is_mass_matrix_diagonal=False → IMM shape (2, d, d)."""
        states, params = self._run(1008, num_chains=2, is_mass_matrix_diagonal=False)
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (
            2,
            10,
            10,
        ), f"Expected (2, 10, 10) for dense multi-chain, got {imm.shape}"
        pos_shape = _position_shape(states)
        assert pos_shape == (2, 10), f"Expected (2, 10), got {pos_shape}"

    def test_all_step_sizes_positive(self) -> None:
        """All per-chain step sizes must be positive."""
        _, params = self._run(1009, num_chains=4)
        ss = jnp.asarray(params["step_size"])
        assert bool(jnp.all(ss > 0)), f"Not all step sizes positive: {ss}"


class TestMclmcTuningMultiChain:
    """P5.0c: mclmc_tuning multi-chain shape contract tests."""

    def _run(self, seed: int, num_chains: int) -> tuple[object, dict]:
        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        warmup_key = jax.random.fold_in(key, 1)
        return WARMUPS["mclmc_tuning"].runner(
            warmup_key,
            init_pos,
            200,
            _MCLMC,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
        )

    def test_default_num_chains_is_4(self) -> None:
        """No num_chains kwarg → default 4."""
        key = jax.random.key(2001)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        states, params = WARMUPS["mclmc_tuning"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _MCLMC,
            logdensity_fn=logdensity_fn,
        )
        leading = _state_leading_dim(states)
        assert leading == 4, f"Default num_chains should be 4, got {leading}"

    def test_num_chains_4_state_shape(self) -> None:
        """num_chains=4 → position leading dim == 4."""
        states, params = self._run(2002, num_chains=4)
        leading = _state_leading_dim(states)
        assert leading == 4, f"Expected leading dim 4, got {leading}"

    def test_num_chains_4_L_shape(self) -> None:
        """num_chains=4 → L has shape (4,)."""
        _, params = self._run(2003, num_chains=4)
        L = jnp.asarray(params["L"])
        assert L.shape == (4,), f"Expected (4,), got {L.shape}"

    def test_num_chains_4_step_size_shape(self) -> None:
        """num_chains=4 → step_size has shape (4,)."""
        _, params = self._run(2004, num_chains=4)
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (4,), f"Expected (4,), got {ss.shape}"

    def test_num_chains_4_imm_shape(self) -> None:
        """num_chains=4 → inverse_mass_matrix has shape (4, 10)."""
        _, params = self._run(2005, num_chains=4)
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (4, 10), f"Expected (4, 10), got {imm.shape}"

    def test_num_chains_1_not_squeezed(self) -> None:
        """num_chains=1 → leading dim 1 (NOT squeezed)."""
        states, params = self._run(2006, num_chains=1)
        leading = _state_leading_dim(states)
        assert leading == 1, f"num_chains=1 should give leading dim 1, got {leading}"

    def test_total_tuning_steps_is_int(self) -> None:
        """_total_tuning_steps is a Python int (not a JAX array)."""
        _, params = self._run(2007, num_chains=4)
        steps = params["_total_tuning_steps"]
        assert isinstance(
            steps, int
        ), f"_total_tuning_steps should be int, got {type(steps)}"
        assert steps > 0, f"_total_tuning_steps={steps} <= 0"

    def test_all_L_positive(self) -> None:
        """All per-chain L values must be positive."""
        _, params = self._run(2008, num_chains=4)
        L = jnp.asarray(params["L"])
        assert bool(jnp.all(L > 0)), f"Not all L values positive: {L}"


class TestNoWarmupMultiChain:
    """P5.0c: no_warmup multi-chain shape contract tests."""

    def _run(self, seed: int, base_method, num_chains: int) -> tuple[object, dict]:
        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        return WARMUPS["no_warmup"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            base_method,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
        )

    def test_default_num_chains_is_4(self) -> None:
        """No num_chains kwarg → default 4."""
        key = jax.random.key(3001)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        states, params = WARMUPS["no_warmup"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _RWM,
            logdensity_fn=logdensity_fn,
        )
        leading = _state_leading_dim(states)
        assert leading == 4, f"Default num_chains should be 4, got {leading}"

    def test_num_chains_4_nuts_state_shape(self) -> None:
        """num_chains=4, NUTS → position leading dim == 4."""
        states, params = self._run(3002, _NUTS, num_chains=4)
        pos_shape = _position_shape(states)
        assert pos_shape == (4, 10), f"Expected (4, 10), got {pos_shape}"
        assert params == {}, f"no_warmup always returns empty dict, got {params}"

    def test_num_chains_4_rwm_state_shape(self) -> None:
        """num_chains=4, RWM → position leading dim == 4."""
        states, params = self._run(3003, _RWM, num_chains=4)
        pos_shape = _position_shape(states)
        assert pos_shape == (4, 10), f"Expected (4, 10), got {pos_shape}"
        assert params == {}

    def test_num_chains_4_mclmc_state_shape(self) -> None:
        """num_chains=4, MCLMC → position leading dim == 4."""
        states, params = self._run(3004, _MCLMC, num_chains=4)
        pos_shape = _position_shape(states)
        assert pos_shape == (4, 10), f"Expected (4, 10), got {pos_shape}"
        assert params == {}

    def test_num_chains_1_not_squeezed(self) -> None:
        """num_chains=1 → leading dim 1 (NOT squeezed)."""
        states, params = self._run(3005, _NUTS, num_chains=1)
        leading = _state_leading_dim(states)
        assert leading == 1, f"num_chains=1 should give leading dim 1, got {leading}"
        assert params == {}

    def test_adapted_params_always_empty(self) -> None:
        """no_warmup always returns {} regardless of num_chains."""
        for nc in (1, 2, 4, 8):
            _, params = self._run(3010 + nc, _NUTS, num_chains=nc)
            assert params == {}, f"num_chains={nc}: expected empty dict, got {params}"


# ---------------------------------------------------------------------------
# 12. P5.4: Pathfinder multi-chain warmup
# ---------------------------------------------------------------------------

_D = 10  # MVN-10 has 10 dimensions


class TestPathfinderMultiChain:
    """P5.4: pathfinder warmup multi-chain shape contract tests.

    Single-path Pathfinder: one independent L-BFGS run per chain (vmap).
    Returns init positions drawn from the per-chain variational surrogate and
    the per-chain alpha (diagonal of L-BFGS inverse Hessian) as IMM.
    """

    def _run(self, seed: int, num_chains: int, **kw):
        from bjx_bench.inference.warmup.pathfinder import ENTRY

        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        return ENTRY.runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
            **kw,
        )

    def test_default_num_chains_equals_4(self) -> None:
        """Default num_chains=4: position leading dim == 4."""
        from bjx_bench.inference.warmup.pathfinder import ENTRY

        key = jax.random.key(4001)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        states, params = ENTRY.runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
        )
        leading = _state_leading_dim(states)
        assert leading == 4, f"Default num_chains should be 4, got {leading}"
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (4,), f"step_size expected (4,), got {ss.shape}"
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (4, _D), f"IMM expected (4, {_D}), got {imm.shape}"

    def test_explicit_num_chains_2(self) -> None:
        """num_chains=2: position (2, d), step_size (2,), IMM (2, d)."""
        states, params = self._run(4002, num_chains=2)
        pos_shape = _position_shape(states)
        assert pos_shape == (2, _D), f"Expected (2, {_D}), got {pos_shape}"
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (2,), f"step_size expected (2,), got {ss.shape}"
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (2, _D), f"IMM expected (2, {_D}), got {imm.shape}"

    def test_pre_batched_init_position_passes_through(self) -> None:
        """Pre-batched init_position (leading dim == num_chains) is not double-broadcast."""
        from bjx_bench.inference.warmup.pathfinder import ENTRY

        num_chains = 4
        key = jax.random.key(4003)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        batched_pos = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_chains,) + x.shape), init_pos
        )
        states, params = ENTRY.runner(
            jax.random.fold_in(key, 1),
            batched_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
        )
        pos_shape = _position_shape(states)
        assert pos_shape == (4, _D), f"Pre-batched: expected (4, {_D}), got {pos_shape}"

    def test_step_size_and_imm_keys_present(self) -> None:
        """adapted_params must contain step_size and inverse_mass_matrix."""
        _, params = self._run(4004, num_chains=2)
        assert "step_size" in params, f"Missing step_size; keys: {list(params)}"
        assert "inverse_mass_matrix" in params, f"Missing IMM; keys: {list(params)}"

    def test_logz_estimate_sidecar_present(self) -> None:
        """_pathfinder_logZ_estimate sidecar must be present."""
        _, params = self._run(4005, num_chains=2)
        assert (
            "_pathfinder_logZ_estimate" in params
        ), f"Missing _pathfinder_logZ_estimate; keys: {list(params)}"
        elbo = jnp.asarray(params["_pathfinder_logZ_estimate"])
        assert elbo.shape == (2,), f"Expected (2,), got {elbo.shape}"

    def test_step_size_is_constant_default(self) -> None:
        """Pathfinder returns a constant step_size_default per chain (no adaptation)."""
        _, params = self._run(4006, num_chains=4)
        ss = jnp.asarray(params["step_size"])
        assert bool(jnp.all(ss == 1.0)), f"All step_sizes should be 1.0, got {ss}"

    def test_num_chains_1_not_squeezed(self) -> None:
        """num_chains=1 → leading dim 1 (NOT squeezed)."""
        states, params = self._run(4007, num_chains=1)
        leading = _state_leading_dim(states)
        assert leading == 1, f"num_chains=1 should give leading dim 1, got {leading}"
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (1,), f"step_size expected (1,), got {ss.shape}"
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (1, _D), f"IMM expected (1, {_D}), got {imm.shape}"

    def test_compatibility_check_raises_for_mclmc(self) -> None:
        """is_compatible('mclmc') returns False; runner raises ValueError for mclmc."""
        from bjx_bench.inference.warmup.pathfinder import ENTRY

        assert not ENTRY.is_compatible(
            "mclmc"
        ), "pathfinder should not be compatible with mclmc"

        key = jax.random.key(4008)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        with pytest.raises(ValueError, match="not compatible with"):
            ENTRY.runner(
                jax.random.fold_in(key, 1),
                init_pos,
                200,
                _MCLMC,
                logdensity_fn=logdensity_fn,
                num_chains=2,
            )


# ---------------------------------------------------------------------------
# 13. P5.4: MultiPathfinder multi-chain warmup
# ---------------------------------------------------------------------------


class TestMultiPathfinderMultiChain:
    """P5.4: multipathfinder warmup multi-chain shape contract tests.

    Multi-path Pathfinder: one multi-path fit feeds num_chains init positions
    drawn from the PSIS-resampled mixture.  The post-PSIS empirical covariance
    diagonal is used as the IMM (same value replicated to each chain).
    """

    def _run(self, seed: int, num_chains: int, **kw):
        from bjx_bench.inference.warmup.multipathfinder import ENTRY

        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        return ENTRY.runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
            **kw,
        )

    def test_default_num_chains_equals_4(self) -> None:
        """Default num_chains=4: position leading dim == 4."""
        from bjx_bench.inference.warmup.multipathfinder import ENTRY

        key = jax.random.key(5001)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        states, params = ENTRY.runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
        )
        leading = _state_leading_dim(states)
        assert leading == 4, f"Default num_chains should be 4, got {leading}"
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (4,), f"step_size expected (4,), got {ss.shape}"
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (4, _D), f"IMM expected (4, {_D}), got {imm.shape}"

    def test_explicit_num_chains_2(self) -> None:
        """num_chains=2: position (2, d), step_size (2,), IMM (2, d)."""
        states, params = self._run(5002, num_chains=2)
        pos_shape = _position_shape(states)
        assert pos_shape == (2, _D), f"Expected (2, {_D}), got {pos_shape}"
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (2,), f"step_size expected (2,), got {ss.shape}"
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (2, _D), f"IMM expected (2, {_D}), got {imm.shape}"

    def test_pre_batched_init_position_passes_through(self) -> None:
        """Pre-batched init_position (leading dim == n_paths) passes through."""
        from bjx_bench.inference.warmup.multipathfinder import ENTRY

        n_paths = 4
        num_chains = 4
        key = jax.random.key(5003)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        # Pre-batch to (n_paths, d); multipathfinder uses this as init_positions.
        batched_pos = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (n_paths,) + x.shape), init_pos
        )
        states, params = ENTRY.runner(
            jax.random.fold_in(key, 1),
            batched_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
            n_paths=n_paths,
            num_chains=num_chains,
        )
        pos_shape = _position_shape(states)
        assert pos_shape == (4, _D), f"Pre-batched: expected (4, {_D}), got {pos_shape}"

    def test_step_size_and_imm_keys_present(self) -> None:
        """adapted_params must contain step_size and inverse_mass_matrix."""
        _, params = self._run(5004, num_chains=2)
        assert "step_size" in params, f"Missing step_size; keys: {list(params)}"
        assert "inverse_mass_matrix" in params, f"Missing IMM; keys: {list(params)}"

    def test_psis_diagnostics_in_calibration_metadata(self) -> None:
        """_multipathfinder_psis_pareto_k sidecar must be present."""
        _, params = self._run(5005, num_chains=2)
        assert (
            "_multipathfinder_psis_pareto_k" in params
        ), f"Missing _multipathfinder_psis_pareto_k; keys: {list(params)}"

    def test_step_size_is_constant_default(self) -> None:
        """MultiPathfinder returns constant step_size_default per chain."""
        _, params = self._run(5006, num_chains=4)
        ss = jnp.asarray(params["step_size"])
        assert bool(jnp.all(ss == 1.0)), f"All step_sizes should be 1.0, got {ss}"

    def test_num_chains_1_not_squeezed(self) -> None:
        """num_chains=1 → leading dim 1 (NOT squeezed)."""
        states, params = self._run(5007, num_chains=1)
        leading = _state_leading_dim(states)
        assert leading == 1, f"num_chains=1 should give leading dim 1, got {leading}"
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (1,), f"step_size expected (1,), got {ss.shape}"
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (1, _D), f"IMM expected (1, {_D}), got {imm.shape}"

    def test_imm_values_are_identical_across_chains(self) -> None:
        """All chains share the same IMM (same post-PSIS empirical variance)."""
        _, params = self._run(5008, num_chains=4)
        imm = params["inverse_mass_matrix"]
        # All rows should be equal (same shared estimate).
        for i in range(1, 4):
            assert jnp.allclose(
                imm[0], imm[i], atol=1e-6
            ), f"IMM row {i} differs from row 0: {imm[0]} vs {imm[i]}"

    def test_compatibility_check_raises_for_mclmc(self) -> None:
        """is_compatible('mclmc') returns False; runner raises ValueError for mclmc."""
        from bjx_bench.inference.warmup.multipathfinder import ENTRY

        assert not ENTRY.is_compatible(
            "mclmc"
        ), "multipathfinder should not be compatible with mclmc"

        key = jax.random.key(5009)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        with pytest.raises(ValueError, match="not compatible with"):
            ENTRY.runner(
                jax.random.fold_in(key, 1),
                init_pos,
                200,
                _MCLMC,
                logdensity_fn=logdensity_fn,
                num_chains=2,
            )


# ---------------------------------------------------------------------------
# 14. P5.5: MEADS multi-chain warmup (GHMC-specific)
# ---------------------------------------------------------------------------


class TestMeadsMultiChain:
    """P5.5: meads warmup multi-chain shape contract tests.

    MEADS is fundamentally multi-chain: a single call handles all num_chains
    chains jointly via cross-validation across num_folds folds.  Unlike
    stan_window (which vmaps per-chain), MEADS is NOT vmapped — one call, all
    chains.

    Adapted parameters are shared (single MEADS estimate) and broadcast to
    (num_chains,) shape to satisfy the multi-chain contract.
    """

    def _run(self, seed: int, num_chains: int, **kw):
        from bjx_bench.inference.warmup.meads import ENTRY

        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        return ENTRY.runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _GHMC,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
            **kw,
        )

    def test_default_num_chains_equals_4_meets_num_folds_4(self) -> None:
        """num_chains=4 == num_folds=4 (default): should not raise; shapes correct."""
        from bjx_bench.inference.warmup.meads import ENTRY

        key = jax.random.key(6001)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        states, params = ENTRY.runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _GHMC,
            logdensity_fn=logdensity_fn,
        )
        leading = _state_leading_dim(states)
        assert (
            leading == 4
        ), f"Default num_chains=4: expected leading dim 4, got {leading}"
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (4,), f"step_size expected (4,), got {ss.shape}"

    def test_num_chains_below_num_folds_raises(self) -> None:
        """num_chains=2 < num_folds=4 must raise ValueError."""
        from bjx_bench.inference.warmup.meads import ENTRY

        key = jax.random.key(6002)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        with pytest.raises(ValueError, match="num_chains"):
            ENTRY.runner(
                jax.random.fold_in(key, 1),
                init_pos,
                200,
                _GHMC,
                logdensity_fn=logdensity_fn,
                num_chains=2,
                num_folds=4,
            )

    def test_explicit_num_chains_8_num_folds_4(self) -> None:
        """num_chains=8, num_folds=4 (chains > folds): shapes correct."""
        states, params = self._run(6003, num_chains=8, num_folds=4)
        leading = _state_leading_dim(states)
        assert leading == 8, f"Expected leading dim 8, got {leading}"
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (8,), f"step_size expected (8,), got {ss.shape}"
        imm = jnp.asarray(params["momentum_inverse_scale"])
        assert imm.shape == (
            8,
            _D,
        ), f"momentum_inverse_scale expected (8, {_D}), got {imm.shape}"

    def test_pre_batched_init_position_passes_through(self) -> None:
        """Pre-batched init_position (leading dim == num_chains) passes through verbatim."""
        from bjx_bench.inference.warmup.meads import ENTRY

        num_chains = 4
        key = jax.random.key(6004)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        batched_pos = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_chains,) + x.shape), init_pos
        )
        states, params = ENTRY.runner(
            jax.random.fold_in(key, 1),
            batched_pos,
            200,
            _GHMC,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
        )
        pos_shape = _position_shape(states)
        assert pos_shape == (4, _D), f"Pre-batched: expected (4, {_D}), got {pos_shape}"

    def test_step_size_alpha_delta_imm_keys_present(self) -> None:
        """adapted_params must contain step_size, momentum_inverse_scale, alpha, delta."""
        _, params = self._run(6005, num_chains=4)
        for key_name in ("step_size", "momentum_inverse_scale", "alpha", "delta"):
            assert (
                key_name in params
            ), f"Missing {key_name!r} in MEADS adapted_params; got: {list(params)}"

    def test_meads_num_folds_sidecar_present(self) -> None:
        """_meads_num_folds sidecar key must be present."""
        _, params = self._run(6006, num_chains=4)
        assert (
            "_meads_num_folds" in params
        ), f"Missing _meads_num_folds sidecar; got: {list(params)}"
        assert params["_meads_num_folds"] == 4

    def test_compatibility_check_raises_for_nuts(self) -> None:
        """MEADS is GHMC-only; is_compatible('nuts') must return False."""
        from bjx_bench.inference.warmup.meads import ENTRY

        assert not ENTRY.is_compatible(
            "nuts"
        ), "meads should not be compatible with nuts"

    def test_meads_is_compatible_with_ghmc(self) -> None:
        """is_compatible('ghmc') returns True."""
        from bjx_bench.inference.warmup.meads import ENTRY

        assert ENTRY.is_compatible("ghmc"), "meads must be compatible with ghmc"

    def test_meads_not_compatible_with_hmc(self) -> None:
        from bjx_bench.inference.warmup.meads import ENTRY

        assert not ENTRY.is_compatible("hmc"), "meads should not be compatible with hmc"

    def test_meads_not_compatible_with_mclmc(self) -> None:
        from bjx_bench.inference.warmup.meads import ENTRY

        assert not ENTRY.is_compatible(
            "mclmc"
        ), "meads should not be compatible with mclmc"


# ---------------------------------------------------------------------------
# 15. P5.6: CHEES multi-chain warmup (dynamic_hmc-specific)
# ---------------------------------------------------------------------------

_DYNAMIC_HMC = BASE_METHODS["dynamic_hmc"]


class TestCheesMultiChain:
    """P5.6: chees warmup multi-chain shape contract tests.

    CHEES is fundamentally multi-chain: a single call handles all num_chains
    chains jointly.  Like MEADS, CHEES is NOT vmapped — one call, all chains.

    Upstream API note: chees_adaptation.run() requires step_size and an optax
    optimizer as positional args (unlike meads_adaptation.run).  This wrapper
    handles that internally.

    Adapted numeric params (step_size, inverse_mass_matrix) are shared CHEES
    estimates broadcast to (num_chains,) / (num_chains, d).  Callable params
    (next_random_arg_fn, integration_steps_fn) are passed through as-is.
    """

    def _run(self, seed: int, num_chains: int, **kw):
        from bjx_bench.inference.warmup.chees import ENTRY

        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        return ENTRY.runner(
            jax.random.fold_in(key, 1),
            init_pos,
            50,  # short warmup for tests
            _DYNAMIC_HMC,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
            **kw,
        )

    def test_default_num_chains_equals_4(self) -> None:
        """Default num_chains=4: position leading dim == 4."""
        from bjx_bench.inference.warmup.chees import ENTRY

        key = jax.random.key(7001)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        states, params = ENTRY.runner(
            jax.random.fold_in(key, 1),
            init_pos,
            50,
            _DYNAMIC_HMC,
            logdensity_fn=logdensity_fn,
        )
        leading = _state_leading_dim(states)
        assert (
            leading == 4
        ), f"Default num_chains=4: expected leading dim 4, got {leading}"
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (4,), f"step_size expected (4,), got {ss.shape}"

    def test_explicit_num_chains_8(self) -> None:
        """num_chains=8: position leading dim == 8."""
        states, params = self._run(7002, num_chains=8)
        leading = _state_leading_dim(states)
        assert leading == 8, f"Expected leading dim 8, got {leading}"
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (8,), f"step_size expected (8,), got {ss.shape}"
        imm = jnp.asarray(params["inverse_mass_matrix"])
        assert imm.shape == (
            8,
            _D,
        ), f"inverse_mass_matrix expected (8, {_D}), got {imm.shape}"

    def test_pre_batched_init_position_passes_through(self) -> None:
        """Pre-batched init_position (leading dim == num_chains) passes through verbatim."""
        from bjx_bench.inference.warmup.chees import ENTRY

        num_chains = 4
        key = jax.random.key(7003)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        batched_pos = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_chains,) + x.shape), init_pos
        )
        states, params = ENTRY.runner(
            jax.random.fold_in(key, 1),
            batched_pos,
            50,
            _DYNAMIC_HMC,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
        )
        pos_shape = _position_shape(states)
        assert pos_shape == (4, _D), f"Pre-batched: expected (4, {_D}), got {pos_shape}"

    def test_step_size_and_imm_keys_present(self) -> None:
        """adapted_params must contain step_size and inverse_mass_matrix."""
        _, params = self._run(7004, num_chains=4)
        assert "step_size" in params, f"Missing step_size; keys: {list(params)}"
        assert (
            "inverse_mass_matrix" in params
        ), f"Missing inverse_mass_matrix; keys: {list(params)}"

    def test_callable_params_present(self) -> None:
        """CHEES adapted_params must contain next_random_arg_fn and integration_steps_fn."""
        _, params = self._run(7005, num_chains=4)
        assert (
            "next_random_arg_fn" in params
        ), f"Missing next_random_arg_fn; keys: {list(params)}"
        assert (
            "integration_steps_fn" in params
        ), f"Missing integration_steps_fn; keys: {list(params)}"
        assert callable(
            params["next_random_arg_fn"]
        ), "next_random_arg_fn must be callable"
        assert callable(
            params["integration_steps_fn"]
        ), "integration_steps_fn must be callable"

    def test_jitter_amount_default_in_sidecar(self) -> None:
        """Sidecar must contain _chees_target_acceptance_rate metadata."""
        _, params = self._run(7006, num_chains=4)
        assert (
            "_chees_target_acceptance_rate" in params
        ), f"Missing _chees_target_acceptance_rate sidecar; keys: {list(params)}"
        assert (
            abs(params["_chees_target_acceptance_rate"] - 0.651) < 1e-6
        ), f"Expected 0.651, got {params['_chees_target_acceptance_rate']}"

    def test_max_leapfrog_steps_sidecar_present(self) -> None:
        """Sidecar must contain _chees_max_leapfrog_steps metadata."""
        _, params = self._run(7007, num_chains=4)
        assert (
            "_chees_max_leapfrog_steps" in params
        ), f"Missing _chees_max_leapfrog_steps sidecar; keys: {list(params)}"
        assert isinstance(
            params["_chees_max_leapfrog_steps"], int
        ), "_chees_max_leapfrog_steps must be a Python int"

    def test_step_size_positive(self) -> None:
        """Adapted step_size must be positive across all chains."""
        _, params = self._run(7008, num_chains=4)
        ss = jnp.asarray(params["step_size"])
        assert bool(jnp.all(ss > 0)), f"Not all step sizes positive: {ss}"

    def test_imm_shape_num_chains_4(self) -> None:
        """num_chains=4 → inverse_mass_matrix has shape (4, d)."""
        _, params = self._run(7009, num_chains=4)
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (4, _D), f"Expected (4, {_D}), got {imm.shape}"

    def test_compatibility_check_raises_for_nuts(self) -> None:
        """CHEES is dynamic_hmc-only; is_compatible('nuts') must return False."""
        from bjx_bench.inference.warmup.chees import ENTRY

        assert not ENTRY.is_compatible(
            "nuts"
        ), "chees should not be compatible with nuts"

    def test_compatibility_check_raises_for_hmc(self) -> None:
        """CHEES is for dynamic_hmc, not fixed-L HMC; is_compatible('hmc') must return False."""
        from bjx_bench.inference.warmup.chees import ENTRY

        assert not ENTRY.is_compatible(
            "hmc"
        ), "chees should not be compatible with hmc (only dynamic_hmc)"

    def test_chees_is_compatible_with_dynamic_hmc(self) -> None:
        """is_compatible('dynamic_hmc') returns True."""
        from bjx_bench.inference.warmup.chees import ENTRY

        assert ENTRY.is_compatible(
            "dynamic_hmc"
        ), "chees must be compatible with dynamic_hmc"


# ---------------------------------------------------------------------------
# 16. P5.7: adjusted_mclmc_tuning warmup (adjusted_mclmc + adjusted_mclmc_dynamic)
# ---------------------------------------------------------------------------


class TestAdjustedMclmcTuning:
    """P5.7: adjusted_mclmc_tuning warmup registry and multi-chain shape contract tests.

    adjusted_mclmc_tuning uses blackjax.adjusted_mclmc_find_L_and_step_size
    (static kernel) to jointly find L, step_size, and a diagonal IMM.
    Compatible with both adjusted_mclmc and adjusted_mclmc_dynamic.
    """

    def test_adjusted_mclmc_tuning_in_registry(self) -> None:
        """P5.7: adjusted_mclmc_tuning must be present in WARMUPS."""
        assert (
            "adjusted_mclmc_tuning" in WARMUPS
        ), f"'adjusted_mclmc_tuning' not in WARMUPS; registered: {sorted(WARMUPS)}"

    def test_adjusted_mclmc_tuning_entry_importable(self) -> None:
        """P5.7: adjusted_mclmc_tuning ENTRY is importable as a Warmup instance."""
        from bjx_bench.inference.warmup.adjusted_mclmc_tuning import ENTRY

        assert isinstance(ENTRY, Warmup), f"ENTRY is not a Warmup: {type(ENTRY)}"
        assert ENTRY.name == "adjusted_mclmc_tuning"

    def test_adjusted_mclmc_tuning_is_warmup_instance(self) -> None:
        from bjx_bench.inference.warmup.adjusted_mclmc_tuning import ENTRY

        assert isinstance(ENTRY, Warmup)

    def test_adjusted_mclmc_tuning_name_matches_key(self) -> None:
        from bjx_bench.inference.warmup.adjusted_mclmc_tuning import ENTRY

        assert ENTRY.name == "adjusted_mclmc_tuning"

    def test_compatible_with_adjusted_mclmc(self) -> None:
        from bjx_bench.inference.warmup.adjusted_mclmc_tuning import ENTRY

        assert ENTRY.is_compatible("adjusted_mclmc")

    def test_compatible_with_adjusted_mclmc_dynamic(self) -> None:
        from bjx_bench.inference.warmup.adjusted_mclmc_tuning import ENTRY

        assert ENTRY.is_compatible("adjusted_mclmc_dynamic")

    def test_not_compatible_with_nuts(self) -> None:
        from bjx_bench.inference.warmup.adjusted_mclmc_tuning import ENTRY

        assert not ENTRY.is_compatible("nuts")

    def test_not_compatible_with_mclmc(self) -> None:
        from bjx_bench.inference.warmup.adjusted_mclmc_tuning import ENTRY

        assert not ENTRY.is_compatible("mclmc")

    def test_single_chain_signature_adjusted_mclmc(self) -> None:
        """Single-chain run on 5-D MVN with adjusted_mclmc."""
        from bjx_bench.inference.base_method.adjusted_mclmc import ENTRY as _ADJ_MCLMC
        from bjx_bench.inference.warmup.adjusted_mclmc_tuning import ENTRY

        key = jax.random.key(8001)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        warmup_key = jax.random.fold_in(key, 1)

        states, params = ENTRY.runner(
            warmup_key,
            init_pos,
            100,
            _ADJ_MCLMC,
            logdensity_fn=logdensity_fn,
            num_chains=1,
        )
        assert states is not None
        assert isinstance(params, dict)
        assert "L" in params, f"L missing; keys={list(params)}"
        assert "step_size" in params, f"step_size missing; keys={list(params)}"
        assert (
            "inverse_mass_matrix" in params
        ), f"inverse_mass_matrix missing; keys={list(params)}"
        assert (
            "_total_tuning_steps" in params
        ), f"_total_tuning_steps missing; keys={list(params)}"

    def test_single_chain_L_and_step_size_positive(self) -> None:
        from bjx_bench.inference.base_method.adjusted_mclmc import ENTRY as _ADJ_MCLMC
        from bjx_bench.inference.warmup.adjusted_mclmc_tuning import ENTRY

        key = jax.random.key(8002)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        warmup_key = jax.random.fold_in(key, 1)

        _, params = ENTRY.runner(
            warmup_key,
            init_pos,
            100,
            _ADJ_MCLMC,
            logdensity_fn=logdensity_fn,
            num_chains=1,
        )
        assert bool(jnp.all(jnp.asarray(params["L"]) > 0)), f"L not > 0: {params['L']}"
        assert bool(
            jnp.all(jnp.asarray(params["step_size"]) > 0)
        ), f"step_size not > 0: {params['step_size']}"

    def test_multi_chain_3_shape(self) -> None:
        """num_chains=3: L/step_size shape (3,), IMM shape (3, d)."""
        from bjx_bench.inference.base_method.adjusted_mclmc import ENTRY as _ADJ_MCLMC
        from bjx_bench.inference.warmup.adjusted_mclmc_tuning import ENTRY

        key = jax.random.key(8003)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        warmup_key = jax.random.fold_in(key, 1)

        states, params = ENTRY.runner(
            warmup_key,
            init_pos,
            100,
            _ADJ_MCLMC,
            logdensity_fn=logdensity_fn,
            num_chains=3,
        )
        # State leading dim == 3
        leaves = jax.tree.leaves(states)
        assert (
            leaves[0].shape[0] == 3
        ), f"Expected leading dim 3, got {leaves[0].shape[0]}"

        # L shape (3,)
        L = jnp.asarray(params["L"])
        assert L.shape == (3,), f"Expected L.shape=(3,), got {L.shape}"

        # step_size shape (3,)
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (3,), f"Expected step_size.shape=(3,), got {ss.shape}"

        # inverse_mass_matrix shape (3, d)
        imm = params["inverse_mass_matrix"]
        d = _D
        assert imm.shape == (3, d), f"Expected IMM.shape=(3, {d}), got {imm.shape}"

    def test_total_tuning_steps_is_python_int(self) -> None:
        from bjx_bench.inference.base_method.adjusted_mclmc import ENTRY as _ADJ_MCLMC
        from bjx_bench.inference.warmup.adjusted_mclmc_tuning import ENTRY

        key = jax.random.key(8004)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        warmup_key = jax.random.fold_in(key, 1)

        _, params = ENTRY.runner(
            warmup_key,
            init_pos,
            100,
            _ADJ_MCLMC,
            logdensity_fn=logdensity_fn,
            num_chains=2,
        )
        steps = params["_total_tuning_steps"]
        assert isinstance(
            steps, int
        ), f"_total_tuning_steps must be int, got {type(steps)}"
        assert steps > 0, f"_total_tuning_steps must be > 0, got {steps}"


# ---------------------------------------------------------------------------
# TestNoWarmupGuards (P5.8)
# ---------------------------------------------------------------------------


class TestNoWarmupGuards:
    """no_warmup._runner raises NotImplementedError for specialised methods."""

    def test_no_warmup_raises_for_elliptical_slice(self) -> None:
        """elliptical_slice.requires_prior_metadata=True → no_warmup raises NotImplementedError."""
        import pytest

        from bjx_bench.inference.base_method.elliptical_slice import (
            ENTRY as _ELLIP_SLICE,
        )
        from bjx_bench.inference.warmup import WARMUPS

        key = jax.random.key(9001)
        init_pos = jnp.zeros(5)

        def dummy_logdensity(x):
            return -0.5 * jnp.sum(x**2)

        with pytest.raises(
            NotImplementedError, match="requires Gaussian-prior metadata"
        ):
            WARMUPS["no_warmup"].runner(
                key,
                init_pos,
                0,
                _ELLIP_SLICE,
                logdensity_fn=dummy_logdensity,
                num_chains=1,
            )

    def test_no_warmup_raises_for_irmh(self) -> None:
        """irmh.requires_proposal_distribution=True → no_warmup raises NotImplementedError."""
        from bjx_bench.inference.base_method.irmh import ENTRY as _IRMH
        from bjx_bench.inference.warmup import WARMUPS

        key = jax.random.key(9002)
        init_pos = jnp.zeros(5)

        def dummy_logdensity(x):
            return -0.5 * jnp.sum(x**2)

        with pytest.raises(
            NotImplementedError, match="independent proposal distribution"
        ):
            WARMUPS["no_warmup"].runner(
                key,
                init_pos,
                0,
                _IRMH,
                logdensity_fn=dummy_logdensity,
                num_chains=1,
            )

    def test_no_warmup_raises_for_synthetic_proposal_distribution_entry(self) -> None:
        """Synthetic BaseMethod(requires_proposal_distribution=True) → NotImplementedError."""
        from bjx_bench.inference.base_method._base import BaseMethod
        from bjx_bench.inference.warmup import WARMUPS

        synthetic = BaseMethod(
            name="synthetic_irmh_like",
            family="mcmc",
            factory=lambda logdensity_fn, **kw: None,
            grad_count_per_step=lambda info: 0,
            default_hp_space=(),
            requires_proposal_distribution=True,
        )

        key = jax.random.key(9003)
        init_pos = jnp.zeros(5)

        def dummy_logdensity(x):
            return -0.5 * jnp.sum(x**2)

        with pytest.raises(
            NotImplementedError, match="independent proposal distribution"
        ):
            WARMUPS["no_warmup"].runner(
                key,
                init_pos,
                0,
                synthetic,
                logdensity_fn=dummy_logdensity,
                num_chains=1,
            )
