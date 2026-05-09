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
"""Tests for the Phase 3 (P3.1) warmup registry.

Covers:
  1. WARMUPS dict has exactly the three expected entries.
  2. is_compatible() for stan_window: hmc/nuts → True; mclmc → False.
  3. is_compatible() for mclmc_tuning: mclmc → True; nuts → False.
  4. is_compatible() for no_warmup: any name → True (sentinel "*").
  5. stan_window smoke: NUTS on 10-D MVN at n_warmup=200.
  6. mclmc_tuning smoke: MCLMC on 10-D MVN at n_warmup=200.
  7. no_warmup smoke: RWM (gradient-free) and NUTS.
  8. Compatibility error via _run_warmup (wrong warmup for algorithm).
  9. Auto-dispatch in tune_algorithm: mclmc → mclmc_tuning, nuts → stan_window,
     rwm → no_warmup (verified via result structure).
 10. tune_algorithm regression: existing calls with warmup_name=None still pass.
"""

import math

import jax
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

    def test_warmups_has_three_entries(self) -> None:
        assert (
            len(WARMUPS) == 3
        ), f"Expected 3 entries, got {len(WARMUPS)}: {sorted(WARMUPS)}"

    def test_warmups_has_stan_window(self) -> None:
        assert "stan_window" in WARMUPS

    def test_warmups_has_mclmc_tuning(self) -> None:
        assert "mclmc_tuning" in WARMUPS

    def test_warmups_has_no_warmup(self) -> None:
        assert "no_warmup" in WARMUPS

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
# 5. stan_window smoke: NUTS on MVN-10 at n_warmup=200
# ---------------------------------------------------------------------------


class TestStanWindowSmoke:
    """stan_window smoke test on NUTS + MVN-10."""

    def test_returns_state_and_adapted_params(self) -> None:
        key = jax.random.key(101)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        warmup_key = jax.random.fold_in(key, 1)
        state, params = WARMUPS["stan_window"].runner(
            warmup_key, init_pos, 200, _NUTS, logdensity_fn=logdensity_fn
        )
        assert state is not None
        assert isinstance(params, dict)

    def test_adapted_params_has_step_size(self) -> None:
        key = jax.random.key(102)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        _, params = WARMUPS["stan_window"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
        )
        assert "step_size" in params, f"params keys: {list(params.keys())}"

    def test_adapted_params_has_inverse_mass_matrix(self) -> None:
        key = jax.random.key(103)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        _, params = WARMUPS["stan_window"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
        )
        assert "inverse_mass_matrix" in params, f"params keys: {list(params.keys())}"

    def test_step_size_positive(self) -> None:
        key = jax.random.key(104)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        _, params = WARMUPS["stan_window"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
        )
        assert float(params["step_size"]) > 0, f"step_size={params['step_size']} <= 0"

    def test_inverse_mass_matrix_shape(self) -> None:
        key = jax.random.key(105)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        _, params = WARMUPS["stan_window"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
        )
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (
            10,
        ), f"inverse_mass_matrix.shape={imm.shape}, expected (10,) for diagonal default"

    def test_dense_mass_matrix_shape(self) -> None:
        """P5.0b: is_mass_matrix_diagonal=False produces a (d, d) IMM."""
        key = jax.random.key(106)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        _, params = WARMUPS["stan_window"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
            is_mass_matrix_diagonal=False,
        )
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (
            10,
            10,
        ), f"inverse_mass_matrix.shape={imm.shape}, expected (10, 10) for dense MM"

    def test_dense_mass_matrix_is_symmetric_positive_definite(self) -> None:
        """Sanity check: dense IMM should be symmetric and PD."""
        import jax.numpy as jnp

        key = jax.random.key(107)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        _, params = WARMUPS["stan_window"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
            is_mass_matrix_diagonal=False,
        )
        imm = params["inverse_mass_matrix"]
        # Symmetry within float tolerance.
        assert jnp.allclose(imm, imm.T, atol=1e-6), "dense IMM must be symmetric"
        # Positive definiteness via Cholesky.
        try:
            jnp.linalg.cholesky(imm)
        except Exception as exc:  # pragma: no cover
            raise AssertionError(f"dense IMM not positive definite: {exc}") from exc


# ---------------------------------------------------------------------------
# 6. mclmc_tuning smoke: MCLMC on MVN-10 at n_warmup=200
# ---------------------------------------------------------------------------


class TestMclmcTuningSmoke:
    """mclmc_tuning smoke test on MCLMC + MVN-10."""

    def _run(self, seed: int) -> tuple[object, dict]:
        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        warmup_key = jax.random.fold_in(key, 1)
        return WARMUPS["mclmc_tuning"].runner(
            warmup_key, init_pos, 200, _MCLMC, logdensity_fn=logdensity_fn
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
        assert float(params["L"]) > 0, f"L={params['L']} <= 0"

    def test_step_size_positive(self) -> None:
        _, params = self._run(207)
        assert float(params["step_size"]) > 0, f"step_size={params['step_size']} <= 0"

    def test_inverse_mass_matrix_shape(self) -> None:
        _, params = self._run(208)
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (
            10,
        ), f"inverse_mass_matrix.shape={imm.shape}, expected (10,)"

    def test_total_tuning_steps_positive(self) -> None:
        _, params = self._run(209)
        steps = int(params["_total_tuning_steps"])
        assert steps > 0, f"_total_tuning_steps={steps} <= 0"


# ---------------------------------------------------------------------------
# 7. no_warmup smoke
# ---------------------------------------------------------------------------


class TestNoWarmupSmoke:
    """no_warmup smoke tests on RWM and NUTS."""

    def test_rwm_returns_state_and_empty_params(self) -> None:
        key = jax.random.key(301)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        state, params = WARMUPS["no_warmup"].runner(
            jax.random.fold_in(key, 1), init_pos, 200, _RWM, logdensity_fn=logdensity_fn
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
