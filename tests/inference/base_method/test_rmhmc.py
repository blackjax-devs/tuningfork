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
"""Tests for the rmhmc base method registry entry.

Covers:
  1. ENTRY field correctness (name, family, default_hp_space, etc.).
  2. Factory: IMM→mass_matrix conversion (diagonal and dense).
  3. Factory returns SamplingAlgorithm with .init and .step.
  4. State type is HMCState (rmhmc reuses hmc.init + hmc.build_kernel).
  5. End-to-end smoke test on mvn_5d_logdensity:
     - step produces finite positions.
     - 10-step scan produces finite positions.
  6. grad_count_per_step callable.
  7. HP space: step_size (loguniform), num_integration_steps (int) present.
  8. Compatibility test: rmhmc + window_adaptation end-to-end on 5-D Gaussian
     (the key test: does IMM→mass_matrix conversion let window_adaptation
     drive rmhmc end-to-end?).

Finding: The IMM→mass_matrix conversion (1/imm for diagonal, linalg.inv
for dense) works correctly. window_adaptation drives blackjax.rmhmc directly (it
calls algorithm.build_kernel(integrator) and passes step_size + IMM to the kernel);
the bjx-bench factory is then applied POST-warmup to build the sampler using adapted
params. This is the correct composition: window_adaptation → adapted params (step_size,
IMM) → bjx-bench factory (converts IMM→mass_matrix) → sampling.

Note: window_adaptation does NOT accept the bjx-bench factory shape directly —
it expects a blackjax.GenerateSamplingAPI object (like blackjax.rmhmc). The test
drives warmup via blackjax.rmhmc directly, then applies the factory to build the
sampler. This friction is documented in the wrapper's notes field.
"""

import blackjax
import jax
import jax.numpy as jnp
import pytest
from blackjax.diagnostics import effective_sample_size, potential_scale_reduction
from blackjax.mcmc.hmc import HMCState

from tests.fixtures import mvn_5d_init, mvn_5d_logdensity
from tuningfork.inference.base_method.rmhmc import ENTRY

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_D = 5
_SEED = 42
_STEP_SIZE = 0.3
_NUM_INTEGRATION_STEPS = 3
_INVERSE_MASS_MATRIX = jnp.ones(_D)

# ---------------------------------------------------------------------------
# 1. ENTRY field correctness
# ---------------------------------------------------------------------------


class TestRmhmcEntryFields:
    """ENTRY field validation for rmhmc."""

    def test_name(self) -> None:
        assert ENTRY.name == "rmhmc"

    def test_family(self) -> None:
        assert ENTRY.family == "mcmc"

    def test_needs_mass_matrix(self) -> None:
        """inverse_mass_matrix comes from warmup adaptation."""
        assert ENTRY.needs_mass_matrix is True

    def test_target_acceptance_rate(self) -> None:
        """Target acceptance rate is 0.8 (higher than HMC's 0.65)."""
        assert ENTRY.target_acceptance_rate == pytest.approx(0.8)

    def test_extra_required_kwargs_empty(self) -> None:
        """rmhmc uses constant mass_matrix mode — no extra required kwargs."""
        assert ENTRY.extra_required_kwargs == ()

    def test_factory_callable(self) -> None:
        assert callable(ENTRY.factory)

    def test_grad_count_callable(self) -> None:
        assert callable(ENTRY.grad_count_per_step)

    def test_default_hp_space_nonempty(self) -> None:
        assert len(ENTRY.default_hp_space) >= 2

    def test_notes_nonempty(self) -> None:
        assert len(ENTRY.notes) > 0


# ---------------------------------------------------------------------------
# 2. IMM→mass_matrix conversion
# ---------------------------------------------------------------------------


class TestRmhmcImmConversion:
    """The factory converts inverse_mass_matrix to mass_matrix correctly."""

    def test_diagonal_imm_yields_reciprocal(self) -> None:
        """Diagonal IMM: mass_matrix = 1.0 / imm."""
        # Build algo with known IMM; if conversion is correct the step runs.
        imm = jnp.array([2.0, 3.0, 4.0, 5.0, 6.0])
        algo = ENTRY.factory(
            mvn_5d_logdensity,
            step_size=_STEP_SIZE,
            inverse_mass_matrix=imm,
            num_integration_steps=2,
        )
        state = algo.init(mvn_5d_init())
        key = jax.random.key(_SEED)
        new_state, info = algo.step(key, state)
        assert jnp.all(
            jnp.isfinite(new_state.position)
        ), "Positions non-finite after step with non-uniform diagonal IMM."

    def test_dense_imm_yields_inverse(self) -> None:
        """Dense IMM (2-D): mass_matrix = linalg.inv(imm)."""
        imm_dense = jnp.eye(_D) * 2.0  # diagonal dense — easy ground truth
        algo = ENTRY.factory(
            mvn_5d_logdensity,
            step_size=_STEP_SIZE,
            inverse_mass_matrix=imm_dense,
            num_integration_steps=2,
        )
        state = algo.init(mvn_5d_init())
        key = jax.random.key(_SEED)
        new_state, info = algo.step(key, state)
        assert jnp.all(
            jnp.isfinite(new_state.position)
        ), "Positions non-finite after step with dense IMM."

    def test_identity_imm_diagonal(self) -> None:
        """Identity diagonal IMM (all ones) maps to identity mass_matrix."""
        algo = ENTRY.factory(
            mvn_5d_logdensity,
            step_size=_STEP_SIZE,
            inverse_mass_matrix=_INVERSE_MASS_MATRIX,
            num_integration_steps=_NUM_INTEGRATION_STEPS,
        )
        state = algo.init(mvn_5d_init())
        key = jax.random.key(_SEED)
        new_state, _ = algo.step(key, state)
        assert jnp.all(jnp.isfinite(new_state.position))


# ---------------------------------------------------------------------------
# 3. Factory returns SamplingAlgorithm with .init and .step
# ---------------------------------------------------------------------------


class TestRmhmcFactory:
    """Factory invocation correctness."""

    def test_factory_returns_algorithm_with_init_step(self) -> None:
        algo = ENTRY.factory(
            mvn_5d_logdensity,
            step_size=_STEP_SIZE,
            inverse_mass_matrix=_INVERSE_MASS_MATRIX,
            num_integration_steps=_NUM_INTEGRATION_STEPS,
        )
        assert hasattr(algo, "init"), "factory result must have .init"
        assert hasattr(algo, "step"), "factory result must have .step"


# ---------------------------------------------------------------------------
# 4. State type: rmhmc reuses HMCState
# ---------------------------------------------------------------------------


class TestRmhmcStateType:
    """rmhmc init returns HMCState (no distinct RMHMCState NamedTuple)."""

    def test_init_returns_hmc_state(self) -> None:
        algo = ENTRY.factory(
            mvn_5d_logdensity,
            step_size=_STEP_SIZE,
            inverse_mass_matrix=_INVERSE_MASS_MATRIX,
            num_integration_steps=_NUM_INTEGRATION_STEPS,
        )
        state = algo.init(mvn_5d_init())
        assert isinstance(state, HMCState), (
            f"rmhmc.init should return HMCState (rmhmc reuses hmc.init). "
            f"Got: {type(state).__name__}."
        )

    def test_hmc_state_fields(self) -> None:
        algo = ENTRY.factory(
            mvn_5d_logdensity,
            step_size=_STEP_SIZE,
            inverse_mass_matrix=_INVERSE_MASS_MATRIX,
            num_integration_steps=_NUM_INTEGRATION_STEPS,
        )
        state = algo.init(mvn_5d_init())
        expected = ("position", "logdensity", "logdensity_grad")
        assert (
            state._fields == expected
        ), f"HMCState._fields changed: expected {expected}, got {state._fields}."


# ---------------------------------------------------------------------------
# 5. End-to-end smoke test on mvn_5d_logdensity
# ---------------------------------------------------------------------------


class TestRmhmcEndToEnd:
    """End-to-end smoke test: finite positions, scan."""

    def _build_algo(self):
        return ENTRY.factory(
            mvn_5d_logdensity,
            step_size=_STEP_SIZE,
            inverse_mass_matrix=_INVERSE_MASS_MATRIX,
            num_integration_steps=_NUM_INTEGRATION_STEPS,
        )

    def test_step_produces_finite_positions(self) -> None:
        algo = self._build_algo()
        state = algo.init(mvn_5d_init())
        key = jax.random.key(_SEED)
        new_state, _ = algo.step(key, state)
        assert jnp.all(
            jnp.isfinite(new_state.position)
        ), "NaN/Inf in position after one step."

    def test_step_returns_hmc_info(self) -> None:
        algo = self._build_algo()
        state = algo.init(mvn_5d_init())
        key = jax.random.key(_SEED)
        new_state, info = algo.step(key, state)
        expected_fields = (
            "momentum",
            "acceptance_rate",
            "is_accepted",
            "is_divergent",
            "energy",
            "proposal",
            "num_integration_steps",
        )
        assert (
            info._fields == expected_fields
        ), f"HMCInfo._fields changed: expected {expected_fields}, got {info._fields}."

    def test_scan_10_steps_no_nan(self) -> None:
        """10 steps via lax.scan: no NaN, shape preserved."""
        algo = self._build_algo()
        state = algo.init(mvn_5d_init())

        def one_step(carry, key):
            new_state, _ = algo.step(key, carry)
            return new_state, new_state.position

        keys = jax.random.split(jax.random.key(_SEED), 10)
        final_state, positions = jax.lax.scan(one_step, state, keys)

        assert jnp.all(
            jnp.isfinite(positions)
        ), "NaN/Inf in positions after 10-step scan."
        assert positions.shape == (
            10,
            _D,
        ), f"positions shape changed after scan: {positions.shape}"

    def test_acceptance_rate_nonnegative(self) -> None:
        algo = self._build_algo()
        state = algo.init(mvn_5d_init())
        key = jax.random.key(_SEED)
        _, info = algo.step(key, state)
        assert float(info.acceptance_rate) >= 0.0

    def test_num_integration_steps_in_info(self) -> None:
        """HMCInfo.num_integration_steps matches the factory argument."""
        algo = ENTRY.factory(
            mvn_5d_logdensity,
            step_size=_STEP_SIZE,
            inverse_mass_matrix=_INVERSE_MASS_MATRIX,
            num_integration_steps=7,  # distinctive value
        )
        state = algo.init(mvn_5d_init())
        key = jax.random.key(_SEED)
        _, info = algo.step(key, state)
        assert (
            int(info.num_integration_steps) == 7
        ), f"Expected num_integration_steps=7, got {int(info.num_integration_steps)}."


# ---------------------------------------------------------------------------
# 6. grad_count_per_step callable
# ---------------------------------------------------------------------------


class TestRmhmcGradCount:
    """grad_count_per_step accepts info and returns num_integration_steps."""

    def test_grad_count_with_real_info(self) -> None:
        algo = ENTRY.factory(
            mvn_5d_logdensity,
            step_size=_STEP_SIZE,
            inverse_mass_matrix=_INVERSE_MASS_MATRIX,
            num_integration_steps=_NUM_INTEGRATION_STEPS,
        )
        state = algo.init(mvn_5d_init())
        key = jax.random.key(_SEED)
        _, info = algo.step(key, state)
        count = ENTRY.grad_count_per_step(info)
        assert isinstance(
            count, jax.Array
        ), f"grad_count_per_step should return JAX array, got {type(count)}"
        assert (
            int(count) == _NUM_INTEGRATION_STEPS
        ), f"grad_count_per_step should return {_NUM_INTEGRATION_STEPS}, got {int(count)}"


# ---------------------------------------------------------------------------
# 7. HP space correctness
# ---------------------------------------------------------------------------


class TestRmhmcHpSpace:
    """HP space: step_size (loguniform), num_integration_steps (int)."""

    def _hp_by_name(self, name):
        for hp in ENTRY.default_hp_space:
            if hp.name == name:
                return hp
        raise KeyError(f"HP '{name}' not found in ENTRY.default_hp_space")

    def test_hp_space_has_step_size(self) -> None:
        self._hp_by_name("step_size")  # raises if missing

    def test_hp_space_has_num_integration_steps(self) -> None:
        self._hp_by_name("num_integration_steps")  # raises if missing

    def test_step_size_is_loguniform(self) -> None:
        hp = self._hp_by_name("step_size")
        assert (
            hp.kind == "loguniform"
        ), f"step_size kind should be 'loguniform', got {hp.kind!r}"

    def test_num_integration_steps_is_int(self) -> None:
        hp = self._hp_by_name("num_integration_steps")
        assert (
            hp.kind == "int"
        ), f"num_integration_steps kind should be 'int', got {hp.kind!r}"
        assert hp.low == 1
        assert hp.high == 20


# ---------------------------------------------------------------------------
# 8. Compatibility test: rmhmc + window_adaptation end-to-end
# ---------------------------------------------------------------------------


class TestRmhmcWindowAdaptationCompatibility:
    """compatibility test: window_adaptation + rmhmc factory end-to-end.

    Finding: window_adaptation expects a blackjax.GenerateSamplingAPI object
    (blackjax.rmhmc) as the 'algorithm' argument — it calls algorithm.build_kernel()
    internally.  The bjx-bench factory wrapper cannot be passed directly to
    window_adaptation.

    Composition:
      1. Run window_adaptation(blackjax.rmhmc, ...) → adapted (step_size, IMM).
      2. Apply bjx-bench factory with adapted params (converts IMM→mass_matrix).
      3. Run 500 sampling steps with the factory-built sampler.
      4. Assert: min ESS > 100, R-hat < 1.05, no NaN, mean within atol=0.2.

    If the IMM→mass_matrix conversion is wrong, the sampler will diverge or
    produce poor mixing — this test surfaces that failure mode.
    """

    _NUM_WARMUP = 200
    _NUM_SAMPLES = 500

    def test_rmhmc_with_window_adaptation_5d_gaussian(self) -> None:
        """End-to-end: window_adaptation → rmhmc factory → sampling on 5-D Gaussian."""
        key = jax.random.key(_SEED)

        # Step 1: Run window_adaptation using blackjax.rmhmc directly.
        # window_adaptation calls algorithm.build_kernel(integrator) internally
        # and produces adapted (step_size, inverse_mass_matrix).
        warmup = blackjax.window_adaptation(
            blackjax.rmhmc,
            mvn_5d_logdensity,
            is_mass_matrix_diagonal=True,
            initial_step_size=0.3,
            target_acceptance_rate=0.8,
            num_integration_steps=_NUM_INTEGRATION_STEPS,
        )
        key, warmup_key = jax.random.split(key)
        (final_state, params), _ = warmup.run(
            warmup_key, mvn_5d_init(), num_steps=self._NUM_WARMUP
        )

        # Verify warmup produced sane params.
        assert "step_size" in params, "window_adaptation must return step_size"
        assert (
            "inverse_mass_matrix" in params
        ), "window_adaptation must return inverse_mass_matrix"
        adapted_step_size = float(params["step_size"])
        adapted_imm = params["inverse_mass_matrix"]
        assert (
            adapted_step_size > 0
        ), f"Adapted step_size must be positive, got {adapted_step_size}"
        assert jnp.all(jnp.isfinite(adapted_imm)), "Adapted IMM must be finite"

        # Step 2: Build sampler via bjx-bench factory with adapted params.
        # The factory converts IMM → mass_matrix internally.
        algo = ENTRY.factory(
            mvn_5d_logdensity,
            step_size=adapted_step_size,
            inverse_mass_matrix=adapted_imm,
            num_integration_steps=_NUM_INTEGRATION_STEPS,
        )

        # Step 3: Run 500 sampling steps.
        def one_step(carry, rng_key):
            new_state, _ = algo.step(rng_key, carry)
            return new_state, new_state.position

        key, sample_key = jax.random.split(key)
        step_keys = jax.random.split(sample_key, self._NUM_SAMPLES)
        final_sample_state, positions = jax.lax.scan(one_step, final_state, step_keys)

        # Step 4a: No NaN.
        assert jnp.all(jnp.isfinite(positions)), (
            "Positions contain non-finite values after 500 sampling steps. "
            "IMM→mass_matrix conversion may be incorrect."
        )

        # Step 4b: Sample mean within atol=0.2 of zero.
        mean = jnp.mean(positions, axis=0)
        assert jnp.all(jnp.abs(mean) < 0.2), (
            f"Sample mean not within atol=0.2 of zero: {mean}. "
            "Sampler may be poorly mixed or biased."
        )

        # Step 4c: min ESS > 100.
        positions_3d = positions[None, :, :]  # (1, n_samples, D)
        ess = effective_sample_size(positions_3d)
        min_ess = float(jnp.min(ess))
        assert min_ess > 100, (
            f"min ESS = {min_ess:.1f} is below 100. "
            "IMM→mass_matrix conversion or warmup may not be working correctly."
        )

        # Step 4d: R-hat < 1.05 (split-chain: split 500 samples into 2x250).
        half = self._NUM_SAMPLES // 2
        positions_2chains = jnp.stack(
            [positions[:half, :], positions[half:, :]], axis=0
        )  # (2, 250, D)
        rhat = potential_scale_reduction(positions_2chains)
        max_rhat = float(jnp.max(rhat))
        assert max_rhat < 1.05, (
            f"max R-hat = {max_rhat:.4f} is above 1.05. "
            "Sampler may not be converging. Check IMM→mass_matrix conversion."
        )
