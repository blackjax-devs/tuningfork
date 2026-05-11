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
"""Tests for the MCLMC algorithm wrapper.

Covers:
1. Registry: BASE_METHODS["mclmc"] is ENTRY; entry fields match spec.
2. HyperparamSpace sanity: 2 HPs (step_size, L), both loguniform with
   positive bounds.
3. Factory smoke: 10-D MVN, init + 5 step() calls, all states finite.
4. grad_count constant: grad_count_per_step returns Array(2).
5. MCLMCInfo schema pin: _fields == ('logdensity', 'kinetic_change',
   'energy_change', 'nonans'). Fires if upstream changes the info shape.
6. PyTree position smoke: dict-of-arrays {"x": zeros(5), "y": zeros(3)}.
7. Empirical grad-count verification: 10 steps → exactly 20 grad calls
   (with tolerance for JIT trace overhead; upper bound 3x = 60).

Init note: blackjax.mclmc requires rng_key at init time for momentum
initialisation.  Calls are therefore kernel.init(position, rng_key) rather
than the key-free form used by HMC/MALA/Barker.
"""

import jax
import jax.numpy as jnp
import pytest

from tuningfork.inference.base_method import BASE_METHODS
from tuningfork.inference.base_method._base import HyperparamSpace
from tuningfork.inference.base_method.mclmc import ENTRY

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DIM = 10
_LOGDENSITY_FN = lambda x: -0.5 * jnp.sum(x["x"] ** 2)
_POSITION = {"x": jnp.zeros(_DIM)}
_L = 1.0
_STEP_SIZE = 0.1


# ===========================================================================
# 1. Registry tests
# ===========================================================================


class TestMclmcRegistry:
    def test_mclmc_registered(self) -> None:
        assert "mclmc" in BASE_METHODS, "BASE_METHODS must contain 'mclmc'"

    def test_entry_identity(self) -> None:
        assert BASE_METHODS["mclmc"] is ENTRY

    def test_entry_name(self) -> None:
        assert ENTRY.name == "mclmc"

    def test_entry_family(self) -> None:
        assert ENTRY.family == "mcmc"

    def test_needs_mass_matrix_false(self) -> None:
        assert ENTRY.needs_mass_matrix is False

    def test_target_acceptance_rate_none(self) -> None:
        """MCLMC is rejection-free; target_acceptance_rate must be None."""
        assert ENTRY.target_acceptance_rate is None

    def test_core_phase4_algorithms_present(self) -> None:
        """The Phase-4 core six must remain present; more entries may be added later.

        Originally `test_algorithms_has_six_entries` (==6); changed to subset
        check (after `ghmc` was added the strict-equality test became
        fragile by design (failed every time a sampler was added).
        """
        core_six = {"hmc", "nuts", "mala", "barker", "rwm", "mclmc"}
        missing = core_six - set(BASE_METHODS.keys())
        assert not missing, f"missing core base methods: {missing}"


# ===========================================================================
# 2. HyperparamSpace sanity
# ===========================================================================


class TestMclmcHyperparamSpace:
    def test_exactly_two_hps(self) -> None:
        assert len(ENTRY.default_hp_space) == 2

    def test_step_size_present(self) -> None:
        names = [hp.name for hp in ENTRY.default_hp_space]
        assert "step_size" in names

    def test_L_present(self) -> None:
        names = [hp.name for hp in ENTRY.default_hp_space]
        assert "L" in names

    def test_all_hps_are_loguniform(self) -> None:
        for hp in ENTRY.default_hp_space:
            assert isinstance(hp, HyperparamSpace)
            assert (
                hp.kind == "loguniform"
            ), f"HP '{hp.name}' must be loguniform, got '{hp.kind}'"

    def test_all_bounds_positive(self) -> None:
        for hp in ENTRY.default_hp_space:
            assert (
                hp.low is not None and hp.low > 0
            ), f"HP '{hp.name}' low bound must be > 0, got {hp.low}"
            assert (
                hp.high is not None and hp.high > 0
            ), f"HP '{hp.name}' high bound must be > 0, got {hp.high}"
            assert (
                hp.low < hp.high
            ), f"HP '{hp.name}': low {hp.low} must be < high {hp.high}"


# ===========================================================================
# 3. Factory smoke — 10-D MVN, init + 5 steps
# ===========================================================================


class TestMclmcFactorySmoke:
    def test_factory_callable(self) -> None:
        assert callable(ENTRY.factory)

    def test_factory_returns_sampling_algorithm(self) -> None:
        kernel = ENTRY.factory(_LOGDENSITY_FN, L=_L, step_size=_STEP_SIZE)
        assert hasattr(kernel, "init"), "kernel must have .init"
        assert hasattr(kernel, "step"), "kernel must have .step"

    def test_init_with_rng_key(self) -> None:
        """MCLMC init requires rng_key (for momentum unit-vector generation)."""
        key = jax.random.key(0)
        kernel = ENTRY.factory(_LOGDENSITY_FN, L=_L, step_size=_STEP_SIZE)
        state = kernel.init(_POSITION, key)
        assert jnp.isfinite(
            state.logdensity
        ), f"init logdensity not finite: {state.logdensity}"

    def test_step_returns_finite_state(self) -> None:
        init_key = jax.random.key(1)
        step_key = jax.random.key(2)
        kernel = ENTRY.factory(_LOGDENSITY_FN, L=_L, step_size=_STEP_SIZE)
        state = kernel.init(_POSITION, init_key)
        new_state, _info = kernel.step(step_key, state)
        assert jnp.isfinite(
            new_state.logdensity
        ), f"step logdensity not finite: {new_state.logdensity}"

    def test_five_step_chain_all_finite(self) -> None:
        key = jax.random.key(42)
        key, init_key = jax.random.split(key)
        kernel = ENTRY.factory(_LOGDENSITY_FN, L=_L, step_size=_STEP_SIZE)
        state = kernel.init(_POSITION, init_key)
        for i in range(5):
            key, subkey = jax.random.split(key)
            state, info = kernel.step(subkey, state)
            assert jnp.isfinite(
                state.logdensity
            ), f"Non-finite logdensity at step {i}: {state.logdensity}"


# ===========================================================================
# 4. grad_count_per_step returns constant 2
# ===========================================================================


class TestMclmcGradCount:
    def test_grad_count_returns_2(self) -> None:
        """grad_count_per_step must return 2 regardless of info contents."""
        from blackjax.mcmc.mclmc import MCLMCInfo

        dummy_info = MCLMCInfo(
            logdensity=0.0, kinetic_change=0.0, energy_change=0.0, nonans=True
        )
        count = ENTRY.grad_count_per_step(dummy_info)
        assert int(jnp.asarray(count)) == 2

    def test_grad_count_with_real_info(self) -> None:
        """grad_count_per_step returns 2 on real step output."""
        init_key = jax.random.key(10)
        step_key = jax.random.key(11)
        kernel = ENTRY.factory(_LOGDENSITY_FN, L=_L, step_size=_STEP_SIZE)
        state = kernel.init(_POSITION, init_key)
        _, info = kernel.step(step_key, state)
        count = ENTRY.grad_count_per_step(info)
        assert int(jnp.asarray(count)) == 2


# ===========================================================================
# 5. MCLMCInfo schema pin
# ===========================================================================


class TestMclmcInfoSchema:
    def test_mclmc_info_fields(self) -> None:
        """Pin MCLMCInfo._fields. Fires if upstream changes the info shape."""
        from blackjax.mcmc.mclmc import MCLMCInfo

        assert MCLMCInfo._fields == (
            "logdensity",
            "kinetic_change",
            "energy_change",
            "nonans",
        ), (
            f"MCLMCInfo._fields changed upstream: {MCLMCInfo._fields}. "
            "Update grad_count_per_step and this test accordingly."
        )

    def test_no_num_integration_steps_field(self) -> None:
        """Confirm MCLMCInfo does NOT have num_integration_steps."""
        from blackjax.mcmc.mclmc import MCLMCInfo

        assert "num_integration_steps" not in MCLMCInfo._fields, (
            "MCLMCInfo gained num_integration_steps — update grad_count_per_step "
            "from constant 2 to 2 * info.num_integration_steps."
        )


# ===========================================================================
# 6. PyTree position smoke (multi-site dict)
# ===========================================================================


class TestMclmcPytreePosition:
    def test_dict_position_init_and_step(self) -> None:
        """MCLMC must handle a multi-site dict position (dim >= 2 total)."""
        position_dict = {"x": jnp.zeros(5), "y": jnp.zeros(3)}
        logdensity_2site = lambda pos: (
            -0.5 * jnp.sum(pos["x"] ** 2) - 0.5 * jnp.sum(pos["y"] ** 2)
        )
        init_key = jax.random.key(50)
        step_key = jax.random.key(51)
        kernel = ENTRY.factory(logdensity_2site, L=_L, step_size=_STEP_SIZE)
        state = kernel.init(position_dict, init_key)
        assert jnp.isfinite(
            state.logdensity
        ), f"PyTree init logdensity not finite: {state.logdensity}"
        new_state, _info = kernel.step(step_key, state)
        assert jnp.isfinite(
            new_state.logdensity
        ), f"PyTree step logdensity not finite: {new_state.logdensity}"
        # Both sites must persist with correct shapes
        assert "x" in new_state.position
        assert "y" in new_state.position
        assert new_state.position["x"].shape == (5,)
        assert new_state.position["y"].shape == (3,)


# ===========================================================================
# 7. Empirical grad-count verification (10 steps → expected 20)
# ===========================================================================


class TestMclmcEmpiricalGradCount:
    """Verify that MCLMC calls value_and_grad exactly 2 times per kernel step.

    Strategy: wrap logdensity_fn in a Python closure counter (eager-mode
    reliable). Run without jit so each Python call is counted faithfully.
    Reset counter after kernel.init() (which calls value_and_grad once).

    Expected: 10 steps × 2 grads/step = 20 calls.
    Upper bound: 3 × 20 = 60 (to absorb any JAX abstract-eval traces).
    """

    def test_empirical_2_grads_per_step(self) -> None:
        grad_calls: list[int] = [0]

        def counting_logdensity(x: dict) -> jax.Array:
            grad_calls[0] += 1
            return -0.5 * jnp.sum(x["x"] ** 2)

        kernel = ENTRY.factory(counting_logdensity, L=_L, step_size=_STEP_SIZE)

        # init calls value_and_grad once for the initial logdensity + grad
        init_key = jax.random.key(77)
        state = kernel.init(_POSITION, init_key)
        grad_calls[0] = 0  # reset counter after init

        n_steps = 10
        key = jax.random.key(78)
        for _ in range(n_steps):
            key, subkey = jax.random.split(key)
            state, _info = kernel.step(subkey, state)

        expected_exact = 2 * n_steps  # = 20
        upper_bound = 3 * expected_exact  # = 60

        assert grad_calls[0] >= expected_exact, (
            f"MCLMC: expected >= {expected_exact} grad calls for {n_steps} steps "
            f"(2 grads/step), got {grad_calls[0]}. "
            "Possible causes: different default integrator or JIT is skipping calls."
        )
        assert grad_calls[0] <= upper_bound, (
            f"MCLMC: unexpectedly many grad calls ({grad_calls[0]}) for "
            f"{n_steps} steps — expected {expected_exact} (2/step), "
            f"upper bound {upper_bound}. "
            "Possible causes: multiple integrator steps per kernel step, or "
            "the integrator costs more than 2 grads/step."
        )

    def test_grad_count_grows_linearly(self) -> None:
        """Grad count must scale linearly with number of steps (not quadratic etc.)."""
        results = {}
        for n_steps in (5, 10):
            grad_calls: list[int] = [0]

            def counting_logdensity(x: dict) -> jax.Array:
                grad_calls[0] += 1
                return -0.5 * jnp.sum(x["x"] ** 2)

            kernel = ENTRY.factory(counting_logdensity, L=_L, step_size=_STEP_SIZE)
            init_key = jax.random.key(80 + n_steps)
            state = kernel.init(_POSITION, init_key)
            grad_calls[0] = 0

            key = jax.random.key(90 + n_steps)
            for _ in range(n_steps):
                key, subkey = jax.random.split(key)
                state, _ = kernel.step(subkey, state)

            results[n_steps] = grad_calls[0]

        # Ratio of grad counts must match ratio of steps (2:1 for steps 10:5)
        ratio = results[10] / results[5]
        assert 1.5 <= ratio <= 2.5, (
            f"Grad count ratio ({results[10]}/{results[5]} = {ratio:.2f}) is not "
            f"near 2.0 — suggests non-linear scaling. "
            f"10-step count: {results[10]}, 5-step count: {results[5]}"
        )
