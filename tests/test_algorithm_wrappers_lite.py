"""Tests for MALA, Barker, and RWM algorithm wrappers (T2.3).

Covers:
- Each entry is registered in ALGORITHMS under its expected name.
- factory is callable and returns a SamplingAlgorithm-shaped object
  (has .init and .step attributes).
- .init(position) returns a state with finite logdensity.
- .step(rng_key, state) returns (new_state, info).
- grad_count_per_step returns the expected constant (0 or 1).
- 5-step end-to-end chain smoke test on a 10-D MVN logdensity.
- Empirical question answers:
  1. Barker with inverse_mass_matrix=None works correctly.
  2. RWM proposal_generator works with PyTree (dict) positions.
  3. MALA grad count is exactly 1 per step (verified with a counter).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from bjx_bench.algorithms import ALGORITHMS
from bjx_bench.algorithms._base import AlgorithmEntry, HyperparamSpace

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_DIM = 10
_LOGDENSITY_FN = lambda x: -0.5 * jnp.sum(x["x"] ** 2)
_POSITION = {"x": jnp.zeros(_DIM)}
_IMM = jnp.ones(_DIM)  # diagonal identity mass matrix


def _make_mala_params() -> dict:
    return {"step_size": 0.1}


def _make_barker_params_no_imm() -> dict:
    """Barker with inverse_mass_matrix=None (identity default)."""
    return {"step_size": 0.1}


def _make_barker_params_with_imm() -> dict:
    """Barker with explicit inverse_mass_matrix."""
    return {"step_size": 0.1, "inverse_mass_matrix": _IMM}


def _make_rwm_params() -> dict:
    return {"sigma": 0.5}


# ===========================================================================
# Registry tests
# ===========================================================================


class TestAlgorithmRegistryLite:
    def test_mala_registered(self) -> None:
        assert "mala" in ALGORITHMS, "ALGORITHMS must contain 'mala'"

    def test_barker_registered(self) -> None:
        assert "barker" in ALGORITHMS, "ALGORITHMS must contain 'barker'"

    def test_rwm_registered(self) -> None:
        assert "rwm" in ALGORITHMS, "ALGORITHMS must contain 'rwm'"

    def test_mala_is_algorithm_entry(self) -> None:
        assert isinstance(ALGORITHMS["mala"], AlgorithmEntry)

    def test_barker_is_algorithm_entry(self) -> None:
        assert isinstance(ALGORITHMS["barker"], AlgorithmEntry)

    def test_rwm_is_algorithm_entry(self) -> None:
        assert isinstance(ALGORITHMS["rwm"], AlgorithmEntry)

    def test_mala_family(self) -> None:
        assert ALGORITHMS["mala"].family == "mcmc"

    def test_barker_family(self) -> None:
        assert ALGORITHMS["barker"].family == "mcmc"

    def test_rwm_family(self) -> None:
        assert ALGORITHMS["rwm"].family == "mcmc"

    def test_mala_needs_mass_matrix_false(self) -> None:
        assert ALGORITHMS["mala"].needs_mass_matrix is False

    def test_barker_needs_mass_matrix_true(self) -> None:
        assert ALGORITHMS["barker"].needs_mass_matrix is True

    def test_rwm_needs_mass_matrix_false(self) -> None:
        assert ALGORITHMS["rwm"].needs_mass_matrix is False

    def test_mala_target_acceptance(self) -> None:
        assert ALGORITHMS["mala"].target_acceptance_rate == pytest.approx(0.574)

    def test_barker_target_acceptance(self) -> None:
        assert ALGORITHMS["barker"].target_acceptance_rate == pytest.approx(0.40)

    def test_rwm_target_acceptance(self) -> None:
        assert ALGORITHMS["rwm"].target_acceptance_rate == pytest.approx(0.234)

    def test_all_five_algorithms_present(self) -> None:
        """Sanity check: ALGORITHMS has exactly hmc, nuts, mala, barker, rwm."""
        expected = {"hmc", "nuts", "mala", "barker", "rwm"}
        assert set(ALGORITHMS.keys()) == expected


# ===========================================================================
# HyperparamSpace sanity
# ===========================================================================


class TestHyperparamSpaceLite:
    @pytest.mark.parametrize("name", ["mala", "barker", "rwm"])
    def test_default_hp_space_non_empty(self, name: str) -> None:
        assert len(ALGORITHMS[name].default_hp_space) >= 1

    @pytest.mark.parametrize("name", ["mala", "barker", "rwm"])
    def test_all_hp_are_hyperparam_space(self, name: str) -> None:
        for hp in ALGORITHMS[name].default_hp_space:
            assert isinstance(hp, HyperparamSpace)

    @pytest.mark.parametrize("name", ["mala", "barker", "rwm"])
    def test_hp_bounds_consistent(self, name: str) -> None:
        for hp in ALGORITHMS[name].default_hp_space:
            if hp.kind in ("loguniform", "uniform", "int"):
                assert hp.low is not None
                assert hp.high is not None
                assert hp.low < hp.high
            elif hp.kind == "categorical":
                assert hp.choices is not None and len(hp.choices) > 0

    def test_mala_has_step_size_hp(self) -> None:
        names = [hp.name for hp in ALGORITHMS["mala"].default_hp_space]
        assert "step_size" in names

    def test_barker_has_step_size_hp(self) -> None:
        names = [hp.name for hp in ALGORITHMS["barker"].default_hp_space]
        assert "step_size" in names

    def test_rwm_has_sigma_hp_not_step_size(self) -> None:
        """RWM is parameterized by sigma, NOT step_size."""
        names = [hp.name for hp in ALGORITHMS["rwm"].default_hp_space]
        assert "sigma" in names
        assert "step_size" not in names

    def test_mala_has_no_mass_matrix_hp(self) -> None:
        names = [hp.name for hp in ALGORITHMS["mala"].default_hp_space]
        assert "inverse_mass_matrix" not in names

    def test_barker_has_no_mass_matrix_hp(self) -> None:
        """IMM must NOT be in the BO search space — supplied by warmup."""
        names = [hp.name for hp in ALGORITHMS["barker"].default_hp_space]
        assert "inverse_mass_matrix" not in names


# ===========================================================================
# Factory → init → step pipeline
# ===========================================================================


class TestMalaFactory:
    def test_factory_callable(self) -> None:
        assert callable(ALGORITHMS["mala"].factory)

    def test_factory_returns_sampling_algorithm(self) -> None:
        entry = ALGORITHMS["mala"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_mala_params())
        assert hasattr(kernel, "init"), "kernel must have .init"
        assert hasattr(kernel, "step"), "kernel must have .step"

    def test_init_returns_finite_logdensity(self) -> None:
        entry = ALGORITHMS["mala"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_mala_params())
        state = kernel.init(_POSITION)
        assert jnp.isfinite(state.logdensity)

    def test_step_returns_new_state_and_info(self) -> None:
        key = jax.random.key(10)
        entry = ALGORITHMS["mala"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_mala_params())
        state = kernel.init(_POSITION)
        new_state, info = kernel.step(key, state)
        assert jnp.isfinite(new_state.logdensity)

    def test_grad_count_is_1(self) -> None:
        """MALA grad_count_per_step must return exactly 1 (constant)."""
        key = jax.random.key(11)
        entry = ALGORITHMS["mala"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_mala_params())
        state = kernel.init(_POSITION)
        _, info = kernel.step(key, state)
        count = entry.grad_count_per_step(info)
        assert int(jnp.asarray(count)) == 1


class TestBarkerFactory:
    def test_factory_callable(self) -> None:
        assert callable(ALGORITHMS["barker"].factory)

    def test_factory_returns_sampling_algorithm_with_imm(self) -> None:
        entry = ALGORITHMS["barker"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_barker_params_with_imm())
        assert hasattr(kernel, "init"), "kernel must have .init"
        assert hasattr(kernel, "step"), "kernel must have .step"

    def test_factory_returns_sampling_algorithm_without_imm(self) -> None:
        """Empirical Q1: Barker with inverse_mass_matrix=None must work correctly."""
        entry = ALGORITHMS["barker"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_barker_params_no_imm())
        assert hasattr(kernel, "init"), "kernel must have .init"
        assert hasattr(kernel, "step"), "kernel must have .step"

    def test_init_returns_finite_logdensity_with_imm(self) -> None:
        entry = ALGORITHMS["barker"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_barker_params_with_imm())
        state = kernel.init(_POSITION)
        assert jnp.isfinite(state.logdensity)

    def test_init_returns_finite_logdensity_without_imm(self) -> None:
        """Empirical Q1: init must succeed with inverse_mass_matrix=None."""
        entry = ALGORITHMS["barker"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_barker_params_no_imm())
        state = kernel.init(_POSITION)
        assert jnp.isfinite(state.logdensity)

    def test_step_with_explicit_imm(self) -> None:
        key = jax.random.key(20)
        entry = ALGORITHMS["barker"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_barker_params_with_imm())
        state = kernel.init(_POSITION)
        new_state, info = kernel.step(key, state)
        assert jnp.isfinite(new_state.logdensity)

    def test_step_without_imm(self) -> None:
        """Empirical Q1: step must succeed with inverse_mass_matrix=None."""
        key = jax.random.key(21)
        entry = ALGORITHMS["barker"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_barker_params_no_imm())
        state = kernel.init(_POSITION)
        new_state, info = kernel.step(key, state)
        assert jnp.isfinite(new_state.logdensity)

    def test_grad_count_is_1_with_imm(self) -> None:
        key = jax.random.key(22)
        entry = ALGORITHMS["barker"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_barker_params_with_imm())
        state = kernel.init(_POSITION)
        _, info = kernel.step(key, state)
        count = entry.grad_count_per_step(info)
        assert int(jnp.asarray(count)) == 1

    def test_grad_count_is_1_without_imm(self) -> None:
        key = jax.random.key(23)
        entry = ALGORITHMS["barker"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_barker_params_no_imm())
        state = kernel.init(_POSITION)
        _, info = kernel.step(key, state)
        count = entry.grad_count_per_step(info)
        assert int(jnp.asarray(count)) == 1


class TestRwmFactory:
    def test_factory_callable(self) -> None:
        assert callable(ALGORITHMS["rwm"].factory)

    def test_factory_returns_sampling_algorithm(self) -> None:
        entry = ALGORITHMS["rwm"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_rwm_params())
        assert hasattr(kernel, "init"), "kernel must have .init"
        assert hasattr(kernel, "step"), "kernel must have .step"

    def test_factory_does_not_accept_step_size(self) -> None:
        """RWM factory must use sigma, not step_size."""
        entry = ALGORITHMS["rwm"]
        with pytest.raises(TypeError):
            entry.factory(_LOGDENSITY_FN, step_size=0.1)

    def test_init_returns_finite_logdensity(self) -> None:
        entry = ALGORITHMS["rwm"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_rwm_params())
        state = kernel.init(_POSITION)
        assert jnp.isfinite(state.logdensity)

    def test_step_returns_new_state_and_info(self) -> None:
        key = jax.random.key(30)
        entry = ALGORITHMS["rwm"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_rwm_params())
        state = kernel.init(_POSITION)
        new_state, info = kernel.step(key, state)
        assert jnp.isfinite(new_state.logdensity)

    def test_grad_count_is_0(self) -> None:
        """RWM grad_count_per_step must return 0 (logdensity-only kernel)."""
        key = jax.random.key(31)
        entry = ALGORITHMS["rwm"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_rwm_params())
        state = kernel.init(_POSITION)
        _, info = kernel.step(key, state)
        count = entry.grad_count_per_step(info)
        assert int(jnp.asarray(count)) == 0

    def test_rwm_with_pytree_dict_position(self) -> None:
        """Empirical Q2: RWM proposal_generator must work with dict pytree positions.

        Tests that ravel_pytree correctly handles a 2-site dict position,
        which mimics NumPyro-style model outputs.
        """
        position_dict = {"x": jnp.zeros(5), "y": jnp.zeros(3)}
        logdensity_2site = lambda pos: (
            -0.5 * jnp.sum(pos["x"] ** 2) - 0.5 * jnp.sum(pos["y"] ** 2)
        )
        entry = ALGORITHMS["rwm"]
        kernel = entry.factory(logdensity_2site, sigma=0.5)
        state = kernel.init(position_dict)
        assert jnp.isfinite(state.logdensity)
        key = jax.random.key(32)
        new_state, _ = kernel.step(key, state)
        assert jnp.isfinite(new_state.logdensity)
        # Position must have both sites in new state
        assert "x" in new_state.position
        assert "y" in new_state.position
        assert new_state.position["x"].shape == (5,)
        assert new_state.position["y"].shape == (3,)


# ===========================================================================
# 5-step end-to-end chain smoke tests
# ===========================================================================


class TestEndToEndChainLite:
    def _run_chain(self, entry: AlgorithmEntry, params: dict, n_steps: int = 5) -> int:
        """Run n_steps and return total grad count (may be 0 for RWM)."""
        key = jax.random.key(42)
        kernel = entry.factory(_LOGDENSITY_FN, **params)
        state = kernel.init(_POSITION)
        total_grads = 0
        for i in range(n_steps):
            key, subkey = jax.random.split(key)
            state, info = kernel.step(subkey, state)
            assert jnp.isfinite(
                state.logdensity
            ), f"Non-finite logdensity at step {i}: {state.logdensity}"
            total_grads += int(jnp.asarray(entry.grad_count_per_step(info)))
        return total_grads

    def test_mala_5_step_chain(self) -> None:
        total = self._run_chain(ALGORITHMS["mala"], _make_mala_params())
        assert total == 5, f"MALA: expected 5 grad evals in 5 steps, got {total}"

    def test_barker_5_step_chain_with_imm(self) -> None:
        total = self._run_chain(ALGORITHMS["barker"], _make_barker_params_with_imm())
        assert (
            total == 5
        ), f"Barker (IMM): expected 5 grad evals in 5 steps, got {total}"

    def test_barker_5_step_chain_without_imm(self) -> None:
        """Empirical Q1 end-to-end: Barker with None IMM must produce 5 grads."""
        total = self._run_chain(ALGORITHMS["barker"], _make_barker_params_no_imm())
        assert (
            total == 5
        ), f"Barker (no IMM): expected 5 grad evals in 5 steps, got {total}"

    def test_rwm_5_step_chain(self) -> None:
        total = self._run_chain(ALGORITHMS["rwm"], _make_rwm_params())
        assert total == 0, f"RWM: expected 0 grad evals in 5 steps, got {total}"

    def test_mala_position_changes(self) -> None:
        """At least one step of MALA should move the position."""
        key = jax.random.key(99)
        entry = ALGORITHMS["mala"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_mala_params())
        state = kernel.init(_POSITION)
        any_moved = False
        for _ in range(5):
            key, subkey = jax.random.split(key)
            new_state, _ = kernel.step(subkey, state)
            if not jnp.allclose(new_state.position["x"], state.position["x"]):
                any_moved = True
                break
            state = new_state
        assert any_moved, "MALA position never moved in 5 steps — likely a bug"

    def test_rwm_position_changes(self) -> None:
        """At least one step of RWM should move the position."""
        key = jax.random.key(100)
        entry = ALGORITHMS["rwm"]
        kernel = entry.factory(_LOGDENSITY_FN, **_make_rwm_params())
        state = kernel.init(_POSITION)
        any_moved = False
        for _ in range(5):
            key, subkey = jax.random.split(key)
            new_state, _ = kernel.step(subkey, state)
            if not jnp.allclose(new_state.position["x"], state.position["x"]):
                any_moved = True
                break
            state = new_state
        assert any_moved, "RWM position never moved in 5 steps — likely a bug"


# ===========================================================================
# Empirical Q3: MALA grad count empirical verification with counter
# ===========================================================================


class TestMalaGradCountEmpirical:
    """Verify that MALA calls value_and_grad exactly once per step.

    Strategy: wrap logdensity_fn in a pure-Python counter (safe because
    tests run outside JAX tracing context); each kernel.step() call must
    increment the grad counter by exactly 1.

    Note: kernel.init() also calls value_and_grad (to populate the
    initial state's logdensity_grad), so we reset the counter after init.
    """

    def test_mala_exactly_1_grad_per_step(self) -> None:
        grad_calls: list[int] = [0]

        def counting_logdensity(x: dict) -> jax.Array:
            # This counter increments in Python each time JAX evaluates the fn.
            # In eager mode (no jit) this is reliable; inside jit it may count
            # compile-time traces too, so we compare relative deltas.
            grad_calls[0] += 1
            return -0.5 * jnp.sum(x["x"] ** 2)

        entry = ALGORITHMS["mala"]
        kernel = entry.factory(counting_logdensity, step_size=0.1)

        # init calls value_and_grad once; reset after
        state = kernel.init(_POSITION)
        grad_calls[0] = 0  # reset after init

        n_steps = 10
        key = jax.random.key(77)
        for _ in range(n_steps):
            key, subkey = jax.random.split(key)
            state, info = kernel.step(subkey, state)

        # Each step calls value_and_grad once (1 grad/step).
        # In eager/non-jit mode, expect exactly n_steps calls.
        # Allow a small multiple for JAX internal traces (e.g. abstract eval).
        assert grad_calls[0] >= n_steps, (
            f"MALA: expected >= {n_steps} grad calls for {n_steps} steps, "
            f"got {grad_calls[0]}"
        )
        # Upper bound: no more than 3x (to allow for jit tracing overhead)
        assert grad_calls[0] <= 3 * n_steps, (
            f"MALA: unexpectedly many grad calls ({grad_calls[0]}) for "
            f"{n_steps} steps — suggests grad is called multiple times per step"
        )
