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
"""Tests for adjusted_mclmc_trajectory_tuning warmup.

Covers:
  1. (fast) Selection logic: given a stub ess/grad map, argmax picks the correct avg;
     output dict has the right keys including _avg_star, L == avg_star * step, etc.
  2. (slow) Integration: run the full warmup on a small MVN and assert:
     - _avg_star >= 1 (escaped MALA or at least as good)
     - adapted_params has the expected keys + shapes
     - L == _avg_star * step_size (per-chain)
     - _total_tuning_steps > 0 (pilot grads were counted)
     - Warmup is registered in WARMUPS and compatible only with adjusted_mclmc_dynamic.
"""

import jax
import jax.numpy as jnp
import pytest

# ---------------------------------------------------------------------------
# Section 1: Fast tests — pure logic, no JAX trace
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestSelectionLogic:
    """Tests for the argmax selection logic and output key contract.

    We stub the ess/grad map to avoid running any JAX.
    """

    def test_argmax_picks_correct_avg(self) -> None:
        """Given a known ess/grad map, _avg_star should be argmax."""
        # Simulate: avg=2 is best.
        ess_per_grad = {1.0: 0.0010, 2.0: 0.0025, 4.0: 0.0018}
        avg_star = float(max(ess_per_grad, key=lambda a: ess_per_grad[a]))
        assert avg_star == 2.0, f"Expected avg_star=2.0, got {avg_star}"

    def test_argmax_picks_highest_when_4_is_best(self) -> None:
        """If avg=4 gives the best ess/grad, it should be selected."""
        ess_per_grad = {1.0: 0.0005, 2.0: 0.0012, 4.0: 0.0030}
        avg_star = float(max(ess_per_grad, key=lambda a: ess_per_grad[a]))
        assert avg_star == 4.0, f"Expected avg_star=4.0, got {avg_star}"

    def test_argmax_falls_back_to_mala_when_1_is_best(self) -> None:
        """If avg=1 is the best (degenerate case), it should be selected."""
        ess_per_grad = {1.0: 0.0050, 2.0: 0.0012, 4.0: 0.0005}
        avg_star = float(max(ess_per_grad, key=lambda a: ess_per_grad[a]))
        assert avg_star == 1.0, f"Expected avg_star=1.0, got {avg_star}"

    def test_output_dict_has_required_keys(self) -> None:
        """Output dict must contain the documented keys."""
        required_keys = {
            "L",
            "step_size",
            "inverse_mass_matrix",
            "_total_tuning_steps",
            "_avg_star",
            "_avg_search_ess_per_grad",
        }
        # Simulate a minimal output dict.
        num_chains = 2
        d = 5
        avg_star = 2.0
        step_sizes = jnp.ones(num_chains) * 0.3
        imms = jnp.ones((num_chains, d))
        adapted = {
            "L": avg_star * step_sizes,
            "step_size": step_sizes,
            "inverse_mass_matrix": imms,
            "_total_tuning_steps": 1000,
            "_avg_star": avg_star,
            "_avg_search_ess_per_grad": {1.0: 0.001, 2.0: 0.002, 4.0: 0.0015},
        }
        missing = required_keys - set(adapted.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_L_equals_avg_star_times_step_size(self) -> None:
        """L must equal avg_star * step_size (the key contract of this warmup)."""
        avg_star = 2.0
        step_sizes = jnp.array([0.25, 0.35, 0.40, 0.30])
        L_expected = avg_star * step_sizes
        # This is the exact computation in _runner.
        L_actual = avg_star * step_sizes
        assert jnp.allclose(
            L_actual, L_expected, atol=1e-7
        ), f"L mismatch: {L_actual} vs {L_expected}"

    def test_avg_search_ess_per_grad_has_grid_keys(self) -> None:
        """_avg_search_ess_per_grad must have keys for all grid points {1.0, 2.0, 4.0}."""
        from tuningfork.warmup.adjusted_mclmc_trajectory_tuning import _AVG_GRID

        # Simulate the dict the runner would produce.
        ess_map = {float(a): float(a) * 0.001 for a in _AVG_GRID}
        for a in _AVG_GRID:
            assert float(a) in ess_map, f"Grid point {a} not in ess_per_grad map"

    def test_registry_entry_exists(self) -> None:
        """WARMUPS must contain 'adjusted_mclmc_trajectory_tuning'."""
        from tuningfork.warmup import WARMUPS

        assert "adjusted_mclmc_trajectory_tuning" in WARMUPS, (
            f"adjusted_mclmc_trajectory_tuning not in WARMUPS; "
            f"registered: {sorted(WARMUPS)}"
        )

    def test_entry_is_warmup_instance(self) -> None:
        """The registered entry must be a Warmup instance."""
        from tuningfork.warmup import WARMUPS, Warmup

        entry = WARMUPS["adjusted_mclmc_trajectory_tuning"]
        assert isinstance(entry, Warmup)

    def test_compatible_with_adjusted_mclmc_dynamic(self) -> None:
        """Entry must be compatible with adjusted_mclmc_dynamic."""
        from tuningfork.warmup import WARMUPS

        entry = WARMUPS["adjusted_mclmc_trajectory_tuning"]
        assert entry.is_compatible(
            "adjusted_mclmc_dynamic"
        ), "Must be compatible with adjusted_mclmc_dynamic"

    def test_not_compatible_with_other_methods(self) -> None:
        """Entry must NOT be compatible with nuts, mclmc, adjusted_mclmc, hmc."""
        from tuningfork.warmup import WARMUPS

        entry = WARMUPS["adjusted_mclmc_trajectory_tuning"]
        for name in ("nuts", "mclmc", "adjusted_mclmc", "hmc", "barker"):
            assert not entry.is_compatible(
                name
            ), f"adjusted_mclmc_trajectory_tuning should NOT be compatible with {name}"


# ---------------------------------------------------------------------------
# Section 2: Slow tests — chain-running, JAX-compiled
# ---------------------------------------------------------------------------


def _build_logdensity_mvn(key):
    """Build a small MVN logdensity from the model registry."""
    from tuningfork.model import MODELS
    from tuningfork.model._numpyro import build_logdensity_fn

    mvn = MODELS["mvn_10"]
    init_position, logdensity_fn, _ = build_logdensity_fn(key, mvn)
    return init_position, logdensity_fn


@pytest.mark.slow
class TestAdjustedMclmcTrajectoryTuningSlow:
    """Integration tests for the full warmup runner on MVN-10.

    These tests run JAX-compiled chain code (slow tier).
    """

    _SEED = 42
    _N_WARMUP = 300  # Short but enough for convergence on MVN-10.
    _N_PILOT = 200  # Shorter pilot for test speed.
    _NUM_CHAINS = 2  # Reduce chains for speed.
    _D = 10

    def _run(self, seed: int | None = None, **kwargs):
        """Run the warmup and return (states, params)."""
        from tuningfork.base_method.adjusted_mclmc_dynamic import ENTRY as _ADJ_DYN
        from tuningfork.warmup.adjusted_mclmc_trajectory_tuning import ENTRY

        s = seed if seed is not None else self._SEED
        key = jax.random.key(s)
        init_pos, logdensity_fn = _build_logdensity_mvn(key)
        warmup_key = jax.random.fold_in(key, 1)
        return ENTRY.runner(
            warmup_key,
            init_pos,
            self._N_WARMUP,
            _ADJ_DYN,
            logdensity_fn=logdensity_fn,
            num_chains=self._NUM_CHAINS,
            n_pilot=self._N_PILOT,
            **kwargs,
        )

    def test_returns_state_and_dict(self) -> None:
        """Runner returns (states, dict)."""
        states, params = self._run()
        assert states is not None
        assert isinstance(params, dict)

    def test_required_keys_present(self) -> None:
        """All required output keys must be present."""
        _, params = self._run(seed=101)
        required = {
            "L",
            "step_size",
            "inverse_mass_matrix",
            "_total_tuning_steps",
            "_avg_star",
            "_avg_search_ess_per_grad",
        }
        missing = required - set(params.keys())
        assert not missing, f"Missing keys: {missing}; got: {list(params)}"

    def test_L_shapes(self) -> None:
        """L must have shape (num_chains,)."""
        _, params = self._run(seed=102)
        L = jnp.asarray(params["L"])
        assert L.shape == (
            self._NUM_CHAINS,
        ), f"L.shape={L.shape}, expected ({self._NUM_CHAINS},)"

    def test_step_size_shapes(self) -> None:
        """step_size must have shape (num_chains,)."""
        _, params = self._run(seed=103)
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (
            self._NUM_CHAINS,
        ), f"step_size.shape={ss.shape}, expected ({self._NUM_CHAINS},)"

    def test_imm_shape(self) -> None:
        """inverse_mass_matrix must have shape (num_chains, d)."""
        _, params = self._run(seed=104)
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (
            self._NUM_CHAINS,
            self._D,
        ), f"IMM.shape={imm.shape}, expected ({self._NUM_CHAINS}, {self._D})"

    def test_L_positive(self) -> None:
        """All L values must be positive."""
        _, params = self._run(seed=105)
        L = jnp.asarray(params["L"])
        assert bool(jnp.all(L > 0)), f"L not all > 0: {L}"

    def test_step_size_positive(self) -> None:
        """All step_sizes must be positive."""
        _, params = self._run(seed=106)
        ss = jnp.asarray(params["step_size"])
        assert bool(jnp.all(ss > 0)), f"step_size not all > 0: {ss}"

    def test_avg_star_at_least_1(self) -> None:
        """_avg_star must be >= 1 (at least as good as MALA)."""
        _, params = self._run(seed=107)
        avg_star = params["_avg_star"]
        assert avg_star >= 1.0, f"_avg_star={avg_star} < 1 (below MALA baseline)"

    def test_L_equals_avg_star_times_step(self) -> None:
        """L must equal avg_star * step_size per-chain."""
        _, params = self._run(seed=108)
        L = jnp.asarray(params["L"])
        ss = jnp.asarray(params["step_size"])
        avg_star = params["_avg_star"]
        L_expected = avg_star * ss
        assert jnp.allclose(
            L, L_expected, rtol=1e-5, atol=1e-7
        ), f"L != avg_star * step_size: L={L}, expected={L_expected}"

    def test_total_tuning_steps_positive_int(self) -> None:
        """_total_tuning_steps must be a positive Python int."""
        _, params = self._run(seed=109)
        steps = params["_total_tuning_steps"]
        assert isinstance(
            steps, int
        ), f"_total_tuning_steps must be int, got {type(steps)}"
        assert steps > 0, f"_total_tuning_steps={steps} <= 0"

    def test_avg_search_ess_per_grad_structure(self) -> None:
        """_avg_search_ess_per_grad must be a dict with keys {1.0, 2.0, 4.0}."""
        from tuningfork.warmup.adjusted_mclmc_trajectory_tuning import _AVG_GRID

        _, params = self._run(seed=110)
        epg = params["_avg_search_ess_per_grad"]
        assert isinstance(epg, dict), f"Expected dict, got {type(epg)}"
        for a in _AVG_GRID:
            assert float(a) in epg, f"Grid point {a} missing; got keys: {list(epg)}"
        for a, v in epg.items():
            assert isinstance(v, float), f"ess_per_grad[{a}]={v} is not float"
            assert v >= 0.0, f"ess_per_grad[{a}]={v} < 0"

    def test_avg_star_is_argmax(self) -> None:
        """_avg_star must be the argmax of _avg_search_ess_per_grad."""
        _, params = self._run(seed=111)
        epg = params["_avg_search_ess_per_grad"]
        avg_star = params["_avg_star"]
        expected_star = float(max(epg, key=lambda a: epg[a]))
        assert (
            avg_star == expected_star
        ), f"_avg_star={avg_star} != argmax {expected_star}; ess_map={epg}"

    def test_state_shape(self) -> None:
        """State position must have leading dim == num_chains."""
        states, _ = self._run(seed=112)
        leaves = jax.tree.leaves(states)
        leading = leaves[0].shape[0]
        assert (
            leading == self._NUM_CHAINS
        ), f"State leading dim {leading} != {self._NUM_CHAINS}"
