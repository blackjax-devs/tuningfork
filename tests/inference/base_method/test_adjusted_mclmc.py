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
"""Tests for the P5.7 adjusted_mclmc base method registry entry.

Covers:
  1. adjusted_mclmc entry fields: name, family, needs_mass_matrix,
     target_acceptance_rate, default_hp_space.
  2. default_params_for(ENTRY) returns dict with step_size and L keys.
  3. ENTRY.factory returns a SamplingAlgorithm with .init and .step.
  4. grad_count_per_step returns 2 * num_integration_steps.
  5. 10-D isotropic Gaussian sanity check: warmup + sampling.
"""

import blackjax
import jax
import jax.numpy as jnp
import pytest

from bjx_bench.calibration.tier_b import default_params_for
from bjx_bench.inference.base_method.adjusted_mclmc import ENTRY
from bjx_bench.model import MODELS

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_MVN = MODELS["mvn_10"]
_D = 10
_SEED = 0
_RNG_KEY = jax.random.key(_SEED)


def _build_logdensity(posterior_entry, key):
    from bjx_bench.model._numpyro import build_logdensity_fn

    init_position, logdensity_fn, _ = build_logdensity_fn(key, posterior_entry)
    return init_position, logdensity_fn


# ---------------------------------------------------------------------------
# 1. Registry: adjusted_mclmc entry exists and has correct fields
# ---------------------------------------------------------------------------


class TestAdjustedMclmcRegistry:
    """adjusted_mclmc entry exists in BASE_METHODS with correct fields (commit-2 check)."""

    def test_entry_name(self) -> None:
        assert ENTRY.name == "adjusted_mclmc"

    def test_entry_family_is_mcmc(self) -> None:
        assert ENTRY.family == "mcmc"

    def test_entry_needs_mass_matrix_true(self) -> None:
        assert ENTRY.needs_mass_matrix is True

    def test_entry_target_acceptance_rate(self) -> None:
        assert ENTRY.target_acceptance_rate == pytest.approx(0.9)

    def test_entry_default_hp_space_has_step_size_and_L(self) -> None:
        hp_names = {hp.name for hp in ENTRY.default_hp_space}
        assert "step_size" in hp_names, f"step_size missing; got {hp_names}"
        assert "L" in hp_names, f"L missing; got {hp_names}"

    def test_entry_step_size_is_loguniform(self) -> None:
        for hp in ENTRY.default_hp_space:
            if hp.name == "step_size":
                assert hp.kind == "loguniform"
                assert hp.low == pytest.approx(1e-3)
                assert hp.high == pytest.approx(1.0)
                break

    def test_entry_L_is_loguniform(self) -> None:
        for hp in ENTRY.default_hp_space:
            if hp.name == "L":
                assert hp.kind == "loguniform"
                assert hp.low == pytest.approx(0.1)
                assert hp.high == pytest.approx(100.0)
                break


# ---------------------------------------------------------------------------
# 2. default_params_for returns step_size and L
# ---------------------------------------------------------------------------


class TestAdjustedMclmcDefaultParams:
    """default_params_for(ENTRY) returns dict with step_size and L."""

    def test_default_params_has_step_size(self) -> None:
        params = default_params_for(ENTRY)
        assert "step_size" in params, f"step_size missing; got {list(params)}"

    def test_default_params_has_L(self) -> None:
        params = default_params_for(ENTRY)
        assert "L" in params, f"L missing; got {list(params)}"

    def test_default_params_step_size_positive(self) -> None:
        params = default_params_for(ENTRY)
        assert float(params["step_size"]) > 0

    def test_default_params_L_positive(self) -> None:
        params = default_params_for(ENTRY)
        assert float(params["L"]) > 0


# ---------------------------------------------------------------------------
# 3. Factory returns a SamplingAlgorithm
# ---------------------------------------------------------------------------


class TestAdjustedMclmcFactory:
    """adjusted_mclmc factory produces a valid BlackJAX SamplingAlgorithm."""

    def test_factory_callable(self) -> None:
        assert callable(ENTRY.factory)

    def test_factory_returns_sampling_algorithm(self) -> None:
        key = jax.random.key(101)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        params = default_params_for(ENTRY)
        algo = ENTRY.factory(logdensity_fn, **params)
        assert hasattr(algo, "init"), "factory result must have .init"
        assert hasattr(algo, "step"), "factory result must have .step"

    def test_factory_init_no_rng_key(self) -> None:
        """adjusted_mclmc.init does NOT require rng_key."""
        key = jax.random.key(102)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        params = default_params_for(ENTRY)
        algo = ENTRY.factory(logdensity_fn, **params)
        # Should work without rng_key (unlike vanilla MCLMC)
        state = algo.init(init_pos)
        assert hasattr(state, "position")

    def test_factory_step_returns_state_and_info(self) -> None:
        key = jax.random.key(103)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        params = default_params_for(ENTRY)
        algo = ENTRY.factory(logdensity_fn, **params)
        state = algo.init(init_pos)
        step_key, _ = jax.random.split(key)
        new_state, info = algo.step(step_key, state)
        assert hasattr(new_state, "position")
        assert hasattr(info, "acceptance_rate")
        assert hasattr(info, "num_integration_steps")

    def test_factory_with_inverse_mass_matrix(self) -> None:
        key = jax.random.key(104)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        algo = ENTRY.factory(
            logdensity_fn,
            step_size=0.1,
            L=1.0,
            inverse_mass_matrix=jnp.ones(_D),
        )
        state = algo.init(init_pos)
        assert hasattr(state, "position")


# ---------------------------------------------------------------------------
# 4. grad_count_per_step returns 2 * num_integration_steps
# ---------------------------------------------------------------------------


class TestAdjustedMclmcGradCount:
    """grad_count_per_step = 2 * info.num_integration_steps."""

    def test_grad_count_synthetic_info(self) -> None:
        """Synthesize an HMCInfo-like NamedTuple with num_integration_steps=5."""
        from blackjax.mcmc.hmc import HMCInfo

        info = HMCInfo(
            momentum=None,
            acceptance_rate=jnp.array(0.9),
            is_accepted=jnp.array(True),
            is_divergent=jnp.array(False),
            energy=jnp.array(0.0),
            proposal=None,
            num_integration_steps=jnp.array(5),
        )
        result = ENTRY.grad_count_per_step(info)
        assert int(result) == 10, f"Expected 10, got {result}"

    def test_grad_count_returns_jax_array(self) -> None:
        from blackjax.mcmc.hmc import HMCInfo

        info = HMCInfo(
            momentum=None,
            acceptance_rate=jnp.array(0.9),
            is_accepted=jnp.array(True),
            is_divergent=jnp.array(False),
            energy=jnp.array(0.0),
            proposal=None,
            num_integration_steps=jnp.array(5),
        )
        result = ENTRY.grad_count_per_step(info)
        assert isinstance(result, jax.Array)

    def test_grad_count_real_step(self) -> None:
        """grad_count on a real step info equals 2 * realized num_integration_steps."""
        key = jax.random.key(301)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        params = default_params_for(ENTRY)
        algo = ENTRY.factory(logdensity_fn, **params)
        state = algo.init(init_pos)
        step_key, _ = jax.random.split(key)
        _, info = algo.step(step_key, state)

        expected = 2 * int(info.num_integration_steps)
        result = int(ENTRY.grad_count_per_step(info))
        assert result == expected, f"Expected {expected}, got {result}"


# ---------------------------------------------------------------------------
# 5. 10-D Gaussian sanity check: warmup + sampling
# ---------------------------------------------------------------------------


class TestAdjustedMclmcSanity:
    """10-D isotropic Gaussian: run adjusted_mclmc_find_L_and_step_size + sampling."""

    def test_posterior_mean_near_zero_and_acceptance_above_half(self) -> None:
        key = jax.random.key(0)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)

        # --- Warmup via adjusted_mclmc_find_L_and_step_size ---
        warmup_key, sample_key = jax.random.split(key)
        adj_mclmc_kernel = blackjax.mcmc.adjusted_mclmc.build_kernel()
        init_state = blackjax.mcmc.adjusted_mclmc.init(init_pos, logdensity_fn)

        state, adaptation_state, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
            adj_mclmc_kernel,
            logdensity_fn=logdensity_fn,
            num_steps=200,
            state=init_state,
            rng_key=warmup_key,
            target=0.9,
            diagonal_preconditioning=True,
        )

        # Build the sampling kernel with adapted params
        n_steps = max(
            1, int(round(float(adaptation_state.L) / float(adaptation_state.step_size)))
        )
        algo = blackjax.adjusted_mclmc(
            logdensity_fn,
            step_size=adaptation_state.step_size,
            integration_steps_params=(n_steps,),
            inverse_mass_matrix=adaptation_state.inverse_mass_matrix,
        )

        # --- Sampling: 500 steps ---
        def one_step(s, k):
            new_s, info = algo.step(k, s)
            return new_s, (new_s, info)

        keys_500 = jax.random.split(sample_key, 500)
        _, (states_hist, infos_hist) = jax.lax.scan(one_step, state, keys_500)

        # Acceptance rate median > 0.5
        ar = jnp.median(infos_hist.acceptance_rate)
        assert float(ar) > 0.5, f"Median acceptance rate {float(ar):.3f} < 0.5"

        # Posterior mean within 0.5 of zero on each dim
        # positions shape: (500, 10)
        positions = jax.tree.leaves(states_hist.position)[0]
        mean_pos = jnp.mean(positions, axis=0)
        assert jnp.all(
            jnp.abs(mean_pos) < 0.5
        ), f"Posterior mean not near zero: {mean_pos}"
