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
"""Tests for the P5.8 mgrad_gaussian base method registry entry.

Covers:
  1. ENTRY field correctness (name, family, requires_prior_metadata, etc.).
  2. default_params_for(ENTRY) returns {"step_size": <midpoint>}.
  3. Direct factory invocation with synthetic prior: init + 200-step scan.
  4. grad_count_per_step returns 1 (one value_and_grad per step).
  5. factory() without prior_cov/prior_mean raises TypeError (contract pin).
"""

import jax
import jax.numpy as jnp
import pytest

from bjx_bench.inference.base_method.mgrad_gaussian import ENTRY

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_D = 5
_SEED = 42
_PRIOR_MEAN = jnp.zeros(_D)
_PRIOR_COV = jnp.eye(_D)

# Gaussian likelihood centered at a non-zero point.
_OBS_CENTER = jnp.array([1.0, -1.0, 0.5, 0.0, 0.0])


def _logdensity_fn(x):
    """Full posterior log-density = log-prior + log-likelihood.

    For mgrad_gaussian, logdensity_fn is the FULL posterior (not likelihood-only).
    The Gaussian prior contribution: -0.5 * x^T cov^{-1} x (for mean=0, cov=I).
    The likelihood: -0.5 * ||x - obs_center||^2.
    """
    log_prior = -0.5 * jnp.sum(x**2)
    log_likelihood = -0.5 * jnp.sum((x - _OBS_CENTER) ** 2)
    return log_prior + log_likelihood


# ---------------------------------------------------------------------------
# 1. ENTRY field correctness
# ---------------------------------------------------------------------------


class TestMgradGaussianEntryFields:
    """ENTRY field validation for mgrad_gaussian."""

    def test_name(self) -> None:
        assert ENTRY.name == "mgrad_gaussian"

    def test_family(self) -> None:
        assert ENTRY.family == "mcmc"

    def test_requires_prior_metadata_true(self) -> None:
        assert ENTRY.requires_prior_metadata is True

    def test_target_acceptance_rate(self) -> None:
        """Upstream guidance: target ≈ 0.5."""
        assert ENTRY.target_acceptance_rate == pytest.approx(0.5)

    def test_needs_mass_matrix_false(self) -> None:
        assert ENTRY.needs_mass_matrix is False

    def test_default_hp_space_has_step_size(self) -> None:
        hp_names = {hp.name for hp in ENTRY.default_hp_space}
        assert "step_size" in hp_names, f"step_size missing; got {hp_names}"

    def test_default_hp_space_step_size_loguniform(self) -> None:
        for hp in ENTRY.default_hp_space:
            if hp.name == "step_size":
                assert hp.kind == "loguniform"
                assert hp.low == pytest.approx(1e-3)
                assert hp.high == pytest.approx(10.0)
                break

    def test_factory_callable(self) -> None:
        assert callable(ENTRY.factory)

    def test_grad_count_callable(self) -> None:
        assert callable(ENTRY.grad_count_per_step)


# ---------------------------------------------------------------------------
# 2. default_params_for returns step_size midpoint
# ---------------------------------------------------------------------------


class TestMgradGaussianDefaultParams:
    """default_params_for(ENTRY) returns {step_size: geometric midpoint of [1e-3, 10]}."""

    def test_default_params_has_step_size(self) -> None:
        from bjx_bench.calibration.tier_b import default_params_for

        params = default_params_for(ENTRY)
        assert "step_size" in params, f"Expected step_size in params, got {params!r}"

    def test_default_params_step_size_is_70th_percentile(self) -> None:
        """default_value_for_space uses 70th-percentile on log-scale (P4.0 tweak):
        low * (high / low) ** 0.7 = 1e-3 * (1e4) ** 0.7 ≈ 0.631.
        """
        from bjx_bench.calibration.tier_b import default_params_for

        params = default_params_for(ENTRY)
        expected = 1e-3 * (10.0 / 1e-3) ** 0.7
        assert params["step_size"] == pytest.approx(expected, rel=1e-6), (
            f"step_size default should be 70th-percentile {expected:.6f}, "
            f"got {params['step_size']}"
        )


# ---------------------------------------------------------------------------
# 3. Direct factory invocation: init + 200-step scan
# ---------------------------------------------------------------------------


class TestMgradGaussianFactory:
    """Factory invocation and kernel smoke test with synthetic prior."""

    def test_factory_returns_sampling_algorithm(self) -> None:
        algo = ENTRY.factory(
            _logdensity_fn, prior_cov=_PRIOR_COV, prior_mean=_PRIOR_MEAN
        )
        assert hasattr(algo, "init"), "factory result must have .init"
        assert hasattr(algo, "step"), "factory result must have .step"

    def test_init_returns_state_with_position(self) -> None:
        algo = ENTRY.factory(
            _logdensity_fn, prior_cov=_PRIOR_COV, prior_mean=_PRIOR_MEAN
        )
        init_pos = jnp.zeros(_D)
        state = algo.init(init_pos)
        assert hasattr(state, "position"), "State must have 'position' field"
        assert state.position.shape == (_D,)

    def test_200_step_scan_preserves_shape_and_finite_logdensity(self) -> None:
        """Run 200 steps via jax.lax.scan; verify shape and finiteness."""
        algo = ENTRY.factory(
            _logdensity_fn,
            prior_cov=_PRIOR_COV,
            prior_mean=_PRIOR_MEAN,
            step_size=1.0,
        )
        init_pos = jnp.zeros(_D)
        state = algo.init(init_pos)

        def one_step(carry, key):
            state = carry
            new_state, info = algo.step(key, state)
            return new_state, info

        keys = jax.random.split(jax.random.key(_SEED), 200)
        final_state, infos = jax.lax.scan(one_step, state, keys)

        assert final_state.position.shape == (
            _D,
        ), f"Position shape changed: {final_state.position.shape}"
        assert jnp.isfinite(
            final_state.logdensity
        ), f"logdensity not finite: {final_state.logdensity}"

    def test_200_steps_mean_acceptance_positive(self) -> None:
        """Mean acceptance_rate over 200 steps must be > 0 (basic sanity)."""
        algo = ENTRY.factory(
            _logdensity_fn,
            prior_cov=_PRIOR_COV,
            prior_mean=_PRIOR_MEAN,
            step_size=1.0,
        )
        state = algo.init(jnp.zeros(_D))

        def one_step(carry, key):
            state = carry
            new_state, info = algo.step(key, state)
            return new_state, info

        keys = jax.random.split(jax.random.key(_SEED), 200)
        _, infos = jax.lax.scan(one_step, state, keys)

        mean_accept = float(jnp.mean(infos.acceptance_rate))
        assert mean_accept > 0.0, (
            f"Mean acceptance_rate over 200 steps = {mean_accept:.4f}; "
            f"expected > 0.0 (kernel is stuck or broken)"
        )

    def test_info_has_expected_fields(self) -> None:
        """MarginalInfo must have (acceptance_rate, is_accepted, proposal)."""
        algo = ENTRY.factory(
            _logdensity_fn, prior_cov=_PRIOR_COV, prior_mean=_PRIOR_MEAN
        )
        state = algo.init(jnp.zeros(_D))
        _, info = algo.step(jax.random.key(_SEED), state)

        assert hasattr(
            info, "acceptance_rate"
        ), "MarginalInfo must have 'acceptance_rate'"
        assert hasattr(info, "is_accepted"), "MarginalInfo must have 'is_accepted'"
        assert hasattr(info, "proposal"), "MarginalInfo must have 'proposal'"

        ar = float(info.acceptance_rate)
        assert 0.0 <= ar <= 1.0, f"acceptance_rate={ar} out of [0, 1]"


# ---------------------------------------------------------------------------
# 4. grad_count_per_step returns 1
# ---------------------------------------------------------------------------


class TestMgradGaussianGradCount:
    """grad_count_per_step returns 1 (one value_and_grad per step)."""

    def test_grad_count_one_with_synthetic_info(self) -> None:
        """Synthesise a minimal MarginalInfo and verify grad count = 1."""
        from blackjax.mcmc.marginal_latent_gaussian import MarginalInfo, MarginalState

        fake_proposal = MarginalState(
            position=jnp.zeros(_D),
            logdensity=jnp.asarray(-1.0),
            logdensity_grad=jnp.zeros(_D),
            U_x=jnp.zeros(_D),
            U_grad_x=jnp.zeros(_D),
        )
        fake_info = MarginalInfo(
            acceptance_rate=jnp.asarray(0.8),
            is_accepted=jnp.asarray(True),
            proposal=fake_proposal,
        )
        result = ENTRY.grad_count_per_step(fake_info)
        assert int(result) == 1, f"Expected grad_count=1, got {result}"

    def test_grad_count_returns_array(self) -> None:
        from blackjax.mcmc.marginal_latent_gaussian import MarginalInfo, MarginalState

        fake_proposal = MarginalState(
            position=jnp.zeros(_D),
            logdensity=jnp.asarray(-1.0),
            logdensity_grad=jnp.zeros(_D),
            U_x=jnp.zeros(_D),
            U_grad_x=jnp.zeros(_D),
        )
        fake_info = MarginalInfo(
            acceptance_rate=jnp.asarray(0.8),
            is_accepted=jnp.asarray(True),
            proposal=fake_proposal,
        )
        result = ENTRY.grad_count_per_step(fake_info)
        assert isinstance(result, jax.Array), f"Expected JAX Array, got {type(result)}"


# ---------------------------------------------------------------------------
# 5. factory() without prior metadata raises TypeError
# ---------------------------------------------------------------------------


class TestMgradGaussianContractPin:
    """Calling factory without prior_cov/prior_mean must raise TypeError."""

    def test_factory_without_prior_raises_typeerror(self) -> None:
        with pytest.raises(TypeError):
            ENTRY.factory(_logdensity_fn)

    def test_factory_without_prior_mean_raises_typeerror(self) -> None:
        with pytest.raises(TypeError):
            ENTRY.factory(_logdensity_fn, prior_cov=_PRIOR_COV)

    def test_factory_without_prior_cov_raises_typeerror(self) -> None:
        with pytest.raises(TypeError):
            ENTRY.factory(_logdensity_fn, prior_mean=_PRIOR_MEAN)
