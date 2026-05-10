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
"""Tests for the P5.8 elliptical_slice base method registry entry.

Covers:
  1. ENTRY field correctness (name, family, extra_required_kwargs, etc.).
  2. default_params_for(ENTRY) returns {} (HP-free).
  3. Direct factory invocation with synthetic prior: init + 200-step scan.
  4. grad_count_per_step returns 0 (gradient-free).
  5. factory() without prior_cov/prior_mean raises TypeError (contract pin).
"""

import jax
import jax.numpy as jnp
import pytest
from blackjax.mcmc.elliptical_slice import EllipSliceState

from bjx_bench.inference.base_method.elliptical_slice import ENTRY

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_D = 5
_SEED = 42
_PRIOR_MEAN = jnp.zeros(_D)
_PRIOR_COV = jnp.eye(_D)

# Gaussian likelihood centered at a non-zero point so posterior shifts from prior.
_OBS_CENTER = jnp.array([1.0, -1.0, 0.5, 0.0, 0.0])


def _loglikelihood(f):
    """Gaussian likelihood: log p(y | f) ∝ -0.5 * ||f - obs_center||^2."""
    return -0.5 * jnp.sum((f - _OBS_CENTER) ** 2)


# ---------------------------------------------------------------------------
# 1. ENTRY field correctness
# ---------------------------------------------------------------------------


class TestEllipSliceEntryFields:
    """ENTRY field validation for elliptical_slice."""

    def test_name(self) -> None:
        assert ENTRY.name == "elliptical_slice"

    def test_family(self) -> None:
        assert ENTRY.family == "mcmc"

    def test_extra_required_kwargs_match(self) -> None:
        assert ENTRY.extra_required_kwargs == ("prior_cov", "prior_mean")

    def test_target_acceptance_rate_none(self) -> None:
        """Slice sampler has no MH step; no target acceptance rate."""
        assert ENTRY.target_acceptance_rate is None

    def test_needs_mass_matrix_false(self) -> None:
        assert ENTRY.needs_mass_matrix is False

    def test_default_hp_space_empty(self) -> None:
        """Elliptical slice is hyperparameter-free."""
        assert ENTRY.default_hp_space == ()

    def test_factory_callable(self) -> None:
        assert callable(ENTRY.factory)

    def test_grad_count_callable(self) -> None:
        assert callable(ENTRY.grad_count_per_step)


# ---------------------------------------------------------------------------
# 2. default_params_for returns {}
# ---------------------------------------------------------------------------


class TestEllipSliceDefaultParams:
    """default_params_for(ENTRY) returns an empty dict (HP-free)."""

    def test_default_params_empty(self) -> None:
        from bjx_bench.calibration.tier_b import default_params_for

        params = default_params_for(ENTRY)
        assert params == {}, f"Expected empty dict, got {params!r}"


# ---------------------------------------------------------------------------
# 3. Direct factory invocation: init + 200-step scan
# ---------------------------------------------------------------------------


class TestEllipSliceFactory:
    """Factory invocation and kernel smoke test with synthetic prior."""

    def test_factory_returns_sampling_algorithm(self) -> None:
        algo = ENTRY.factory(
            _loglikelihood, prior_cov=_PRIOR_COV, prior_mean=_PRIOR_MEAN
        )
        assert hasattr(algo, "init"), "factory result must have .init"
        assert hasattr(algo, "step"), "factory result must have .step"

    def test_init_returns_ellip_slice_state(self) -> None:
        algo = ENTRY.factory(
            _loglikelihood, prior_cov=_PRIOR_COV, prior_mean=_PRIOR_MEAN
        )
        init_pos = jnp.zeros(_D)
        state = algo.init(init_pos)
        assert isinstance(
            state, EllipSliceState
        ), f"Expected EllipSliceState, got {type(state)}"
        assert state.position.shape == (_D,)

    def test_200_step_scan_preserves_shape_and_finite_logdensity(self) -> None:
        """Run 200 steps via jax.lax.scan; verify shape and finiteness."""
        algo = ENTRY.factory(
            _loglikelihood, prior_cov=_PRIOR_COV, prior_mean=_PRIOR_MEAN
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

    def test_info_has_expected_fields(self) -> None:
        """EllipSliceInfo must have (momentum, theta, subiter); no acceptance_rate."""
        algo = ENTRY.factory(
            _loglikelihood, prior_cov=_PRIOR_COV, prior_mean=_PRIOR_MEAN
        )
        state = algo.init(jnp.zeros(_D))
        _, info = algo.step(jax.random.key(_SEED), state)

        assert hasattr(info, "momentum"), "EllipSliceInfo must have 'momentum'"
        assert hasattr(info, "theta"), "EllipSliceInfo must have 'theta'"
        assert hasattr(info, "subiter"), "EllipSliceInfo must have 'subiter'"
        assert not hasattr(
            info, "acceptance_rate"
        ), "EllipSliceInfo must NOT have 'acceptance_rate' (slice sampler always accepts)"


# ---------------------------------------------------------------------------
# 4. grad_count_per_step returns 0 (gradient-free)
# ---------------------------------------------------------------------------


class TestEllipSliceGradCount:
    """grad_count_per_step returns 0 for gradient-free elliptical slice."""

    def test_grad_count_zero_with_synthetic_info(self) -> None:
        """Synthesise a minimal EllipSliceInfo and verify grad count = 0."""
        from blackjax.mcmc.elliptical_slice import EllipSliceInfo

        fake_info = EllipSliceInfo(
            momentum=jnp.zeros(_D),
            theta=jnp.asarray(0.5),
            subiter=jnp.asarray(1),
        )
        result = ENTRY.grad_count_per_step(fake_info)
        assert int(result) == 0, f"Expected grad_count=0, got {result}"

    def test_grad_count_returns_array(self) -> None:
        from blackjax.mcmc.elliptical_slice import EllipSliceInfo

        fake_info = EllipSliceInfo(
            momentum=jnp.zeros(_D),
            theta=jnp.asarray(0.5),
            subiter=jnp.asarray(1),
        )
        result = ENTRY.grad_count_per_step(fake_info)
        assert isinstance(result, jax.Array), f"Expected JAX Array, got {type(result)}"


# ---------------------------------------------------------------------------
# 5. factory() without prior metadata raises TypeError
# ---------------------------------------------------------------------------


class TestEllipSliceContractPin:
    """Calling factory without prior_cov/prior_mean must raise TypeError."""

    def test_factory_without_prior_raises_typeerror(self) -> None:
        with pytest.raises(TypeError):
            ENTRY.factory(_loglikelihood)

    def test_factory_without_prior_mean_raises_typeerror(self) -> None:
        with pytest.raises(TypeError):
            ENTRY.factory(_loglikelihood, prior_cov=_PRIOR_COV)

    def test_factory_without_prior_cov_raises_typeerror(self) -> None:
        with pytest.raises(TypeError):
            ENTRY.factory(_loglikelihood, prior_mean=_PRIOR_MEAN)
