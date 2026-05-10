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
"""Tests for the P5.15 additive_step_random_walk base method registry entry.

Covers:
  1. ENTRY field correctness (name, family, extra_required_kwargs, etc.).
  2. factory with inline Gaussian proposal_generator: init + 500-step scan,
     shape preservation, acceptance_rate finite, mean acceptance > 0.
  3. no_warmup runner raises NotImplementedError if invoked without proposal_generator
     (proves P5.14a extra_required_kwargs schema integration).
  4. grad_count_per_step returns 0 (gradient-free).
  5. factory() without proposal_generator raises TypeError (contract pin).
"""

import jax
import jax.numpy as jnp
import pytest
from blackjax.mcmc.random_walk import RWState

from bjx_bench.inference.base_method.additive_step_random_walk import ENTRY
from tests.fixtures import mvn_5d_logdensity

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_D = 5
_SEED = 42
_PROPOSAL_SCALE = 0.1


def _gaussian_proposal(key: jax.Array, position: jax.Array) -> jax.Array:
    """Standard Gaussian additive step: step ~ N(0, _PROPOSAL_SCALE^2 * I).

    This is the inline proposal_generator used across all tests.  Must be
    symmetric: P(step | position) = P(-step | position + step).
    """
    return _PROPOSAL_SCALE * jax.random.normal(key, position.shape)


# ---------------------------------------------------------------------------
# 1. ENTRY field correctness
# ---------------------------------------------------------------------------


class TestAddStepRwEntryFields:
    """ENTRY field validation for additive_step_random_walk."""

    def test_name(self) -> None:
        assert ENTRY.name == "additive_step_random_walk"

    def test_family(self) -> None:
        assert ENTRY.family == "mcmc"

    def test_extra_required_kwargs_match(self) -> None:
        """P5.14a schema: extra_required_kwargs=('proposal_generator',)."""
        assert ENTRY.extra_required_kwargs == ("proposal_generator",)

    def test_target_acceptance_rate_none(self) -> None:
        """No universal optimal rate; depends on proposal-vs-target overlap."""
        assert ENTRY.target_acceptance_rate is None

    def test_needs_mass_matrix_false(self) -> None:
        assert ENTRY.needs_mass_matrix is False

    def test_default_hp_space_empty(self) -> None:
        """HP-free: proposal_generator encodes its own scale."""
        assert ENTRY.default_hp_space == ()

    def test_factory_callable(self) -> None:
        assert callable(ENTRY.factory)

    def test_grad_count_callable(self) -> None:
        assert callable(ENTRY.grad_count_per_step)


# ---------------------------------------------------------------------------
# 2. Factory with inline Gaussian proposal_generator
# ---------------------------------------------------------------------------


class TestAddStepRwFactory:
    """Factory invocation with a Gaussian proposal_generator."""

    def test_factory_returns_algorithm_with_init_step(self) -> None:
        algo = ENTRY.factory(
            mvn_5d_logdensity,
            proposal_generator=_gaussian_proposal,
        )
        assert hasattr(algo, "init"), "factory result must have .init"
        assert hasattr(algo, "step"), "factory result must have .step"

    def test_init_returns_rwstate(self) -> None:
        algo = ENTRY.factory(
            mvn_5d_logdensity,
            proposal_generator=_gaussian_proposal,
        )
        state = algo.init(jnp.zeros(_D))
        assert isinstance(state, RWState), f"Expected RWState, got {type(state)}"
        assert state.position.shape == (_D,)

    def test_500_step_scan_preserves_shape(self) -> None:
        """500 steps via lax.scan: shape preserved and logdensity finite."""
        algo = ENTRY.factory(
            mvn_5d_logdensity,
            proposal_generator=_gaussian_proposal,
        )
        state = algo.init(jnp.zeros(_D))

        def one_step(carry, key):
            new_state, info = algo.step(key, carry)
            return new_state, info

        keys = jax.random.split(jax.random.key(_SEED), 500)
        final_state, infos = jax.lax.scan(one_step, state, keys)

        assert final_state.position.shape == (
            _D,
        ), f"Position shape changed: {final_state.position.shape}"
        assert jnp.isfinite(
            final_state.logdensity
        ), f"logdensity not finite: {final_state.logdensity}"

    def test_acceptance_rate_finite_per_step(self) -> None:
        """All per-step acceptance_rates must be finite."""
        algo = ENTRY.factory(
            mvn_5d_logdensity,
            proposal_generator=_gaussian_proposal,
        )
        state = algo.init(jnp.zeros(_D))

        def one_step(carry, key):
            new_state, info = algo.step(key, carry)
            return new_state, info

        keys = jax.random.split(jax.random.key(_SEED), 500)
        _, infos = jax.lax.scan(one_step, state, keys)

        assert jnp.all(
            jnp.isfinite(infos.acceptance_rate)
        ), "Some per-step acceptance_rates are not finite."

    def test_mean_acceptance_rate_positive(self) -> None:
        """Proposal and target overlap enough that mean acceptance > 0."""
        algo = ENTRY.factory(
            mvn_5d_logdensity,
            proposal_generator=_gaussian_proposal,
        )
        state = algo.init(jnp.zeros(_D))

        def one_step(carry, key):
            new_state, info = algo.step(key, carry)
            return new_state, info

        keys = jax.random.split(jax.random.key(_SEED), 500)
        _, infos = jax.lax.scan(one_step, state, keys)

        mean_accept = float(jnp.mean(infos.acceptance_rate))
        assert (
            mean_accept > 0.0
        ), f"Mean acceptance rate is {mean_accept:.4f}; expected > 0"


# ---------------------------------------------------------------------------
# 3. no_warmup raises NotImplementedError without proposal_generator
# ---------------------------------------------------------------------------


class TestAddStepRwNoWarmupSchema:
    """Verifies P5.14a extra_required_kwargs schema: no_warmup raises NotImplementedError."""

    def test_no_warmup_raises_not_implemented_error(self) -> None:
        """no_warmup._runner raises NotImplementedError for any entry with
        non-empty extra_required_kwargs.  This proves P5.14a schema integration.
        """
        from bjx_bench.inference.warmup.no_warmup import _runner

        with pytest.raises(NotImplementedError, match="extra kwargs"):
            _runner(
                rng_key=jax.random.key(_SEED),
                init_position=jnp.zeros(_D),
                n_warmup=0,
                base_method=ENTRY,
                logdensity_fn=mvn_5d_logdensity,
                num_chains=1,
            )

    def test_no_warmup_error_mentions_method_name(self) -> None:
        """NotImplementedError message mentions the algorithm name."""
        from bjx_bench.inference.warmup.no_warmup import _runner

        with pytest.raises(NotImplementedError, match="additive_step_random_walk"):
            _runner(
                rng_key=jax.random.key(_SEED),
                init_position=jnp.zeros(_D),
                n_warmup=0,
                base_method=ENTRY,
                logdensity_fn=mvn_5d_logdensity,
                num_chains=1,
            )


# ---------------------------------------------------------------------------
# 4. grad_count_per_step returns 0 (gradient-free)
# ---------------------------------------------------------------------------


class TestAddStepRwGradCount:
    """grad_count_per_step returns 0 for gradient-free additive step RW."""

    def test_grad_count_zero_with_none_info(self) -> None:
        """grad_count_per_step(None) returns 0 (constant; gradient-free)."""
        result = ENTRY.grad_count_per_step(None)
        assert int(result) == 0, f"Expected grad_count=0, got {result}"

    def test_grad_count_returns_array(self) -> None:
        """grad_count_per_step returns a JAX array."""
        result = ENTRY.grad_count_per_step(None)
        assert isinstance(result, jax.Array), f"Expected JAX Array, got {type(result)}"


# ---------------------------------------------------------------------------
# 5. factory() without proposal_generator raises TypeError (contract pin)
# ---------------------------------------------------------------------------


class TestAddStepRwContractPin:
    """Calling factory without proposal_generator must raise TypeError."""

    def test_factory_without_proposal_generator_raises_typeerror(self) -> None:
        with pytest.raises(TypeError):
            ENTRY.factory(mvn_5d_logdensity)
