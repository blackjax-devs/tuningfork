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
"""Tests for the irmh base method registry entry.

Covers:
  1. ENTRY field correctness (name, family, extra_required_kwargs, etc.).
  2. default_params_for(ENTRY) returns {} (HP-free).
  3. Direct factory invocation with synthetic proposal: init + 500-step scan,
     shape preservation, acceptance_rate finite, mean acceptance > 0.
  4. Symmetric vs non-symmetric proposal path.
  5. grad_count_per_step returns 0 (gradient-free), synthesised via RWInfo.
  6. factory() without proposal_distribution raises TypeError (contract pin).
"""

import jax
import jax.numpy as jnp
import pytest
from blackjax.mcmc.random_walk import RWInfo, RWState

from tuningfork.inference.base_method.irmh import ENTRY

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_D = 5
_SEED = 42

# Gaussian target shifted off the standard-normal proposal mean.
_OBS_CENTER = jnp.array([1.0, -1.0, 0.5, 0.0, 0.0])


def _logdensity_fn(x):
    """Gaussian log-density centered at _OBS_CENTER."""
    return -0.5 * jnp.sum((x - _OBS_CENTER) ** 2)


def _proposal_distribution(key):
    """Standard normal proposal: q(y) = N(0, I); independent of current state."""
    return jax.random.normal(key, (_D,))


def _proposal_logdensity_fn(new_state, prev_state):
    """Log-density of the standard normal proposal: log q(new_state | prev_state).

    Note: blackjax.irmh passes proposal_logdensity_fn(new_state, prev_state) where
    both arguments are RWState NamedTuples (position, logdensity).  For an
    independent proposal q(y) = N(0, I), the log-density is only a function of
    new_state.position.
    """
    x = new_state.position
    return -0.5 * jnp.sum(x**2) - 0.5 * _D * jnp.log(2 * jnp.pi)


# ---------------------------------------------------------------------------
# 1. ENTRY field correctness
# ---------------------------------------------------------------------------


class TestIRMHEntryFields:
    """ENTRY field validation for irmh."""

    def test_name(self) -> None:
        assert ENTRY.name == "irmh"

    def test_family(self) -> None:
        assert ENTRY.family == "mcmc"

    def test_extra_required_kwargs_match(self) -> None:
        assert ENTRY.extra_required_kwargs == ("proposal_distribution",)

    def test_prior_cov_not_in_extra_required_kwargs(self) -> None:
        assert "prior_cov" not in ENTRY.extra_required_kwargs

    def test_target_acceptance_rate_none(self) -> None:
        """IRMH has no universal optimal acceptance rate."""
        assert ENTRY.target_acceptance_rate is None

    def test_needs_mass_matrix_false(self) -> None:
        assert ENTRY.needs_mass_matrix is False

    def test_default_hp_space_empty(self) -> None:
        """IRMH is hyperparameter-free; proposal is a full callable."""
        assert ENTRY.default_hp_space == ()

    def test_factory_callable(self) -> None:
        assert callable(ENTRY.factory)

    def test_grad_count_callable(self) -> None:
        assert callable(ENTRY.grad_count_per_step)


# ---------------------------------------------------------------------------
# 2. default_params_for returns {}
# ---------------------------------------------------------------------------


class TestIRMHDefaultParams:
    """default_params_for(ENTRY) returns an empty dict (HP-free)."""

    def test_default_params_empty(self) -> None:
        from tuningfork.calibration.tune import default_params_for

        params = default_params_for(ENTRY)
        assert params == {}, f"Expected empty dict, got {params!r}"


# ---------------------------------------------------------------------------
# 3. Direct factory invocation: init + 500-step scan
# ---------------------------------------------------------------------------


class TestIRMHFactory:
    """Factory invocation and kernel smoke test with synthetic proposal."""

    def test_factory_returns_sampling_algorithm(self) -> None:
        algo = ENTRY.factory(
            _logdensity_fn, proposal_distribution=_proposal_distribution
        )
        assert hasattr(algo, "init"), "factory result must have .init"
        assert hasattr(algo, "step"), "factory result must have .step"

    def test_init_returns_rw_state(self) -> None:
        algo = ENTRY.factory(
            _logdensity_fn, proposal_distribution=_proposal_distribution
        )
        init_pos = jnp.zeros(_D)
        state = algo.init(init_pos)
        assert isinstance(state, RWState), f"Expected RWState, got {type(state)}"
        assert state.position.shape == (_D,)

    def test_500_step_scan_preserves_shape(self) -> None:
        """Run 500 steps via jax.lax.scan; verify shape and finiteness."""
        algo = ENTRY.factory(
            _logdensity_fn, proposal_distribution=_proposal_distribution
        )
        init_pos = jnp.zeros(_D)
        state = algo.init(init_pos)

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
        """All per-step acceptance_rates must be finite scalars."""
        algo = ENTRY.factory(
            _logdensity_fn, proposal_distribution=_proposal_distribution
        )
        state = algo.init(jnp.zeros(_D))

        def one_step(carry, key):
            new_state, info = algo.step(key, carry)
            return new_state, info

        keys = jax.random.split(jax.random.key(_SEED), 500)
        _, infos = jax.lax.scan(one_step, state, keys)

        assert jnp.all(
            jnp.isfinite(infos.acceptance_rate)
        ), "Some per-step acceptance_rates are not finite"

    def test_mean_acceptance_rate_positive(self) -> None:
        """Proposal and target overlap enough that mean acceptance > 0."""
        algo = ENTRY.factory(
            _logdensity_fn, proposal_distribution=_proposal_distribution
        )
        state = algo.init(jnp.zeros(_D))

        def one_step(carry, key):
            new_state, info = algo.step(key, carry)
            return new_state, info

        keys = jax.random.split(jax.random.key(_SEED), 500)
        _, infos = jax.lax.scan(one_step, state, keys)

        mean_accept = jnp.mean(infos.acceptance_rate)
        assert (
            float(mean_accept) > 0.0
        ), f"Mean acceptance rate is {float(mean_accept):.4f}; expected > 0"


# ---------------------------------------------------------------------------
# 4. Symmetric vs non-symmetric proposal path
# ---------------------------------------------------------------------------


class TestIRMHProposalPaths:
    """Symmetric (no proposal_logdensity_fn) and non-symmetric paths both work."""

    def test_symmetric_path_runs_without_error(self) -> None:
        """proposal_logdensity_fn=None (default): symmetric MH ratio, no correction."""
        algo = ENTRY.factory(
            _logdensity_fn,
            proposal_distribution=_proposal_distribution,
            proposal_logdensity_fn=None,
        )
        state = algo.init(jnp.zeros(_D))
        new_state, info = algo.step(jax.random.key(_SEED), state)
        assert jnp.isfinite(
            info.acceptance_rate
        ), f"acceptance_rate not finite in symmetric path: {info.acceptance_rate}"

    def test_nonsymmetric_path_runs_without_error(self) -> None:
        """Non-trivial proposal_logdensity_fn: MH ratio includes proposal correction."""
        algo = ENTRY.factory(
            _logdensity_fn,
            proposal_distribution=_proposal_distribution,
            proposal_logdensity_fn=_proposal_logdensity_fn,
        )
        state = algo.init(jnp.zeros(_D))
        new_state, info = algo.step(jax.random.key(_SEED), state)
        assert jnp.isfinite(
            info.acceptance_rate
        ), f"acceptance_rate not finite in non-symmetric path: {info.acceptance_rate}"

    def test_nonsymmetric_acceptance_differs_from_symmetric(self) -> None:
        """The two paths produce different (but both finite) acceptance behavior.

        Note: since _proposal_distribution is symmetric (N(0,I)) but we also pass
        a non-None proposal_logdensity_fn, the MH correction is applied differently,
        leading to different acceptance_rate values over a scan.
        """
        algo_sym = ENTRY.factory(
            _logdensity_fn,
            proposal_distribution=_proposal_distribution,
            proposal_logdensity_fn=None,
        )
        algo_nonsym = ENTRY.factory(
            _logdensity_fn,
            proposal_distribution=_proposal_distribution,
            proposal_logdensity_fn=_proposal_logdensity_fn,
        )

        state0 = algo_sym.init(jnp.zeros(_D))
        state1 = algo_nonsym.init(jnp.zeros(_D))

        def scan_sym(carry, key):
            new_state, info = algo_sym.step(key, carry)
            return new_state, info

        def scan_nonsym(carry, key):
            new_state, info = algo_nonsym.step(key, carry)
            return new_state, info

        keys = jax.random.split(jax.random.key(_SEED + 1), 200)
        _, infos_sym = jax.lax.scan(scan_sym, state0, keys)
        _, infos_nonsym = jax.lax.scan(scan_nonsym, state1, keys)

        # Both must be finite
        assert jnp.all(
            jnp.isfinite(infos_sym.acceptance_rate)
        ), "Symmetric path: some acceptance_rates are not finite"
        assert jnp.all(
            jnp.isfinite(infos_nonsym.acceptance_rate)
        ), "Non-symmetric path: some acceptance_rates are not finite"

        # They must differ (passing proposal_logdensity_fn changes the MH correction)
        mean_sym = float(jnp.mean(infos_sym.acceptance_rate))
        mean_nonsym = float(jnp.mean(infos_nonsym.acceptance_rate))
        assert mean_sym != mean_nonsym, (
            f"Symmetric and non-symmetric paths produced identical mean acceptance "
            f"({mean_sym:.6f}); expected them to differ."
        )


# ---------------------------------------------------------------------------
# 5. grad_count_per_step returns 0 (gradient-free)
# ---------------------------------------------------------------------------


class TestIRMHGradCount:
    """grad_count_per_step returns 0 for gradient-free IRMH."""

    def test_grad_count_zero_with_synthetic_info(self) -> None:
        """Synthesise a minimal RWInfo and verify grad count = 0."""
        fake_proposal = RWState(position=jnp.zeros(_D), logdensity=jnp.asarray(0.0))
        fake_info = RWInfo(
            acceptance_rate=jnp.asarray(0.5),
            is_accepted=jnp.asarray(True),
            proposal=fake_proposal,
        )
        result = ENTRY.grad_count_per_step(fake_info)
        assert int(result) == 0, f"Expected grad_count=0, got {result}"

    def test_grad_count_returns_array(self) -> None:
        fake_proposal = RWState(position=jnp.zeros(_D), logdensity=jnp.asarray(0.0))
        fake_info = RWInfo(
            acceptance_rate=jnp.asarray(0.5),
            is_accepted=jnp.asarray(True),
            proposal=fake_proposal,
        )
        result = ENTRY.grad_count_per_step(fake_info)
        assert isinstance(result, jax.Array), f"Expected JAX Array, got {type(result)}"


# ---------------------------------------------------------------------------
# 6. factory() without proposal_distribution raises TypeError
# ---------------------------------------------------------------------------


class TestIRMHContractPin:
    """Calling factory without proposal_distribution must raise TypeError."""

    def test_factory_without_proposal_distribution_raises_typeerror(self) -> None:
        with pytest.raises(TypeError):
            ENTRY.factory(_logdensity_fn)
