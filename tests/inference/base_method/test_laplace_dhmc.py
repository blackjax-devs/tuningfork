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
"""Tests for the laplace_dhmc base method registry entry.

Covers:
  1. ENTRY field correctness (name, family, extra_required_kwargs, etc.).
  2. Factory smoke: constructs without error.
  3. End-to-end smoke: 50-step scan, no NaN, finite acceptance rate.
  4. no_warmup runner raises NotImplementedError mentioning extra_required_kwargs.

Toy hierarchical model (Gaussian-Gaussian, Laplace is exact):
    phi   ~ N(0, 10^2)                   [hyperparameter, scalar]
    theta | phi ~ N(0, exp(phi/2)^2 * I) [latent, n-vector]
    y     | theta ~ N(theta, I)           [observations, n-vector]

Note: laplace_dhmc uses dynamic (quasi-random) step count.  The init signature
is ``.init(phi_init, rng_key)`` (extra rng_key argument for step-count seeding).
"""

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import pytest

from bjx_bench.inference.base_method.laplace_dhmc import ENTRY

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Toy hierarchical model
# ---------------------------------------------------------------------------

_N_LATENT = 3
_SEED = 42
_Y = jnp.array([0.5, -0.3, 1.2])  # fixed synthetic observations


def _log_joint(theta, log_sigma):
    """Full log joint for Gaussian-Gaussian hierarchical model.

    Parameters
    ----------
    theta
        Latent variables, shape (N_LATENT,).
    log_sigma
        Hyperparameter (log std), scalar.
    """
    sigma = jnp.exp(log_sigma)
    log_prior_phi = stats.norm.logpdf(log_sigma, 0.0, 10.0)
    log_prior_theta = stats.norm.logpdf(theta, 0.0, sigma).sum()
    log_lik = stats.norm.logpdf(_Y, theta, 1.0).sum()
    return log_prior_phi + log_prior_theta + log_lik


_THETA_INIT = jnp.zeros(_N_LATENT)
_PHI_INIT = jnp.array(0.0)  # scalar log-sigma
_INVERSE_MASS_MATRIX = jnp.ones(1)  # phi is scalar


# ---------------------------------------------------------------------------
# 1. ENTRY field correctness
# ---------------------------------------------------------------------------


class TestLaplaceDHMCEntryFields:
    """ENTRY field validation for laplace_dhmc."""

    def test_name(self) -> None:
        assert ENTRY.name == "laplace_dhmc"

    def test_family(self) -> None:
        assert ENTRY.family == "mcmc"

    def test_extra_required_kwargs_match(self) -> None:
        assert ENTRY.extra_required_kwargs == ("log_joint_fn", "theta_init")

    def test_target_acceptance_rate(self) -> None:
        assert ENTRY.target_acceptance_rate == pytest.approx(0.8)

    def test_needs_mass_matrix_true(self) -> None:
        assert ENTRY.needs_mass_matrix is True

    def test_default_hp_space_has_step_size_only(self) -> None:
        """Dynamic variant: only step_size is BO-tunable (trajectory length is adapted)."""
        hp_names = {hp.name for hp in ENTRY.default_hp_space}
        assert "step_size" in hp_names
        assert (
            "num_integration_steps" not in hp_names
        ), "laplace_dhmc is dynamic: num_integration_steps should NOT be in HP space"

    def test_factory_callable(self) -> None:
        assert callable(ENTRY.factory)

    def test_grad_count_callable(self) -> None:
        assert callable(ENTRY.grad_count_per_step)


# ---------------------------------------------------------------------------
# 2. Factory smoke: constructs without error
# ---------------------------------------------------------------------------


class TestLaplaceDHMCFactorySmoke:
    """Factory smoke tests: constructs a kernel and init state without error."""

    def test_factory_returns_sampling_algorithm(self) -> None:
        algo = ENTRY.factory(
            None,  # logdensity_fn: NOT USED
            log_joint_fn=_log_joint,
            theta_init=_THETA_INIT,
            step_size=0.1,
            inverse_mass_matrix=_INVERSE_MASS_MATRIX,
            maxiter=50,
        )
        assert hasattr(algo, "init"), "factory result must have .init"
        assert hasattr(algo, "step"), "factory result must have .step"

    def test_factory_without_log_joint_fn_raises(self) -> None:
        with pytest.raises(TypeError):
            ENTRY.factory(
                None,
                theta_init=_THETA_INIT,
                step_size=0.1,
                inverse_mass_matrix=_INVERSE_MASS_MATRIX,
            )

    def test_factory_without_theta_init_raises(self) -> None:
        with pytest.raises(TypeError):
            ENTRY.factory(
                None,
                log_joint_fn=_log_joint,
                step_size=0.1,
                inverse_mass_matrix=_INVERSE_MASS_MATRIX,
            )

    def test_init_requires_rng_key_returns_state_with_theta_star(self) -> None:
        """laplace_dhmc.init takes (phi_init, rng_key) — dynamic step-count seeding."""
        algo = ENTRY.factory(
            None,
            log_joint_fn=_log_joint,
            theta_init=_THETA_INIT,
            step_size=0.1,
            inverse_mass_matrix=_INVERSE_MASS_MATRIX,
            maxiter=50,
        )
        state = algo.init(_PHI_INIT, jax.random.key(_SEED))
        assert hasattr(state, "position"), "State must have 'position' field"
        assert hasattr(
            state, "theta_star"
        ), "LaplaceDynamicHMCState must have 'theta_star'"
        assert hasattr(
            state, "random_generator_arg"
        ), "LaplaceDynamicHMCState must have 'random_generator_arg'"
        assert jnp.isfinite(
            state.logdensity
        ), f"logdensity not finite: {state.logdensity}"
        assert jnp.all(
            jnp.isfinite(state.theta_star)
        ), "theta_star has non-finite values"


# ---------------------------------------------------------------------------
# 3. End-to-end smoke: 50-step scan, no NaN, finite acceptance rate
# ---------------------------------------------------------------------------


class TestLaplaceDHMCEndToEnd:
    """End-to-end smoke test: 50-step scan, no NaN, finite acceptance rate."""

    def test_50_step_scan_no_nan(self) -> None:
        algo = ENTRY.factory(
            None,
            log_joint_fn=_log_joint,
            theta_init=_THETA_INIT,
            step_size=0.05,
            inverse_mass_matrix=_INVERSE_MASS_MATRIX,
            maxiter=50,
        )
        state = algo.init(_PHI_INIT, jax.random.key(_SEED))

        def one_step(carry, key):
            new_state, info = algo.step(key, carry)
            return new_state, info

        keys = jax.random.split(jax.random.key(_SEED + 1), 50)
        final_state, infos = jax.lax.scan(one_step, state, keys)

        assert jnp.isfinite(
            final_state.logdensity
        ), f"logdensity not finite after 50 steps: {final_state.logdensity}"
        assert jnp.all(
            jnp.isfinite(final_state.position)
        ), "Final phi position has non-finite values"
        assert jnp.all(
            jnp.isfinite(final_state.theta_star)
        ), "Final theta_star has non-finite values"

    def test_50_step_acceptance_rate_finite_and_positive(self) -> None:
        algo = ENTRY.factory(
            None,
            log_joint_fn=_log_joint,
            theta_init=_THETA_INIT,
            step_size=0.05,
            inverse_mass_matrix=_INVERSE_MASS_MATRIX,
            maxiter=50,
        )
        state = algo.init(_PHI_INIT, jax.random.key(_SEED))

        def one_step(carry, key):
            new_state, info = algo.step(key, carry)
            return new_state, info

        keys = jax.random.split(jax.random.key(_SEED + 1), 50)
        _, infos = jax.lax.scan(one_step, state, keys)

        assert jnp.all(
            jnp.isfinite(infos.acceptance_rate)
        ), "Some acceptance_rates are not finite"
        mean_ar = float(jnp.mean(infos.acceptance_rate))
        assert mean_ar > 0.0, f"Mean acceptance rate = {mean_ar:.4f}; expected > 0"

    def test_info_has_num_integration_steps(self) -> None:
        """HMCInfo must have num_integration_steps for grad_count_per_step."""
        algo = ENTRY.factory(
            None,
            log_joint_fn=_log_joint,
            theta_init=_THETA_INIT,
            step_size=0.05,
            inverse_mass_matrix=_INVERSE_MASS_MATRIX,
            maxiter=50,
        )
        state = algo.init(_PHI_INIT, jax.random.key(_SEED))
        _, info = algo.step(jax.random.key(_SEED + 2), state)
        assert hasattr(
            info, "num_integration_steps"
        ), "HMCInfo must have 'num_integration_steps' for grad_count_per_step"
        grad_count = ENTRY.grad_count_per_step(info)
        assert int(grad_count) > 0, f"grad_count must be positive, got {grad_count}"


# ---------------------------------------------------------------------------
# 4. no_warmup runner raises NotImplementedError (schema wired correctly)
# ---------------------------------------------------------------------------


class TestLaplaceDHMCNoWarmup:
    """Proves the extra_required_kwargs schema is wired: no_warmup raises NotImplementedError."""

    def test_no_warmup_raises_not_implemented(self) -> None:
        from bjx_bench.inference.warmup.no_warmup import _runner

        with pytest.raises(NotImplementedError, match="extra kwargs"):
            _runner(
                jax.random.key(0),
                _PHI_INIT,
                0,
                ENTRY,
                logdensity_fn=lambda x: -0.5 * jnp.sum(x**2),
                num_chains=1,
            )
