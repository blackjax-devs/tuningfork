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
"""Tests for the fullrank_vi warmup wrapper.

Covers:
  1. ENTRY field correctness (name, compatible_methods).
  2. is_compatible() returns True for hmc/nuts/mala/rwm/barker.
  3. is_compatible() returns False for mclmc.
  4. _runner shape check: states.position shape (num_chains, D) after run.
  5. adapted_params keys: step_size, inverse_mass_matrix (dense), _frvi_elbo.
  6. inverse_mass_matrix shape: (num_chains, D, D) — dense covariance.
  7. inverse_mass_matrix is positive definite (all eigenvalues > 0).
  8. Sidecar _frvi_elbo is finite.
  9. End-to-end: 5-D std normal; _frvi_elbo converges; dense IMM near I_D.
 10. Incompatible base_method raises ValueError.

Single seed make_rng(42) per kickoff decision.
"""

import math

import jax
import jax.numpy as jnp
import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.warmup.fullrank_vi import ENTRY

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_D = 5
_SEED = 42
_NUM_CHAINS = 2  # keep low for test speed


def _logdensity_fn(x: jax.Array) -> jax.Array:
    """Standard 5-D isotropic Gaussian log-density."""
    return -0.5 * jnp.sum(x**2)


_NUTS = BASE_METHODS["nuts"]
_HMC = BASE_METHODS["hmc"]
_MCLMC = BASE_METHODS["mclmc"]


# ---------------------------------------------------------------------------
# 1. ENTRY field correctness
# ---------------------------------------------------------------------------


class TestFRVIWarmupEntry:
    """ENTRY field validation for fullrank_vi warmup."""

    def test_name(self) -> None:
        assert ENTRY.name == "fullrank_vi"

    def test_compatible_methods_has_nuts(self) -> None:
        assert "nuts" in ENTRY.compatible_methods

    def test_compatible_methods_has_hmc(self) -> None:
        assert "hmc" in ENTRY.compatible_methods

    def test_compatible_methods_has_mala(self) -> None:
        assert "mala" in ENTRY.compatible_methods

    def test_compatible_methods_has_rwm(self) -> None:
        assert "rwm" in ENTRY.compatible_methods

    def test_compatible_methods_has_barker(self) -> None:
        assert "barker" in ENTRY.compatible_methods

    def test_runner_callable(self) -> None:
        assert callable(ENTRY.runner)


# ---------------------------------------------------------------------------
# 2-3. is_compatible()
# ---------------------------------------------------------------------------


class TestFRVIWarmupCompatibility:
    """is_compatible() contract for fullrank_vi warmup."""

    def test_compatible_with_nuts(self) -> None:
        assert ENTRY.is_compatible("nuts")

    def test_compatible_with_hmc(self) -> None:
        assert ENTRY.is_compatible("hmc")

    def test_compatible_with_mala(self) -> None:
        assert ENTRY.is_compatible("mala")

    def test_compatible_with_rwm(self) -> None:
        assert ENTRY.is_compatible("rwm")

    def test_compatible_with_barker(self) -> None:
        assert ENTRY.is_compatible("barker")

    def test_not_compatible_with_mclmc(self) -> None:
        assert not ENTRY.is_compatible("mclmc")

    def test_not_compatible_with_unknown(self) -> None:
        assert not ENTRY.is_compatible("unknown_algo")


# ---------------------------------------------------------------------------
# 4-8. _runner shape and content checks
# ---------------------------------------------------------------------------


class TestFRVIWarmupRunner:
    """_runner() output shape and content checks."""

    def _run(self, num_chains: int = _NUM_CHAINS, num_optimization_steps: int = 200):
        """Helper: run the warmup and return (states, adapted_params)."""
        key = jax.random.key(_SEED)
        init_pos = jnp.zeros(_D)
        states, adapted_params = ENTRY.runner(
            key,
            init_pos,
            500,  # n_warmup (unused by VI warmup)
            _NUTS,
            logdensity_fn=_logdensity_fn,
            num_chains=num_chains,
            num_optimization_steps=num_optimization_steps,
        )
        return states, adapted_params

    def test_states_position_shape(self) -> None:
        states, _ = self._run()
        assert states.position.shape == (_NUM_CHAINS, _D), (
            f"Expected position shape ({_NUM_CHAINS}, {_D}), "
            f"got {states.position.shape}"
        )

    def test_adapted_params_has_step_size(self) -> None:
        _, adapted_params = self._run()
        assert "step_size" in adapted_params, "adapted_params missing 'step_size'"

    def test_adapted_params_has_inverse_mass_matrix(self) -> None:
        _, adapted_params = self._run()
        assert (
            "inverse_mass_matrix" in adapted_params
        ), "adapted_params missing 'inverse_mass_matrix'"

    def test_adapted_params_has_frvi_elbo_sidecar(self) -> None:
        _, adapted_params = self._run()
        assert (
            "_frvi_elbo" in adapted_params
        ), "adapted_params missing '_frvi_elbo' sidecar"

    def test_inverse_mass_matrix_shape(self) -> None:
        """Dense IMM must be shape (num_chains, D, D)."""
        _, adapted_params = self._run()
        imm = adapted_params["inverse_mass_matrix"]
        assert imm.shape == (
            _NUM_CHAINS,
            _D,
            _D,
        ), f"Expected IMM shape ({_NUM_CHAINS}, {_D}, {_D}), got {imm.shape}"

    def test_inverse_mass_matrix_positive_definite(self) -> None:
        """Dense IMM must be positive definite (all eigenvalues > 0)."""
        _, adapted_params = self._run()
        imm = adapted_params["inverse_mass_matrix"]
        # Check the first chain's IMM; they should all be identical.
        imm_chain0 = imm[0]  # (D, D)
        eigenvalues = jnp.linalg.eigvalsh(imm_chain0)
        assert jnp.all(
            eigenvalues > 0
        ), f"Dense IMM must be positive definite; min eigenvalue={jnp.min(eigenvalues)}"

    def test_step_size_shape(self) -> None:
        """step_size must be (num_chains,) array."""
        _, adapted_params = self._run()
        ss = adapted_params["step_size"]
        assert ss.shape == (
            _NUM_CHAINS,
        ), f"Expected step_size shape ({_NUM_CHAINS},), got {ss.shape}"

    def test_frvi_elbo_is_finite(self) -> None:
        """Sidecar _frvi_elbo must be a finite scalar."""
        _, adapted_params = self._run()
        elbo = adapted_params["_frvi_elbo"]
        assert jnp.isfinite(elbo), f"_frvi_elbo is not finite: {elbo}"

    def test_frvi_elbo_is_scalar(self) -> None:
        """Sidecar _frvi_elbo must be a scalar (shape ())."""
        _, adapted_params = self._run()
        elbo = jnp.asarray(adapted_params["_frvi_elbo"])
        assert elbo.shape == (), f"Expected _frvi_elbo shape (), got {elbo.shape}"

    def test_incompatible_base_method_raises(self) -> None:
        """Using mclmc (incompatible) must raise ValueError."""
        key = jax.random.key(_SEED)
        with pytest.raises(ValueError, match="compatible"):
            ENTRY.runner(
                key,
                jnp.zeros(_D),
                100,
                _MCLMC,
                logdensity_fn=_logdensity_fn,
                num_chains=2,
                num_optimization_steps=50,
            )


# ---------------------------------------------------------------------------
# 9. End-to-end: 5-D std normal convergence
# ---------------------------------------------------------------------------


class TestFRVIWarmupEndToEnd:
    """End-to-end convergence test on 5-D standard normal target.

    Uses 5_000 optimisation steps (test default from kickoff).
    """

    def test_frvi_elbo_converges(self) -> None:
        """_frvi_elbo > -0.5*5*(1+log(2π)) - 1.0 after 5_000 steps."""
        key = jax.random.key(_SEED)
        _, adapted_params = ENTRY.runner(
            key,
            jnp.zeros(_D),
            5_000,
            _NUTS,
            logdensity_fn=_logdensity_fn,
            num_chains=2,
            num_optimization_steps=5_000,
        )
        final_elbo = float(adapted_params["_frvi_elbo"])
        threshold = -0.5 * _D * (1 + math.log(2 * math.pi)) - 1.0
        assert final_elbo > threshold, (
            f"ELBO did not converge: final_elbo={final_elbo:.4f}, "
            f"threshold={threshold:.4f}"
        )

    def test_dense_imm_near_identity_for_std_normal(self) -> None:
        """For std normal target, dense IMM should be close to I_D."""
        key = jax.random.key(_SEED)
        _, adapted_params = ENTRY.runner(
            key,
            jnp.zeros(_D),
            5_000,
            _NUTS,
            logdensity_fn=_logdensity_fn,
            num_chains=2,
            num_optimization_steps=5_000,
        )
        imm = adapted_params["inverse_mass_matrix"]
        imm_chain0 = imm[0]  # (D, D) — same for all chains
        # For std normal, optimal FRVI recovers Cov = I_D.
        # Allow generous tolerance since test uses reduced step count.
        assert jnp.allclose(imm_chain0, jnp.eye(_D), atol=0.5), (
            f"Dense IMM for std normal should be near I_D; "
            f"max_abs_diff={jnp.max(jnp.abs(imm_chain0 - jnp.eye(_D))):.4f}"
        )

    def test_chains_have_identical_imm(self) -> None:
        """All chains must share the same IMM (VI fit is shared)."""
        key = jax.random.key(_SEED)
        _, adapted_params = ENTRY.runner(
            key,
            jnp.zeros(_D),
            500,
            _NUTS,
            logdensity_fn=_logdensity_fn,
            num_chains=2,
            num_optimization_steps=200,
        )
        imm = adapted_params["inverse_mass_matrix"]  # (num_chains, D, D)
        # Chains should have identical IMM (broadcast from shared fit).
        assert jnp.allclose(
            imm[0], imm[1]
        ), "All chains must share the same IMM (VI fit is shared, not per-chain)"
