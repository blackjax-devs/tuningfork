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
"""Tests for the P5.12 meanfield_vi warmup wrapper.

Covers:
  1. ENTRY field correctness (name, compatible_methods).
  2. is_compatible() returns True for hmc/nuts/mala/rwm/barker.
  3. is_compatible() returns False for mclmc.
  4. _runner shape check: states.position shape (num_chains, D) after run.
  5. adapted_params keys: step_size, inverse_mass_matrix (diagonal), _mfvi_elbo.
  6. inverse_mass_matrix shape: (num_chains, D) — diagonal per coord.
  7. inverse_mass_matrix values: positive (variance-derived, must be > 0).
  8. Sidecar _mfvi_elbo is finite.
  9. End-to-end: 5-D std normal; _mfvi_elbo converges; diag IMM near 1.0.
 10. Incompatible base_method raises ValueError.

Single seed make_rng(42) per kickoff decision.
"""

import math

import jax
import jax.numpy as jnp
import pytest

from bjx_bench.inference.base_method import BASE_METHODS
from bjx_bench.inference.warmup.meanfield_vi import ENTRY

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


class TestMFVIWarmupEntry:
    """ENTRY field validation for meanfield_vi warmup."""

    def test_name(self) -> None:
        assert ENTRY.name == "meanfield_vi"

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


class TestMFVIWarmupCompatibility:
    """is_compatible() contract for meanfield_vi warmup."""

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


class TestMFVIWarmupRunner:
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

    def test_adapted_params_has_mfvi_elbo_sidecar(self) -> None:
        _, adapted_params = self._run()
        assert (
            "_mfvi_elbo" in adapted_params
        ), "adapted_params missing '_mfvi_elbo' sidecar"

    def test_inverse_mass_matrix_shape(self) -> None:
        """Diagonal IMM must be shape (num_chains, D)."""
        _, adapted_params = self._run()
        imm = adapted_params["inverse_mass_matrix"]
        assert imm.shape == (
            _NUM_CHAINS,
            _D,
        ), f"Expected IMM shape ({_NUM_CHAINS}, {_D}), got {imm.shape}"

    def test_inverse_mass_matrix_positive(self) -> None:
        """Diagonal IMM must be positive (variance-derived via exp(2*rho))."""
        _, adapted_params = self._run()
        imm = adapted_params["inverse_mass_matrix"]
        assert jnp.all(imm > 0), f"IMM must be positive; got min={jnp.min(imm)}"

    def test_step_size_shape(self) -> None:
        """step_size must be (num_chains,) array."""
        _, adapted_params = self._run()
        ss = adapted_params["step_size"]
        assert ss.shape == (
            _NUM_CHAINS,
        ), f"Expected step_size shape ({_NUM_CHAINS},), got {ss.shape}"

    def test_mfvi_elbo_is_finite(self) -> None:
        """Sidecar _mfvi_elbo must be a finite scalar."""
        _, adapted_params = self._run()
        elbo = adapted_params["_mfvi_elbo"]
        assert jnp.isfinite(elbo), f"_mfvi_elbo is not finite: {elbo}"

    def test_mfvi_elbo_is_scalar(self) -> None:
        """Sidecar _mfvi_elbo must be a scalar (shape ())."""
        _, adapted_params = self._run()
        elbo = jnp.asarray(adapted_params["_mfvi_elbo"])
        assert elbo.shape == (), f"Expected _mfvi_elbo shape (), got {elbo.shape}"

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


class TestMFVIWarmupEndToEnd:
    """End-to-end convergence test on 5-D standard normal target.

    Uses 2_000 optimisation steps (test default from kickoff).
    """

    def test_mfvi_elbo_converges(self) -> None:
        """_mfvi_elbo > -0.5*5*(1+log(2π)) - 1.0 after 2_000 steps."""
        key = jax.random.key(_SEED)
        _, adapted_params = ENTRY.runner(
            key,
            jnp.zeros(_D),
            2_000,
            _NUTS,
            logdensity_fn=_logdensity_fn,
            num_chains=2,
            num_optimization_steps=2_000,
        )
        final_elbo = float(adapted_params["_mfvi_elbo"])
        threshold = -0.5 * _D * (1 + math.log(2 * math.pi)) - 1.0
        assert final_elbo > threshold, (
            f"ELBO did not converge: final_elbo={final_elbo:.4f}, "
            f"threshold={threshold:.4f}"
        )

    def test_diagonal_imm_near_one_for_std_normal(self) -> None:
        """For std normal target, diagonal IMM should be close to 1.0."""
        key = jax.random.key(_SEED)
        _, adapted_params = ENTRY.runner(
            key,
            jnp.zeros(_D),
            2_000,
            _NUTS,
            logdensity_fn=_logdensity_fn,
            num_chains=2,
            num_optimization_steps=2_000,
        )
        imm = adapted_params["inverse_mass_matrix"]
        # For std normal, optimal MFVI recovers sigma=1 -> exp(2*rho)≈1
        # Allow generous tolerance since test uses reduced step count.
        assert jnp.allclose(
            imm, jnp.ones_like(imm), atol=0.5
        ), f"Diagonal IMM for std normal should be near 1.0; got mean={jnp.mean(imm):.4f}"
