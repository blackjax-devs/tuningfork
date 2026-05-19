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
"""Tests for the multipathfinder_window_adaptation warmup entry.

Covers:
  1. Registry: entry is present in WARMUPS and is a Warmup instance.
  2. Compatibility: nuts/hmc/mala/rwm/barker → True; mclmc → False.
  3. Smoke test: NUTS on MVN-10 at n_warmup=100, num_chains=1; shape contract.
  4. Multi-chain shape contract: num_chains=2 returns (2, d, d) IMM.
  5. Sidecar key _multipathfinder_psis_pareto_k present and finite (or nan).
  6. step_size all positive.
  7. Incompatible base_method raises ValueError.
"""

import jax
import jax.numpy as jnp
import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.model import MODELS
from tuningfork.warmup import WARMUPS, Warmup

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_MVN = MODELS["mvn_10"]
_NUTS = BASE_METHODS["nuts"]
_HMC = BASE_METHODS["hmc"]
_MCLMC = BASE_METHODS["mclmc"]
_SEED = 77


def _build_logdensity(posterior_entry, key):
    from tuningfork.model._numpyro import build_logdensity_fn

    init_position, logdensity_fn, _ = build_logdensity_fn(key, posterior_entry)
    return init_position, logdensity_fn


def _run(seed: int, num_chains: int = 1, n_paths: int | None = None, **kw):
    """Helper: run multipathfinder_window_adaptation on MVN-10."""
    key = jax.random.key(seed)
    init_pos, logdensity_fn = _build_logdensity(_MVN, key)
    warmup_key = jax.random.fold_in(key, 99)
    runner_kwargs = {}
    if n_paths is not None:
        runner_kwargs["n_paths"] = n_paths
    runner_kwargs.update(kw)
    return WARMUPS["multipathfinder_window_adaptation"].runner(
        warmup_key,
        init_pos,
        100,  # n_warmup — small for tests
        _NUTS,
        logdensity_fn=logdensity_fn,
        num_chains=num_chains,
        **runner_kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Registry presence
# ---------------------------------------------------------------------------


class TestRegistry:
    """Registry structure tests (fast; no chain runs)."""

    def test_entry_registered(self) -> None:
        """multipathfinder_window_adaptation is in WARMUPS."""
        assert (
            "multipathfinder_window_adaptation" in WARMUPS
        ), f"WARMUPS keys: {sorted(WARMUPS)}"

    def test_entry_is_warmup_instance(self) -> None:
        assert isinstance(WARMUPS["multipathfinder_window_adaptation"], Warmup)

    def test_entry_name_matches_key(self) -> None:
        entry = WARMUPS["multipathfinder_window_adaptation"]
        assert entry.name == "multipathfinder_window_adaptation"


# ---------------------------------------------------------------------------
# 2. Compatibility
# ---------------------------------------------------------------------------


class TestCompatibility:
    """is_compatible() for multipathfinder_window_adaptation."""

    def test_compatible_with_nuts(self) -> None:
        assert WARMUPS["multipathfinder_window_adaptation"].is_compatible("nuts")

    def test_compatible_with_hmc(self) -> None:
        assert WARMUPS["multipathfinder_window_adaptation"].is_compatible("hmc")

    def test_compatible_with_mala(self) -> None:
        assert WARMUPS["multipathfinder_window_adaptation"].is_compatible("mala")

    def test_compatible_with_rwm(self) -> None:
        assert WARMUPS["multipathfinder_window_adaptation"].is_compatible("rwm")

    def test_compatible_with_barker(self) -> None:
        assert WARMUPS["multipathfinder_window_adaptation"].is_compatible("barker")

    def test_not_compatible_with_mclmc(self) -> None:
        assert not WARMUPS["multipathfinder_window_adaptation"].is_compatible("mclmc")


# ---------------------------------------------------------------------------
# 3. Smoke test: shape contract, num_chains=1
# ---------------------------------------------------------------------------


class TestSmokeNumChains1:
    """Smoke tests with num_chains=1 on MVN-10 / NUTS."""

    def test_returns_tuple(self) -> None:
        result = _run(101, num_chains=1, n_paths=2)
        assert isinstance(result, tuple) and len(result) == 2

    def test_adapted_params_has_step_size(self) -> None:
        _, params = _run(102, num_chains=1, n_paths=2)
        assert "step_size" in params, f"params keys: {list(params.keys())}"

    def test_adapted_params_has_imm(self) -> None:
        _, params = _run(103, num_chains=1, n_paths=2)
        assert "inverse_mass_matrix" in params, f"params keys: {list(params.keys())}"

    def test_step_size_shape_single_chain(self) -> None:
        """step_size should have shape (1,) for num_chains=1."""
        _, params = _run(104, num_chains=1, n_paths=2)
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (1,), f"step_size.shape={ss.shape}, expected (1,)"

    def test_imm_shape_single_chain(self) -> None:
        """IMM should have shape (1, 10, 10) for num_chains=1, d=10."""
        _, params = _run(105, num_chains=1, n_paths=2)
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (
            1,
            10,
            10,
        ), f"inverse_mass_matrix.shape={imm.shape}, expected (1, 10, 10)"

    def test_step_size_positive(self) -> None:
        _, params = _run(106, num_chains=1, n_paths=2)
        ss = jnp.asarray(params["step_size"])
        assert bool(jnp.all(ss > 0)), f"step_size={ss} not all > 0"

    def test_imm_is_dense_matrix(self) -> None:
        """IMM should be 2-D square (dense) per chain."""
        _, params = _run(107, num_chains=1, n_paths=2)
        imm = params["inverse_mass_matrix"]
        assert imm.ndim == 3, f"Expected 3-D (num_chains, d, d), got ndim={imm.ndim}"


# ---------------------------------------------------------------------------
# 4. Multi-chain shape contract: num_chains=2
# ---------------------------------------------------------------------------


class TestMultiChain:
    """Multi-chain shape contract: num_chains=2."""

    def test_step_size_shape_2chains(self) -> None:
        _, params = _run(201, num_chains=2, n_paths=2)
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (2,), f"step_size.shape={ss.shape}, expected (2,)"

    def test_imm_shape_2chains(self) -> None:
        _, params = _run(202, num_chains=2, n_paths=2)
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (
            2,
            10,
            10,
        ), f"inverse_mass_matrix.shape={imm.shape}, expected (2, 10, 10)"

    def test_states_position_leading_dim(self) -> None:
        """states.position should have leading dim num_chains=2."""
        states, _ = _run(203, num_chains=2, n_paths=2)
        pos = states.position
        leaves = jax.tree.leaves(pos)
        assert leaves, "states.position is empty"
        leading_dim = leaves[0].shape[0]
        assert (
            leading_dim == 2
        ), f"states.position leading dim={leading_dim}, expected 2"


# ---------------------------------------------------------------------------
# 5. Sidecar: _multipathfinder_psis_pareto_k present and finite-or-nan
# ---------------------------------------------------------------------------


class TestSidecar:
    """PSIS Pareto-k sidecar key is present and well-formed."""

    def test_pareto_k_present(self) -> None:
        _, params = _run(301, num_chains=1, n_paths=2)
        assert (
            "_multipathfinder_psis_pareto_k" in params
        ), f"params keys: {list(params.keys())}"

    def test_pareto_k_is_scalar(self) -> None:
        _, params = _run(302, num_chains=1, n_paths=2)
        k = jnp.asarray(params["_multipathfinder_psis_pareto_k"])
        assert k.ndim == 0, f"pareto_k should be scalar, got shape {k.shape}"

    def test_pareto_k_finite_or_nan(self) -> None:
        """Pareto-k may be NaN for degenerate fits but must not be inf."""
        _, params = _run(303, num_chains=1, n_paths=2)
        k = jnp.asarray(params["_multipathfinder_psis_pareto_k"])
        assert not bool(jnp.isinf(k)), f"pareto_k={k} is inf (not finite, not nan)"


# ---------------------------------------------------------------------------
# 6. Incompatible base method raises ValueError
# ---------------------------------------------------------------------------


class TestIncompatibleMethod:
    def test_mclmc_raises_value_error(self) -> None:
        key = jax.random.key(999)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        warmup_key = jax.random.fold_in(key, 88)
        with pytest.raises(ValueError, match="not compatible"):
            WARMUPS["multipathfinder_window_adaptation"].runner(
                warmup_key,
                init_pos,
                100,
                _MCLMC,
                logdensity_fn=logdensity_fn,
                num_chains=1,
            )
