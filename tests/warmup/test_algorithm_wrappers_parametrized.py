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
"""Parametrized tests for algorithm wrapper compatibility across 8 algorithms.

Consolidates the identical Registry/HyperparamSpace pattern from
test_algorithm_wrappers.py (NUTS/HMC), test_algorithm_wrappers_lite.py
(MALA/Barker/RWM), and test_algorithm_wrappers_mclmc.py (MCLMC).

Special cases (Barker inverse_mass_matrix=None, MCLMC rng_key threading) are
preserved separately in the per-file unique-invariant tests.
"""

import jax
import jax.numpy as jnp
import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.base_method._base import BaseMethod, HyperparamSpace

pytestmark = pytest.mark.slow

# Algorithms to test via parametrization (the common pattern)
_ALGORITHMS_TO_TEST = ["nuts", "hmc", "mala", "barker", "rwm", "mclmc"]

_DIM = 10
_LOGDENSITY_FN = lambda x: -0.5 * jnp.sum(x["x"] ** 2)
_POSITION = {"x": jnp.zeros(_DIM)}
_IMM = jnp.ones(_DIM)


# ===========================================================================
# Parametrized Registry tests
# ===========================================================================


@pytest.mark.parametrize("algo_name", _ALGORITHMS_TO_TEST)
def test_algorithm_registered(algo_name: str) -> None:
    """Algorithm is registered in BASE_METHODS."""
    assert (
        algo_name in BASE_METHODS
    ), f"BASE_METHODS must contain '{algo_name}'"


@pytest.mark.parametrize("algo_name", _ALGORITHMS_TO_TEST)
def test_algorithm_is_entry(algo_name: str) -> None:
    """Registered entry is a BaseMethod instance."""
    assert isinstance(BASE_METHODS[algo_name], BaseMethod)


@pytest.mark.parametrize("algo_name", _ALGORITHMS_TO_TEST)
def test_algorithm_family_mcmc(algo_name: str) -> None:
    """Family is mcmc."""
    assert BASE_METHODS[algo_name].family == "mcmc"


# ===========================================================================
# Parametrized HyperparamSpace tests
# ===========================================================================


@pytest.mark.parametrize("algo_name", _ALGORITHMS_TO_TEST)
def test_default_hp_space_non_empty(algo_name: str) -> None:
    """default_hp_space is non-empty."""
    entry = BASE_METHODS[algo_name]
    assert len(entry.default_hp_space) >= 1


@pytest.mark.parametrize("algo_name", _ALGORITHMS_TO_TEST)
def test_all_hp_are_hyperparam_space(algo_name: str) -> None:
    """All HPs in default_hp_space are HyperparamSpace instances."""
    for hp in BASE_METHODS[algo_name].default_hp_space:
        assert isinstance(hp, HyperparamSpace)


@pytest.mark.parametrize("algo_name", _ALGORITHMS_TO_TEST)
def test_hp_bounds_consistent(algo_name: str) -> None:
    """HP bounds are consistent (low < high for numeric, choices non-empty for categorical)."""
    for hp in BASE_METHODS[algo_name].default_hp_space:
        if hp.kind in ("loguniform", "uniform", "int"):
            assert hp.low is not None
            assert hp.high is not None
            assert hp.low < hp.high
        elif hp.kind == "categorical":
            assert hp.choices is not None and len(hp.choices) > 0


@pytest.mark.parametrize("algo_name", _ALGORITHMS_TO_TEST)
def test_step_size_hp_present(algo_name: str) -> None:
    """step_size is in default_hp_space (common across all)."""
    names = [hp.name for hp in BASE_METHODS[algo_name].default_hp_space]
    assert "step_size" in names


# ===========================================================================
# Special unique cases (not dropped; preserved from original files)
# ===========================================================================


class TestNutsHmcSpecialCases:
    """Unique to NUTS/HMC: num_integration_steps HP."""

    def test_nuts_has_only_step_size_hp(self) -> None:
        """NUTS has only step_size (trajectory length is dynamic)."""
        names = [hp.name for hp in BASE_METHODS["nuts"].default_hp_space]
        assert names == ["step_size"]

    def test_hmc_has_step_size_and_num_steps(self) -> None:
        """HMC has step_size and num_integration_steps."""
        names = [hp.name for hp in BASE_METHODS["hmc"].default_hp_space]
        assert "step_size" in names
        assert "num_integration_steps" in names


class TestBarkerSpecialCases:
    """Unique to Barker: inverse_mass_matrix=None option."""

    def test_barker_inverse_mass_matrix_optional(self) -> None:
        """Barker accepts inverse_mass_matrix=None (identity default)."""
        entry = BASE_METHODS["barker"]
        algo = entry.factory(
            _LOGDENSITY_FN,
            step_size=0.1,
            inverse_mass_matrix=None,
        )
        assert hasattr(algo, "init")
        assert hasattr(algo, "step")


class TestMclmcSpecialCases:
    """Unique to MCLMC: rng_key threading at init."""

    def test_mclmc_rng_key_required_at_init(self) -> None:
        """MCLMC.init requires rng_key for momentum initialization."""
        entry = BASE_METHODS["mclmc"]
        algo = entry.factory(_LOGDENSITY_FN, step_size=0.1, L=1.0)
        key = jax.random.key(0)
        state = algo.init(_POSITION, rng_key=key)
        assert hasattr(state, "position")

    def test_mclmc_has_l_hp(self) -> None:
        """MCLMC has step_size and L HPs."""
        names = [hp.name for hp in BASE_METHODS["mclmc"].default_hp_space]
        assert "step_size" in names
        assert "L" in names
