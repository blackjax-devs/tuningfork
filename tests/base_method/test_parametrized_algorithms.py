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
"""Parametrized tests for base method registry entries.

This module tests the common registry/factory/kernel properties across all
base methods using parametrization instead of per-file duplication.

Coverage (10 algos via parametrization):
  - Registry entry existence and naming
  - Factory callability
  - default_hp_space structure
  - needs_mass_matrix flag
  - target_acceptance_rate
"""

import jax
import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.model import MODELS

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_MVN = MODELS["mvn_10"]
_D = 10  # MVN-10 has 10 dimensions
_SEED = 42
_RNG_KEY = jax.random.key(_SEED)

# List of algorithms that require special kwargs (per BaseMethod.extra_required_kwargs)
# These are skipped from parametrized tests as they need custom factory logic
_ALGOS_WITH_EXTRA_REQUIRED_KWARGS = {
    "mgrad_gaussian": ("prior_cov", "prior_mean"),
    "elliptical_slice": ("prior_cov", "prior_mean"),
    "irmh": ("proposal_distribution",),
    "additive_step_random_walk": ("proposal_generator",),
    "laplace_hmc": ("log_joint_fn", "theta_init"),
    "laplace_dhmc": ("log_joint_fn", "theta_init"),
    "laplace_mhmc": ("log_joint_fn", "theta_init"),
    "laplace_dmhmc": ("log_joint_fn", "theta_init"),
}

# List of algorithms that need per-chain L parameter (skipped from generic tests)
_ALGOS_WITH_PER_CHAIN_L = {"mclmc", "adjusted_mclmc", "adjusted_mclmc_dynamic"}

# List of algorithms to skip from parametrized tests
# (tested separately or via other files)
_SKIP_ALGOS = (
    {"hmc", "nuts", "mala", "barker", "rwm"}
    | _ALGOS_WITH_EXTRA_REQUIRED_KWARGS.keys()
    | _ALGOS_WITH_PER_CHAIN_L
)


def _build_logdensity(posterior_entry, key):
    from tuningfork.model._numpyro import build_logdensity_fn

    init_position, logdensity_fn, _ = build_logdensity_fn(key, posterior_entry)
    return init_position, logdensity_fn


def _get_test_algos():
    """Return list of algorithms to test, excluding those in _SKIP_ALGOS."""
    return [
        algo
        for algo in sorted(BASE_METHODS.keys())
        if algo not in _SKIP_ALGOS and algo not in _ALGOS_WITH_EXTRA_REQUIRED_KWARGS
    ]


# ---------------------------------------------------------------------------
# Parametrized tests: Registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo_name", _get_test_algos())
def test_algorithm_entry_in_registry(algo_name) -> None:
    """Entry exists in BASE_METHODS registry."""
    assert (
        algo_name in BASE_METHODS
    ), f"'{algo_name}' not found in BASE_METHODS; registered: {sorted(BASE_METHODS)}"


@pytest.mark.parametrize("algo_name", _get_test_algos())
def test_algorithm_name_field_matches_key(algo_name) -> None:
    """Name field matches registry key."""
    entry = BASE_METHODS[algo_name]
    assert entry.name == algo_name, f"Expected name='{algo_name}', got {entry.name!r}"


@pytest.mark.parametrize("algo_name", _get_test_algos())
def test_algorithm_family_is_mcmc_or_vi(algo_name) -> None:
    """Family is either 'mcmc' or 'vi'."""
    entry = BASE_METHODS[algo_name]
    assert entry.family in (
        "mcmc",
        "vi",
    ), f"Expected family='mcmc' or 'vi', got {entry.family!r}"


# ---------------------------------------------------------------------------
# Parametrized tests: Factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo_name", _get_test_algos())
def test_algorithm_factory_callable(algo_name) -> None:
    """Factory method is callable."""
    entry = BASE_METHODS[algo_name]
    assert callable(entry.factory), f"{algo_name}.factory must be callable"


@pytest.mark.parametrize("algo_name", _get_test_algos())
def test_algorithm_factory_is_defined(algo_name) -> None:
    """Factory field exists (detailed factory tests in per-algo files)."""
    entry = BASE_METHODS[algo_name]
    assert entry.factory is not None, f"{algo_name}.factory must be defined"
    assert callable(entry.factory), f"{algo_name}.factory must be callable"


# ---------------------------------------------------------------------------
# Parametrized tests: HP space
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo_name", _get_test_algos())
def test_algorithm_has_default_hp_space(algo_name) -> None:
    """default_hp_space is defined."""
    entry = BASE_METHODS[algo_name]
    assert (
        entry.default_hp_space is not None
    ), f"{algo_name}.default_hp_space must be defined"
    assert (
        len(entry.default_hp_space) > 0
    ), f"{algo_name}.default_hp_space must not be empty"


@pytest.mark.parametrize("algo_name", _get_test_algos())
def test_algorithm_hp_space_has_valid_names(algo_name) -> None:
    """Each HP in default_hp_space has a name."""
    entry = BASE_METHODS[algo_name]
    for hp in entry.default_hp_space:
        assert hasattr(hp, "name"), f"{algo_name} HP missing 'name' field"
        assert isinstance(
            hp.name, str
        ), f"{algo_name} HP name must be str, got {type(hp.name)}"


# ---------------------------------------------------------------------------
# Parametrized tests: Flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo_name", _get_test_algos())
def test_algorithm_needs_mass_matrix_is_bool(algo_name) -> None:
    """needs_mass_matrix is a boolean."""
    entry = BASE_METHODS[algo_name]
    assert isinstance(
        entry.needs_mass_matrix, bool
    ), f"{algo_name}.needs_mass_matrix must be bool, got {type(entry.needs_mass_matrix)}"


@pytest.mark.parametrize("algo_name", _get_test_algos())
def test_algorithm_target_acceptance_rate_is_valid(algo_name) -> None:
    """target_acceptance_rate is either None (VI/orbital) or numeric in [0, 1]."""
    entry = BASE_METHODS[algo_name]
    if entry.target_acceptance_rate is not None:
        assert isinstance(
            entry.target_acceptance_rate, (int, float)
        ), f"{algo_name}.target_acceptance_rate must be numeric or None, got {type(entry.target_acceptance_rate)}"
        assert (
            0.0 <= entry.target_acceptance_rate <= 1.0
        ), f"{algo_name}.target_acceptance_rate out of [0, 1]"
