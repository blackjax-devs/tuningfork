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
"""Parametrized tests for SMC method entry-field correctness.

Consolidates the identical ENTRY field pattern from:
- test_adaptive_tempered.py, test_tempered.py, test_partial_posteriors.py,
  test_inner_kernel_tuning.py, test_persistent_sampling.py,
  test_adaptive_persistent_sampling.py

Covers shared assertions:
  1. ENTRY field correctness (name, family, default_inner_method, num_particles_default, HP space)
  2. _COMPATIBLE_INNER exclusions (mclmc family always excluded)
  3. HP space structure and bounds

Per-method end-to-end factory tests are preserved in the individual files.
"""

import pytest

from tuningfork.smc import SMC_METHODS

pytestmark = pytest.mark.fast

# All 6 SMC methods to test
_SMC_METHODS = [
    "adaptive_tempered_smc",
    "tempered_smc",
    "partial_posteriors_smc",
    "inner_kernel_tuning",
    "persistent_sampling_smc",
    "adaptive_persistent_sampling_smc",
]


# ===========================================================================
# Parametrized ENTRY field tests
# ===========================================================================


@pytest.mark.parametrize("smc_name", _SMC_METHODS)
def test_smc_method_registered(smc_name: str) -> None:
    """SMC method is registered in SMC_METHODS."""
    assert (
        smc_name in SMC_METHODS
    ), f"SMC_METHODS must contain '{smc_name}'; registered: {sorted(SMC_METHODS)}"


@pytest.mark.parametrize("smc_name", _SMC_METHODS)
def test_smc_method_name_correct(smc_name: str) -> None:
    """ENTRY.name matches registry key."""
    entry = SMC_METHODS[smc_name]
    assert entry.name == smc_name, f"Expected name={smc_name}, got {entry.name!r}"


@pytest.mark.parametrize("smc_name", _SMC_METHODS)
def test_smc_method_family_is_smc(smc_name: str) -> None:
    """ENTRY.family is 'smc'."""
    entry = SMC_METHODS[smc_name]
    assert entry.family == "smc", f"Expected family='smc', got {entry.family!r}"


@pytest.mark.parametrize("smc_name", _SMC_METHODS)
def test_smc_method_has_default_inner_method(smc_name: str) -> None:
    """ENTRY.default_inner_method is defined."""
    entry = SMC_METHODS[smc_name]
    assert (
        entry.default_inner_method is not None
    ), f"{smc_name}.default_inner_method must be defined"
    assert isinstance(
        entry.default_inner_method, str
    ), f"{smc_name}.default_inner_method must be str"


@pytest.mark.parametrize("smc_name", _SMC_METHODS)
def test_smc_method_num_particles_default_positive(smc_name: str) -> None:
    """ENTRY.num_particles_default is a positive integer."""
    entry = SMC_METHODS[smc_name]
    assert isinstance(
        entry.num_particles_default, int
    ), f"{smc_name}.num_particles_default must be int"
    assert (
        entry.num_particles_default > 0
    ), f"{smc_name}.num_particles_default must be positive"


@pytest.mark.parametrize("smc_name", _SMC_METHODS)
def test_smc_method_factory_is_callable(smc_name: str) -> None:
    """ENTRY.factory is callable."""
    entry = SMC_METHODS[smc_name]
    assert callable(entry.factory), f"{smc_name}.factory must be callable"


@pytest.mark.parametrize("smc_name", _SMC_METHODS)
def test_smc_method_hp_space_non_empty(smc_name: str) -> None:
    """ENTRY.default_hp_space is non-empty."""
    entry = SMC_METHODS[smc_name]
    assert (
        len(entry.default_hp_space) > 0
    ), f"{smc_name}.default_hp_space must be non-empty"


@pytest.mark.parametrize("smc_name", _SMC_METHODS)
def test_smc_method_notes_non_empty(smc_name: str) -> None:
    """ENTRY.notes is non-empty."""
    entry = SMC_METHODS[smc_name]
    assert len(entry.notes) > 0, f"{smc_name}.notes must be non-empty"


@pytest.mark.parametrize("smc_name", _SMC_METHODS)
def test_smc_method_compatible_inner_non_empty(smc_name: str) -> None:
    """ENTRY.compatible_inner_methods is non-empty."""
    entry = SMC_METHODS[smc_name]
    assert (
        len(entry.compatible_inner_methods) > 0
    ), f"{smc_name}.compatible_inner_methods must be non-empty"


@pytest.mark.parametrize("smc_name", _SMC_METHODS)
def test_smc_method_default_inner_in_compatible(smc_name: str) -> None:
    """ENTRY.default_inner_method is in compatible_inner_methods."""
    entry = SMC_METHODS[smc_name]
    assert (
        entry.default_inner_method in entry.compatible_inner_methods
    ), f"{smc_name}: default_inner_method={entry.default_inner_method} not in compatible_inner_methods"


# ===========================================================================
# Parametrized _COMPATIBLE_INNER exclusion tests
# ===========================================================================


@pytest.mark.parametrize("smc_name", _SMC_METHODS)
def test_smc_method_mclmc_excluded(smc_name: str) -> None:
    """mclmc is excluded from compatible_inner_methods (microcanonical invariance violated by tempering)."""
    entry = SMC_METHODS[smc_name]
    assert (
        "mclmc" not in entry.compatible_inner_methods
    ), f"{smc_name}: mclmc must be excluded from compatible_inner_methods"


@pytest.mark.parametrize("smc_name", _SMC_METHODS)
def test_smc_method_adjusted_mclmc_excluded(smc_name: str) -> None:
    """adjusted_mclmc is excluded from compatible_inner_methods."""
    entry = SMC_METHODS[smc_name]
    assert (
        "adjusted_mclmc" not in entry.compatible_inner_methods
    ), f"{smc_name}: adjusted_mclmc must be excluded from compatible_inner_methods"


@pytest.mark.parametrize("smc_name", _SMC_METHODS)
def test_smc_method_adjusted_mclmc_dynamic_excluded(smc_name: str) -> None:
    """adjusted_mclmc_dynamic is excluded from compatible_inner_methods."""
    entry = SMC_METHODS[smc_name]
    assert (
        "adjusted_mclmc_dynamic" not in entry.compatible_inner_methods
    ), f"{smc_name}: adjusted_mclmc_dynamic must be excluded from compatible_inner_methods"
