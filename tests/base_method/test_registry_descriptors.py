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
"""Tests for T2.3 registry descriptors on BaseMethod entries.

Verifies that every entry in BASE_METHODS has explicitly-set descriptor
values (per_chain_param_keys, reinit_state, extra_kwarg_builder) rather
than relying on defaults.  This test exists to catch new entries that
forget to set descriptor values.

Expected descriptor values are specified here so that changes to any
entry's descriptors require a deliberate update to this file.
"""

import pytest

from tuningfork.base_method import BASE_METHODS

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Ground-truth table: algorithm_name -> (per_chain_param_keys, reinit_state,
#                                         extra_kwarg_builder_is_none)
# ---------------------------------------------------------------------------
# per_chain_param_keys variants:
#   HMC_KEYS = ("step_size", "inverse_mass_matrix")
#   MCLMC_KEYS = ("step_size", "inverse_mass_matrix", "L")
#   GF_KEYS = ()  — gradient-free / no adapted params
#
# reinit_state: True for kernels whose state type differs from warmup output.
# extra_kwarg_builder: None for all current entries (model-specific logic
#   cannot be portably captured as a descriptor builder at this time).

HMC_KEYS = ("step_size", "inverse_mass_matrix")
MCLMC_KEYS = ("step_size", "inverse_mass_matrix", "L")
GF_KEYS = ()  # gradient-free / no adapted params

_EXPECTED: dict[str, tuple[tuple[str, ...], bool, bool]] = {
    # name -> (per_chain_param_keys, reinit_state, extra_kwarg_builder_is_none)
    "hmc": (HMC_KEYS, False, True),
    "mhmc": (HMC_KEYS, False, True),
    "nuts": (HMC_KEYS, False, True),
    "mala": (HMC_KEYS, False, True),
    "barker": (HMC_KEYS, False, True),
    "dynamic_hmc": (HMC_KEYS, True, True),  # needs DynamicHMCState
    "dmhmc": (HMC_KEYS, True, True),  # needs DynamicHMCState
    "ghmc": (HMC_KEYS, False, True),  # GHMCState OK from MEADS (no reinit in runner)
    "orbital_hmc": (HMC_KEYS, False, True),
    "rmhmc": (HMC_KEYS, False, True),
    "meanfield_vi": (GF_KEYS, False, True),  # no_warmup → no ss/imm
    "fullrank_vi": (GF_KEYS, False, True),  # no_warmup → no ss/imm
    "mclmc": (MCLMC_KEYS, False, True),  # L per-chain from warmup
    "adjusted_mclmc": (MCLMC_KEYS, False, True),  # L per-chain from warmup
    "adjusted_mclmc_dynamic": (MCLMC_KEYS, True, True),  # needs DynamicHMCState
    "laplace_hmc": (HMC_KEYS, True, True),  # needs LaplaceHMCState
    "laplace_dhmc": (HMC_KEYS, True, True),  # needs LaplaceDynamicHMCState
    "laplace_mhmc": (HMC_KEYS, True, True),  # needs LaplaceHMCState
    "laplace_dmhmc": (HMC_KEYS, True, True),  # needs LaplaceDynamicHMCState
    "elliptical_slice": (GF_KEYS, False, True),  # no adapted params
    "mgrad_gaussian": (GF_KEYS, False, True),  # no adapted ss/imm
    "irmh": (GF_KEYS, False, True),  # no adapted params
    "additive_step_random_walk": (GF_KEYS, False, True),  # no adapted params
    "rwm": (GF_KEYS, False, True),  # uses sigma not step_size
}


def test_all_entries_have_expected_descriptors() -> None:
    """Every entry in BASE_METHODS must be in the expected table."""
    missing = set(BASE_METHODS.keys()) - set(_EXPECTED.keys())
    extra = set(_EXPECTED.keys()) - set(BASE_METHODS.keys())
    assert not missing, (
        f"Entries in BASE_METHODS missing from _EXPECTED descriptor table: {sorted(missing)}. "
        "Add them to test_registry_descriptors.py with explicit descriptor values."
    )
    assert not extra, (
        f"Entries in _EXPECTED table not in BASE_METHODS: {sorted(extra)}. "
        "Remove stale rows from _EXPECTED."
    )


@pytest.mark.parametrize("name,expected", list(_EXPECTED.items()))
def test_entry_descriptors(name: str, expected: tuple) -> None:
    """Verify per_chain_param_keys, reinit_state, extra_kwarg_builder for each entry."""
    exp_keys, exp_reinit, exp_builder_none = expected
    entry = BASE_METHODS[name]
    assert (
        entry.per_chain_param_keys == exp_keys
    ), f"{name}.per_chain_param_keys: expected {exp_keys!r}, got {entry.per_chain_param_keys!r}"
    assert (
        entry.reinit_state == exp_reinit
    ), f"{name}.reinit_state: expected {exp_reinit}, got {entry.reinit_state}"
    assert (entry.extra_kwarg_builder is None) == exp_builder_none, (
        f"{name}.extra_kwarg_builder: expected is_none={exp_builder_none}, "
        f"got {entry.extra_kwarg_builder!r}"
    )


# imm_kwarg_name: single source of truth for the factory-kwarg / batched_params
# key that carries the adapted mass-matrix-like parameter. ghmc is the sole
# exception (blackjax.ghmc calls it momentum_inverse_scale); every other entry
# must keep the default "inverse_mass_matrix". A HARD-KEEP guard: silently
# reverting ghmc's override (or adding a new mismatched kernel without setting
# this field) would reintroduce the TypeError this field was added to fix.
_IMM_KWARG_NAME_OVERRIDES: dict[str, str] = {"ghmc": "momentum_inverse_scale"}


@pytest.mark.parametrize("name", sorted(BASE_METHODS.keys()))
def test_imm_kwarg_name_matches_expected(name: str) -> None:
    """Every entry's imm_kwarg_name is the default, except ghmc's override."""
    entry = BASE_METHODS[name]
    expected = _IMM_KWARG_NAME_OVERRIDES.get(name, "inverse_mass_matrix")
    assert (
        entry.imm_kwarg_name == expected
    ), f"{name}.imm_kwarg_name: expected {expected!r}, got {entry.imm_kwarg_name!r}"
