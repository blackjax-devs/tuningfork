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
"""Tests for the SMCMethod dataclass.

Covers:
1. Valid construction (with all required fields) succeeds; field values
   round-trip correctly.
2. Empty ``name`` raises with the exact message
   ``"SMCMethod: 'name' must be a non-empty string"``.
3. ``family != "smc"`` raises with the right message; only ``"smc"`` is allowed.
4. Empty ``compatible_inner_methods`` raises.
5. ``default_inner_method`` NOT in ``compatible_inner_methods`` raises.
6. Empty ``default_hp_space`` raises (parallel to BaseMethod's check).
7. Default values: ``num_particles_default == 1000``,
   ``step_kwargs_schema == ()``, ``notes == ""``.
"""

import pytest

from tuningfork.inference.base_method._base import HyperparamSpace
from tuningfork.inference.smc._base import SMCMethod

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Minimal helpers for constructing valid instances.
# ---------------------------------------------------------------------------

_MINIMAL_HP = (HyperparamSpace("target_ess", "uniform", low=0.3, high=0.95),)
_MINIMAL_INNER = ("rwm", "mala")


def _make_smc_entry(**overrides: object) -> SMCMethod:
    """Return a valid SMCMethod, applying any overrides."""
    defaults: dict[str, object] = dict(
        name="test_smc_algo",
        family="smc",
        factory=lambda logprior_fn, loglikelihood_fn, **kw: None,
        compatible_inner_methods=_MINIMAL_INNER,
        default_inner_method="rwm",
        default_hp_space=_MINIMAL_HP,
    )
    defaults.update(overrides)
    return SMCMethod(**defaults)  # type: ignore[arg-type]


# ===========================================================================
# 1. Valid construction
# ===========================================================================


class TestSMCMethodConstruction:
    def test_minimal_construction_succeeds(self) -> None:
        entry = _make_smc_entry()
        assert entry.name == "test_smc_algo"
        assert entry.family == "smc"
        assert entry.compatible_inner_methods == _MINIMAL_INNER
        assert entry.default_inner_method == "rwm"

    def test_full_construction_round_trips(self) -> None:
        hp_space = (
            HyperparamSpace("target_ess", "uniform", low=0.3, high=0.95),
            HyperparamSpace("num_mcmc_steps", "int", low=1, high=50),
        )
        entry = _make_smc_entry(
            name="my_smc",
            compatible_inner_methods=("rwm", "nuts", "mala"),
            default_inner_method="nuts",
            num_particles_default=500,
            default_hp_space=hp_space,
            step_kwargs_schema=("data_mask",),
            notes="Test notes.",
        )
        assert entry.name == "my_smc"
        assert entry.family == "smc"
        assert entry.compatible_inner_methods == ("rwm", "nuts", "mala")
        assert entry.default_inner_method == "nuts"
        assert entry.num_particles_default == 500
        assert len(entry.default_hp_space) == 2
        assert entry.step_kwargs_schema == ("data_mask",)
        assert entry.notes == "Test notes."

    def test_factory_is_callable(self) -> None:
        entry = _make_smc_entry()
        assert callable(entry.factory)

    def test_frozen(self) -> None:
        entry = _make_smc_entry()
        with pytest.raises(Exception):  # FrozenInstanceError
            entry.name = "other"  # type: ignore[misc]


# ===========================================================================
# 2. Empty name raises
# ===========================================================================


class TestSMCMethodEmptyName:
    def test_empty_name_raises_exact_message(self) -> None:
        with pytest.raises(
            ValueError, match="SMCMethod: 'name' must be a non-empty string"
        ):
            _make_smc_entry(name="")


# ===========================================================================
# 3. family != "smc" raises
# ===========================================================================


class TestSMCMethodFamilyValidation:
    def test_family_mcmc_raises(self) -> None:
        with pytest.raises(ValueError, match="family must be 'smc', got 'mcmc'"):
            _make_smc_entry(family="mcmc")  # type: ignore[arg-type]

    def test_family_vi_raises(self) -> None:
        with pytest.raises(ValueError, match="family must be 'smc'"):
            _make_smc_entry(family="vi")  # type: ignore[arg-type]

    def test_family_random_string_raises(self) -> None:
        with pytest.raises(ValueError, match="family must be 'smc'"):
            _make_smc_entry(family="nuts")  # type: ignore[arg-type]

    def test_family_smc_succeeds(self) -> None:
        entry = _make_smc_entry(family="smc")
        assert entry.family == "smc"


# ===========================================================================
# 4. Empty compatible_inner_methods raises
# ===========================================================================


class TestSMCMethodCompatibleInnerMethods:
    def test_empty_compatible_inner_methods_raises(self) -> None:
        with pytest.raises(
            ValueError,
            match="'compatible_inner_methods' must be a non-empty tuple",
        ):
            _make_smc_entry(
                compatible_inner_methods=(),
                default_inner_method="rwm",
            )

    def test_single_inner_method_ok(self) -> None:
        entry = _make_smc_entry(
            compatible_inner_methods=("rwm",),
            default_inner_method="rwm",
        )
        assert entry.compatible_inner_methods == ("rwm",)


# ===========================================================================
# 5. default_inner_method not in compatible_inner_methods raises
# ===========================================================================


class TestSMCMethodDefaultInnerMethod:
    def test_default_not_in_compatible_raises(self) -> None:
        with pytest.raises(
            ValueError,
            match="default_inner_method 'hmc' not in compatible_inner_methods",
        ):
            _make_smc_entry(
                compatible_inner_methods=("rwm", "mala"),
                default_inner_method="hmc",
            )

    def test_default_in_compatible_succeeds(self) -> None:
        entry = _make_smc_entry(
            compatible_inner_methods=("rwm", "mala", "nuts"),
            default_inner_method="mala",
        )
        assert entry.default_inner_method == "mala"


# ===========================================================================
# 6. Empty default_hp_space raises
# ===========================================================================


class TestSMCMethodHpSpace:
    def test_empty_hp_space_raises(self) -> None:
        with pytest.raises(
            ValueError,
            match="'default_hp_space' must contain at least one HyperparamSpace entry",
        ):
            _make_smc_entry(default_hp_space=())

    def test_multiple_hp_spaces_ok(self) -> None:
        hp_space = (
            HyperparamSpace("target_ess", "uniform", low=0.3, high=0.95),
            HyperparamSpace("num_mcmc_steps", "int", low=1, high=50),
        )
        entry = _make_smc_entry(default_hp_space=hp_space)
        assert len(entry.default_hp_space) == 2


# ===========================================================================
# 7. Default values
# ===========================================================================


class TestSMCMethodDefaults:
    def test_num_particles_default_is_1000(self) -> None:
        entry = _make_smc_entry()
        assert entry.num_particles_default == 1000

    def test_step_kwargs_schema_default_is_empty_tuple(self) -> None:
        entry = _make_smc_entry()
        assert entry.step_kwargs_schema == ()

    def test_notes_default_is_empty_string(self) -> None:
        entry = _make_smc_entry()
        assert entry.notes == ""

    def test_num_particles_default_overrideable(self) -> None:
        entry = _make_smc_entry(num_particles_default=500)
        assert entry.num_particles_default == 500

    def test_step_kwargs_schema_overrideable(self) -> None:
        entry = _make_smc_entry(step_kwargs_schema=("data_mask",))
        assert entry.step_kwargs_schema == ("data_mask",)
