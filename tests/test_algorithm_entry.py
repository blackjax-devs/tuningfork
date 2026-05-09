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
"""Tests for BaseMethod and HyperparamSpace dataclasses.

Covers:
- Construction with defaults and full fields.
- __post_init__ validator failures for BaseMethod (empty name, bad
  family, empty default_hp_space).
- __post_init__ validator failures for HyperparamSpace (bad kind,
  missing low/high for numeric kinds, missing choices for categorical).
- Smoke test: build an entry with a trivial grad_count_per_step and
  confirm the callable is usable.
"""

import pytest

from bjx_bench.inference.base_method._base import BaseMethod, HyperparamSpace

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Minimal helpers for constructing valid instances.
# ---------------------------------------------------------------------------

_MINIMAL_HP = (HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),)


def _make_entry(**overrides: object) -> BaseMethod:
    """Return a valid BaseMethod, applying any overrides."""
    defaults: dict[str, object] = dict(
        name="test_algo",
        family="mcmc",
        factory=lambda logdensity_fn, **kw: None,
        grad_count_per_step=lambda info: 1,
        default_hp_space=_MINIMAL_HP,
    )
    defaults.update(overrides)
    return BaseMethod(**defaults)  # type: ignore[arg-type]


# ===========================================================================
# HyperparamSpace tests
# ===========================================================================


class TestHyperparamSpaceConstruction:
    def test_loguniform(self) -> None:
        hp = HyperparamSpace("step_size", "loguniform", low=1e-4, high=1.0)
        assert hp.name == "step_size"
        assert hp.kind == "loguniform"
        assert hp.low == 1e-4
        assert hp.high == 1.0
        assert hp.choices is None

    def test_uniform(self) -> None:
        hp = HyperparamSpace("alpha", "uniform", low=0.0, high=1.0)
        assert hp.kind == "uniform"

    def test_int(self) -> None:
        hp = HyperparamSpace("num_leapfrog", "int", low=1, high=128)
        assert hp.kind == "int"

    def test_categorical(self) -> None:
        hp = HyperparamSpace("integrator", "categorical", choices=("verlet", "yoshida"))
        assert hp.choices == ("verlet", "yoshida")
        assert hp.low is None
        assert hp.high is None

    def test_frozen(self) -> None:
        hp = HyperparamSpace("step_size", "uniform", low=0.0, high=1.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            hp.name = "other"  # type: ignore[misc]


class TestHyperparamSpaceValidation:
    def test_bad_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="kind must be one of"):
            HyperparamSpace("x", "exponential", low=0.0, high=1.0)  # type: ignore[arg-type]

    def test_loguniform_missing_low_raises(self) -> None:
        with pytest.raises(ValueError, match="requires both 'low' and 'high'"):
            HyperparamSpace("x", "loguniform", high=1.0)

    def test_uniform_missing_high_raises(self) -> None:
        with pytest.raises(ValueError, match="requires both 'low' and 'high'"):
            HyperparamSpace("x", "uniform", low=0.0)

    def test_int_missing_both_raises(self) -> None:
        with pytest.raises(ValueError, match="requires both 'low' and 'high'"):
            HyperparamSpace("x", "int")

    def test_categorical_missing_choices_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a non-empty 'choices'"):
            HyperparamSpace("x", "categorical")

    def test_categorical_empty_choices_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a non-empty 'choices'"):
            HyperparamSpace("x", "categorical", choices=())


# ===========================================================================
# BaseMethod tests
# ===========================================================================


class TestBaseMethodConstruction:
    def test_minimal_construction(self) -> None:
        entry = _make_entry()
        assert entry.name == "test_algo"
        assert entry.family == "mcmc"
        assert not entry.needs_mass_matrix
        assert entry.target_acceptance_rate is None
        assert entry.notes == ""

    def test_full_construction(self) -> None:
        entry = _make_entry(
            name="hmc",
            family="mcmc",
            needs_mass_matrix=True,
            target_acceptance_rate=0.65,
            notes="Beskos et al. optimal accept ≈ 0.65.",
            default_hp_space=(
                HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
                HyperparamSpace("num_integration_steps", "int", low=1, high=128),
            ),
        )
        assert entry.name == "hmc"
        assert entry.needs_mass_matrix
        assert entry.target_acceptance_rate == 0.65
        assert len(entry.default_hp_space) == 2

    def test_vi_family(self) -> None:
        entry = _make_entry(name="vi_algo", family="vi")
        assert entry.family == "vi"

    def test_smc_family(self) -> None:
        entry = _make_entry(name="smc_algo", family="smc")
        assert entry.family == "smc"

    def test_frozen(self) -> None:
        entry = _make_entry()
        with pytest.raises(Exception):  # FrozenInstanceError
            entry.name = "other"  # type: ignore[misc]

    def test_multiple_hp_spaces(self) -> None:
        hp_space = (
            HyperparamSpace("step_size", "loguniform", low=1e-4, high=1.0),
            HyperparamSpace("n_steps", "int", low=1, high=64),
            HyperparamSpace("integrator", "categorical", choices=("verlet",)),
        )
        entry = _make_entry(default_hp_space=hp_space)
        assert len(entry.default_hp_space) == 3


class TestBaseMethodValidation:
    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="'name' must be a non-empty string"):
            _make_entry(name="")

    def test_bad_family_raises(self) -> None:
        with pytest.raises(ValueError, match="family must be one of"):
            _make_entry(family="nuts")  # type: ignore[arg-type]

    def test_empty_hp_space_raises(self) -> None:
        with pytest.raises(ValueError, match="'default_hp_space' must contain"):
            _make_entry(default_hp_space=())


class TestBaseMethodSmoke:
    def test_grad_count_callable(self) -> None:
        """Trivial smoke: grad_count_per_step must be callable and return a value."""

        class FakeInfo:
            num_integration_steps = 5

        entry = _make_entry(grad_count_per_step=lambda info: info.num_integration_steps)
        result = entry.grad_count_per_step(FakeInfo())
        assert result == 5

    def test_constant_one_grad(self) -> None:
        entry = _make_entry(grad_count_per_step=lambda info: 1)
        assert entry.grad_count_per_step(object()) == 1

    def test_zero_grad_rwm(self) -> None:
        entry = _make_entry(name="rwm", grad_count_per_step=lambda info: 0)
        assert entry.grad_count_per_step(object()) == 0
