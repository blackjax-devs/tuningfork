"""Boundary tests for BaseMethod and HyperparamSpace descriptors."""

from dataclasses import fields

import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.base_method._base import BaseMethod, HyperparamSpace

pytestmark = pytest.mark.fast

_HP = (HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),)


def _entry(**overrides: object) -> BaseMethod:
    values: dict[str, object] = {
        "name": "test_algo",
        "family": "mcmc",
        "grad_count_per_step": lambda info: 1,
        "default_hp_space": _HP,
    }
    values.update(overrides)
    return BaseMethod(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kind", "kwargs"),
    [
        ("loguniform", {"low": 1e-4, "high": 1.0}),
        ("uniform", {"low": 0.0, "high": 1.0}),
        ("int", {"low": 1, "high": 8}),
        ("categorical", {"choices": ("a", "b")}),
    ],
)
def test_hyperparam_space_valid_kinds(kind: str, kwargs: dict[str, object]) -> None:
    space = HyperparamSpace("x", kind, **kwargs)  # type: ignore[arg-type]
    assert space.name == "x"
    assert space.kind == kind


@pytest.mark.parametrize(
    ("kind", "kwargs", "message"),
    [
        ("exponential", {"low": 0.0, "high": 1.0}, "kind must be one of"),
        ("loguniform", {"high": 1.0}, "requires both 'low' and 'high'"),
        ("uniform", {"low": 0.0}, "requires both 'low' and 'high'"),
        ("int", {}, "requires both 'low' and 'high'"),
        ("categorical", {}, "requires a non-empty 'choices'"),
        ("categorical", {"choices": ()}, "requires a non-empty 'choices'"),
    ],
)
def test_hyperparam_space_invalid(
    kind: str, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        HyperparamSpace("x", kind, **kwargs)  # type: ignore[arg-type]


def test_base_method_valid_defaults() -> None:
    entry = _entry()
    assert entry.name == "test_algo"
    assert entry.family == "mcmc"
    assert entry.needs_mass_matrix is False
    assert entry.target_acceptance_rate is None


def test_base_method_surface_is_descriptor_only() -> None:
    assert {field.name for field in fields(BaseMethod)} == {
        "name",
        "family",
        "grad_count_per_step",
        "default_hp_space",
        "needs_mass_matrix",
        "target_acceptance_rate",
        "notes",
        "grad_count_convention",
        "extra_required_kwargs",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", "", "'name' must be a non-empty string"),
        ("family", "nuts", "family must be one of"),
    ],
)
def test_base_method_invalid_identity(field: str, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _entry(**{field: value})


def test_empty_hp_space_requires_specialized_kwargs() -> None:
    with pytest.raises(ValueError, match="'default_hp_space' must contain"):
        _entry(default_hp_space=())
    entry = _entry(default_hp_space=(), extra_required_kwargs=("prior_cov",))
    assert entry.default_hp_space == ()


def test_grad_count_oracle() -> None:
    class Info:
        num_integration_steps = 5

    entry = _entry(grad_count_per_step=lambda info: info.num_integration_steps)
    assert entry.grad_count_per_step(Info()) == 5


def test_registry_specialized_entries_keep_extra_kwargs() -> None:
    assert BASE_METHODS["mgrad_gaussian"].extra_required_kwargs == (
        "prior_cov",
        "prior_mean",
    )
