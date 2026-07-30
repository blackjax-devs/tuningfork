"""Structural checks for sampler step-policy constructor wiring."""

import ast

import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.recipes._emit._sampler import emit_sampler

pytestmark = pytest.mark.fast


def _constructor_calls(source: str, name: str) -> list[ast.Call]:
    tree = ast.parse(source)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


@pytest.mark.parametrize("name", ["dynamic_hmc", "dmhmc"])
def test_dynamic_sampler_wires_step_policy_into_kernel_and_reinit(name: str):
    source = emit_sampler(BASE_METHODS[name], {})
    calls = _constructor_calls(source, name)

    assert len(calls) == 2
    assert all(
        any(
            keyword.arg == "integration_steps_fn"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "_integration_steps_fn"
            for keyword in call.keywords
        )
        for call in calls
    )


@pytest.mark.parametrize(
    "name",
    ["nuts", "hmc", "mhmc", "rmhmc", "ghmc", "laplace_dhmc", "laplace_dmhmc"],
)
def test_other_sampler_families_do_not_receive_dynamic_step_policy(name: str):
    source = emit_sampler(BASE_METHODS[name], {})

    assert "integration_steps_fn=_integration_steps_fn" not in source
