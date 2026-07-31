"""Integration checks for recipe configuration wiring in emitted scripts."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.catalog import emit_script, load_recipe
from tuningfork.recipes._emit._sampler import emit_sampler

pytestmark = pytest.mark.fast

_CATALOG = Path(__file__).resolve().parents[2] / "tuningfork" / "catalog"


def _recipe(relative: str):
    return load_recipe(_CATALOG / relative)


def _calls(source: str, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    ]


def _first_arg_names(calls: list[ast.Call]) -> set[str]:
    return {
        argument.id
        for call in calls
        if call.args and isinstance((argument := call.args[0]), ast.Name)
    }


@pytest.mark.parametrize("name", ["dynamic_hmc", "dmhmc"])
def test_dynamic_sampler_constructor_and_reinit_receive_step_policy(name: str) -> None:
    calls = [
        call
        for call in _calls(emit_sampler(BASE_METHODS[name], {}), name)
        if any(keyword.arg == "integration_steps_fn" for keyword in call.keywords)
    ]
    assert len(calls) >= 2
    assert all(
        any(
            keyword.arg == "integration_steps_fn"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "_integration_steps_fn"
            for keyword in call.keywords
        )
        for call in calls
    )


def test_uniform_perchain_catalog_recipe_keeps_batched_init_and_reinit() -> None:
    source = emit_script(
        _recipe(
            "lotka_volterra/recipes/medium__dmhmc__window_adaptation_dense_imm__"
            "gt_informed_init.json"
        ),
        num_warmup=1,
        num_samples=1,
    )
    assignments = {
        target.id: node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and target.id
        in {"_init_position_is_prebatched", "_init_positions", "_warmup_is_perchain"}
    }
    assert isinstance(assignments["_init_position_is_prebatched"], ast.Constant)
    assert assignments["_init_position_is_prebatched"].value is True
    assert isinstance(assignments["_init_positions"], ast.Name)
    assert assignments["_init_positions"].id == "init_position"
    warmup_is_perchain = assignments["_warmup_is_perchain"]
    assert isinstance(warmup_is_perchain, ast.Constant)
    assert warmup_is_perchain.value is True
    assert "ss" in _first_arg_names(_calls(source, "_state_reinit"))


@pytest.mark.parametrize("warmup_num_chains, perchain", [(None, True), ([1], False)])
def test_laplace_catalog_recipe_reinit_matches_warmup_topology(
    warmup_num_chains: list[int] | None, perchain: bool
) -> None:
    recipe = _recipe(
        "eight_schools_ncp/recipes/low__laplace_dmhmc__window_adaptation_diag_imm.json"
    )
    source = emit_script(
        recipe,
        num_samples=2,
        num_chains=4,
        warmup_num_chains=warmup_num_chains,
    )
    assignments = {
        target.id: node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and target.id
        in {"_warmup_is_perchain", "_batched_step_size", "_shared_step_size"}
    }
    warmup_is_perchain = assignments["_warmup_is_perchain"]
    assert isinstance(warmup_is_perchain, ast.Constant)
    assert warmup_is_perchain.value is perchain
    assert ("_batched_step_size" in assignments) is perchain
    assert ("_shared_step_size" in assignments) is not perchain
    reinit_first_args = _first_arg_names(_calls(source, "_state_reinit"))
    expected, absent = (
        ("ss", "_shared_step_size") if perchain else ("_shared_step_size", "ss")
    )
    assert expected in reinit_first_args
    assert absent not in reinit_first_args


def test_resolver_rejects_perchain_init_with_single_warmup_chain() -> None:
    recipe = _recipe(
        "lotka_volterra/recipes/medium__dmhmc__window_adaptation_dense_imm__"
        "gt_informed_init.json"
    )
    with pytest.raises(ValueError, match="uniform_perchain.*W=S"):
        emit_script(recipe, warmup_num_chains=[1], num_warmup=1, num_samples=1)


def test_resolver_rejects_step_policy_on_non_dynamic_sampler() -> None:
    recipe = _recipe(
        "lotka_volterra/recipes/low__hmc__window_adaptation_diag_imm__inner_nuts.json"
    )
    invalid = dataclasses.replace(
        recipe,
        step_policy={"kind": "uniform_int", "low": 1, "high": 10},
    )
    with pytest.raises(ValueError, match="only executable for dynamic_hmc and dmhmc"):
        emit_script(invalid, num_warmup=1, num_samples=1)
