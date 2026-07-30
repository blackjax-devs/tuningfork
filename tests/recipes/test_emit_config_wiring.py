"""Integration checks for recipe configuration wiring in emitted scripts."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from tuningfork.catalog import emit_script, load_recipe

pytestmark = pytest.mark.fast

_CATALOG = Path(__file__).resolve().parents[2] / "tuningfork" / "catalog"


def _recipe(relative: str):
    return load_recipe(_CATALOG / relative)


def _calls(source: str, name: str) -> list[ast.Call]:
    tree = ast.parse(source)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


def test_lotka_uniform_perchain_init_is_wired_before_warmup() -> None:
    recipe = _recipe(
        "lotka_volterra/recipes/medium__dmhmc__window_adaptation_dense_imm__"
        "gt_informed_init.json"
    )
    source = emit_script(recipe, num_warmup=1, num_samples=1)

    init = source.index("# === INITIAL POSITION ===")
    warmup = source.index("# === WARMUP:")
    assert source.index("_warmup_t0") < init < warmup
    assert "_init_positions = init_position" in source
    assert "jnp.broadcast_to(x[None]" not in source[init:warmup]


@pytest.mark.parametrize("sampler", ["dynamic_hmc", "dmhmc"])
def test_empirical_policy_is_defined_before_sampler_and_passed_to_both_calls(
    sampler: str,
) -> None:
    filename = (
        "lotka_volterra/recipes/low__dynamic_hmc__window_adaptation_dense_imm.json"
        if sampler == "dynamic_hmc"
        else "lotka_volterra/recipes/medium__dmhmc__window_adaptation_diag_imm__"
        "policy_v2-long.json"
    )
    source = emit_script(_recipe(filename), num_warmup=1, num_samples=1)

    policy = source.index("def _integration_steps_fn")
    sampler_calls = _calls(source, sampler)
    assert len(sampler_calls) == 2
    assert all(
        any(
            kw.arg == "integration_steps_fn"
            and isinstance(kw.value, ast.Name)
            and kw.value.id == "_integration_steps_fn"
            for kw in call.keywords
        )
        for call in sampler_calls
    )
    assert policy < source.index(f"blackjax.{sampler}", policy)


def test_uniform_int_policy_emits_catalog_bounds() -> None:
    recipe = _recipe(
        "lotka_volterra/recipes/medium__dmhmc__window_adaptation_diag_imm__"
        "policy_v2-long.json"
    )
    source = emit_script(recipe, num_warmup=1, num_samples=1)
    policy = source.index("def _integration_steps_fn")
    assert "jax.random.randint(key, (), 50, 200)" in source[policy:]


def test_dynamic_none_policy_emits_default_bounds() -> None:
    recipe = _recipe(
        "german_credit/recipes/low__dmhmc__window_adaptation_dense_imm.json"
    )
    source = emit_script(recipe, num_warmup=1, num_samples=1)
    assert "jax.random.randint(key, (), 1, 10)" in source


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
