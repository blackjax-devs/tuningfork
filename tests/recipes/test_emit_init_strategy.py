"""Structural tests for initial-position source emission."""

import ast

import jax
import jax.numpy as jnp
import pytest

from tuningfork.recipes._emit._init_strategy import emit_init_strategy
from tuningfork.recipes._init_strategy import apply_init_strategy


def _tree_calls(source: str) -> list[str]:
    tree = ast.parse(source)
    return [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]


@pytest.mark.fast
@pytest.mark.parametrize(
    ("strategy", "batched"),
    [
        (None, False),
        ({"type": "prior_sample"}, False),
        ({"type": "zero"}, False),
        ({"type": "uniform", "low": -1, "high": 1}, False),
        ({"type": "zero_perchain"}, True),
        ({"type": "uniform_perchain", "low": -2, "high": 3}, True),
    ],
)
def test_emits_valid_source_for_all_strategies(strategy, batched):
    source = emit_init_strategy(strategy, num_chains=4)
    ast.parse(source)
    assert "init_position" in source
    assert f"_init_position_is_prebatched = {batched}" in source
    assert "tuningfork" not in source
    if strategy and strategy.get("type") in {
        "uniform",
        "zero_perchain",
        "uniform_perchain",
    }:
        assert "_init_key" in source


@pytest.mark.fast
def test_perchain_emission_uses_explicit_chain_count_and_default_jitter():
    source = emit_init_strategy({"type": "zero_perchain"}, 3)
    assert "num_chains * len(_init_leaves)" in source
    assert "0.5" in source
    assert "normal" in _tree_calls(source)


@pytest.mark.fast
@pytest.mark.parametrize(
    "strategy",
    [
        {"type": "wat"},
        {},
        "zero",
        {"type": "uniform", "low": 1, "high": 1},
        {"type": "uniform_perchain", "low": 0},
        {"type": "zero_perchain", "jitter": -1},
    ],
)
def test_rejects_unknown_or_malformed_strategy(strategy):
    with pytest.raises(ValueError):
        emit_init_strategy(strategy, 2)


@pytest.mark.slow
@pytest.mark.parametrize(
    "strategy",
    [
        None,
        {"type": "prior_sample"},
        {"type": "zero"},
        {"type": "uniform", "low": -1.0, "high": 1.0},
        {"type": "zero_perchain"},
        {"type": "uniform_perchain", "low": -1.0, "high": 1.0},
    ],
)
def test_emitted_source_matches_apply_init_strategy(strategy):
    seed = 17
    num_chains = 3
    init = {"a": jnp.array([1.0, -2.0]), "b": jnp.array(0.5)}
    init_key = jax.random.split(jax.random.key(seed), 3)[0]
    expected = apply_init_strategy(
        strategy or {"type": "prior_sample"},
        init,
        jax.random.fold_in(init_key, 42),
        num_chains,
    )
    namespace = {
        "jax": jax,
        "jnp": jnp,
        "init_position": init,
        "_init_key": init_key,
        "num_chains": num_chains,
    }
    exec(emit_init_strategy(strategy, num_chains), namespace)
    actual = namespace["init_position"]
    assert namespace["_init_position_is_prebatched"] == (
        strategy is not None
        and strategy.get("type") in {"zero_perchain", "uniform_perchain"}
    )
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected)
    ):
        assert actual_leaf.shape == expected_leaf.shape
        assert jnp.array_equal(actual_leaf, expected_leaf)
