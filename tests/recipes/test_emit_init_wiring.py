"""Structural tests for emitted initial-key and warmup-position wiring."""

import ast

import pytest

from tuningfork.recipes._emit._preamble import emit_preamble
from tuningfork.recipes._emit._warmup import emit_warmup


pytestmark = pytest.mark.fast


def _preamble_ctx() -> dict[str, object]:
    return {
        "recipe_id": "test-recipe",
        "model_name": "mvn_10",
        "base_method_name": "nuts",
        "warmup_name": "window_adaptation_diag_imm",
        "recipe_hash": "abc123",
        "effort": "low",
        "verdict": "green",
        "x64_config_line": "",
        "tuning_seed": 17,
        "num_chains": 2,
    }


def _warmup_ctx(*, prebatched: bool) -> dict[str, object]:
    return {
        "target_acceptance_rate": 0.8,
        "n_warmup": 10,
        "tuning_seed": 17,
        "warmup_algorithm": "blackjax.nuts",
        "warmup_extra_kwargs": "",
        "window_adaptation_fn": "blackjax.window_adaptation",
        "window_adaptation_extra_kwargs": "",
        "warmup_progress_bar": False,
        "init_position_is_prebatched": prebatched,
    }


def test_preamble_binds_and_reuses_init_key() -> None:
    source = emit_preamble(_preamble_ctx())
    ast.parse(source)
    assert "_init_key = jax.random.key(17)" in source
    assert "    _init_key, posterior" in source
    assert "jax.random.key(17), posterior" not in source


@pytest.mark.parametrize("prebatched", [True, False])
def test_multichain_warmup_resolves_initial_position_topology(prebatched: bool) -> None:
    source = emit_warmup(
        "window_adaptation_diag_imm",
        base_method=object(),  # descriptor is unused by this warmup family
        ctx={**_warmup_ctx(prebatched=prebatched), "_warmup_is_multichain": True},
    )
    ast.parse(source)
    if prebatched:
        assert "_init_positions = init_position" in source
        assert "jnp.broadcast_to(x[None]" not in source
    else:
        assert "jnp.broadcast_to(x[None]" in source
        assert "_init_positions = init_position" not in source
    assert "if init_position_is_prebatched" not in source
