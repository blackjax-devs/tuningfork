import pytest

from tuningfork.catalog.inspect import load_recipe
from tuningfork.recipes._emit._diagnostics import (
    emit_diagnostics,
    emit_diagnostics_close,
)
from tuningfork.recipes._emit_script import emit_script

pytestmark = pytest.mark.fast


def test_emit_diagnostics_is_lazy_and_has_cleanup_contract() -> None:
    source = emit_diagnostics(
        {
            "model_name": "mvn_10",
            "base_method_name": "nuts",
            "sampler_seed": 42,
            "max_num_doublings": 7,
        }
    )
    assert "TUNINGFORK_TAP_DIAGNOSTICS" in source
    assert "is_algorithm_tap_compatible" in source
    assert "_tap_atexit.register" in source
    assert "tap_diagnostics_context" in source
    assert emit_diagnostics_close() == "_tap_stack.close()"


def test_emitted_recipe_wraps_warmup_and_closes_after_sampling_sync() -> None:
    recipe = load_recipe(
        "tuningfork/catalog/mvn_10/recipes/"
        "low__nuts__window_adaptation_diag_imm.json"
    )
    source = emit_script(recipe, num_samples=2, num_warmup=2, num_chains=2)

    setup = source.index("_tap_stack = _tap_contextlib.ExitStack()")
    warmup = source.index("# === WARMUP:")
    sync = source.index("jax.block_until_ready((_samples, _infos))")
    close = source.index("_tap_stack.close()", sync)
    persist = source.index("# Persist draws as .npz")

    assert setup < warmup < sync < close < persist
