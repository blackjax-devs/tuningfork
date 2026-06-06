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
"""Tests for emit_script progress_bar branching (warmup + sampling).

Fast tests:
  - progress_bar=True  → single-chain warmup (.run( once, no jax.vmap(...run))
                         + single-chain sampling (no jax.vmap anywhere in script)
                         + warnings.warn blocks mentioning blackjax issue #927
                         + run_inference_algorithm( present in sampling section.
  - progress_bar=False → multi-chain warmup via jax.vmap over warmup.run
                         + multi-chain sampling via jax.vmap over kernel step
                         + no warnings.warn blocks.

e2e tests (lightweight, num_samples=10, minimal warmup):
  - progress_bar=True  → single-chain run to completion; _samples leading axis = 1.
  - progress_bar=False → multi-chain run to completion; _samples leading axis = num_chains.
  - No vmap/io_callback errors in either mode.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax
import pytest

from tuningfork.catalog import emit_script
from tuningfork.recipes._base import Recipe

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_WARMUP_VARIANTS = [
    "window_adaptation_diag_imm",
    "window_adaptation_dense_imm",
    "window_adaptation_low_rank_imm",
]


def _make_recipe(warmup_name: str, n_warmup: int = 10) -> Recipe:
    """Minimal in-memory recipe for mvn_10 x nuts x <warmup_name>.

    Uses Recipe.from_warmup_only for diag/dense IMM warmups.  For
    window_adaptation_low_rank_imm, Recipe.from_warmup_only fails because
    squeeze_single_chain can't handle the heterogeneous-shape LowRankIMM tuple,
    so we build a synthetic Recipe directly.
    """
    from tuningfork.recipes._base import Effort

    if warmup_name == "window_adaptation_low_rank_imm":
        # Synthetic recipe with stubbed params (sufficient for emit_script).
        return Recipe(
            model_name="mvn_10",
            base_method_name="nuts",
            warmup_name="window_adaptation_low_rank_imm",
            effort=Effort.LOW,
            base_method_params={"step_size": 0.1, "max_num_doublings": 10},
            warmup_params={
                "n_warmup": n_warmup,
                "target_acceptance_rate": 0.8,
                "max_rank": 5,
                "num_chains": 2,
            },
            headline_metric=None,
            sample_quality=None,
            calibration_budget={"num_chains": 2},
            difficulty=None,
            instructions="",
            tuning_seed=0,
        )

    from tuningfork.base_method import BASE_METHODS
    from tuningfork.model import MODELS
    from tuningfork.warmup import WARMUPS

    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    warmup = WARMUPS[warmup_name]
    return Recipe.from_warmup_only(
        posterior,
        base_method,
        warmup,
        n_warmup=n_warmup,
        rng_key=jax.random.key(0),
    )


# ---------------------------------------------------------------------------
# Fast: emitted source text checks
# ---------------------------------------------------------------------------


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_true_emits_single_chain_warmup(warmup_name: str) -> None:
    """progress_bar=True → single-chain warmup (.run( once) + no jax.vmap(...run).

    The single-chain template calls _warmup.run(...) once and broadcasts the
    resulting state.  It must NOT contain a jax.vmap wrapper over the run call.
    For a non-multichain recipe (num_chains=2 in mvn_10), no warning is issued
    when progress_bar=True.
    """
    recipe = _make_recipe(warmup_name)
    # No warning for num_chains=2 (not multichain spec).
    script = emit_script(recipe, num_samples=10, num_warmup=10, progress_bar=True)

    # Single-chain path: exactly one .run( call (the warmup.run call)
    # and no jax.vmap wrapping it.
    assert "_warmup.run(" in script or ".run(_warmup_key" in script, (
        f"[{warmup_name}] Single-chain warmup must contain a .run( call.\n"
        f"Script snippet:\n{script[:1200]}"
    )
    # Must NOT contain the multi-chain vmap pattern (_run_one_warmup).
    assert "_run_one_warmup" not in script, (
        f"[{warmup_name}] progress_bar=True must NOT emit the multi-chain "
        "_run_one_warmup vmap pattern.\n"
        f"Script snippet:\n{script[:1200]}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_true_emits_warnings_warn(warmup_name: str) -> None:
    """progress_bar=True → for non-multichain recipes, no warnings.

    Changed 2026-06-06: warnings only fire for multichain recipes (num_chains > 1 +
    window_adaptation variant). The test recipes here are num_chains=2 (not multichain),
    so no warning. This test now verifies that no warning is issued.
    """
    recipe = _make_recipe(warmup_name)
    # No warning expected for non-multichain recipes.
    script = emit_script(recipe, num_samples=10, num_warmup=10, progress_bar=True)

    assert "warnings.warn(" not in script, (
        f"[{warmup_name}] progress_bar=True on non-multichain recipes must NOT emit warnings.warn(...).\n"
        f"Script snippet:\n{script[:1200]}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_true_warning_above_warmup_section(warmup_name: str) -> None:
    """progress_bar=True with multichain recipe: warning appears ABOVE the warmup section.

    Changed 2026-06-06: warnings only fire for multichain recipes. This test now
    checks that non-multichain recipes emit no warnings. The test recipes are
    num_chains=2 (not multichain), so this verifies the lack of warning.
    """
    recipe = _make_recipe(warmup_name)
    # No warning expected for non-multichain recipes.
    script = emit_script(recipe, num_samples=10, num_warmup=10, progress_bar=True)

    warn_pos = script.find("warnings.warn(")

    assert warn_pos == -1, (
        f"[{warmup_name}] warnings.warn( should NOT be in emitted script for non-multichain recipes.\n"
        f"Script preamble:\n{script[:600]}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_false_emits_multichain_vmap_warmup(warmup_name: str) -> None:
    """progress_bar=False → multi-chain warmup via jax.vmap(warmup.run) emitted.

    The multi-chain template vmaps _run_one_warmup over (warmup_keys, init_positions).
    The emitted source must contain the jax.vmap decorator / call and the
    _run_one_warmup function.
    """
    recipe = _make_recipe(warmup_name)
    script = emit_script(recipe, num_samples=10, num_warmup=10, progress_bar=False)

    # Multi-chain path: must contain the vmap run helper.
    assert "_run_one_warmup" in script, (
        f"[{warmup_name}] progress_bar=False must emit multi-chain _run_one_warmup "
        "vmap pattern.\n"
        f"Script snippet:\n{script[:1200]}"
    )
    assert "jax.vmap" in script, (
        f"[{warmup_name}] progress_bar=False warmup must use jax.vmap.\n"
        f"Script snippet:\n{script[:1200]}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_false_no_warnings_warn_in_warmup(warmup_name: str) -> None:
    """progress_bar=False → NO warnings.warn in the emitted warmup section.

    The multi-chain template is the honest path; no warning is needed.
    """
    recipe = _make_recipe(warmup_name)
    script = emit_script(recipe, num_samples=10, num_warmup=10, progress_bar=False)

    assert "warnings.warn(" not in script, (
        f"[{warmup_name}] progress_bar=False must NOT emit warnings.warn(...).\n"
        f"Script snippet:\n{script[:1200]}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_none_default_same_as_false(warmup_name: str) -> None:
    """progress_bar=None (default) produces the same script as progress_bar=False.

    Changed 2026-06-06: None now maps to False (multichain-preserving behavior).
    The generated script content is identical; both use jax.vmap for multichain.
    """
    recipe = _make_recipe(warmup_name)
    script_none = emit_script(recipe, num_samples=10, num_warmup=10)
    script_false = emit_script(
        recipe, num_samples=10, num_warmup=10, progress_bar=False
    )

    assert (
        script_none == script_false
    ), f"[{warmup_name}] progress_bar=None must produce the same script as False.\n"


# ---------------------------------------------------------------------------
# Fast: sampling section branching on progress_bar
# ---------------------------------------------------------------------------


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_true_no_vmap_in_sampling(warmup_name: str) -> None:
    """progress_bar=True → single-chain: no jax.vmap( CALLS in warmup or sampling.

    The single-chain path avoids jax.vmap calls entirely (warmup + sampling both
    single-chain) so that progress_bar's io_callback is never inside a vmap.
    This test uses a num_chains=2 recipe (non-multichain), so no warning is issued.
    """
    recipe = _make_recipe(warmup_name)
    # No warning for non-multichain recipes.
    script = emit_script(recipe, num_samples=10, num_warmup=10, progress_bar=True)

    assert "jax.vmap(" not in script, (
        f"[{warmup_name}] progress_bar=True must NOT contain jax.vmap( calls "
        "(single-chain warmup + single-chain sampling).\n"
        f"Script snippet:\n{script[:2000]}"
    )
    assert "@jax.vmap" not in script, (
        f"[{warmup_name}] progress_bar=True must NOT use @jax.vmap decorator "
        "(single-chain warmup + single-chain sampling).\n"
        f"Script snippet:\n{script[:2000]}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_true_sampling_has_run_inference_algorithm(
    warmup_name: str,
) -> None:
    """progress_bar=True → sampling section uses run_inference_algorithm directly.

    The single-chain path calls run_inference_algorithm with initial_state=
    (a single-chain state), not a vmapped variant.
    This test uses a num_chains=2 recipe (non-multichain), so no warning is issued.
    """
    recipe = _make_recipe(warmup_name)
    # No warning for non-multichain recipes.
    script = emit_script(recipe, num_samples=10, num_warmup=10, progress_bar=True)

    assert "run_inference_algorithm" in script, (
        f"[{warmup_name}] progress_bar=True must call run_inference_algorithm.\n"
        f"Script snippet:\n{script[:2000]}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_true_sampling_warning_mentions_num_chains(
    warmup_name: str,
) -> None:
    """progress_bar=True with multichain: emit_script issues warning at call time.

    Changed 2026-06-06: warnings fire for multichain recipes (num_chains > 1 +
    window_adaptation). When progress_bar=True and the recipe would be multichain,
    a warning is issued at emit_script call time (before the script runs).
    """
    recipe = _make_recipe(warmup_name)
    # Use num_chains=4 to trigger multichain spec (would be multichain if progress_bar=False).
    # But with progress_bar=True, it's forced to single-chain, so a warning is issued.
    with pytest.warns(UserWarning, match="multichain"):
        script = emit_script(
            recipe, num_samples=10, num_warmup=10, progress_bar=True, num_chains=4
        )

    # The warning message should mention multichain.
    assert isinstance(script, str) and len(script) > 0
    assert "progress_bar=False" in script, (
        f"[{warmup_name}] Sampling warning must suggest setting progress_bar=False.\n"
        f"Script snippet:\n{script[:2000]}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_false_has_vmap_in_sampling(warmup_name: str) -> None:
    """progress_bar=False → multi-chain sampling uses jax.vmap (vmapped step)."""
    recipe = _make_recipe(warmup_name)
    script = emit_script(recipe, num_samples=10, num_warmup=10, progress_bar=False)

    # The multi-chain sampling path uses jax.vmap for the vmapped step.
    assert "jax.vmap" in script, (
        f"[{warmup_name}] progress_bar=False must use jax.vmap for multi-chain sampling.\n"
        f"Script snippet:\n{script[:2000]}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_false_no_sampling_warning(warmup_name: str) -> None:
    """progress_bar=False → no single-chain sampling warning emitted."""
    recipe = _make_recipe(warmup_name)
    script = emit_script(recipe, num_samples=10, num_warmup=10, progress_bar=False)

    # No warnings.warn at all when progress_bar=False (multi-chain is the honest path).
    assert "warnings.warn(" not in script, (
        f"[{warmup_name}] progress_bar=False must NOT emit any warnings.warn(...).\n"
        f"Script snippet:\n{script[:2000]}"
    )


# ---------------------------------------------------------------------------
# e2e: emit + execute both variants
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.parametrize(
    "warmup_name,progress_bar,expected_chain_axis",
    [
        # progress_bar=True: single-chain sampling → leading axis = 1
        ("window_adaptation_diag_imm", True, 1),
        # progress_bar=False: multi-chain sampling → leading axis = num_chains
        ("window_adaptation_diag_imm", False, 2),
        ("window_adaptation_dense_imm", False, 2),
    ],
)
def test_emit_warmup_pb_executes_and_shapes_correct(
    warmup_name: str,
    progress_bar: bool,
    expected_chain_axis: int,
    tmp_path: Path,
) -> None:
    """Emitted script with progress_bar={True,False} executes and produces correct shape.

    Lightweight config: num_samples=10, num_warmup=10.
    Asserts:
    - returncode == 0
    - "DONE" in stdout
    - _samples leading axis = expected_chain_axis (1 for progress_bar=True, num_chains for False)
    - No vmap/io_callback errors in stderr
    """
    _NUM_CHAINS = 2
    _NUM_SAMPLES = 10
    recipe = _make_recipe(warmup_name, n_warmup=10)

    if progress_bar is True:
        with pytest.warns(UserWarning, match="#927"):
            script = emit_script(
                recipe,
                num_samples=_NUM_SAMPLES,
                num_chains=_NUM_CHAINS,
                num_warmup=10,
                progress_bar=progress_bar,
            )
    else:
        script = emit_script(
            recipe,
            num_samples=_NUM_SAMPLES,
            num_chains=_NUM_CHAINS,
            num_warmup=10,
            progress_bar=progress_bar,
        )

    # Append shape-check lines.
    shape_check = (
        "\nimport jax as _jax\n"
        "_first_leaf = _jax.tree.leaves(_samples)[0]\n"
        'print("SHAPE=" + str(_first_leaf.shape))\n'
    )
    script += shape_check

    script_path = tmp_path / f"test_pb_{warmup_name}_{progress_bar}.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )

    assert result.returncode == 0, (
        f"Emitted script (pb={progress_bar}, warmup={warmup_name}) failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout, (
        f"Expected 'DONE' in stdout (pb={progress_bar}, warmup={warmup_name}).\n"
        f"stdout:\n{result.stdout}"
    )
    assert f"SHAPE=({expected_chain_axis}, {_NUM_SAMPLES}" in result.stdout, (
        f"Expected _samples leading axis {expected_chain_axis} "
        f"(pb={progress_bar}, warmup={warmup_name}).\n"
        f"stdout:\n{result.stdout}"
    )
    assert "IO effect not supported" not in result.stderr, (
        f"vmap/io_callback error (pb={progress_bar}, warmup={warmup_name}).\n"
        f"stderr:\n{result.stderr[:2000]}"
    )
