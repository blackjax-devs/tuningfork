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
"""Tests for emit_script progress_bar wrapping (warmup + sampling).

Changed 2026-07-08 (stage 2, blackjax #964): blackjax.progress_bar() is
vmap-safe, so ``progress_bar`` no longer selects TOPOLOGY at all -- warmup and
sampling are unconditionally multi-chain (jax.vmap) regardless of the flag.
The only knob that still selects single-chain topology is the independent
``warmup_num_chains=[1]`` override (see test_warmup_num_chains_schema.py),
which is unrelated to progress bars.

Fast tests:
  - progress_bar=True  → still multi-chain warmup + sampling (jax.vmap present,
                         _run_one_warmup present, _warmup_is_perchain = True)
                         PLUS `with blackjax.progress_bar():` wraps around the
                         warmup and sampling run calls.
  - progress_bar=False → multi-chain warmup + sampling, NO progress_bar() wrap.
  - Neither case emits a warnings.warn() block (no forcing to warn about).

e2e tests (lightweight, num_samples=10, minimal warmup):
  - Both progress_bar=True and progress_bar=False run to completion with
    _samples leading axis = num_chains (topology is unaffected by the flag).
  - No vmap/io_callback errors in either mode.
"""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

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

    Uses a declarative default recipe with the requested warmup identity; no
    sampling is needed for these emitted-source assertions.
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

    recipe = Recipe.from_default_config(MODELS["mvn_10"], BASE_METHODS["nuts"])
    return dataclasses.replace(
        recipe,
        warmup_name=warmup_name,
        warmup_params={"n_warmup": n_warmup, "num_chains": 2},
        warmups=[
            {
                "name": warmup_name,
                "params": {"n_warmup": n_warmup, "num_chains": 2},
            }
        ],
    )


# ---------------------------------------------------------------------------
# Fast: emitted source text checks -- warmup section
# ---------------------------------------------------------------------------


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_true_still_emits_multichain_warmup(warmup_name: str) -> None:
    """progress_bar=True must NOT change warmup topology -- still multi-chain.

    No warning is issued: progress_bar no longer forces single-chain, so there
    is nothing to warn about even for a multichain recipe.
    """
    recipe = _make_recipe(warmup_name)
    script = emit_script(
        recipe, num_samples=10, num_warmup=10, progress_bar=True, num_chains=2
    )

    assert "_run_one_warmup" in script, (
        f"[{warmup_name}] progress_bar=True must still emit the multi-chain "
        "_run_one_warmup vmap pattern (topology is unaffected by the flag).\n"
        f"Script snippet:\n{script[:1200]}"
    )
    assert "_warmup_is_perchain = True" in script, (
        f"[{warmup_name}] progress_bar=True must still mark warmup as per-chain.\n"
        f"Script snippet:\n{script[:1200]}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_true_wraps_warmup_run_call(warmup_name: str) -> None:
    """progress_bar=True wraps the warmup run call in blackjax.progress_bar()."""
    recipe = _make_recipe(warmup_name)
    script = emit_script(
        recipe, num_samples=10, num_warmup=10, progress_bar=True, num_chains=2
    )

    assert 'with blackjax.progress_bar(label="warmup"):' in script, (
        f"[{warmup_name}] progress_bar=True must wrap the warmup run call in "
        "blackjax.progress_bar().\n"
        f"Script snippet:\n{script[:1500]}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_true_no_warnings_warn(warmup_name: str) -> None:
    """progress_bar=True never emits (nor triggers) a warnings.warn() call.

    Unlike stage 1, progress_bar no longer forces single-chain topology, so
    there is nothing to warn about -- neither in the emitted script nor at
    emit_script() call time. pytest.ini's filterwarnings=error means an
    unexpected UserWarning here would already fail the test outright; no
    explicit pytest.warns(...) context is needed.
    """
    recipe = _make_recipe(warmup_name)
    script = emit_script(
        recipe, num_samples=10, num_warmup=10, progress_bar=True, num_chains=2
    )
    assert "warnings.warn(" not in script, (
        f"[{warmup_name}] progress_bar=True must NOT emit warnings.warn() in "
        f"the script.\nScript snippet:\n{script[:1200]}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_false_emits_multichain_vmap_warmup_no_wrap(
    warmup_name: str,
) -> None:
    """progress_bar=False → multi-chain warmup, no blackjax.progress_bar() wrap."""
    recipe = _make_recipe(warmup_name)
    script = emit_script(recipe, num_samples=10, num_warmup=10, progress_bar=False)

    assert "_run_one_warmup" in script, (
        f"[{warmup_name}] progress_bar=False must emit multi-chain _run_one_warmup "
        "vmap pattern.\n"
        f"Script snippet:\n{script[:1200]}"
    )
    assert "jax.vmap" in script, (
        f"[{warmup_name}] progress_bar=False warmup must use jax.vmap.\n"
        f"Script snippet:\n{script[:1200]}"
    )
    assert "blackjax.progress_bar(" not in script, (
        f"[{warmup_name}] progress_bar=False must NOT wrap the warmup call in "
        "blackjax.progress_bar().\n"
        f"Script snippet:\n{script[:1200]}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_false_no_warnings_warn_in_warmup(warmup_name: str) -> None:
    """progress_bar=False → NO warnings.warn in the emitted warmup section."""
    recipe = _make_recipe(warmup_name)
    script = emit_script(recipe, num_samples=10, num_warmup=10, progress_bar=False)

    assert "warnings.warn(" not in script, (
        f"[{warmup_name}] progress_bar=False must NOT emit warnings.warn(...).\n"
        f"Script snippet:\n{script[:1200]}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_none_default_same_as_false(warmup_name: str) -> None:
    """progress_bar=None (default) produces the same script as progress_bar=False."""
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
def test_progress_bar_true_still_has_vmap_in_sampling(warmup_name: str) -> None:
    """progress_bar=True must NOT remove jax.vmap from the sampling section.

    Sampling topology is unconditionally multi-chain regardless of the flag.
    """
    recipe = _make_recipe(warmup_name)
    script = emit_script(
        recipe, num_samples=10, num_warmup=10, progress_bar=True, num_chains=2
    )

    assert "jax.vmap" in script, (
        f"[{warmup_name}] progress_bar=True must still use jax.vmap for "
        "multi-chain sampling.\n"
        f"Script snippet:\n{script[:2000]}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_true_wraps_sampling_run_call(warmup_name: str) -> None:
    """progress_bar=True wraps the sampling run_inference_algorithm call."""
    recipe = _make_recipe(warmup_name)
    script = emit_script(
        recipe, num_samples=10, num_warmup=10, progress_bar=True, num_chains=2
    )

    assert "run_inference_algorithm" in script, (
        f"[{warmup_name}] progress_bar=True must call run_inference_algorithm.\n"
        f"Script snippet:\n{script[:2000]}"
    )
    assert 'with blackjax.progress_bar(label="sampling"):' in script, (
        f"[{warmup_name}] progress_bar=True must wrap the sampling call in "
        "blackjax.progress_bar().\n"
        f"Script snippet:\n{script[:2000]}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_false_has_vmap_in_sampling_no_wrap(warmup_name: str) -> None:
    """progress_bar=False → multi-chain sampling uses jax.vmap, no wrap."""
    recipe = _make_recipe(warmup_name)
    script = emit_script(recipe, num_samples=10, num_warmup=10, progress_bar=False)

    assert "jax.vmap" in script, (
        f"[{warmup_name}] progress_bar=False must use jax.vmap for multi-chain sampling.\n"
        f"Script snippet:\n{script[:2000]}"
    )
    assert "blackjax.progress_bar(" not in script, (
        f"[{warmup_name}] progress_bar=False must NOT wrap the sampling call in "
        "blackjax.progress_bar().\n"
        f"Script snippet:\n{script[:2000]}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("warmup_name", _WARMUP_VARIANTS)
def test_progress_bar_false_no_sampling_warning(warmup_name: str) -> None:
    """progress_bar=False → no single-chain sampling warning emitted."""
    recipe = _make_recipe(warmup_name)
    script = emit_script(recipe, num_samples=10, num_warmup=10, progress_bar=False)

    # No warnings.warn at all when progress_bar=False.
    assert "warnings.warn(" not in script, (
        f"[{warmup_name}] progress_bar=False must NOT emit any warnings.warn(...).\n"
        f"Script snippet:\n{script[:2000]}"
    )


# ---------------------------------------------------------------------------
# e2e: emit + execute both variants
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.parametrize(
    "warmup_name,progress_bar",
    [
        # Both progress_bar=True and False must produce the SAME (multichain)
        # leading axis -- topology is unaffected by the flag (stage 2).
        ("window_adaptation_diag_imm", True),
        ("window_adaptation_diag_imm", False),
        ("window_adaptation_dense_imm", False),
    ],
)
def test_emit_warmup_pb_executes_and_shapes_correct(
    warmup_name: str,
    progress_bar: bool,
    tmp_path: Path,
) -> None:
    """Emitted script with progress_bar={True,False} executes and produces correct shape.

    Lightweight config: num_samples=10, num_warmup=10.
    Asserts:
    - returncode == 0
    - "DONE" in stdout
    - _samples leading axis = num_chains regardless of progress_bar (topology
      is unaffected by the flag as of stage 2).
    - No vmap/io_callback errors in stderr (progress_bar() is vmap-safe).
    """
    _NUM_CHAINS = 2
    _NUM_SAMPLES = 10
    recipe = _make_recipe(warmup_name, n_warmup=10)

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
    assert f"SHAPE=({_NUM_CHAINS}, {_NUM_SAMPLES}" in result.stdout, (
        f"Expected _samples leading axis {_NUM_CHAINS} (multichain, unaffected "
        f"by progress_bar) (pb={progress_bar}, warmup={warmup_name}).\n"
        f"stdout:\n{result.stdout}"
    )
    assert "IO effect not supported" not in result.stderr, (
        f"vmap/io_callback error (pb={progress_bar}, warmup={warmup_name}).\n"
        f"stderr:\n{result.stderr[:2000]}"
    )
