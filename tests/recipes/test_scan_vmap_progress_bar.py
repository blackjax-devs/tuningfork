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
"""Regression guard for the vmap(scan)-with-progress_bar bug.

Before the run_inference_algorithm(vmapped input) refactor, emit_script
produced a vmap(scan) pattern for sampling.  When progress_bar=True,
blackjax uses ``io_callback`` inside the scan body to update the progress bar.
``io_callback`` is not supported inside ``jax.vmap``, causing the JAX error:

    "IO effect not supported in vmap-of-cond"

The current fix (PR #72):
  - Warmup templates: single-chain warmup (run once, broadcast state).
  - Sampling: run_inference_algorithm(vmapped input) — ONE kernel built with
    shared params; the step function vmaps over chains internally so
    run_inference_algorithm is NOT vmapped.  The progress_bar io_callback
    is at the scan level (inside run_inference_algorithm) and NOT inside vmap.

This file contains one @pytest.mark.e2e regression test: emit a multi-chain
recipe (mvn_10 × nuts × window_adaptation_diag_imm, num_chains=2,
n_warmup=10, n_samples=10) with progress_bar=True (implied by the template),
execute the emitted script via subprocess, and assert:
  - returncode == 0
  - "IO effect not supported" not in stderr
  - "vmap-of-cond" not in stderr
  - "DONE" in stdout

NOTE: e2e emit-execute tests run a minimal 10-sample / minimal-warmup config;
they assert the emitted script executes (structure correct, no vmap/io_callback
errors), NOT inference quality. This keeps the e2e gate fast and memory-safe.
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

pytestmark = pytest.mark.e2e


def _make_mvn10_nuts_wadapt_recipe(n_warmup: int = 10) -> Recipe:
    """Minimal in-memory recipe for mvn_10 × nuts × window_adaptation_diag_imm."""
    from tuningfork.base_method import BASE_METHODS
    from tuningfork.model import MODELS

    recipe = Recipe.from_default_config(MODELS["mvn_10"], BASE_METHODS["nuts"])
    return dataclasses.replace(
        recipe,
        warmup_name="window_adaptation_diag_imm",
        warmup_params={"n_warmup": n_warmup, "num_chains": 2},
        warmups=[
            {
                "name": "window_adaptation_diag_imm",
                "params": {"n_warmup": n_warmup, "num_chains": 2},
            }
        ],
    )


def test_multichain_progress_bar_no_vmap_of_cond_error(tmp_path: Path) -> None:
    """Regression guard: emitted multi-chain script with progress_bar=True runs cleanly.

    Emits a 2-chain recipe (mvn_10 × nuts × window_adaptation_diag_imm) and
    executes it via subprocess.  Asserts that:
    - The subprocess exits with returncode=0 (no Python exception).
    - stderr does NOT contain "IO effect not supported" (vmap-of-cond regression).
    - stderr does NOT contain "vmap-of-cond" (any vmap+io_callback error variant).
    - stdout contains "DONE" (postamble reached).

    This is the CI regression guard for the exact vmap(scan)+progress_bar bug
    that motivated the scan(vmap) refactor.

    Lightweight config: num_samples=10, num_warmup=10 (minimal but sufficient).
    """
    recipe = _make_mvn10_nuts_wadapt_recipe(n_warmup=10)
    # num_chains=2 triggers multi-chain path; num_samples=10, num_warmup=10 for e2e speed.
    script = emit_script(recipe, num_samples=10, num_chains=2, num_warmup=10)
    script_path = tmp_path / "test_pb_regression.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )

    assert "IO effect not supported" not in result.stderr, (
        "Regression: 'IO effect not supported' appeared in stderr — "
        "the vmap(scan) with io_callback bug is back.\n"
        f"stderr:\n{result.stderr[:2000]}"
    )
    assert "vmap-of-cond" not in result.stderr, (
        "Regression: 'vmap-of-cond' appeared in stderr — "
        "io_callback is inside a vmap somewhere.\n"
        f"stderr:\n{result.stderr[:2000]}"
    )
    assert result.returncode == 0, (
        f"Emitted multi-chain progress_bar script exited with code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert (
        "DONE" in result.stdout
    ), f"Expected 'DONE' in stdout.\nstdout:\n{result.stdout}"
