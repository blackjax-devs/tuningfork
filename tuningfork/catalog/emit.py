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
"""Public recipe emission and evidence-preserving execution helpers.

The implementation lives in ``tuningfork.recipes._emit_script``. This module
re-exports the function for user-facing convenience under
``tuningfork.catalog.emit``.

Example usage::

    from tuningfork.catalog import load_recipe, emit_script, execute_recipe
    from pathlib import Path

    recipe = load_recipe("tuningfork/catalog/eight_schools_ncp/groundtruth.json")
    script = emit_script(recipe, num_samples=500)
    Path("reproduce_eight_schools.py").write_text(script)

    result = execute_recipe(recipe, Path("runs"), num_samples=500)

``emit_script`` is pure: it returns the generated source and performs no I/O.
``execute_recipe`` uses that same source and launches it through the receipt-
preserving runner, so callers receive a typed :class:`LaunchResult` and can
inspect evidence when execution fails with :class:`GeneratedProgramError`.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tuningfork.recipes._emit_script import emit_script
from tuningfork.recipes._launcher import (
    ExecutionTimings,
    GeneratedProgramError,
    LaunchResult,
    launch_generated_program,
)

if TYPE_CHECKING:
    from tuningfork.recipes._base import Recipe


def execute_recipe(
    recipe: Recipe,
    run_root: Path,
    *,
    num_samples: int | None = None,
    sampler_seed: int | None = None,
    reinit_seed: int | None = None,
    num_chains: int | None = None,
    num_warmup: int | list[int] | None = None,
    progress_bar: bool | None = None,
    warmup_num_chains: list[int] | None = None,
    timeout: float | None = None,
    python_executable: str = sys.executable,
    env: Mapping[str, str] | None = None,
    reference_identity: Mapping[str, Any] | None = None,
    diagnostics: bool | None = None,
) -> LaunchResult:
    """Emit a recipe and execute the generated program with a verified receipt.

    The recipe-generation overrides are forwarded to :func:`emit_script`.
    Launcher controls are forwarded to :func:`launch_generated_program`.
    Emission always completes first; any emission error prevents launching.
    ``diagnostics`` optionally overrides the child tap-diagnostics environment.
    """
    if diagnostics is not None:
        if not isinstance(diagnostics, bool):
            raise TypeError("diagnostics must be a bool or None")
        if env is not None and "TUNINGFORK_TAP_DIAGNOSTICS" in env:
            raise ValueError(
                "diagnostics conflicts with explicit TUNINGFORK_TAP_DIAGNOSTICS"
            )

    source = emit_script(
        recipe,
        num_samples=num_samples,
        sampler_seed=sampler_seed,
        reinit_seed=reinit_seed,
        num_chains=num_chains,
        num_warmup=num_warmup,
        progress_bar=progress_bar,
        warmup_num_chains=warmup_num_chains,
    )
    launch_env: Mapping[str, str] | None
    if diagnostics is not None:
        launch_env = dict(env or {})
        launch_env["TUNINGFORK_TAP_DIAGNOSTICS"] = "1" if diagnostics else "0"
    else:
        launch_env = env
    return launch_generated_program(
        source,
        run_root,
        timeout=timeout,
        python_executable=python_executable,
        env=launch_env,
        reference_identity=reference_identity,
    )


__all__ = [
    "ExecutionTimings",
    "GeneratedProgramError",
    "LaunchResult",
    "emit_script",
    "execute_recipe",
]
