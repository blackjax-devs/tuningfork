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

``emit_script`` dispatches typed MCMC and SMC recipes, returns generated source,
and performs no I/O. ``execute_recipe`` uses that same source and launches it
through the receipt-preserving runner, so callers receive a typed
:class:`LaunchResult` and can inspect evidence when execution fails with
:class:`GeneratedProgramError`.
"""

from __future__ import annotations

import copy
import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tuningfork.recipes._base import Recipe
from tuningfork.recipes._base_smc import SMCRecipe
from tuningfork.recipes._emit_script import emit_script as _emit_mcmc_script
from tuningfork.recipes._emit_smc_script import emit_smc_script
from tuningfork.recipes._execution_plan import canonical_json
from tuningfork.recipes._launcher import (
    ExecutionTimings,
    GeneratedProgramError,
    LaunchResult,
    launch_generated_program,
)

RECIPE_EVIDENCE_KEY = "tuningfork_recipe_evidence"
RECIPE_EVIDENCE_SCHEMA = "tuningfork.recipe-evidence.v1"
RECIPE_EVIDENCE_HASH_DOMAIN = RECIPE_EVIDENCE_SCHEMA + "\0"
_NONFINITE_TAG = "\u0000tuningfork_recipe_evidence_nonfinite_float"


def emit_script(
    recipe: Recipe | SMCRecipe,
    *,
    tuning_seed: int | None = None,
    num_samples: int | None = None,
    sampler_seed: int | None = None,
    reinit_seed: int | None = None,
    num_chains: int | None = None,
    num_warmup: int | list[int] | None = None,
    progress_bar: bool | None = None,
    warmup_num_chains: list[int] | None = None,
) -> str:
    """Emit the standalone program selected by a typed recipe."""
    if isinstance(recipe, SMCRecipe):
        if any(
            value is not None
            for value in (
                tuning_seed,
                num_samples,
                sampler_seed,
                reinit_seed,
                num_chains,
                num_warmup,
                progress_bar,
                warmup_num_chains,
            )
        ):
            raise TypeError("MCMC-only emission overrides are not valid for SMCRecipe")
        return emit_smc_script(recipe)
    if not isinstance(recipe, Recipe):
        raise TypeError("recipe must be a Recipe or SMCRecipe")
    return _emit_mcmc_script(
        recipe,
        tuning_seed=tuning_seed,
        num_samples=num_samples,
        sampler_seed=sampler_seed,
        reinit_seed=reinit_seed,
        num_chains=num_chains,
        num_warmup=num_warmup,
        progress_bar=progress_bar,
        warmup_num_chains=warmup_num_chains,
    )


def canonical_recipe_snapshot(recipe: Recipe | SMCRecipe) -> dict[str, Any]:
    """Return the lossless, JSON-safe recipe identity used in receipts.

    Receipt production and certification binding must compare the same
    materialized representation.  In particular, this normalizes namedtuple
    and tuple values to JSON arrays and tags non-finite floats before hashing.
    """
    if isinstance(recipe, SMCRecipe):
        snapshot = recipe.to_dict()
    elif isinstance(recipe, Recipe):
        snapshot = recipe.to_dict(include_legacy_warmup_fields=True)
    else:
        raise TypeError("recipe must be a Recipe or SMCRecipe")
    if not isinstance(snapshot, Mapping):
        raise TypeError("recipe.to_dict() must return a mapping")
    return _receipt_snapshot_value(copy.deepcopy(dict(snapshot)))


def _receipt_snapshot_value(value: Any) -> Any:
    """Convert non-finite floats to strict-JSON, tagged values recursively."""
    if isinstance(value, float):
        if value != value:
            return {_NONFINITE_TAG: "nan"}
        if value == float("inf"):
            return {_NONFINITE_TAG: "+inf"}
        if value == float("-inf"):
            return {_NONFINITE_TAG: "-inf"}
        return value
    if isinstance(value, Mapping):
        return {key: _receipt_snapshot_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_receipt_snapshot_value(item) for item in value]
    return value


def _recipe_reference_identity(
    recipe: Recipe | SMCRecipe,
    caller_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the receipt provenance envelope for a recipe execution.

    The snapshot uses the complete family-specific schema, including MCMC
    legacy warmup keys. This keeps the receipt self-contained across schema
    migrations, failed attempts, and unknown extension fields. A caller's
    identity is nested so generated evidence cannot overwrite it.
    """
    if caller_identity is not None and not isinstance(caller_identity, Mapping):
        raise GeneratedProgramError("reference_identity must be a mapping or None")
    if caller_identity is not None and RECIPE_EVIDENCE_KEY in caller_identity:
        raise ValueError(f"reference_identity key {RECIPE_EVIDENCE_KEY!r} is reserved")
    snapshot = canonical_recipe_snapshot(recipe)
    canonical_snapshot = canonical_json(snapshot)
    snapshot_hash = hashlib.sha256(
        (RECIPE_EVIDENCE_HASH_DOMAIN + canonical_snapshot).encode("utf-8")
    ).hexdigest()
    envelope: dict[str, Any] = {
        "schema": RECIPE_EVIDENCE_SCHEMA,
        "snapshot": snapshot,
        "snapshot_sha256": snapshot_hash,
    }
    if caller_identity is not None:
        try:
            canonical_json(caller_identity)
        except (TypeError, ValueError) as exc:
            raise GeneratedProgramError(f"invalid reference_identity: {exc}") from exc
        envelope["caller_reference_identity"] = copy.deepcopy(dict(caller_identity))
    return {RECIPE_EVIDENCE_KEY: envelope}


def execute_recipe(
    recipe: Recipe | SMCRecipe,
    run_root: Path,
    *,
    tuning_seed: int | None = None,
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

    if isinstance(recipe, SMCRecipe) and diagnostics is not None:
        raise TypeError("MCMC-only diagnostics are not valid for SMCRecipe")
    source = emit_script(
        recipe,
        tuning_seed=tuning_seed,
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
    recipe_identity = _recipe_reference_identity(recipe, reference_identity)
    return launch_generated_program(
        source,
        run_root,
        timeout=timeout,
        python_executable=python_executable,
        env=launch_env,
        reference_identity=recipe_identity,
    )


__all__ = [
    "ExecutionTimings",
    "GeneratedProgramError",
    "LaunchResult",
    "emit_script",
    "emit_smc_script",
    "execute_recipe",
    "canonical_recipe_snapshot",
    "RECIPE_EVIDENCE_HASH_DOMAIN",
    "RECIPE_EVIDENCE_KEY",
    "RECIPE_EVIDENCE_SCHEMA",
]
