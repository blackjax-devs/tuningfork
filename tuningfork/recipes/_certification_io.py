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
"""Persistence helpers for generated certification attempts.

This module owns the parts of certification that only merge or write catalog
artifacts.  Execution and evaluation remain in ``_certification_runner``.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from tuningfork.recipes._base import Recipe
from tuningfork.recipes._certification_intent import CertificationIntent
from tuningfork.recipes._certification_record import import_legacy_current_view
from tuningfork.recipes._execution_plan import canonical_json
from tuningfork.recipes._instructions import render_instructions


def merge_existing_recipe(
    intent: CertificationIntent,
    existing: Recipe | None,
    ground_truth: dict[str, Any] | None,
) -> Recipe:
    """Merge legacy/current fields from an existing recipe into a new intent."""
    if existing is None:
        return intent.recipe
    migrated = import_legacy_current_view(existing, ground_truth=ground_truth)
    return replace(
        intent.recipe,
        attempted_configurations=copy.deepcopy(migrated.attempted_configurations),
        notes=migrated.notes,
        workflow=migrated.workflow,
        difficulty=copy.deepcopy(migrated.difficulty),
        gt_schema_version=migrated.gt_schema_version,
        summary_v2_path=migrated.summary_v2_path,
        _extra_fields=copy.deepcopy(migrated._extra_fields),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_geometry_sidecar(
    recipe: Recipe,
    intent: CertificationIntent,
    catalog_root: Path,
    attempt_id: str,
) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]:
    """Materialize large geometry under an immutable attempt-specific name."""
    params = copy.deepcopy(recipe.base_method_params)
    imm = params.get("inverse_mass_matrix")
    if imm is None or isinstance(imm, str):
        return params, None, None
    is_low_rank = hasattr(imm, "_fields") and "sigma" in getattr(imm, "_fields", ())
    array = None if is_low_rank else np.asarray(imm)
    if not is_low_rank and array is not None and array.size <= 50:
        return params, None, None

    attempt_hash = hashlib.sha256(attempt_id.encode()).hexdigest()[:16]
    suffix = f"attempt_{attempt_hash}"
    sidecar_tag = (
        suffix if intent.filename_tag is None else f"{intent.filename_tag}__{suffix}"
    )
    recipe_dir = catalog_root / recipe.model_name / "recipes"
    target = recipe_dir / (f"{recipe.catalog_stem(filename_tag=sidecar_tag)}.imm.npz")
    if target.exists():
        raise FileExistsError(f"attempt sidecar already exists: {target}")
    relative = recipe.save_imm_sidecar(
        catalog_root,
        imm,
        filename_tag=sidecar_tag,
        model=recipe.model_name,
        seed=recipe.tuning_seed,
        note="Adapted geometry from generated certification attempt",
    )
    if Path(relative) != target.relative_to(catalog_root):
        raise RuntimeError("sidecar writer returned an unexpected path")
    if is_low_rank:
        params.pop("inverse_mass_matrix")
    else:
        params["inverse_mass_matrix"] = "sidecar"
    evidence = {
        "path": relative,
        "sha256": _sha256(target),
        "source": "generated execution telemetry",
    }
    return params, relative, evidence


def persist_recipe_atomically(
    recipe: Recipe,
    intent: CertificationIntent,
    catalog_root: Path,
) -> tuple[Recipe, Path, str | None]:
    """Render and atomically persist a recipe at its intent-resolved path."""
    recipe = replace(recipe, instructions=render_instructions(recipe))
    canonical_json(recipe.to_dict())
    path = recipe.save(
        catalog_root,
        filename_tag=intent.filename_tag,
        imm_sidecar=False,
    )
    if path != intent.recipe_path:
        raise RuntimeError(
            f"recipe writer resolved {path}, expected {intent.recipe_path}"
        )
    return recipe, path, recipe.inverse_mass_matrix_path


__all__ = [
    "merge_existing_recipe",
    "persist_recipe_atomically",
    "prepare_geometry_sidecar",
]
