# Copyright 2026- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Fail-closed code-generation contract for every committed recipe."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tuningfork._version import __version__
from tuningfork.catalog import emit_script, load_recipe
from tuningfork.recipes import Recipe
from tuningfork.recipes._base_smc import SMCRecipe
from tuningfork.recipes._execution_manifest import ExecutionManifest
from tuningfork.recipes._execution_plan import ExecutionOverrides
from tuningfork.recipes._resolve_execution_plan import resolve_execution_plan
from tuningfork.recipes._smc_execution_plan import resolve_smc_execution_plan

pytestmark = pytest.mark.fast

_CATALOG_ROOT = Path(__file__).resolve().parents[2] / "tuningfork" / "catalog"
_ALLOWED_TUNINGFORK_IMPORTS = frozenset(
    {
        "tuningfork.model",
        "tuningfork.model._numpyro",
        "tuningfork.diagnostics._tap",
    }
)

# Rules describe each current explicit generation failure; the frozen label set
# prevents a broad or stale glob from silently classifying a new recipe.
_FAILURE_RULES = (
    (
        "*/recipes/low__mclmc_lrd__mclmc_lrd_tuning.json",
        ValueError,
        "No-warmup replay requires pinned inverse_mass_matrix for mclmc",
    ),
    (
        "*/recipes/failed__laplace_*__window_adaptation_low_rank_imm.json",
        ValueError,
        "num_chains must be a positive integer",
    ),
    (
        "gmm_25/recipes/failed__hmc__window_adaptation_low_rank_imm.json",
        ValueError,
        "num_chains must be a positive integer",
    ),
    (
        "gp_regression/recipes/failed__elliptical_slice__no_warmup.json",
        FileNotFoundError,
        "elliptical_slice.py.tmpl",
    ),
    (
        "mvn_10/recipes/failed__elliptical_slice__no_warmup.json",
        FileNotFoundError,
        "elliptical_slice.py.tmpl",
    ),
    (
        "mvn_10/recipes/failed__laplace_hmc__no_warmup.json",
        ValueError,
        "no phi/theta split is registered",
    ),
    (
        "radon/recipes/failed__nuts__fullrank_vi.json",
        KeyError,
        "wp_num_optimization_steps",
    ),
)
_EXPECTED_FAILURE_LABELS = frozenset(
    {
        "banana/recipes/failed__laplace_dhmc__window_adaptation_low_rank_imm.json",
        "banana/recipes/failed__laplace_dmhmc__window_adaptation_low_rank_imm.json",
        "banana/recipes/failed__laplace_hmc__window_adaptation_low_rank_imm.json",
        "banana/recipes/failed__laplace_mhmc__window_adaptation_low_rank_imm.json",
        "german_credit/recipes/"
        "failed__laplace_dhmc__window_adaptation_low_rank_imm.json",
        "german_credit/recipes/"
        "failed__laplace_dmhmc__window_adaptation_low_rank_imm.json",
        "german_credit/recipes/"
        "failed__laplace_hmc__window_adaptation_low_rank_imm.json",
        "german_credit/recipes/"
        "failed__laplace_mhmc__window_adaptation_low_rank_imm.json",
        "german_credit/recipes/low__mclmc_lrd__mclmc_lrd_tuning.json",
        "gmm_25/recipes/failed__hmc__window_adaptation_low_rank_imm.json",
        "gmm_25/recipes/failed__laplace_dhmc__window_adaptation_low_rank_imm.json",
        "gmm_25/recipes/failed__laplace_dmhmc__window_adaptation_low_rank_imm.json",
        "gmm_25/recipes/failed__laplace_hmc__window_adaptation_low_rank_imm.json",
        "gmm_25/recipes/failed__laplace_mhmc__window_adaptation_low_rank_imm.json",
        "gp_regression/recipes/failed__elliptical_slice__no_warmup.json",
        "ill_cond_50/recipes/failed__laplace_dhmc__window_adaptation_low_rank_imm.json",
        "ill_cond_50/recipes/"
        "failed__laplace_dmhmc__window_adaptation_low_rank_imm.json",
        "ill_cond_50/recipes/failed__laplace_hmc__window_adaptation_low_rank_imm.json",
        "ill_cond_50/recipes/failed__laplace_mhmc__window_adaptation_low_rank_imm.json",
        "ill_cond_50/recipes/low__mclmc_lrd__mclmc_lrd_tuning.json",
        "logistic_synthetic/recipes/"
        "failed__laplace_dhmc__window_adaptation_low_rank_imm.json",
        "logistic_synthetic/recipes/"
        "failed__laplace_dmhmc__window_adaptation_low_rank_imm.json",
        "logistic_synthetic/recipes/"
        "failed__laplace_hmc__window_adaptation_low_rank_imm.json",
        "logistic_synthetic/recipes/"
        "failed__laplace_mhmc__window_adaptation_low_rank_imm.json",
        "mvn_10/recipes/failed__elliptical_slice__no_warmup.json",
        "mvn_10/recipes/failed__laplace_dhmc__window_adaptation_low_rank_imm.json",
        "mvn_10/recipes/failed__laplace_dmhmc__window_adaptation_low_rank_imm.json",
        "mvn_10/recipes/failed__laplace_hmc__no_warmup.json",
        "mvn_10/recipes/failed__laplace_hmc__window_adaptation_low_rank_imm.json",
        "mvn_10/recipes/failed__laplace_mhmc__window_adaptation_low_rank_imm.json",
        "neals_funnel/recipes/"
        "failed__laplace_dhmc__window_adaptation_low_rank_imm.json",
        "neals_funnel/recipes/"
        "failed__laplace_dmhmc__window_adaptation_low_rank_imm.json",
        "neals_funnel/recipes/"
        "failed__laplace_hmc__window_adaptation_low_rank_imm.json",
        "neals_funnel/recipes/"
        "failed__laplace_mhmc__window_adaptation_low_rank_imm.json",
        "radon/recipes/failed__nuts__fullrank_vi.json",
    }
)


def _recipe_paths() -> list[Path]:
    recipes = sorted(_CATALOG_ROOT.rglob("recipes/*.json"))
    groundtruth = sorted(_CATALOG_ROOT.rglob("groundtruth.json"))
    return [*groundtruth, *recipes]


def _relative(path: Path) -> str:
    return path.relative_to(_CATALOG_ROOT).as_posix()


def _expected_generation_failure(path: Path) -> tuple[type[Exception], str] | None:
    import fnmatch

    label = _relative(path)
    for pattern, error_type, reason in _FAILURE_RULES:
        if label in _EXPECTED_FAILURE_LABELS and fnmatch.fnmatch(label, pattern):
            return error_type, reason
    return None


def _embedded_manifest(source: str, label: str) -> tuple[str, ExecutionManifest]:
    tree = ast.parse(source, filename=label)
    values = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "EXECUTION_MANIFEST_JSON"
            for target in node.targets
        )
    ]
    assert len(values) == 1, f"{label}: expected one embedded execution manifest"
    encoded = ast.literal_eval(values[0])
    assert isinstance(encoded, str)
    manifest = ExecutionManifest.from_dict(json.loads(encoded))
    assert manifest.to_json() == encoded, f"{label}: manifest JSON is not canonical"
    return encoded, manifest


def _imports(source: str) -> set[str]:
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _expected_manifest(recipe: Recipe | SMCRecipe) -> ExecutionManifest:
    if isinstance(recipe, SMCRecipe):
        return ExecutionManifest.from_plan(
            resolve_smc_execution_plan(recipe),
            generator_version=__version__,
        )
    return ExecutionManifest.from_plan(
        resolve_execution_plan(recipe, ExecutionOverrides()),
        generator_version=__version__,
    )


@pytest.mark.parametrize("recipe_path", _recipe_paths(), ids=_relative)
def test_committed_recipe_is_a_compilable_manifest_bound_program(
    recipe_path: Path,
) -> None:
    label = _relative(recipe_path)
    recipe = load_recipe(recipe_path)
    expected_failure_labels = {
        _relative(path)
        for path in _recipe_paths()
        if _expected_generation_failure(path) is not None
    }
    assert expected_failure_labels == _EXPECTED_FAILURE_LABELS
    expected_failure = _expected_generation_failure(recipe_path)
    if expected_failure is not None:
        error_type, reason = expected_failure
        with pytest.raises(error_type, match=reason):
            emit_script(recipe)
        return
    source = emit_script(recipe)
    compile(source, label, "exec")
    modules = _imports(source)
    forbidden = sorted(
        module
        for module in modules
        if module.startswith("tuningfork") and module not in _ALLOWED_TUNINGFORK_IMPORTS
    )
    assert not forbidden, f"{label}: forbidden tuningfork imports: {forbidden}"
    _, embedded = _embedded_manifest(source, label)
    assert embedded.to_json() == _expected_manifest(recipe).to_json()
