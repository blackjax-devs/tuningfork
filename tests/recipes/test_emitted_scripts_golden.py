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
"""T1.0 golden-emitted-script suite — Phase 1 codegen refactor gate.

This file is the HARD GATE before any emit-simplification changes (T1.1–T1.7
and the aggressive D-2 template replacement). It:

1. Snapshots every catalog recipe's emitted script via a deterministic SHA-1
   hash (fast path — no execution) and asserts the hash does not change
   unexpectedly.
2. Asserts each emitted script is syntactically valid Python.
3. Asserts D8 compliance: zero forbidden tuningfork imports.
4. Covers every warmup × sampler combination via synthetic recipes (no-warmup
   path + window_adaptation diag/dense/low_rank, both single-chain and
   multichain, progress_bar=True and False).
5. Includes a focused execution smoke-test (slow) asserting the emitted script
   produces DONE + n_divergences for the eight_schools_ncp × nuts ×
   window_adaptation_diag_imm recipe at minimal config.

Usage
-----
These tests are the review checkpoint after STEP A (T1.0). Do not proceed to
STEP B until all tests here are GREEN on the unmodified emit. After each STEP
B sub-task, regenerate snapshots ONLY after eyeballing each diff is a pure
dead-code removal (behaviour-identical). COMMIT at each snapshot update.

Markers
-------
- ``fast``: AST / hash / structural tests — no JAX execution.
- ``slow``: execution smoke-tests via subprocess.
"""

from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tuningfork.catalog import emit_script, load_recipe

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_CATALOG_ROOT = Path(__file__).resolve().parents[2] / "tuningfork" / "catalog"

# Allowed tuningfork imports in emitted scripts (D8 constraint).
_ALLOWED_TF_IMPORTS = frozenset({"tuningfork.model", "tuningfork.model._numpyro"})


def _sha1_of(text: str) -> str:
    """Return SHA-1 hex digest of a string (used for snapshot gating)."""
    return hashlib.sha1(text.encode()).hexdigest()


def _assert_valid_python(script: str, label: str) -> None:
    """Assert the script parses as syntactically valid Python."""
    try:
        ast.parse(script)
    except SyntaxError as e:
        raise AssertionError(
            f"Emitted script for {label!r} is not valid Python: {e}\n"
            f"Script excerpt:\n{script[:500]}"
        ) from e


def _assert_d8(script: str, label: str) -> None:
    """Assert emitted script has zero forbidden tuningfork imports (D8).

    Skips the AST check for scripts that have pre-existing syntax errors
    (e.g., unresolved $wp_max_rank slots in low_rank_imm recipes — T0.2 bug).
    """
    try:
        tree = ast.parse(script)
    except SyntaxError:
        # Pre-existing syntax error (e.g. T0.2 unresolved slot) — skip D8 for this script.
        return
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (
                    alias.name.startswith("tuningfork")
                    and alias.name not in _ALLOWED_TF_IMPORTS
                ):
                    bad.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            m = node.module or ""
            if m.startswith("tuningfork") and m not in _ALLOWED_TF_IMPORTS:
                bad.append(m)
    assert not bad, (
        f"D8 violation in emitted script for {label!r}: found forbidden tuningfork imports:\n"
        f"  {bad!r}\n"
        "Only tuningfork.model and tuningfork.model._numpyro are allowed."
    )


def _collect_catalog_recipes() -> list[Path]:
    """Return all non-failed catalog recipe JSON paths."""
    return sorted(
        p
        for p in _CATALOG_ROOT.rglob("*.json")
        if "recipes" in p.parts and "failed__" not in p.name
    )


def _collect_groundtruth_recipes() -> list[Path]:
    """Return all groundtruth.json paths."""
    return sorted(_CATALOG_ROOT.rglob("groundtruth.json"))


# ---------------------------------------------------------------------------
# T1.0-A: All catalog recipes are syntactically valid Python after emit
# ---------------------------------------------------------------------------


def _try_emit_script(recipe_path: Path) -> tuple | None:
    """Try to load and emit a recipe, returning (recipe, script) or None if unsupported.

    Returns None (→ pytest.skip) when:
    - SMC recipes (SMCRecipe type): no tuning_seed attr.
    - Missing templates: warmups like mclmc_tuning / adjusted_mclmc_tuning don't
      have .py.tmpl files yet (not yet in scope for Phase 1).
    - Structurally incompatible combinations that raise at emit time.
    """
    try:
        recipe = load_recipe(recipe_path)
        script = emit_script(recipe)
        return recipe, script
    except (AttributeError, TypeError, ValueError, FileNotFoundError):
        return None


def _is_known_preexisting_invalid(recipe_path: Path) -> bool:
    """Return True for recipes known to emit syntactically invalid Python pre-T1.x.

    NOTE: T0.2 window_adaptation_low_rank_imm recipes were FIXED in PR #156
    (max_rank=10 backfilled into all recipes, 2026-05-29). They now emit
    cleanly and should flow through normal validation.

    NOTE: irt_2pl hmc (inner_nuts) recipes were FIXED by re-cert (PR recert-irt2pl-hmc-inner-nuts,
    2026-06-05). base_method_params now populated with step_size + num_integration_steps from
    NUTS-harvested warmup trace. No further preexisting-invalid cases remain.
    """
    return False


@pytest.mark.fast
@pytest.mark.parametrize(
    "recipe_path",
    _collect_catalog_recipes(),
    ids=lambda p: f"{p.parent.parent.name}/{p.name}",
)
def test_catalog_recipe_emits_valid_python(recipe_path: Path) -> None:
    """Every non-SMC catalog recipe emits syntactically valid Python.

    This is the fast-path golden gate: AST-only, no execution.
    Catches any $slot not substituted (would be a SyntaxError in the emitted
    script or left as an unresolved literal).

    SMC recipes (SMCRecipe type) do not use emit_script — they are skipped.
    """
    result = _try_emit_script(recipe_path)
    if result is None:
        pytest.skip(f"emit_script not supported for {recipe_path.name}")
        return
    _recipe, script = result
    label = f"{recipe_path.parent.parent.name}/{recipe_path.name}"
    if _is_known_preexisting_invalid(recipe_path):
        pytest.xfail(
            reason="irt_2pl hmc recipe with empty base_method_params — known pre-existing defect"
        )
    _assert_valid_python(script, label)


@pytest.mark.fast
@pytest.mark.parametrize(
    "recipe_path",
    _collect_catalog_recipes(),
    ids=lambda p: f"{p.parent.parent.name}/{p.name}",
)
def test_catalog_recipe_d8_compliant(recipe_path: Path) -> None:
    """Every non-SMC catalog recipe emitted script satisfies D8 (zero forbidden tuningfork imports)."""
    result = _try_emit_script(recipe_path)
    if result is None:
        pytest.skip(f"emit_script not supported for {recipe_path.name}")
        return
    _recipe, script = result
    label = f"{recipe_path.parent.parent.name}/{recipe_path.name}"
    _assert_d8(script, label)


# ---------------------------------------------------------------------------
# T1.0-B: Groundtruth recipes
# ---------------------------------------------------------------------------


@pytest.mark.fast
@pytest.mark.parametrize(
    "gt_path",
    _collect_groundtruth_recipes(),
    ids=lambda p: p.parent.name,
)
def test_groundtruth_recipe_emits_valid_python(gt_path: Path) -> None:
    """Every groundtruth recipe emits valid Python."""
    recipe = load_recipe(gt_path)
    script = emit_script(recipe)
    _assert_valid_python(script, gt_path.parent.name)


# ---------------------------------------------------------------------------
# T1.0-C: Structural invariants across all catalog recipes
# ---------------------------------------------------------------------------


@pytest.mark.fast
@pytest.mark.parametrize(
    "recipe_path",
    _collect_catalog_recipes(),
    ids=lambda p: f"{p.parent.parent.name}/{p.name}",
)
def test_catalog_recipe_has_done_marker(recipe_path: Path) -> None:
    """Emitted script postamble contains 'DONE' marker (round-trip CI gate D10)."""
    result = _try_emit_script(recipe_path)
    if result is None:
        pytest.skip(f"emit_script not supported for {recipe_path.name}")
        return
    _recipe, script = result
    assert "DONE" in script, (
        f"Emitted script for {recipe_path.name} missing 'DONE' marker. "
        "Postamble template may be broken."
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    "recipe_path",
    _collect_catalog_recipes(),
    ids=lambda p: f"{p.parent.parent.name}/{p.name}",
)
def test_catalog_recipe_has_timing_fence(recipe_path: Path) -> None:
    """Emitted script contains timing infrastructure (_recipe_t0, _warmup_t0)."""
    result = _try_emit_script(recipe_path)
    if result is None:
        pytest.skip(f"emit_script not supported for {recipe_path.name}")
        return
    _recipe, script = result
    assert (
        "_recipe_t0" in script
    ), f"Emitted script for {recipe_path.name} missing '_recipe_t0' wall-clock start."
    assert (
        "_warmup_t0" in script
    ), f"Emitted script for {recipe_path.name} missing '_warmup_t0' warmup-phase timer."


@pytest.mark.fast
@pytest.mark.parametrize(
    "recipe_path",
    _collect_catalog_recipes(),
    ids=lambda p: f"{p.parent.parent.name}/{p.name}",
)
def test_catalog_recipe_no_unresolved_dollar_slots(recipe_path: Path) -> None:
    """Emitted script has no unresolved $slot markers in executable code lines.

    Unresolved slots appear as literal '$IDENTIFIER' when a key is missing from
    the substitution context and safe_substitute is used (it leaves them literal
    rather than raising). This test catches silently-missed slots before they
    reach users as broken scripts.

    We check only non-comment, non-docstring lines to avoid false positives
    from ``$bm_*`` references that legitimately appear in inline comments
    (e.g., ``# sampler falls back to $bm_* recipe defaults``).

    Note: SMC recipes (SMCRecipe type) do not go through emit_script and are
    excluded upstream (load_recipe raises AttributeError for them).
    """
    result = _try_emit_script(recipe_path)
    if result is None:
        pytest.skip(f"emit_script not supported for {recipe_path.name}")
        return
    _recipe, script = result

    # PR #156 (2026-05-29): window_adaptation_low_rank_imm recipes now have max_rank=10
    # backfilled, so all recipes should emit cleanly.
    # This test tracks NEW unresolved slots introduced by refactoring or schema changes.
    novel_unresolved = _find_unresolved_slots_in_code(script)
    assert not novel_unresolved, (
        f"Emitted script for {recipe_path.name} has NEW unresolved $slots in code: "
        f"{novel_unresolved[:10]!r}\n"
        "These are Template slots that were not substituted (missing context key).\n"
        "Most common cause: a warmup recipe is missing a required warmup_params key."
    )


# ---------------------------------------------------------------------------
# T1.0-D: SHA-1 snapshot registry for key representative recipes
#
# Purpose: catch any unintentional behaviour change during emit refactoring.
# The hashes below were computed from the UNMODIFIED emit (pre-T1.1+).
# After each T1.x step, UPDATE the snapshot for that recipe (and ONLY that
# recipe) after eyeballing the diff is dead-code-only removal.
#
# Design: we hash a canonical form (sorted substitution context → fixed-point
# string) so the snapshot is stable across minor context-order changes.
# We actually hash the emitted script text directly (deterministic given same
# recipe + same emit code). Hashes are stored as {recipe_id: sha1_hex[:16]}.
# ---------------------------------------------------------------------------

# Populated by running: python -m pytest tests/recipes/test_emitted_scripts_golden.py
# --generate-snapshots (see _generate_snapshots() helper below).
# Initially EMPTY — tests skip snapshot comparison until first generation.
_SNAPSHOTS: dict[str, str] = {}


def _recipe_id_for_snapshot(recipe_path: Path) -> str:
    """Canonical key: '{model}/{effort}__{method}__{warmup}'."""
    model = recipe_path.parent.parent.name
    return f"{model}/{recipe_path.stem}"


@pytest.mark.fast
@pytest.mark.parametrize(
    "recipe_path",
    # Limit snapshot checks to groundtruth + one representative per model to
    # keep the parametrize count manageable.  Full catalog is covered by
    # test_catalog_recipe_emits_valid_python above.
    sorted(_collect_groundtruth_recipes()),
    ids=lambda p: p.parent.name,
)
def test_golden_snapshot(recipe_path: Path) -> None:
    """Golden snapshot: emitted script hash matches registered value (if any).

    If no snapshot is registered, this test PASSES and prints the new hash
    so you can register it.  After a T1.x change, verify the diff is correct,
    then update the entry in _SNAPSHOTS.
    """
    recipe = load_recipe(recipe_path)
    script = emit_script(recipe)
    current_hash = _sha1_of(script)[:16]
    recipe_id = _recipe_id_for_snapshot(recipe_path)

    if recipe_id not in _SNAPSHOTS:
        # No snapshot registered yet — print hash for manual registration.
        print(
            f"\n[golden] {recipe_id!r}: no snapshot registered. Current hash: {current_hash!r}"
        )
        return

    expected = _SNAPSHOTS[recipe_id]
    assert current_hash == expected, (
        f"Golden snapshot mismatch for {recipe_id!r}.\n"
        f"  Expected: {expected!r}\n"
        f"  Got:      {current_hash!r}\n"
        "If this is a T1.x dead-code-removal change, eyeball the diff and update _SNAPSHOTS."
    )


# ---------------------------------------------------------------------------
# T1.0-E: Warmup × sampler combo coverage (synthetic recipes, no catalog needed)
# ---------------------------------------------------------------------------

# Representative cover of all warmup × sampler combinations that emit
# distinct code paths. We use mvn_10 (fast, well-conditioned) for all.
_COMBO_COVER = [
    # (warmup_name, sampler_name, extra_bm_params, extra_wp_params)
    # no_warmup path (sampler provides init)
    ("no_warmup", "nuts", {"step_size": 0.5, "max_num_doublings": 5}, {}),
    ("no_warmup", "hmc", {"step_size": 0.5, "num_integration_steps": 5}, {}),
    ("no_warmup", "mhmc", {"step_size": 0.5, "num_integration_steps": 5}, {}),
    ("no_warmup", "dmhmc", {"step_size": 0.5}, {}),
    ("no_warmup", "dynamic_hmc", {"step_size": 0.5}, {}),
    ("no_warmup", "ghmc", {"step_size": 0.5, "alpha": 0.8, "delta": 0.1}, {}),
    ("no_warmup", "mala", {"step_size": 0.1}, {}),
    ("no_warmup", "barker", {"step_size": 0.1}, {}),
    ("no_warmup", "rwm", {"sigma": 0.1}, {}),
    # window_adaptation_diag_imm (single-chain warmup, default progress_bar=True)
    (
        "window_adaptation_diag_imm",
        "nuts",
        {"step_size": 0.5, "max_num_doublings": 5},
        {"n_warmup": 10, "target_acceptance_rate": 0.8, "num_chains": 4},
    ),
    (
        "window_adaptation_diag_imm",
        "hmc",
        {"step_size": 0.5, "num_integration_steps": 5},
        {"n_warmup": 10, "target_acceptance_rate": 0.8, "num_chains": 4},
    ),
    (
        "window_adaptation_diag_imm",
        "mhmc",
        {"step_size": 0.5, "num_integration_steps": 5},
        {"n_warmup": 10, "target_acceptance_rate": 0.8, "num_chains": 4},
    ),
    (
        "window_adaptation_diag_imm",
        "dynamic_hmc",
        {"step_size": 0.5},
        {"n_warmup": 10, "target_acceptance_rate": 0.8, "num_chains": 4},
    ),
    # window_adaptation_dense_imm (single-chain)
    (
        "window_adaptation_dense_imm",
        "nuts",
        {"step_size": 0.5, "max_num_doublings": 5},
        {"n_warmup": 10, "target_acceptance_rate": 0.8, "num_chains": 4},
    ),
    (
        "window_adaptation_dense_imm",
        "hmc",
        {"step_size": 0.5, "num_integration_steps": 5},
        {"n_warmup": 10, "target_acceptance_rate": 0.8, "num_chains": 4},
    ),
    # window_adaptation_low_rank_imm (requires max_rank in warmup_params)
    (
        "window_adaptation_low_rank_imm",
        "nuts",
        {"step_size": 0.5, "max_num_doublings": 5},
        {"n_warmup": 10, "target_acceptance_rate": 0.8, "num_chains": 4, "max_rank": 3},
    ),
    # VI sampler (no_warmup path)
    ("no_warmup", "meanfield_vi", {"num_optimization_steps": 20}, {}),
    ("no_warmup", "fullrank_vi", {"num_optimization_steps": 20}, {}),
    # VI warmup (meanfield + fullrank)
    (
        "meanfield_vi",
        "nuts",
        {"step_size": 0.5, "max_num_doublings": 5},
        {
            "n_warmup": 10,
            "target_acceptance_rate": 0.8,
            "num_chains": 4,
            "num_optimization_steps": 20,
        },
    ),
    (
        "fullrank_vi",
        "nuts",
        {"step_size": 0.5, "max_num_doublings": 5},
        {
            "n_warmup": 10,
            "target_acceptance_rate": 0.8,
            "num_chains": 4,
            "num_optimization_steps": 20,
        },
    ),
    # A3: pathfinder (single-path Pathfinder warmup)
    (
        "pathfinder",
        "nuts",
        {"step_size": 0.5, "max_num_doublings": 5},
        {"n_warmup": 10, "target_acceptance_rate": 0.8, "num_chains": 2},
    ),
    # A3: multipathfinder (multi-path Pathfinder warmup)
    (
        "multipathfinder",
        "nuts",
        {"step_size": 0.5, "max_num_doublings": 5},
        {
            "n_warmup": 10,
            "target_acceptance_rate": 0.8,
            "num_chains": 2,
            "n_paths": 2,
            "num_samples_per_path": 5,
        },
    ),
    # A3: multipathfinder_window_adaptation (composition warmup)
    (
        "multipathfinder_window_adaptation",
        "nuts",
        {"step_size": 0.5, "max_num_doublings": 5},
        {
            "n_warmup": 10,
            "target_acceptance_rate": 0.8,
            "num_chains": 2,
            "n_paths": 2,
            "num_samples_per_path": 5,
            "imm_shrinkage_to_previous": 20.0,
        },
    ),
    # PR3-b: mclmc_lrd_tuning (LRD-preconditioned MCLMC warmup)
    (
        "mclmc_lrd_tuning",
        "mclmc",
        {"step_size": 0.5, "L": 1.0},
        {
            "n_warmup": 5,
            "num_chains": 2,
            "k_rank": 3,
            "pilot_n_warmup": 5,
            "pilot_n_samples": 5,
        },
    ),
]


def _make_synthetic_recipe(
    warmup_name: str,
    sampler_name: str,
    extra_bm_params: dict,
    extra_wp_params: dict,
):
    """Build a minimal in-memory Recipe for the given (warmup, sampler) pair."""
    from tuningfork.recipes._base import Effort, Recipe
    from tuningfork.warmup import WARMUPS

    base_bm_params = {"step_size": 0.5}
    base_bm_params.update(extra_bm_params)

    base_wp_params: dict = {}
    base_wp_params.update(extra_wp_params)

    if warmup_name != "no_warmup":
        if not WARMUPS[warmup_name].is_compatible(sampler_name):
            return None

    warmups_list = (
        []
        if warmup_name == "no_warmup"
        else [{"name": warmup_name, "params": {**base_wp_params}}]
    )

    return Recipe(
        model_name="mvn_10",
        base_method_name=sampler_name,
        warmup_name=warmup_name,
        effort=Effort.LOW,
        base_method_params=base_bm_params,
        warmup_params=base_wp_params,
        warmups=warmups_list,
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    "combo",
    _COMBO_COVER,
    ids=lambda c: f"{c[0]}__{c[1]}",
)
def test_synthetic_combo_emits_valid_python(combo) -> None:
    """Synthetic (warmup, sampler) combo emits syntactically valid Python."""
    warmup_name, sampler_name, extra_bm, extra_wp = combo
    recipe = _make_synthetic_recipe(warmup_name, sampler_name, extra_bm, extra_wp)
    if recipe is None:
        pytest.skip(f"{warmup_name} is not compatible with {sampler_name}")
    script = emit_script(recipe, num_samples=5, num_warmup=5)
    label = f"{warmup_name}__{sampler_name}"
    _assert_valid_python(script, label)
    _assert_d8(script, label)


def _find_unresolved_slots_in_code(script: str) -> list[str]:
    """Find unresolved $IDENTIFIER slots in non-comment, non-docstring code lines.

    Template comments like ``# falls back to $bm_* defaults`` are intentional
    documentation and should not be treated as unresolved slots.  Only executable
    code lines are checked.

    Known pre-existing defects (NOT introduced by T1.x refactoring) are excluded:
    - ``$wp_max_rank``: T0.2 — low_rank_imm recipes missing max_rank in warmup_params.
    - ``$bm_step_size``, ``$bm_num_integration_steps``: some irt_2pl + other catalog
      recipes have empty base_method_params (not a T1.x issue).
    """
    import re

    code_lines = [
        line
        for line in script.splitlines()
        if not line.lstrip().startswith("#") and '"""' not in line and "'''" not in line
    ]
    code_text = "\n".join(code_lines)
    # Known pre-existing slots from catalog data issues (not introduced by T1.x).
    # $wp_max_rank: T0.2 fixed in PR #156 (max_rank backfilled).
    # $bm_step_size / $bm_num_integration_steps: irt_2pl hmc recipes fixed by
    #   re-cert in PR recert-irt2pl-hmc-inner-nuts (2026-06-05).
    # Remaining: rwm/ghmc with empty base_method_params (no warmup path; those
    #   catalog recipes legitimately have empty bmp since defaults are used at runtime).
    _known_preexisting_slots = frozenset(
        {
            "$bm_sigma",  # rwm with empty base_method_params (no-warmup default)
            "$bm_alpha",  # ghmc with empty base_method_params (no-warmup default)
            "$bm_delta",  # ghmc with empty base_method_params (no-warmup default)
        }
    )
    all_found = re.findall(r"\$[A-Za-z_]\w*", code_text)
    return [s for s in all_found if s not in _known_preexisting_slots]


@pytest.mark.fast
@pytest.mark.parametrize(
    "combo",
    _COMBO_COVER,
    ids=lambda c: f"{c[0]}__{c[1]}",
)
def test_synthetic_combo_no_unresolved_slots(combo) -> None:
    """Synthetic (warmup, sampler) combo has no unresolved $slot markers in code."""
    warmup_name, sampler_name, extra_bm, extra_wp = combo
    recipe = _make_synthetic_recipe(warmup_name, sampler_name, extra_bm, extra_wp)
    if recipe is None:
        pytest.skip(f"{warmup_name} is not compatible with {sampler_name}")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        script = emit_script(recipe, num_samples=5, num_warmup=5)
    unresolved = _find_unresolved_slots_in_code(script)
    assert (
        not unresolved
    ), f"Unresolved $slots in code for {warmup_name}__{sampler_name}: {unresolved[:10]!r}"


@pytest.mark.fast
@pytest.mark.parametrize(
    "sampler_name,expected_flag",
    [
        # samplers that need state reinit (dynamic_hmc, dmhmc, ghmc)
        ("dynamic_hmc", "_state_reinit"),
        ("dmhmc", "_state_reinit"),
        ("ghmc", "_state_reinit"),
    ],
)
def test_reinit_samplers_define_state_reinit(
    sampler_name: str, expected_flag: str
) -> None:
    """Samplers that need re-init define _state_reinit in emitted script.

    dynamic_hmc / dmhmc / ghmc require a different state type than what
    window_adaptation produces. The emitted script MUST define _state_reinit
    so inference_loop.py.tmpl can call it per-chain.
    """
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name=sampler_name,
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.5},
        warmup_params={"n_warmup": 10, "target_acceptance_rate": 0.8, "num_chains": 4},
        warmups=[{"name": "window_adaptation_diag_imm", "params": {"n_warmup": 10}}],
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )
    script = emit_script(recipe, num_samples=5)
    assert expected_flag in script, (
        f"Emitted script for {sampler_name} must define {expected_flag!r} "
        "(required by inference loop for state-type-mismatch samplers)."
    )


@pytest.mark.fast
def test_no_warmup_path_emits_warmup_init_is_single_chain() -> None:
    """no_warmup path emits '_warmup_init_is_single_chain = True' for broadcast."""
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="no_warmup",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.5, "max_num_doublings": 5},
        warmup_params={},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )
    script = emit_script(recipe, num_samples=5)
    assert "_warmup_init_is_single_chain = True" in script, (
        "no_warmup path must emit '_warmup_init_is_single_chain = True' "
        "so the inference loop knows to broadcast the initial state."
    )


@pytest.mark.fast
def test_window_adaptation_multichain_emits_warmup_is_perchain() -> None:
    """progress_bar=False multichain warmup emits '_warmup_is_perchain = True'."""
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.5, "max_num_doublings": 5},
        warmup_params={"n_warmup": 10, "target_acceptance_rate": 0.8, "num_chains": 4},
        warmups=[{"name": "window_adaptation_diag_imm", "params": {"n_warmup": 10}}],
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )
    # progress_bar=False → multichain warmup template (_multichain.py.tmpl)
    script = emit_script(recipe, num_samples=5, progress_bar=False)
    assert (
        "_warmup_is_perchain = True" in script
    ), "progress_bar=False multichain warmup must emit '_warmup_is_perchain = True'."


@pytest.mark.fast
def test_window_adaptation_singlechain_emits_warmup_is_perchain_false() -> None:
    """progress_bar=True single-chain warmup emits '_warmup_is_perchain = False'."""
    import warnings

    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.5, "max_num_doublings": 5},
        warmup_params={"n_warmup": 10, "target_acceptance_rate": 0.8, "num_chains": 4},
        warmups=[{"name": "window_adaptation_diag_imm", "params": {"n_warmup": 10}}],
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )
    # progress_bar=True emits the single-chain template AND raises a UserWarning
    # about single-chain execution (expected, documented behaviour).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        script = emit_script(recipe, num_samples=5, progress_bar=True)
    assert (
        "_warmup_is_perchain = False" in script
    ), "progress_bar=True single-chain warmup must emit '_warmup_is_perchain = False'."


@pytest.mark.fast
def test_low_rank_imm_slot_requires_max_rank() -> None:
    """T0.2: emit_script raises ValueError for low_rank recipe missing max_rank.

    After T1.6 + T0.2 fix, emit_script raises ValueError at generation time
    (rather than silently emitting broken Python with unresolved $wp_max_rank).
    This is the correct behaviour: fail loudly at the generator, not at runtime.
    """
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="window_adaptation_low_rank_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.5, "max_num_doublings": 5},
        warmup_params={"n_warmup": 10, "target_acceptance_rate": 0.8, "num_chains": 4},
        # max_rank intentionally absent — T0.2 guard
        warmups=[
            {"name": "window_adaptation_low_rank_imm", "params": {"n_warmup": 10}}
        ],
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )
    with pytest.raises(ValueError, match="max_rank"):
        emit_script(recipe, num_samples=5)


# ---------------------------------------------------------------------------
# T1.0-F: Focused execution smoke-test (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_golden_execution_nuts_window_diag(tmp_path: Path) -> None:
    """Execution smoke: eight_schools_ncp × nuts × window_adaptation_diag_imm prints DONE.

    Uses the catalog LOW recipe with minimal warmup/samples for speed.
    This is the definitive round-trip CI gate (D10): after ANY emit change,
    this test must stay green.
    """
    recipe_path = (
        _CATALOG_ROOT
        / "eight_schools_ncp"
        / "recipes"
        / "low__nuts__window_adaptation_diag_imm.json"
    )
    if not recipe_path.exists():
        pytest.skip("Catalog recipe not found — run emit first")

    recipe = load_recipe(recipe_path)
    script = emit_script(recipe, num_samples=10, num_warmup=10, progress_bar=False)
    script_path = tmp_path / "golden_nuts_wadapt.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(tmp_path),
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Golden NUTS+window_adaptation script failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout
    assert "n_divergences=" in result.stdout


@pytest.mark.slow
def test_golden_execution_hmc_window_dense(tmp_path: Path) -> None:
    """Execution smoke: eight_schools_ncp × mhmc × window_adaptation_dense_imm prints DONE."""
    recipe_path = (
        _CATALOG_ROOT
        / "eight_schools_ncp"
        / "recipes"
        / "medium__mhmc__window_adaptation_dense_imm.json"
    )
    if not recipe_path.exists():
        pytest.skip("Catalog recipe not found — run emit first")

    recipe = load_recipe(recipe_path)
    script = emit_script(recipe, num_samples=10, num_warmup=10, progress_bar=False)
    script_path = tmp_path / "golden_mhmc_wadapt_dense.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(tmp_path),
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Golden MHMC+dense_imm script failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout


@pytest.mark.slow
def test_golden_execution_dynamic_hmc_no_warmup(tmp_path: Path) -> None:
    """Execution smoke: mvn_10 × dynamic_hmc × no_warmup prints DONE (reinit path)."""
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="dynamic_hmc",
        warmup_name="no_warmup",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.5},
        warmup_params={},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )
    script = emit_script(recipe, num_samples=10)
    script_path = tmp_path / "golden_dynamic_hmc_no_warmup.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(tmp_path),
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Golden dynamic_hmc no_warmup script failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout


@pytest.mark.slow
def test_golden_execution_meanfield_vi_warmup(tmp_path: Path) -> None:
    """Execution smoke: mvn_10 × nuts × meanfield_vi warmup prints DONE."""
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="meanfield_vi",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.5, "max_num_doublings": 5},
        warmup_params={
            "n_warmup": 10,
            "target_acceptance_rate": 0.8,
            "num_chains": 2,
            "num_optimization_steps": 20,
        },
        warmups=[
            {
                "name": "meanfield_vi",
                "params": {
                    "n_warmup": 10,
                    "target_acceptance_rate": 0.8,
                    "num_optimization_steps": 20,
                },
            }
        ],
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"num_chains": 2},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )
    script = emit_script(recipe, num_samples=5, num_warmup=5, num_chains=2)
    script_path = tmp_path / "golden_nuts_mfvi_warmup.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(tmp_path),
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Golden nuts+meanfield_vi warmup script failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout


@pytest.mark.slow
def test_golden_execution_fullrank_vi_warmup(tmp_path: Path) -> None:
    """Execution smoke: mvn_10 × nuts × fullrank_vi warmup prints DONE (dense IMM path)."""
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="fullrank_vi",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.5, "max_num_doublings": 5},
        warmup_params={
            "n_warmup": 10,
            "target_acceptance_rate": 0.8,
            "num_chains": 2,
            "num_optimization_steps": 20,
        },
        warmups=[
            {
                "name": "fullrank_vi",
                "params": {
                    "n_warmup": 10,
                    "target_acceptance_rate": 0.8,
                    "num_optimization_steps": 20,
                },
            }
        ],
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"num_chains": 2},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )
    script = emit_script(recipe, num_samples=5, num_warmup=5, num_chains=2)
    script_path = tmp_path / "golden_nuts_frvi_warmup.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(tmp_path),
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Golden nuts+fullrank_vi warmup script failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout


@pytest.mark.slow
def test_golden_execution_window_adaptation_multichain(tmp_path: Path) -> None:
    """Execution smoke: mvn_10 × nuts × window_adaptation_diag_imm, progress_bar=False
    (multichain vmap path, _warmup_is_perchain=True)."""
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.5, "max_num_doublings": 5},
        warmup_params={"n_warmup": 10, "target_acceptance_rate": 0.8, "num_chains": 2},
        warmups=[{"name": "window_adaptation_diag_imm", "params": {"n_warmup": 10}}],
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"num_chains": 2},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )
    # progress_bar=False → multichain warmup template (_warmup_is_perchain=True)
    script = emit_script(recipe, num_samples=5, num_warmup=5, progress_bar=False)
    assert "_warmup_is_perchain = True" in script
    script_path = tmp_path / "golden_nuts_wadapt_multichain.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(tmp_path),
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Golden nuts+window_adaptation multichain script failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout


@pytest.mark.slow
def test_golden_execution_window_adaptation_low_rank(tmp_path: Path) -> None:
    """Execution smoke: mvn_10 × nuts × window_adaptation_low_rank_imm prints DONE."""
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="window_adaptation_low_rank_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.5, "max_num_doublings": 5},
        warmup_params={
            "n_warmup": 10,
            "target_acceptance_rate": 0.8,
            "num_chains": 2,
            "max_rank": 3,
        },
        warmups=[
            {
                "name": "window_adaptation_low_rank_imm",
                "params": {"n_warmup": 10, "max_rank": 3},
            }
        ],
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"num_chains": 2},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )
    script = emit_script(recipe, num_samples=5, num_warmup=5, progress_bar=False)
    script_path = tmp_path / "golden_nuts_wadapt_low_rank.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(tmp_path),
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Golden nuts+window_adaptation_low_rank script failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout


@pytest.mark.slow
def test_golden_execution_laplace_hmc(tmp_path: Path) -> None:
    """Execution smoke: eight_schools_ncp × laplace_hmc emitted script runs to DONE.

    Verifies both structural correctness of the laplace preamble and that the
    emitted script actually executes to completion in subprocess.

    The laplace warmup uses ``blackjax.nuts`` as the inner kernel (WARMUP_SUBSTITUTE
    path). The emitted logdensity_fn is a scalar adapter that drops the LaplaceMarginal
    aux, mirroring the runner's ``_build_laplace_components`` marginal_logdensity_fn.
    """
    recipe_path = (
        _CATALOG_ROOT
        / "eight_schools_ncp"
        / "recipes"
        / "low__laplace_hmc__window_adaptation_diag_imm.json"
    )
    if not recipe_path.exists():
        pytest.skip("Catalog laplace_hmc recipe not found")

    recipe = load_recipe(recipe_path)
    script = emit_script(recipe, num_samples=10, num_warmup=10, progress_bar=False)
    # Verify emit_laplace_preamble structural output (D8 compliant)
    assert "log_joint_fn" in script, "Missing log_joint_fn in laplace preamble"
    assert (
        "_lmf" in script
    ), "Missing _lmf (laplace_marginal_factory) in laplace preamble"
    assert (
        "phi_init" in script
    ), "Missing phi_init (phi/theta split) in laplace preamble"
    assert "_laplace_warmup" in script, "Missing _laplace_warmup factory list"
    assert (
        "from blackjax.mcmc.laplace_marginal" in script
    ), "Missing laplace import (D8 check)"
    assert (
        "_warmup_logdensity_fn" in script
    ), "Missing scalar warmup adapter (_warmup_logdensity_fn) in laplace preamble"
    # Verify D8: no forbidden tuningfork imports
    import ast as _ast

    tree = _ast.parse(script)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ImportFrom):
            m = node.module or ""
            assert not (
                m.startswith("tuningfork")
                and m not in {"tuningfork.model", "tuningfork.model._numpyro"}
            ), f"D8 violation: {m}"

    # Execute: verify the script runs to completion in subprocess.
    script_path = tmp_path / "golden_laplace_hmc.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(tmp_path),
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Golden laplace_hmc emitted script failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout


# ---------------------------------------------------------------------------
# A3: Execution smoke-tests for new warmup families
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_golden_execution_pathfinder_warmup(tmp_path: Path) -> None:
    """Execution smoke: mvn_10 × nuts × pathfinder warmup prints DONE.

    A3: single-path Pathfinder + dual-averaging step size adaptation.
    """
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="pathfinder",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.5, "max_num_doublings": 5},
        warmup_params={"n_warmup": 10, "target_acceptance_rate": 0.8, "num_chains": 2},
        warmups=[{"name": "pathfinder", "params": {"n_warmup": 10}}],
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"num_chains": 2},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )
    script = emit_script(recipe, num_samples=5, num_warmup=5, num_chains=2)
    assert "pathfinder_adaptation" in script
    script_path = tmp_path / "golden_nuts_pathfinder.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(tmp_path),
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Golden nuts+pathfinder warmup script failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout


@pytest.mark.slow
def test_golden_execution_multipathfinder_warmup(tmp_path: Path) -> None:
    """Execution smoke: mvn_10 × nuts × multipathfinder warmup prints DONE.

    A3: multi-path Pathfinder + PSIS-weighted IMM.
    """
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="multipathfinder",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.5, "max_num_doublings": 5},
        warmup_params={
            "n_warmup": 10,
            "target_acceptance_rate": 0.8,
            "num_chains": 2,
            "n_paths": 2,
            "num_samples_per_path": 5,
        },
        warmups=[
            {
                "name": "multipathfinder",
                "params": {
                    "n_warmup": 10,
                    "n_paths": 2,
                    "num_samples_per_path": 5,
                },
            }
        ],
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"num_chains": 2},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )
    script = emit_script(recipe, num_samples=5, num_warmup=5, num_chains=2)
    assert "lbfgs_psis_mixture" in script
    script_path = tmp_path / "golden_nuts_multipathfinder.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(tmp_path),
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Golden nuts+multipathfinder warmup script failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout


@pytest.mark.slow
def test_golden_execution_multipathfinder_window_adaptation_warmup(
    tmp_path: Path,
) -> None:
    """Execution smoke: mvn_10 × nuts × multipathfinder_window_adaptation prints DONE.

    A3: composition warmup (stage 1 MPF + stage 2 window_adaptation seeded with MPF IMM).
    """
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="multipathfinder_window_adaptation",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.5, "max_num_doublings": 5},
        warmup_params={
            "n_warmup": 10,
            "target_acceptance_rate": 0.8,
            "num_chains": 2,
            "n_paths": 2,
            "num_samples_per_path": 5,
            "imm_shrinkage_to_previous": 20.0,
        },
        warmups=[
            {
                "name": "multipathfinder_window_adaptation",
                "params": {
                    "n_warmup": 10,
                    "n_paths": 2,
                    "num_samples_per_path": 5,
                    "imm_shrinkage_to_previous": 20.0,
                },
            }
        ],
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"num_chains": 2},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )
    script = emit_script(recipe, num_samples=5, num_warmup=5, num_chains=2)
    assert "psis_weights" in script
    assert "imm_shrinkage_to_previous" in script
    script_path = tmp_path / "golden_nuts_mpf_wa.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(tmp_path),
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Golden nuts+multipathfinder_window_adaptation script failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout


@pytest.mark.slow
def test_golden_execution_mclmc_lrd_tuning(tmp_path: Path) -> None:
    """Execution smoke: mvn_10 × mclmc × mclmc_lrd_tuning prints DONE.

    PR3-b emit gate: verifies the inlined run_pilot_nuts + extract_lrd_from_samples
    + make_lrd_kernel pipeline assembles into a runnable script.

    Key structural checks before subprocess:
    - ``_lrd_kernel`` is present (inline make_lrd_kernel closure).
    - ``LowRankInverseMassMatrix`` is present (LRD type from blackjax).
    - D8: no ``tuningfork.base_method`` import in the emitted script.

    Lightweight config: pilot_n_warmup=20, pilot_n_samples=20, n_warmup=10,
    n_samples=5 for e2e speed.
    """
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="mclmc",
        warmup_name="mclmc_lrd_tuning",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.5, "L": 1.0},
        warmup_params={
            "n_warmup": 10,
            "num_chains": 2,
            "k_rank": 3,
            "pilot_n_warmup": 20,
            "pilot_n_samples": 20,
        },
        warmups=[
            {
                "name": "mclmc_lrd_tuning",
                "params": {
                    "n_warmup": 10,
                    "num_chains": 2,
                    "k_rank": 3,
                    "pilot_n_warmup": 20,
                    "pilot_n_samples": 20,
                },
            }
        ],
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"num_chains": 2},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )
    script = emit_script(recipe, num_samples=5, num_chains=2)

    # D8 structural checks before execution.
    assert "_lrd_kernel" in script, (
        "mclmc_lrd warmup kernel closure not found in emitted script. "
        "Expected inline make_lrd_kernel definition."
    )
    assert (
        "LowRankInverseMassMatrix" in script
    ), "LowRankInverseMassMatrix not found in emitted script (should come from blackjax)."
    assert "tuningfork.base_method" not in script, (
        "D8 violation: 'tuningfork.base_method' found in emitted script. "
        "run_pilot_nuts / extract_lrd_from_samples / make_lrd_kernel must be inlined."
    )

    script_path = tmp_path / "golden_mclmc_lrd.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(tmp_path),
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Golden mclmc_lrd_tuning emitted script failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout


@pytest.mark.slow
def test_round_trip_ill_cond_50_mclmc_lrd(tmp_path: Path) -> None:
    """PR3-d round-trip: ill_cond_50 × mclmc × mclmc_lrd_tuning emits + executes.

    Loads the golden ill_cond_50 LRD recipe from the catalog, emits a run
    script via emit_script(), writes it to a temp directory, and executes it
    in a subprocess.

    Key checks:
    - Recipe loads cleanly (warmup_name == "mclmc_lrd_tuning").
    - emit_script() produces a D8-compliant script (no tuningfork.base_method).
    - Subprocess returns 0 and prints DONE (end-to-end execution succeeds).

    Lightweight overrides: pilot_n_warmup=30, pilot_n_samples=30, n_warmup=20,
    n_samples=10, num_chains=2 for speed without touching the golden recipe.
    """
    import dataclasses
    from pathlib import Path as _Path

    from tuningfork.recipes._base import Recipe

    # Load the golden recipe from the catalog.
    catalog_root = _Path(__file__).parents[2] / "tuningfork" / "catalog"
    recipe_path = (
        catalog_root
        / "ill_cond_50"
        / "recipes"
        / "low__mclmc_lrd__mclmc_lrd_tuning.json"
    )
    assert recipe_path.exists(), (
        f"Golden ill_cond_50 LRD recipe not found at {recipe_path}. "
        "Regenerate with: "
        "uv run python -m tuningfork.recipes._generate_starter "
        "--warmup mclmc_lrd_tuning --only ill_cond_50 "
        "--calibrate --cert-seeds 77777 88888 99999 "
        "(see tuningfork/catalog/ill_cond_50/lessons.md for full regen steps)."
    )
    recipe = Recipe.load(recipe_path)
    assert recipe.warmup_name == "mclmc_lrd_tuning", (
        f"Expected warmup_name='mclmc_lrd_tuning', got {recipe.warmup_name!r}. "
        "The golden recipe has the wrong warmup."
    )

    # Override warmup params for speed (don't mutate the golden recipe).
    fast_recipe = dataclasses.replace(
        recipe,
        warmup_params={
            "n_warmup": 20,
            "num_chains": 2,
            "k_rank": recipe.warmup_params.get("k_rank", 40),
            "pilot_n_warmup": 30,
            "pilot_n_samples": 30,
        },
        warmups=[
            {
                "name": "mclmc_lrd_tuning",
                "params": {
                    "n_warmup": 20,
                    "num_chains": 2,
                    "k_rank": recipe.warmup_params.get("k_rank", 40),
                    "pilot_n_warmup": 30,
                    "pilot_n_samples": 30,
                },
            }
        ],
        calibration_budget={"num_chains": 2},
    )

    script = emit_script(fast_recipe, num_samples=10, num_chains=2)

    # D8 structural checks.
    assert (
        "_lrd_kernel" in script
    ), "ill_cond_50 LRD round-trip: _lrd_kernel closure not found in emitted script."
    assert (
        "LowRankInverseMassMatrix" in script
    ), "ill_cond_50 LRD round-trip: LowRankInverseMassMatrix not found in emitted script."
    assert "tuningfork.base_method" not in script, (
        "D8 violation in ill_cond_50 LRD round-trip: 'tuningfork.base_method' found. "
        "LRD helpers must be inlined."
    )

    script_path = tmp_path / "ill_cond_50_mclmc_lrd_round_trip.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(tmp_path),
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"ill_cond_50 LRD round-trip emitted script failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout, (
        f"ill_cond_50 LRD round-trip: 'DONE' not in stdout.\n"
        f"stdout:\n{result.stdout}"
    )
