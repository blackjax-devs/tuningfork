"""Regression coverage for the lossless gmm_25 SMC recipe migration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tuningfork.catalog import emit_script
from tuningfork.recipes._attempt_evidence import _canonical_json, _validate_envelope
from tuningfork.recipes._base_smc import SMCRecipe
from tuningfork.recipes._ground_truth_reference import load_ground_truth_reference

pytestmark = pytest.mark.fast

_RECIPE = (
    Path(__file__).parents[2]
    / "tuningfork"
    / "catalog"
    / "gmm_25"
    / "recipes"
    / "smc__adaptive_tempered_smc__rwm.json"
)
_ORIGINAL_VIEW_SHA256 = (
    "50d1f54ea8adc74063ba510c9ea4d65ae0cc2dd883ae11b6a20923e788306456"
)


def test_gmm25_smc_migration_is_lossless_and_codegen_admissible() -> None:
    raw = json.loads(_RECIPE.read_text())
    recipe = SMCRecipe.load(_RECIPE)

    assert recipe.parameter_update_strategy == "none"
    assert recipe.parameter_update_strategy_kwargs == {}
    legacy = next(
        item
        for item in recipe.attempted_configurations
        if item.get("attempt_id") == "legacy-current-view"
    )
    legacy_view = legacy["metrics"]["legacy_current_view"]
    assert hashlib.sha256(_canonical_json(legacy_view)).hexdigest() == (
        _ORIGINAL_VIEW_SHA256
    )
    assert legacy_view["parameter_update_strategy"] == (
        "step_size_and_imm_from_particles"
    )
    assert legacy_view["parameter_update_strategy_kwargs"] == {
        "target_acceptance": 0.65
    }
    assert legacy_view["gate_evidence"] == raw["gate_evidence"]
    assert legacy_view["calibration_budget"] == {
        "n_particles": 1000,
        "n_smc_steps": 1,
        "num_mcmc_steps": 25,
        "lambda_final": 1.0,
        "wall_seconds_total": 1.0,
        "wall_seconds_run": 0.966,
    }

    migration = next(
        item
        for item in recipe.attempted_configurations
        if item.get("attempt_id") == "migration-adaptive-tempered-rwm"
    )
    assert recipe.calibration_budget["selected_attempt_id"] == migration["attempt_id"]
    assert migration["execution"] is None
    assert migration["automatic_verdict"] == "PASS"
    assert migration["metrics"]["dropped_json_pointers"] == []
    mapping = migration["metrics"]["parameter_update_strategy_mapping"]
    assert mapping["/parameter_update_strategy"]["new"] == "none"
    assert mapping["/parameter_update_strategy_kwargs"]["new"] == {}
    assert (
        "passed only to inner_kernel_tuning"
        in mapping["/parameter_update_strategy"]["reason"]
    )

    canonical_gt = load_ground_truth_reference(_RECIPE.parents[2], "gmm_25").identity
    assert legacy["ground_truth"] == canonical_gt
    assert migration["ground_truth"] == canonical_gt
    for attempt in (legacy, migration):
        assert _validate_envelope(attempt) == attempt

    source = emit_script(recipe)
    compile(source, "gmm25_smc_migration.py", "exec")
