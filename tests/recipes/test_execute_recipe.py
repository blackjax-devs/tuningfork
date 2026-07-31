import copy
import hashlib
from collections import namedtuple
from pathlib import Path

import pytest

import tuningfork.catalog as catalog
import tuningfork.catalog.emit as emit_module
from tuningfork.recipes import Effort, Recipe
from tuningfork.recipes._execution_plan import canonical_json
from tuningfork.recipes._launcher import GeneratedProgramError

pytestmark = pytest.mark.fast


class _Recipe:
    def __init__(self):
        self.snapshot = {
            "model_name": "demo",
            "notes": "failed then recovered",
            "workflow": [{"step": "retry", "outcome": "failed"}],
            "failure_diagnosis": "hard_direction",
            "attempted_configurations": [{"seed": 3}],
            "unknown_annotation": {"preserve": True},
            "warmup_name": "no_warmup",
            "warmup_params": {},
        }

    def to_dict(self, *, include_legacy_warmup_fields=False):
        assert include_legacy_warmup_fields is True
        return copy.deepcopy(self.snapshot)


def test_execute_recipe_forwards_options_and_preserves_launch_result(monkeypatch):
    calls = []
    expected = object()

    def fake_emit(recipe, **kwargs):
        calls.append(("emit", recipe, kwargs))
        return "generated-source"

    def fake_launch(source, run_root, **kwargs):
        calls.append(("launch", source, run_root, kwargs))
        return expected

    monkeypatch.setattr(emit_module, "emit_script", fake_emit)
    monkeypatch.setattr(emit_module, "launch_generated_program", fake_launch)
    recipe = _Recipe()
    run_root = Path("runs")

    result = emit_module.execute_recipe(
        recipe,
        run_root,
        tuning_seed=14,
        num_samples=11,
        sampler_seed=12,
        reinit_seed=13,
        num_chains=2,
        num_warmup=[3, 4],
        progress_bar=False,
        warmup_num_chains=[1, 1],
        timeout=5.0,
        python_executable="python-test",
        env={"TOKEN": "secret"},
        reference_identity={"recipe": "demo"},
    )

    assert result is expected
    assert calls == [
        (
            "emit",
            recipe,
            {
                "tuning_seed": 14,
                "num_samples": 11,
                "sampler_seed": 12,
                "reinit_seed": 13,
                "num_chains": 2,
                "num_warmup": [3, 4],
                "progress_bar": False,
                "warmup_num_chains": [1, 1],
            },
        ),
        (
            "launch",
            "generated-source",
            run_root,
            {
                "timeout": 5.0,
                "python_executable": "python-test",
                "env": {"TOKEN": "secret"},
                "reference_identity": {
                    "tuningfork_recipe_evidence": {
                        "schema": "tuningfork.recipe-evidence.v1",
                        "snapshot": recipe.snapshot,
                        "snapshot_sha256": hashlib.sha256(
                            (
                                emit_module.RECIPE_EVIDENCE_HASH_DOMAIN
                                + canonical_json(recipe.snapshot)
                            ).encode()
                        ).hexdigest(),
                        "caller_reference_identity": {"recipe": "demo"},
                    }
                },
            },
        ),
    ]


def test_execute_recipe_does_not_launch_when_emission_fails(monkeypatch):
    error = ValueError("emit failed")
    launched = False

    def fake_emit(*args, **kwargs):
        raise error

    def fake_launch(*args, **kwargs):
        nonlocal launched
        launched = True

    monkeypatch.setattr(emit_module, "emit_script", fake_emit)
    monkeypatch.setattr(emit_module, "launch_generated_program", fake_launch)

    with pytest.raises(ValueError) as caught:
        emit_module.execute_recipe(object(), Path("runs"))
    assert caught.value is error
    assert launched is False


def test_execute_recipe_recipe_evidence_preserves_negative_and_unknown_fields(
    monkeypatch,
):
    seen = []
    recipe = _Recipe()
    monkeypatch.setattr(emit_module, "emit_script", lambda *args, **kwargs: "source")
    monkeypatch.setattr(
        emit_module,
        "launch_generated_program",
        lambda *args, **kwargs: (seen.append(kwargs) or object()),
    )

    emit_module.execute_recipe(recipe, Path("runs"))
    evidence = seen[0]["reference_identity"]["tuningfork_recipe_evidence"]
    assert evidence["snapshot"] == recipe.snapshot
    assert evidence["snapshot"]["failure_diagnosis"] == "hard_direction"
    assert evidence["snapshot"]["attempted_configurations"] == [{"seed": 3}]
    assert evidence["snapshot"]["unknown_annotation"] == {"preserve": True}
    expected = hashlib.sha256(
        (
            emit_module.RECIPE_EVIDENCE_HASH_DOMAIN + canonical_json(recipe.snapshot)
        ).encode()
    ).hexdigest()
    assert evidence["snapshot_sha256"] == expected


def test_execute_recipe_real_recipe_encodes_nonfinite_gate_evidence(monkeypatch):
    calls = []
    recipe = Recipe(
        model_name="demo",
        base_method_name="hmc",
        warmup_name="no_warmup",
        effort=Effort.FAILED,
        base_method_params={},
        warmup_params={},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        gate_evidence={"pass_hi": float("inf"), "pass_lo": float("-inf")},
        failure_diagnosis="hard_direction",
        attempted_configurations=[{"seed": 4}],
        _extra_fields={
            "unknown_annotation": {"keep": True},
        },
    )
    monkeypatch.setattr(emit_module, "emit_script", lambda *args, **kwargs: "source")
    monkeypatch.setattr(
        emit_module,
        "launch_generated_program",
        lambda *args, **kwargs: (calls.append(kwargs) or object()),
    )

    emit_module.execute_recipe(recipe, Path("runs"))
    snapshot = calls[0]["reference_identity"]["tuningfork_recipe_evidence"]["snapshot"]
    assert snapshot["gate_evidence"] == {
        "pass_hi": {"\u0000tuningfork_recipe_evidence_nonfinite_float": "+inf"},
        "pass_lo": {"\u0000tuningfork_recipe_evidence_nonfinite_float": "-inf"},
    }
    assert snapshot["unknown_annotation"] == {"keep": True}
    assert canonical_json(snapshot)


def test_canonical_recipe_snapshot_matches_receipt_for_structured_values(monkeypatch):
    point = namedtuple("Point", "x y")(1, 2)
    recipe = _Recipe()
    recipe.snapshot["structured"] = point
    seen = []
    monkeypatch.setattr(emit_module, "emit_script", lambda *args, **kwargs: "source")
    monkeypatch.setattr(
        emit_module,
        "launch_generated_program",
        lambda *args, **kwargs: (seen.append(kwargs) or object()),
    )

    emit_module.execute_recipe(recipe, Path("runs"))

    snapshot = seen[0]["reference_identity"][emit_module.RECIPE_EVIDENCE_KEY][
        "snapshot"
    ]
    assert snapshot == emit_module.canonical_recipe_snapshot(recipe)
    assert snapshot["structured"] == [1, 2]


def test_execute_recipe_recipe_evidence_is_immutable_and_preserves_caller_identity(
    monkeypatch,
):
    seen = []
    recipe = _Recipe()
    caller = {"source": {"tag": "original"}}
    monkeypatch.setattr(emit_module, "emit_script", lambda *args, **kwargs: "source")
    monkeypatch.setattr(
        emit_module,
        "launch_generated_program",
        lambda *args, **kwargs: (seen.append(kwargs) or object()),
    )

    emit_module.execute_recipe(recipe, Path("runs"), reference_identity=caller)
    caller["source"]["tag"] = "mutated"
    recipe.snapshot["notes"] = "mutated"
    identity = seen[0]["reference_identity"]
    assert identity["tuningfork_recipe_evidence"]["caller_reference_identity"] == {
        "source": {"tag": "original"}
    }
    assert identity["tuningfork_recipe_evidence"]["snapshot"]["notes"] == (
        "failed then recovered"
    )


def test_execute_recipe_rejects_reserved_or_invalid_reference_identity(monkeypatch):
    monkeypatch.setattr(emit_module, "emit_script", lambda *args, **kwargs: "source")
    with pytest.raises(ValueError, match="reserved"):
        emit_module.execute_recipe(
            _Recipe(),
            Path("runs"),
            reference_identity={"tuningfork_recipe_evidence": {}},
        )
    with pytest.raises(GeneratedProgramError, match="mapping"):
        emit_module.execute_recipe(_Recipe(), Path("runs"), reference_identity="bad")


def test_execute_recipe_rejects_smc_until_generated_smc_is_supported(monkeypatch):
    from tuningfork.recipes._base_smc import SMCRecipe

    launched = False

    def fake_emit(*args, **kwargs):
        nonlocal launched
        launched = True

    monkeypatch.setattr(emit_module, "emit_script", fake_emit)
    recipe = SMCRecipe(
        model_name="gmm_25",
        smc_method_name="adaptive_tempered_smc",
        inner_method_name="rwm",
        num_particles=8,
        max_steps=3,
    )
    with pytest.raises(TypeError, match="generated SMC.*capability"):
        emit_module.execute_recipe(recipe, Path("runs"))  # type: ignore[arg-type]
    assert launched is False


def test_execute_recipe_propagates_generated_program_error(monkeypatch):
    error = GeneratedProgramError("launch failed")
    monkeypatch.setattr(emit_module, "emit_script", lambda *args, **kwargs: "source")
    monkeypatch.setattr(
        emit_module,
        "launch_generated_program",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(GeneratedProgramError) as caught:
        emit_module.execute_recipe(_Recipe(), Path("runs"))
    assert caught.value is error


def test_execute_recipe_public_exports():
    assert catalog.execute_recipe is emit_module.execute_recipe
    assert catalog.ExecutionTimings is emit_module.ExecutionTimings
    assert catalog.LaunchResult is emit_module.LaunchResult
    assert catalog.GeneratedProgramError is emit_module.GeneratedProgramError


def test_execute_recipe_diagnostics_sets_child_environment(monkeypatch):
    calls = []
    monkeypatch.setattr(emit_module, "emit_script", lambda *args, **kwargs: "source")
    monkeypatch.setattr(
        emit_module,
        "launch_generated_program",
        lambda *args, **kwargs: (calls.append(kwargs) or object()),
    )

    emit_module.execute_recipe(_Recipe(), Path("runs"), diagnostics=True)
    assert calls[0]["env"] == {"TUNINGFORK_TAP_DIAGNOSTICS": "1"}


def test_execute_recipe_diagnostics_rejects_conflicting_environment():
    with pytest.raises(ValueError, match="diagnostics conflicts"):
        emit_module.execute_recipe(
            _Recipe(),
            Path("runs"),
            diagnostics=False,
            env={"TUNINGFORK_TAP_DIAGNOSTICS": "1"},
        )
