from pathlib import Path

import pytest

import tuningfork.catalog as catalog
import tuningfork.catalog.emit as emit_module
from tuningfork.recipes._launcher import GeneratedProgramError

pytestmark = pytest.mark.fast


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
    recipe = object()
    run_root = Path("runs")

    result = emit_module.execute_recipe(
        recipe,
        run_root,
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
                "reference_identity": {"recipe": "demo"},
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


def test_execute_recipe_propagates_generated_program_error(monkeypatch):
    error = GeneratedProgramError("launch failed")
    monkeypatch.setattr(emit_module, "emit_script", lambda *args, **kwargs: "source")
    monkeypatch.setattr(
        emit_module,
        "launch_generated_program",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(GeneratedProgramError) as caught:
        emit_module.execute_recipe(object(), Path("runs"))
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

    emit_module.execute_recipe(object(), Path("runs"), diagnostics=True)
    assert calls[0]["env"] == {"TUNINGFORK_TAP_DIAGNOSTICS": "1"}


def test_execute_recipe_diagnostics_rejects_conflicting_environment():
    with pytest.raises(ValueError, match="diagnostics conflicts"):
        emit_module.execute_recipe(
            object(),
            Path("runs"),
            diagnostics=False,
            env={"TUNINGFORK_TAP_DIAGNOSTICS": "1"},
        )
