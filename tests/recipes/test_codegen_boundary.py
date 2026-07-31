from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_TOOL = Path(__file__).parents[2] / "tools" / "check_codegen_boundary.py"
_SPEC = importlib.util.spec_from_file_location("check_codegen_boundary", _TOOL)
assert _SPEC and _SPEC.loader
boundary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(boundary)

_PRODUCTION = Path(__file__).parents[2] / "tuningfork"


def test_repository_matches_exact_baseline() -> None:
    assert boundary.scan_source(_PRODUCTION) == boundary.baseline()
    assert boundary.check(_PRODUCTION) == (
        True,
        "no alternate recipe-sampling debt remains",
    )


def test_new_call_in_temp_source_is_reported(tmp_path: Path) -> None:
    (tmp_path / "new.py").write_text(
        "from blackjax.util import run_inference_algorithm as ria\n"
        "def generated_gap():\n"
        "    return ria(kernel, state, 1)\n"
    )
    ok, message = boundary.check(tmp_path)
    assert not ok
    assert "new.py" in message
    assert "run_inference_algorithm" in message


def test_qualified_run_inference_algorithm_is_reported(tmp_path: Path) -> None:
    (tmp_path / "qualified.py").write_text(
        "import blackjax\n"
        "def sample():\n"
        "    return blackjax.util.run_inference_algorithm(kernel, state, 1)\n"
    )
    hits = boundary.scan_source(tmp_path)
    assert hits[("qualified.py", "sample", "run_inference_algorithm")] == 1


def test_aliased_run_smc_is_reported(tmp_path: Path) -> None:
    (tmp_path / "smc.py").write_text(
        "from another.module import run_smc as execute_smc\n"
        "def sample():\n"
        "    return execute_smc(kernel)\n"
    )
    hits = boundary.scan_source(tmp_path)
    assert hits[("smc.py", "sample", "run_smc")] == 1


def test_qualified_run_smc_is_reported(tmp_path: Path) -> None:
    (tmp_path / "qualified_smc.py").write_text(
        "import tuningfork.runner.smc as smc\n"
        "def sample():\n"
        "    return smc.run_smc(kernel)\n"
    )
    hits = boundary.scan_source(tmp_path)
    assert hits[("qualified_smc.py", "sample", "run_smc")] == 1


def test_chained_assignment_alias_is_reported(tmp_path: Path) -> None:
    (tmp_path / "aliases.py").write_text(
        "def sample(base_method):\n"
        "    first = base_method.factory\n"
        "    second = first\n"
        "    return second(logdensity_fn)\n"
    )
    hits = boundary.scan_source(tmp_path)
    assert hits[("aliases.py", "sample", "factory")] == 1


def test_unknown_assignment_alias_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "unknown.py").write_text(
        "def sample():\n" "    helper = object()\n" "    return helper(logdensity_fn)\n"
    )
    assert not boundary.scan_source(tmp_path)


def test_multiplicity_catches_extra_call_in_known_shape(tmp_path: Path) -> None:
    path = tmp_path / "warmup" / "no_warmup.py"
    path.parent.mkdir()
    path.write_text(
        "def _runner():\n"
        "    base_method.factory(logdensity_fn)\n"
        "    base_method.factory(logdensity_fn)\n"
    )
    result = boundary.report(tmp_path)
    key = ("warmup/no_warmup.py", "_runner", "factory")
    assert result["additions"][key] == 1


def test_comments_strings_and_docstrings_do_not_trigger(tmp_path: Path) -> None:
    (tmp_path / "noise.py").write_text(
        '"""run_smc(); obj.runner(); obj.factory()"""\n'
        "# run_smc(); obj.runner(); obj.factory()\n"
        "text = 'run_inference_algorithm(obj); obj.runner()'\n"
    )
    assert not boundary.scan_source(tmp_path)
