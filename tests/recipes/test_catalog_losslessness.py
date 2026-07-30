"""Corpus-level regression checks for lossless recipe evidence I/O."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tuningfork.recipes import Recipe
from tuningfork.recipes._base_smc import SMCRecipe

pytestmark = pytest.mark.fast

_ROOT = Path(__file__).parents[2]


def _committed_recipe_paths() -> list[Path]:
    """Return committed catalog recipe artifacts, excluding reference JSON."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "tuningfork/catalog"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
    )
    paths = [
        _ROOT / name
        for name in result.stdout.decode().split("\0")
        if name.endswith(".json")
        and (
            Path(name).name == "groundtruth.json" or Path(name).parent.name == "recipes"
        )
    ]
    return sorted(paths)


def _pointer(key: str) -> str:
    return "/" + key.replace("~", "~0").replace("/", "~1")


def _public_fields(recipe_type: type[Recipe] | type[SMCRecipe]) -> set[str]:
    """Dataclass fields that are part of the public schema."""
    return {
        field.name
        for field in dataclasses.fields(recipe_type)
        if not field.name.startswith("_")
    }


def _recipe_kind(raw: dict[str, Any], source: Path) -> type[Recipe] | type[SMCRecipe]:
    if "smc_method_name" in raw:
        return SMCRecipe
    if "effort" in raw:
        return Recipe
    raise AssertionError(
        f"{source}: expected recipe JSON with /effort or /smc_method_name"
    )


@pytest.mark.parametrize(
    "source",
    _committed_recipe_paths(),
    ids=lambda path: str(path.relative_to(_ROOT)),
)
def test_catalog_recipe_evidence_survives_roundtrip(
    source: Path, tmp_path: Path
) -> None:
    """Failed-attempt evidence, diagnoses, and legacy annotations survive I/O."""
    raw = json.loads(source.read_text())
    recipe_type = _recipe_kind(raw, source)
    loaded = recipe_type.load(source)
    saved = loaded.save(tmp_path)
    roundtripped = json.loads(saved.read_text())

    if "attempted_configurations" in raw:
        assert (
            "attempted_configurations" in roundtripped
        ), f"{source} /attempted_configurations was dropped after load/save"
        assert (
            roundtripped.get("attempted_configurations")
            == raw["attempted_configurations"]
        ), f"{source} /attempted_configurations changed after load/save"

    if "failure_diagnosis" in raw:
        assert (
            "failure_diagnosis" in roundtripped
        ), f"{source} /failure_diagnosis was dropped after load/save"
        assert (
            roundtripped.get("failure_diagnosis") == raw["failure_diagnosis"]
        ), f"{source} /failure_diagnosis changed after load/save"

    unknown = set(raw) - _public_fields(recipe_type)
    for key in sorted(unknown):
        assert (
            key in roundtripped
        ), f"{source} {_pointer(key)} was dropped after load/save"
        assert (
            roundtripped[key] == raw[key]
        ), f"{source} {_pointer(key)} changed after load/save"
