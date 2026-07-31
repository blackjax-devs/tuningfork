"""Corpus-level regression checks for lossless recipe evidence I/O."""

from __future__ import annotations

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


_NORMALIZED_POINTER_PREFIXES = {
    "/warmup_name": "/warmups/0/name",
    "/warmup_params": "/warmups/0/params",
}


def _pointer(key: str) -> str:
    return "/" + key.replace("~", "~0").replace("/", "~1")


def _leaf_pointers(value: Any, prefix: str = "") -> dict[str, Any]:
    """Return every JSON leaf (including empty containers) by JSON pointer."""
    if isinstance(value, dict):
        if not value:
            return {prefix or "/": value}
        leaves: dict[str, Any] = {}
        for key, child in value.items():
            child_pointer = f"{prefix}/{_pointer(key)[1:]}"
            leaves.update(_leaf_pointers(child, child_pointer))
        return leaves
    if isinstance(value, list):
        if not value:
            return {prefix or "/": value}
        leaves = {}
        for index, child in enumerate(value):
            leaves.update(_leaf_pointers(child, f"{prefix}/{index}"))
        return leaves
    return {prefix or "/": value}


def _normalized_pointer(pointer: str, raw: dict[str, Any]) -> str:
    for source, target in _NORMALIZED_POINTER_PREFIXES.items():
        if source[1:] in raw and (
            pointer == source or pointer.startswith(source + "/")
        ):
            return target + pointer[len(source) :]
    return pointer


def _same_json_leaf(left: Any, right: Any) -> bool:
    """Compare JSON leaves without equating booleans, integers, and floats."""
    if type(left) is not type(right):
        return False
    if isinstance(left, float) and isinstance(right, float):
        if left != left and right != right:  # NaN is not equal to itself.
            return True
    return left == right


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

    original = _leaf_pointers(raw)
    serialized = _leaf_pointers(roundtripped)
    dropped = {
        pointer
        for pointer, value in original.items()
        if _normalized_pointer(pointer, raw) not in serialized
        or not _same_json_leaf(value, serialized[_normalized_pointer(pointer, raw)])
    }
    assert not dropped, f"{source}: lossy leaves after load/save: {sorted(dropped)}"


def test_nested_unknown_field_sentinel_roundtrip(
    tmp_path: Path,
) -> None:
    """An opaque nested annotation remains exact through typed load/save."""
    source = _committed_recipe_paths()[0]
    raw = json.loads(source.read_text())
    raw["legacy_nested_sentinel"] = {
        "attempts": [{"status": "FAIL", "diagnosis": ["keep", {"n": 7}]}],
        "provenance": {"sidecar": "opaque-id", "order": [3, 1, 2]},
    }
    input_path = tmp_path / source.name
    input_path.write_text(json.dumps(raw, allow_nan=True) + "\n")
    recipe_type = _recipe_kind(raw, input_path)
    loaded = recipe_type.load(input_path)
    saved = loaded.save(tmp_path / "out")
    serialized = json.loads(saved.read_text())
    assert serialized["legacy_nested_sentinel"] == raw["legacy_nested_sentinel"]
