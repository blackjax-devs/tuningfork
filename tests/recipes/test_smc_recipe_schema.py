"""Focused schema and persistence checks for :class:`SMCRecipe`."""

import json
from pathlib import Path

import pytest

from tuningfork.recipes._base_smc import SMCRecipe

pytestmark = pytest.mark.fast


def _recipe(**kwargs: object) -> SMCRecipe:
    values: dict[str, object] = {
        "model_name": "mvn_10",
        "smc_method_name": "tempered_smc",
        "inner_method_name": "hmc",
        "num_particles": 16,
        "max_steps": 4,
    }
    values.update(kwargs)
    return SMCRecipe(**values)  # type: ignore[arg-type]


def test_unknown_fields_and_public_to_dict_roundtrip(tmp_path: Path) -> None:
    recipe = _recipe(_extra_fields={"legacy_note": ["kept", {"exact": True}]})
    assert recipe.to_dict()["legacy_note"] == ["kept", {"exact": True}]
    path = recipe.save(tmp_path)
    loaded = SMCRecipe.load(path)
    assert loaded.to_dict() == recipe.to_dict()


def test_extension_collision_is_rejected() -> None:
    recipe = _recipe(_extra_fields={"model_name": "collision"})
    with pytest.raises(ValueError, match="collides"):
        recipe.to_dict()


def test_failed_serialization_leaves_existing_target_unchanged(tmp_path: Path) -> None:
    target = _recipe().save(tmp_path)
    before = target.read_bytes()
    invalid = _recipe(_extra_fields={"not_json": object()})
    with pytest.raises(TypeError):
        invalid.save(tmp_path)
    assert target.read_bytes() == before


def test_nonfinite_values_are_rejected_before_overwrite(tmp_path: Path) -> None:
    target = _recipe().save(tmp_path)
    before = target.read_bytes()
    invalid = _recipe(smc_params={"target_ess": float("nan")})
    with pytest.raises(ValueError, match="Out of range float values"):
        invalid.save(tmp_path)
    assert target.read_bytes() == before


def test_atomic_successful_roundtrip_writes_newline(tmp_path: Path) -> None:
    recipe = _recipe(smc_params={"target_ess": 0.5})
    path = recipe.save(tmp_path)
    payload = path.read_bytes()
    assert payload.endswith(b"\n")
    assert json.loads(payload) == recipe.to_dict()
    assert SMCRecipe.load(path).to_dict() == recipe.to_dict()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", ""),
        ("num_particles", 0),
        ("max_steps", True),
        ("seed", -1),
    ],
)
def test_basic_plan_types_are_validated(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _recipe(**{field: value})
