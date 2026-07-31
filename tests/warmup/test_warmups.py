"""Descriptor-only contracts for the warmup registry."""

import pytest

from tuningfork.warmup import WARMUPS

pytestmark = pytest.mark.fast

EXPECTED_NAMES = {
    "window_adaptation_diag_imm",
    "window_adaptation_dense_imm",
    "window_adaptation_low_rank_imm",
    "mclmc_tuning",
    "mclmc_lrd_tuning",
    "adjusted_mclmc_tuning",
    "adjusted_mclmc_trajectory_tuning",
    "no_warmup",
    "pathfinder",
    "multipathfinder",
    "multipathfinder_window_adaptation",
    "meads",
    "chees",
    "meanfield_vi",
    "fullrank_vi",
}


def test_registry_is_exact_descriptor_set() -> None:
    assert set(WARMUPS) == EXPECTED_NAMES
    for name, descriptor in WARMUPS.items():
        assert descriptor.name == name
        assert descriptor.notes.strip()
        assert descriptor.compatible_methods


def test_descriptor_compatibility_including_wildcard() -> None:
    assert WARMUPS["no_warmup"].is_compatible("anything")
    assert WARMUPS["window_adaptation_diag_imm"].is_compatible("nuts")
    assert not WARMUPS["window_adaptation_diag_imm"].is_compatible("mclmc")


@pytest.mark.parametrize(
    "name",
    [
        "meanfield_vi",
        "fullrank_vi",
        "window_adaptation_low_rank_imm",
        "adjusted_mclmc_trajectory_tuning",
    ],
)
def test_specialized_warmups_declare_nonempty_hp_descriptors(name: str) -> None:
    assert WARMUPS[name].default_hp_space
    assert all(space.name and space.kind for space in WARMUPS[name].default_hp_space)
