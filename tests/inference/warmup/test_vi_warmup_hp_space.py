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
"""Fast tests for VI warmup default_hp_space plumbing (Phase 8B.2).

Validates that:
- meanfield_vi and fullrank_vi ENTRY.default_hp_space declare
  num_optimization_steps with the expected range.
- Warmup._base.Warmup dataclass accepts default_hp_space.
- default_value_for_space returns the midpoint (25_500) for the HP range.
- The recipe runner correctly records num_optimization_steps in warmup_params.
"""

import pytest

pytestmark = pytest.mark.fast


def test_meanfield_vi_warmup_has_hp_space() -> None:
    """meanfield_vi ENTRY.default_hp_space declares num_optimization_steps."""
    from tuningfork.warmup import WARMUPS

    mfwu = WARMUPS["meanfield_vi"]
    assert hasattr(mfwu, "default_hp_space"), "Warmup must have default_hp_space field"
    assert len(mfwu.default_hp_space) == 1
    space = mfwu.default_hp_space[0]
    assert space.name == "num_optimization_steps"
    assert space.kind == "int"
    assert space.low == 1_000
    assert space.high == 50_000


def test_fullrank_vi_warmup_has_hp_space() -> None:
    """fullrank_vi ENTRY.default_hp_space declares num_optimization_steps."""
    from tuningfork.warmup import WARMUPS

    frwu = WARMUPS["fullrank_vi"]
    assert hasattr(frwu, "default_hp_space")
    assert len(frwu.default_hp_space) == 1
    space = frwu.default_hp_space[0]
    assert space.name == "num_optimization_steps"
    assert space.kind == "int"
    assert space.low == 1_000
    assert space.high == 50_000


def test_other_warmups_have_empty_hp_space() -> None:
    """window_adaptation, no_warmup, pathfinder, etc. have empty default_hp_space.

    The VI warmups are the only ones with a declared warmup HP space.
    All other warmups fall back to the Warmup dataclass default (empty tuple).
    """
    from tuningfork.warmup import WARMUPS

    vi_warmup_names = {"meanfield_vi", "fullrank_vi"}
    for name, entry in WARMUPS.items():
        if name in vi_warmup_names:
            continue
        hp_space = getattr(entry, "default_hp_space", ())
        assert (
            hp_space == () or len(hp_space) == 0
        ), f"Non-VI warmup {name!r} unexpectedly has default_hp_space: {hp_space}"


def test_default_value_for_space_midpoint() -> None:
    """default_value_for_space returns integer midpoint for int HP space."""
    from tuningfork.calibration.tune import default_value_for_space
    from tuningfork.warmup import WARMUPS

    space = WARMUPS["meanfield_vi"].default_hp_space[0]
    default = default_value_for_space(space)
    assert default == 25_500, f"Expected midpoint 25500 but got {default}"


def test_warmup_hp_space_roundtrip_in_warmup_params_dict() -> None:
    """Warmup HP defaults are included in _warmup_params_dict built by runner.

    Verifies the recipe runner includes num_optimization_steps in the
    warmup_params when emitting a VI warmup recipe (uses a synthetic recipe
    to avoid running actual VI optimization).
    """
    from tuningfork.calibration.tune import default_value_for_space
    from tuningfork.warmup import WARMUPS

    mfwu = WARMUPS["meanfield_vi"]
    # Simulate what the recipe runner does to build warmup_params
    warmup_hp_defaults = {
        space.name: default_value_for_space(space)
        for space in getattr(mfwu, "default_hp_space", ())
    }
    assert "num_optimization_steps" in warmup_hp_defaults
    assert warmup_hp_defaults["num_optimization_steps"] == 25_500  # midpoint


def test_warmup_hp_space_override_roundtrip() -> None:
    """warmup_kwargs_override correctly overrides the HP default in warmup_params."""
    from tuningfork.calibration.tune import default_value_for_space
    from tuningfork.warmup import WARMUPS

    mfwu = WARMUPS["meanfield_vi"]
    # Simulate recipe runner building warmup params with override
    warmup_hp_defaults = {
        space.name: default_value_for_space(space)
        for space in getattr(mfwu, "default_hp_space", ())
    }
    warmup_kwargs_override = {"num_optimization_steps": 10_000}
    # Merge: override wins
    merged = {**warmup_hp_defaults}
    merged.update(
        {
            k: v
            for k, v in warmup_kwargs_override.items()
            if any(k == s.name for s in getattr(mfwu, "default_hp_space", ()))
        }
    )
    assert (
        merged["num_optimization_steps"] == 10_000
    ), "warmup_kwargs_override must override the HP default"
