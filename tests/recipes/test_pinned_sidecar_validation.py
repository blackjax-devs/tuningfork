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
"""Fast validation tests for legacy inverse-mass-matrix sidecars."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.fast


def _recipe():
    from tuningfork.recipes._base import Effort, Recipe

    return Recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.FAILED,
        base_method_params={"step_size": 0.1, "inverse_mass_matrix": "sidecar"},
        warmup_params={"n_warmup": 100, "num_chains": 4},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"trials": 0, "wall_seconds_estimate": 1.0},
        difficulty=None,
        instructions="test",
        inverse_mass_matrix_path="mvn_10/imm.npz",
    )


def _write_sidecar(root: Path, value: object) -> None:
    summary = root / "mvn_10" / "reference" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text('{"mean": {"x": [0]}, "std": {"x": [1]}}')
    np.savez(root / "mvn_10" / "imm.npz", imm=np.asarray(value))


@pytest.mark.parametrize(
    "value, detail",
    [
        ([], "non-empty"),
        (1.0, "rank"),
        ([[[1.0]]], "rank"),
        ([[1.0, 0.0]], "square"),
        ([1.0, 0.0], "strictly positive"),
        ([[1.0, 2.0], [0.0, 1.0]], "symmetric"),
        ([[1.0, 2.0], [2.0, 1.0]], "positive definite"),
    ],
)
def test_prepare_pinned_replay_rejects_malformed_legacy_imm(
    tmp_path: Path, value: object, detail: str
) -> None:
    from tuningfork.catalog import prepare_pinned_replay

    _write_sidecar(tmp_path, value)
    with pytest.raises(ValueError, match=detail):
        prepare_pinned_replay(_recipe(), catalog_root=tmp_path)


@pytest.mark.parametrize(
    "value",
    [
        [1.0, 2.0],
        [[2.0, 0.1], [0.1, 3.0]],
    ],
)
def test_prepare_pinned_replay_accepts_valid_legacy_imm(
    tmp_path: Path, value: object
) -> None:
    from tuningfork.catalog import prepare_pinned_replay

    _write_sidecar(tmp_path, value)
    replay = prepare_pinned_replay(_recipe(), catalog_root=tmp_path)
    np.testing.assert_allclose(replay.base_method_params["inverse_mass_matrix"], value)
