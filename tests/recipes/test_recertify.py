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
"""Regression guard: a re-emission must not drop a pinned kernel kwarg.

``horseshoe/medium__nuts__window_adaptation_diag_imm.json`` is the case study.
Its 2026-07-30 recert forwarded ``seed`` and ``target_acceptance`` to
``emit_low_recipe_for_cell()`` by hand and never built a
``sampler_kwargs_override``, so the committed ``max_num_doublings=15`` silently
reverted to the registry default (10) and the gate PASSed on an easier cell
than the one the recipe claimed to reproduce.

``tests/recipes/test_reemit_sweep.py`` already guards the PLAN side:
``reconstruct()`` + ``config_fidelity_violations()`` prove a reconstruction is
faithful to a committed artifact, without ever running anything. That is blind
by design to the executor -- a driver that builds the right plan and then
drops the argument passes every plan-side check. This module closes the loop
from the EMISSION side: it actually re-runs a cell through
``reemit_sweep.recertify()`` and checks what landed on disk, so a regression
in the fix itself (not just in some future hand-assembled call) is still
caught.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _REPO_ROOT / "tools" / "reemit_sweep.py"


def _load_driver():
    """Import the driver from tools/, mirroring test_reemit_sweep.py's loader."""
    spec = importlib.util.spec_from_file_location("reemit_sweep", _TOOL)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["reemit_sweep"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


driver = _load_driver()


def _fixture_recipe(tmp_path: Path) -> Path:
    """A committed-looking cell pinning a non-default kernel kwarg.

    Built from the REAL mvn_10 low__nuts__window_adaptation_diag_imm.json --
    the cheapest real (model, warmup, sampler) triple in the registry -- with
    ``max_num_doublings=15`` injected (the NUTS registry default is 10, so
    this is genuinely an override, not a value the emit path would produce on
    its own). Its ``tuning_seed`` (682737) already derives from
    ``RECIPE_SEED`` via the real file, so the seed-recovery check in
    ``reconstruct()`` passes without any further doctoring.
    """
    real = (
        _REPO_ROOT
        / "tuningfork"
        / "catalog"
        / "mvn_10"
        / "recipes"
        / "low__nuts__window_adaptation_diag_imm.json"
    )
    doc = json.loads(real.read_text())
    doc["base_method_params"] = dict(doc["base_method_params"])
    doc["base_method_params"]["max_num_doublings"] = 15
    recipes_dir = tmp_path / "mvn_10" / "recipes"
    recipes_dir.mkdir(parents=True)
    path = recipes_dir / "low__nuts__window_adaptation_diag_imm.json"
    path.write_text(json.dumps(doc))
    return path


class TestRecertifyRoundTripsKernelKwargs:
    def test_recertify_preserves_a_non_default_kernel_kwarg(
        self, tmp_path: Path
    ) -> None:
        """The fixed re-emission path: recertify() must not drop the override.

        Reads target_acceptance / sampler_kwargs_override / everything else
        off the fixture via reconstruct() rather than a hand-typed call, so a
        reseed (11111, chosen to differ from the fixture's own recovered
        seed) cannot also silently drop a kernel kwarg.
        """
        recipe_path = _fixture_recipe(tmp_path)
        result = driver.recertify(recipe_path, seed=11111, catalog_root=tmp_path)
        assert result.verdict == "PASS", result.note
        written = json.loads(result.recipe_path.read_text())
        assert written["base_method_params"].get("max_num_doublings") == 15, (
            "recertify() dropped a pinned kernel kwarg during re-emission -- "
            "this is the exact defect class that silently reverted horseshoe's "
            "max_num_doublings to the registry default on 2026-07-30"
        )

    def test_a_hand_assembled_call_is_the_defect_recertify_closes(
        self, tmp_path: Path
    ) -> None:
        """Documents the failure mode recertify() exists to prevent.

        Mirrors the actual 2026-07-30 recert call shape: seed and
        target_acceptance forwarded explicitly, ``sampler_kwargs_override``
        never built. This is a standing record of the defect's shape, not a
        gate on its own -- the gate is the previous test, which exercises the
        fixed path. If this test ever stops failing to reproduce the drop
        (e.g. because ``max_num_doublings`` gained a registry default of 15),
        the fixture needs a different non-default kwarg, not a relaxed
        assertion.
        """
        from tuningfork.recipes._base import Effort
        from tuningfork.recipes._recipe_runner import emit_low_recipe_for_cell

        result = emit_low_recipe_for_cell(
            model_name="mvn_10",
            warmup_name="window_adaptation_diag_imm",
            sampler_name="nuts",
            n_warmup=1000,
            n_samples=1000,
            num_chains=4,
            seed=11111,
            target_acceptance=0.8,
            effort=Effort.LOW,
            catalog_root=tmp_path,
            verbose=False,
            # sampler_kwargs_override intentionally omitted -- this is the bug:
            # a caller who forwards seed/target_acceptance by hand has to also
            # remember to build this dict, and nothing forces them to.
        )
        assert result.verdict == "PASS", result.note
        assert result.recipe_path is not None
        written = json.loads(result.recipe_path.read_text())
        assert "max_num_doublings" not in written["base_method_params"], (
            "fixture no longer reproduces the defect shape -- pick another "
            "kwarg the naive call would drop, or this test proves nothing"
        )
