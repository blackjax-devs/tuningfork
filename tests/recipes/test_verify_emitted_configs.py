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
"""The artifact-side configuration gate, run as part of the suite.

The plan-side gate in ``test_reemit_sweep.py`` reconstructs each recipe from
itself, so it proves the artifact is self-reproducible.  It is blind by design to
the executor: a driver that builds the right plan and then drops an argument
writes a plausible recipe under the wrong settings, and every plan-side check
stays green.  That is not hypothetical — it dropped each variational warmup's
recorded optimisation budget and five cells re-ran at the registry default.

This closes the loop from the other end, comparing what is on disk against the
pinned baseline revision.  It lives in the suite rather than in a script because
the bug class recurs: the next warmup family that grows a hyperparameter will be
dropped the same way, and a tool nobody runs catches nothing.

Marked ``slow``: one ``git show`` subprocess per stamped cell, ~140 of them.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _REPO_ROOT / "tools" / "verify_emitted_configs.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("verify_emitted_configs", _TOOL)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["verify_emitted_configs"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


tool = _load_tool()


@pytest.fixture(scope="module")
def report():
    """Compare the catalog on disk against the pinned baseline, once."""
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{tool.BASELINE_REVISION}^{{commit}}"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if resolved.returncode != 0:
        # Deliberately a failure, not a skip.  A skip here is how this check went
        # vacuous the first time; a shallow clone is a fixable CI setting
        # (``fetch-depth: 0``), not a reason to stop comparing.
        pytest.fail(
            f"baseline {tool.BASELINE_REVISION} is not in this checkout, so no "
            f"artifact can be compared against what it claims to reproduce. "
            f"On CI this means the clone is shallow: set `fetch-depth: 0`.\n"
            f"{resolved.stderr.strip()}"
        )
    return tool.verify()


def test_emitted_configs_match_the_baseline(report) -> None:
    """No re-emitted cell ran under a configuration other than the one it records."""
    assert not report.failures, "\n".join(report.failures)


def test_the_comparison_is_not_vacuous(report) -> None:
    """A mismatch count of zero means nothing without the count behind it.

    Both routes to a hollow pass are covered: stamps disappearing from artifacts
    (``stamped`` falls) and cells the driver declines to reconstruct
    (``checked`` falls below ``stamped``).
    """
    assert report.vacuous is None, report.vacuous
    assert report.checked == report.stamped, (
        f"{report.stamped - report.checked} stamped cells were not compared; "
        f"they should appear in unreconstructable or missing_baseline"
    )


def test_a_planted_mismatch_is_caught(tmp_path: Path) -> None:
    """The comparison has power: perturb a recorded parameter and it fires.

    Runs against a copy so the catalog is never written to.  Without this the
    suite would assert only that the gate reports zero, which an unwired gate
    also does.
    """
    driver_spec = importlib.util.spec_from_file_location(
        "reemit_sweep", _REPO_ROOT / "tools" / "reemit_sweep.py"
    )
    driver = importlib.util.module_from_spec(driver_spec)  # type: ignore[arg-type]
    sys.modules["reemit_sweep"] = driver
    driver_spec.loader.exec_module(driver)  # type: ignore[union-attr]

    catalog = _REPO_ROOT / "tuningfork" / "catalog"
    source = (
        catalog
        / "banana"
        / "recipes"
        / ("medium__dynamic_hmc__window_adaptation_diag_imm__policy_v1-medium.json")
    )
    if not source.exists():
        pytest.skip("fixture recipe not in the catalog")

    cfg = driver.reconstruct(source)
    assert isinstance(cfg, driver.CellConfig)

    committed = json.loads(source.read_text())
    assert not driver.config_fidelity_violations(
        cfg, committed
    ), "fixture already mismatches itself, so the planted change proves nothing"

    # Derived from what was reconstructed rather than hard-coded, so the plant is
    # guaranteed to differ whatever the fixture records.
    committed["warmups"][0]["params"]["target_acceptance"] = (
        cfg.target_acceptance or 0.8
    ) / 2.0
    bad = driver.config_fidelity_violations(cfg, committed)
    assert any("target_acceptance" in v for v in bad), bad
