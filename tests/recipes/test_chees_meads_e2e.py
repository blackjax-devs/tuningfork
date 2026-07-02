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
"""End-to-end wiring regression tests for the ChEES-HMC and MEADS warmups
through the real ``_recipe_runner.py`` pipeline.

Both warmups have full wrappers + registry entries but were never exercised
end-to-end through ``emit_low_recipe_for_cell`` — the generic dispatch was
built and validated only against the ``window_adaptation`` output contract
(step_size + inverse_mass_matrix, per-chain broadcast, no callables). These
tests pin the wiring defects a statistician A/B investigation located so
they cannot silently regress. The tests assert an HONEST gate verdict is
reached (PASS, REVIEW, or FAIL are all acceptable outcomes) — the point is
that the pipeline runs to completion without TypeError/NaN-crash, not that
the cell PASSES the statistician gate (a later statistician pass tunes for
that).
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def test_chees_dynamic_hmc_e2e_no_crash(tmp_path: Path) -> None:
    """emit_low_recipe_for_cell(mvn_10, chees, dynamic_hmc) runs to completion.

    Regression guard for bug 2 (target_acceptance_rate=None TypeError in
    upstream chees_adaptation.py: `target_acceptance_rate - harmonic_mean`)
    and bug 3 (np.isfinite TypeError on the Python-callable leaves CHEES
    legitimately returns in adapted_params: next_random_arg_fn,
    integration_steps_fn). Both bugs fire deterministically on every CHEES
    cell through the default (no-override) call path — this is the exact
    call shape used in production recipe emission.
    """
    from tuningfork.recipes._recipe_runner import emit_low_recipe_for_cell

    result = emit_low_recipe_for_cell(
        "mvn_10",
        "chees",
        "dynamic_hmc",
        n_warmup=50,
        n_samples=50,
        num_chains=4,
        catalog_root=tmp_path,
        verbose=False,
    )
    assert result.verdict in ("PASS", "REVIEW", "FAIL"), (
        f"Expected an honest gate verdict, got {result.verdict!r} "
        f"(note={result.note!r})"
    )
    # FAIL from a genuine gate reason (rhat/ess/div/NaN-in-sampler) is fine at
    # this tiny n_warmup/n_samples; a crash-derived FAIL (TypeError/KeyError
    # text in the note) is what we're guarding against.
    if result.verdict == "FAIL":
        assert "Error" not in (
            result.note or ""
        ), f"FAIL note looks crash-derived, not gate-derived: {result.note}"
