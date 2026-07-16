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
"""Constants for the statistician auto-gate.

Contains threshold dicts, the ESS advisory ceiling, and verdict-rank lookup
tables.  Everything here is pure data — no imports beyond the stdlib.
"""

import math

# ---------------------------------------------------------------------------
# Z-test verdict boundary
# ---------------------------------------------------------------------------

Z_VERDICT_ESS_CEILING: int = 6400
"""ESS ceiling above which z-test is advisory (ensemble scale), not verdict-bearing.

Rationale: above this resolution, a point-null z-test rejects any fixed discrepancy
including the GT's own MC error (issue #223). The boundary equals (4/0.05)² where
z≥4 crosses a 0.05σ effect — the gate's own MCSE at ESS=400 PASS floor.

Below Z_VERDICT_ESS_CEILING (small-realm sampling): z-test drives FAIL verdicts as
before. Above it (ensemble scale): z-driven FAILs demote to REVIEW, and z is marked
advisory in margins with human-readable bias_sigma diagnostics.
"""

# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, dict[str, tuple]] = {
    # 3-band semantics:
    #   metric within ``pass`` band  → PASS contribution
    #   within ``review`` band       → REVIEW contribution
    #   else                          → FAIL contribution
    # Verdict is the WORST contribution across all evaluated metrics.
    # Missing metrics (e.g. max_abs_mean_z with no ground truth) are skipped.
    "rhat_max": {
        "pass": (0.0, 1.01),  # x < 1.01  → PASS
        "review": (1.01, 1.05),  # 1.01 ≤ x < 1.05 → REVIEW
        # else FAIL
    },
    "min_bulk_ess": {
        "pass": (400.0, math.inf),  # x ≥ 400 → PASS
        "review": (100.0, 400.0),  # 100 ≤ x < 400 → REVIEW
        # else FAIL
    },
    "n_divergences": {
        # Amended: strict zero relaxed to small absolute count for PASS.
        # A few divergences in a long chain reflects geometry (e.g.
        # funnel-neck visits), not adaptation failure.
        "pass": (0, 6),  # x ≤ 5 → PASS (interval [0,6) i.e. x < 6)
        "review": (6, 40),  # 6 ≤ x < 40 → REVIEW
        # else FAIL
    },
    "max_abs_mean_z": {
        # NOTE: this fixed (0.0, 2.0) PASS band is the *d=1* special case of
        # the dimension-aware Šidák band — see ``sidak_t_pass``.  ``auto_gate``
        # overrides this "pass" tuple at call time with
        # ``(0.0, sidak_t_pass(n_dims))`` before classifying, so the dict
        # value here only matters when ``max_abs_mean_z`` is classified
        # directly against ``DEFAULT_THRESHOLDS`` without going through
        # ``auto_gate`` (e.g. ``resolve_thresholds`` callers, docs, tests).
        # See sidak_t_pass for the dimension-aware band derivation.
        "pass": (0.0, 2.0),  # x < 2 → PASS (d=1 case; d>1 loosens via Šidák)
        "review": (2.0, 4.0),  # 2 ≤ x < 4 → REVIEW
        # else FAIL
    },
}

# ---------------------------------------------------------------------------
# Verdict ranking
# ---------------------------------------------------------------------------

_VERDICT_RANK: dict[str, int] = {"PASS": 0, "SKIP": 0, "REVIEW": 1, "FAIL": 2}
"""Verdict rank lookup.

"SKIP" maps to rank 0 (same as "PASS") so ``_worst("PASS", "SKIP")`` and
``_worst("SKIP", "SKIP")`` return "PASS".  This prevents a KeyError when
``compute_w1_realm`` returns ``verdict="SKIP"`` (no-site-overlap case) and
``_assemble_verdict`` folds it via ``_worst``.
"""
_RANK_VERDICT: dict[int, str] = {0: "PASS", 1: "REVIEW", 2: "FAIL"}
