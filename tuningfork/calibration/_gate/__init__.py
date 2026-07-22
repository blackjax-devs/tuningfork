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
"""Internal gate package — implementation stages for the statistician auto-gate.

Public consumers should import from ``tuningfork.calibration.statistician_gate``,
not from here.  This package is an implementation detail.

Modules
-------
constants  — ``Z_VERDICT_ESS_CEILING``, ``DEFAULT_THRESHOLDS``, rank tables.
bands      — ``sidak_t_pass``, ``resolve_thresholds``, classify/margin helpers.
layout     — ``_samples_to_multichain`` (single→multichain reshaping).
mixing     — ``_compute_mixing_stats`` (R̂, ESS, divergences).
gt_compare — ``_compute_gt_compare`` (z-scores, bias-sigma fields).
verdict    — ``AutoGateVerdict``, ``_assemble_verdict`` (classify + aggregate).
w1_realm   — ``W1RealmResult``, ``compute_w1_realm`` (W1/σ two-prong gate).
"""

from .bands import (
    _apply_vi_mode_thresholds,
    _build_margin,
    _classify_metric,
    _worst,
    resolve_thresholds,
    sidak_t_pass,
)
from .constants import DEFAULT_THRESHOLDS, Z_VERDICT_ESS_CEILING
from .gt_compare import _compute_gt_compare, _GtCompareResult
from .layout import _samples_to_multichain
from .marginal_z import (
    _DEFAULT_NU,
    _SE_FLOOR,
    _TAU_SCI,
    bonferroni_z_crit,
    marginal_z_verdict,
)
from .mixing import _compute_mixing_stats
from .verdict import AutoGateVerdict, _assemble_verdict
from .w1_realm import (
    W1RealmResult,
    _build_floor,
    _compute_tau_frac,
    _khat_max,
    _loo_conservatism_check,
    _w1_1d,
    compute_w1_realm,
)

__all__ = [
    "AutoGateVerdict",
    "DEFAULT_THRESHOLDS",
    "W1RealmResult",
    "Z_VERDICT_ESS_CEILING",
    "_DEFAULT_NU",
    "_GtCompareResult",
    "_SE_FLOOR",
    "_TAU_SCI",
    "_apply_vi_mode_thresholds",
    "_assemble_verdict",
    "_build_floor",
    "_build_margin",
    "_classify_metric",
    "_compute_gt_compare",
    "_compute_mixing_stats",
    "_compute_tau_frac",
    "_khat_max",
    "_loo_conservatism_check",
    "_samples_to_multichain",
    "_w1_1d",
    "_worst",
    "bonferroni_z_crit",
    "compute_w1_realm",
    "marginal_z_verdict",
    "resolve_thresholds",
    "sidak_t_pass",
]
