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
"""Calibration modules for tuningfork.

Reference-draw certification, BO tuning, and the Statistician auto-gate.

    - ``certify_reference_analytic``: analytic-sampler certifier (Path A).
    - ``certify_reference``: long-NUTS certifier (Path B).
    - ``tune``: Optuna BO over sampler / warmup hyperparameters.
    - ``statistician_gate``: auto-gate verdict from sample diagnostics.
    - ``_summary``: Summaries dataclass + compute_summaries helper.
"""
