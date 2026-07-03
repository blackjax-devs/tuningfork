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
"""Telemetry saturation detection for NUTS reference-certification chains.

Tripwire for detecting when `num_integration_steps` (trajectory length)
is saturated or degenerate despite passing certification gates. A "certified"
chain is certified for *sample quality* (R̂, ESS, divergence, E-BFMI) but
may have unusable telemetry if the sampler's adaptive machinery (tree-depth
control via max_num_doublings) is pinned at a cap, yielding censored
near-constant trajectories instead of geometry-driven signals.

See worklog/lessons/process/2026-07-03-cert-gate-blind-to-treedepth-saturation.md
"""

import numpy as np

__all__ = ["check_telemetry_saturation"]


def check_telemetry_saturation(num_integration_steps: np.ndarray) -> dict:
    """Check whether num_integration_steps telemetry is saturated or degenerate.

    A saturated chain has a modal value (most frequent value) that dominates
    (mass > 0.90) and exhibits telemetry-useless behavior: either the modal
    value is a power-of-two minus one (2^k - 1, indicating tree-depth capping)
    or the total unique-value count is ≤ 3 (near-degenerate).

    Parameters
    ----------
    num_integration_steps
        1-D array of num_integration_steps telemetry from post-warmup NUTS
        samples. Typically loaded from chain_stats.npz['num_integration_steps'].

    Returns
    -------
    dict
        Keys:
        - 'saturated' : bool
            True iff saturation rule triggers: modal_mass > 0.90 AND
            (modal_value == 2^k - 1 for some k, OR n_unique <= 3).
        - 'modal_value' : int
            Most-frequent value in the input array.
        - 'modal_mass' : float
            Fraction [0, 1] of samples at the modal value.
        - 'is_power2_minus1' : bool
            True iff modal_value == 2^k - 1 for some non-negative integer k.
        - 'n_unique' : int
            Number of distinct values in the input array.

    Notes
    -----
    If the input array is empty, all fields are set to sensible defaults
    (saturated=False, modal_value=0, modal_mass=0.0, is_power2_minus1=False, n_unique=0).
    """
    if len(num_integration_steps) == 0:
        return {
            "saturated": False,
            "modal_value": 0,
            "modal_mass": 0.0,
            "is_power2_minus1": False,
            "n_unique": 0,
        }

    # Count unique values and their frequencies
    unique_vals, counts = np.unique(num_integration_steps, return_counts=True)
    n_unique = len(unique_vals)

    # Modal value and its mass
    modal_idx = np.argmax(counts)
    modal_value = int(unique_vals[modal_idx])
    modal_mass = float(counts[modal_idx] / len(num_integration_steps))

    # Check if modal_value is 2^k - 1 for some k >= 0
    # If modal_value == 2^k - 1, then modal_value + 1 == 2^k is a power of two.
    # A number n is a power of two iff n > 0 and (n & (n - 1)) == 0.
    is_power2_minus1 = (modal_value + 1) > 0 and (
        (modal_value + 1) & (modal_value)
    ) == 0

    # Saturation rule: modal_mass > 0.90 AND (power2_minus1 OR n_unique <= 3)
    saturated = modal_mass > 0.90 and (is_power2_minus1 or n_unique <= 3)

    return {
        "saturated": saturated,
        "modal_value": modal_value,
        "modal_mass": modal_mass,
        "is_power2_minus1": is_power2_minus1,
        "n_unique": n_unique,
    }
