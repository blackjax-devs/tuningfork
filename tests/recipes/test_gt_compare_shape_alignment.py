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
"""Regression tests for structured-event-shape ground-truth alignment.

A ground-truth summary may store a parameter flattened (``(1600,)``) while the
sampler returns it in its structured event shape (``(40, 40)`` for a 2-D grid
model).  ``_compute_gt_compare`` must align the two before differencing.

The three tests below are ordered by what they can catch:

1. ``test_..._no_broadcast_error`` — the crash guard.  Fails with
   ``ValueError: operands could not be broadcast together`` when GT arrays are
   left flat.
2. ``test_..._preserves_dim_order`` — the alignment guard.  A crash guard alone
   is satisfied by *any* reshape; this one fails if the flat→structured mapping
   permutes dimensions, because the planted GT offset would land on the wrong
   grid cell.
3. ``test_..._legacy_single_chain_path`` — same alignment on the legacy
   ``summary.json`` branch (no ``between_chain_se``), which takes a different
   code path for ``se_gt``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.fast

GRID = (5, 8)
D = GRID[0] * GRID[1]


def _grid_samples(
    n_chains: int = 2,
    n_draws: int = 400,
    *,
    offset: float = 0.0,
    offset_flat_idx: int | None = None,
) -> dict[str, np.ndarray]:
    """Samples shaped ``(n_chains, n_draws, *GRID)``, optionally biased at one cell."""
    rng = np.random.default_rng(7)
    arr = rng.normal(0.0, 1.0, (n_chains, n_draws, *GRID))
    if offset_flat_idx is not None:
        cell = np.unravel_index(offset_flat_idx, GRID)
        arr[:, :, cell[0], cell[1]] += offset
    return {"z": arr}


def _flat_gt(
    *,
    multichain: bool,
    mean_offset: float = 0.0,
    offset_flat_idx: int | None = None,
) -> dict[str, dict]:
    """Ground truth stored FLAT — ``(D,)`` — as ``lgcp``'s summary_v2 does."""
    mean = np.zeros(D)
    if offset_flat_idx is not None:
        mean[offset_flat_idx] = mean_offset
    gt: dict = {"mean": mean.tolist(), "std": np.ones(D).tolist()}
    if multichain:
        gt["between_chain_se"] = (np.full(D, 0.002)).tolist()
        gt["bulk_ess"] = (np.full(D, 40000.0)).tolist()
        gt["n_total"] = 40000
    else:
        gt["n_samples"] = 40000
    return {"z": gt}


def test_gt_compare_structured_event_shape_no_broadcast_error() -> None:
    """Flat GT vs structured samples must not raise on the multichain path.

    Without shape alignment this raises
    ``ValueError: operands could not be broadcast together with shapes (5,8) (40,)``
    at the pooled-SE denominator.
    """
    from tuningfork.calibration._gate.gt_compare import _compute_gt_compare

    mc = _grid_samples()
    gt = _flat_gt(multichain=True)

    result = _compute_gt_compare(mc, gt, min_bulk_ess=None)

    assert result.max_abs_mean_z is not None
    assert np.isfinite(result.max_abs_mean_z)
    assert result.n_dims == D, (
        f"every grid cell must contribute one z-score; got n_dims={result.n_dims}"
    )
    assert result.calibrated_D_total == D


def test_gt_compare_structured_event_shape_preserves_dim_order() -> None:
    """The flat→structured mapping must be element-preserving, not merely shaped.

    Positive control: GT carries a large mean offset at flat index ``k`` and the
    samples carry the same offset at grid cell ``unravel_index(k, GRID)``.  A
    C-order reshape cancels them, so every z stays small and the gate PASSes.

    Negative control: move the sample-side offset to a different cell — the
    mismatch must show up as a hard FAIL.  Without it, the positive control
    would also pass for a gate that ignores GT entirely.
    """
    from tuningfork.calibration._gate.gt_compare import _compute_gt_compare

    k = 30  # unravel_index(30, (5, 8)) == (3, 6)
    offset = 25.0

    aligned = _compute_gt_compare(
        _grid_samples(offset=offset, offset_flat_idx=k),
        _flat_gt(multichain=True, mean_offset=offset, offset_flat_idx=k),
        min_bulk_ess=None,
    )
    assert aligned.calibrated_pass is True, (
        "GT offset at flat index k and sample offset at unravel_index(k) must "
        f"cancel; got max_z={aligned.max_abs_mean_z}, "
        f"n_fail={aligned.calibrated_n_fail}"
    )
    assert aligned.calibrated_n_fail == 0

    misaligned = _compute_gt_compare(
        _grid_samples(offset=offset, offset_flat_idx=k + 1),
        _flat_gt(multichain=True, mean_offset=offset, offset_flat_idx=k),
        min_bulk_ess=None,
    )
    assert misaligned.calibrated_pass is False, (
        "a one-cell misalignment must be detected as a hard FAIL"
    )


def test_gt_compare_structured_event_shape_legacy_single_chain_path() -> None:
    """Same alignment on the legacy summary.json branch (no between_chain_se)."""
    from tuningfork.calibration._gate.gt_compare import _compute_gt_compare

    k = 12
    offset = 25.0

    result = _compute_gt_compare(
        _grid_samples(offset=offset, offset_flat_idx=k),
        _flat_gt(multichain=False, mean_offset=offset, offset_flat_idx=k),
        min_bulk_ess=None,
    )

    assert result.max_abs_mean_z is not None
    assert np.isfinite(result.max_abs_mean_z)
    assert result.calibrated_pass is True
    assert result.n_dims == D
