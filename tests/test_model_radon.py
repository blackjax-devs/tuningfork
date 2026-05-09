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
"""Tests for the radon model (390-D NCP varying-intercept hierarchical, radon_all).

Tests
-----
1. test_dim               : MODELS['radon'].dim == 390
2. test_data_shape        : N_COUNTIES == 386 and len(LOG_RADON) >= 919
3. test_county_idx_range  : COUNTY_IDX.min() == 0 and COUNTY_IDX.max() == 385
4. test_logdensity_finite : logdensity_fn returns finite float at zeros init
5. test_posteriordb_id_set: MODELS['radon'].posteriordb_id == 'radon-radon_hierarchical_centered'

Notes
-----
All tests are @pytest.mark.fast (no MCMC).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bjx_bench.model import MODELS, build_logdensity_fn
from bjx_bench.model.hierarchical.radon import (
    COUNTY_IDX,
    DIM,
    ENTRY,
    FLOOR_X,
    LOG_RADON,
    LOG_URANIUM,
    N_COUNTIES,
)

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Test 1: dim
# ---------------------------------------------------------------------------


def test_dim() -> None:
    """MODELS['radon'].dim must equal 390."""
    assert MODELS["radon"].dim == 390
    assert ENTRY.dim == DIM == 390


# ---------------------------------------------------------------------------
# Test 2: data shape
# ---------------------------------------------------------------------------


def test_data_shape() -> None:
    """N_COUNTIES must be 386 and len(LOG_RADON) >= 919."""
    assert N_COUNTIES == 386, f"Expected N_COUNTIES=386, got {N_COUNTIES}"
    assert len(LOG_RADON) >= 919, f"Expected len(LOG_RADON)>=919, got {len(LOG_RADON)}"
    # Verify consistent lengths across arrays
    assert len(FLOOR_X) == len(
        LOG_RADON
    ), f"FLOOR_X length {len(FLOOR_X)} != LOG_RADON length {len(LOG_RADON)}"
    assert len(COUNTY_IDX) == len(
        LOG_RADON
    ), f"COUNTY_IDX length {len(COUNTY_IDX)} != LOG_RADON length {len(LOG_RADON)}"
    assert len(LOG_URANIUM) == len(
        LOG_RADON
    ), f"LOG_URANIUM length {len(LOG_URANIUM)} != LOG_RADON length {len(LOG_RADON)}"


# ---------------------------------------------------------------------------
# Test 3: county_idx range
# ---------------------------------------------------------------------------


def test_county_idx_range() -> None:
    """COUNTY_IDX must be 0-indexed: min==0 and max==385."""
    ci = np.asarray(COUNTY_IDX)
    assert ci.min() == 0, f"Expected county_idx min=0, got {ci.min()}"
    assert ci.max() == 385, f"Expected county_idx max=385, got {ci.max()}"


# ---------------------------------------------------------------------------
# Test 4: logdensity finite at zeros init
# ---------------------------------------------------------------------------


def test_logdensity_finite() -> None:
    """build_logdensity_fn must return finite log-density at the zeros init."""
    key = jax.random.key(0)
    init_pos, logdensity_fn, _ = build_logdensity_fn(key, ENTRY)
    ld = logdensity_fn(init_pos)
    assert jnp.isfinite(ld), f"Expected finite log-density at init, got {ld}"


# ---------------------------------------------------------------------------
# Test 5: posteriordb_id set
# ---------------------------------------------------------------------------


def test_posteriordb_id_set() -> None:
    """MODELS['radon'].posteriordb_id must equal 'radon-radon_hierarchical_centered'."""
    assert MODELS["radon"].posteriordb_id == "radon-radon_hierarchical_centered", (
        f"Expected posteriordb_id='radon-radon_hierarchical_centered', "
        f"got {MODELS['radon'].posteriordb_id!r}"
    )
