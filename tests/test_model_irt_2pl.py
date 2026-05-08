"""Tests for the irt_2pl model (144-D NCP 2PL IRT, J=100 students, I=20 items).

Tests
-----
1. test_dim               : MODELS['irt_2pl'].dim == 144
2. test_data_shape        : N_STUDENTS==100 and N_ITEMS==20; total responses=2000
3. test_response_binary   : RESPONSE values are exactly {0, 1}
4. test_logdensity_finite : logdensity_fn returns finite float at zeros init
5. test_posteriordb_id_none: MODELS['irt_2pl'].posteriordb_id is None (no xcheck)

Notes
-----
All tests are @pytest.mark.fast (no MCMC).
posteriordb_id is explicitly None: 'irt_2pl-irt_2pl' has no reference posterior
draws upstream. Tier-A uses Long-NUTS self-check only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bjx_bench.model import MODELS, build_logdensity_fn
from bjx_bench.model.hierarchical.irt_2pl import (
    DIM,
    ENTRY,
    N_ITEMS,
    N_STUDENTS,
    RESPONSE,
)

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Test 1: dim
# ---------------------------------------------------------------------------


def test_dim() -> None:
    """MODELS['irt_2pl'].dim must equal 144."""
    assert MODELS["irt_2pl"].dim == 144
    assert ENTRY.dim == DIM == 144


# ---------------------------------------------------------------------------
# Test 2: data shape
# ---------------------------------------------------------------------------


def test_data_shape() -> None:
    """N_STUDENTS==100 and N_ITEMS==20; total responses must equal 2000."""
    assert N_STUDENTS == 100, f"Expected N_STUDENTS=100, got {N_STUDENTS}"
    assert N_ITEMS == 20, f"Expected N_ITEMS=20, got {N_ITEMS}"
    assert RESPONSE.shape == (
        100,
        20,
    ), f"Expected RESPONSE shape (100, 20), got {RESPONSE.shape}"
    assert N_STUDENTS * N_ITEMS == 2000, "Total responses must be 2000"


# ---------------------------------------------------------------------------
# Test 3: response binary
# ---------------------------------------------------------------------------


def test_response_binary() -> None:
    """RESPONSE values must be exactly {0, 1}."""
    vals = set(np.unique(np.asarray(RESPONSE)).tolist())
    assert vals == {0.0, 1.0} or vals == {
        0,
        1,
    }, f"Expected response values {{0, 1}}, got {vals}"


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
# Test 5: posteriordb_id is None
# ---------------------------------------------------------------------------


def test_posteriordb_id_none() -> None:
    """MODELS['irt_2pl'].posteriordb_id must be None.

    'irt_2pl-irt_2pl' has reference_posterior_name: null in posteriordb metadata.
    There are no Stan reference draws available for cross-checking.
    Tier-A uses Long-NUTS self-check (split-R̂) only.
    """
    assert (
        MODELS["irt_2pl"].posteriordb_id is None
    ), f"Expected posteriordb_id=None, got {MODELS['irt_2pl'].posteriordb_id!r}"
