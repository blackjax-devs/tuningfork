"""Tests for the stoch_vol model (P4.9, Block D: 503-D NCP AR(1) state-space).

All tests are marked ``fast`` — no chain sampling, only structural checks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bjx_bench.model import MODELS, build_logdensity_fn
from bjx_bench.model.latent_gaussian.stoch_vol import ENTRY, RETURNS, T_LENGTH

pytestmark = pytest.mark.fast


def test_dim() -> None:
    """Registry entry has dim == 503 (mu+phi+log_sigma+h_raw[500])."""
    assert MODELS["stoch_vol"].dim == 503
    assert ENTRY.dim == 503


def test_data_shape() -> None:
    """RETURNS has shape (500,) and T_LENGTH == 500."""
    assert RETURNS.shape == (500,)
    assert T_LENGTH == 500


def test_returns_realistic_scale() -> None:
    """Synthetic returns are in daily-financial-vol range (~0.005–0.03 mean abs).

    mu_true = -10 → exp(mu/2) ≈ 0.0067, so mean_abs should be well below 0.1.
    """
    mean_abs = float(np.abs(np.array(RETURNS)).mean())
    assert (
        mean_abs < 0.1
    ), f"mean_abs={mean_abs:.4f} unexpectedly large (expected < 0.1)"


def test_logdensity_finite() -> None:
    """Log-density is finite at the NumPyro-initialised position."""
    key = jax.random.key(0)
    init_pos, logdensity_fn, _ = build_logdensity_fn(key, ENTRY)
    ld = logdensity_fn(init_pos)
    assert jnp.isfinite(ld), f"Expected finite log-density at init, got {ld}"


def test_posteriordb_id_none() -> None:
    """posteriordb_id is None — no upstream reference draws available."""
    assert MODELS["stoch_vol"].posteriordb_id is None
