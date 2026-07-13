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
"""Tests for per-chain init_strategy types (uniform_perchain, zero_perchain).

Tests cover:
- Validation of new per-chain types
- Unit tests for _apply_init_strategy with per-chain semantics
- Legacy bit-for-bit tests to ensure old behavior is unchanged
- Integration tests with warmup replication logic

All tests are @pytest.mark.fast — pure logic / JAX operations without long chains.
"""

import jax
import jax.numpy as jnp
import pytest

from tuningfork.recipes._base import validate_init_strategy
from tuningfork.recipes._recipe_runner import _apply_init_strategy

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Validation tests for per-chain types
# ---------------------------------------------------------------------------


def test_validate_zero_perchain_passes() -> None:
    """{'type': 'zero_perchain'} with default jitter is valid."""
    validate_init_strategy({"type": "zero_perchain"})


def test_validate_zero_perchain_with_jitter_passes() -> None:
    """{'type': 'zero_perchain', 'jitter': 0.3} is valid."""
    validate_init_strategy({"type": "zero_perchain", "jitter": 0.3})


def test_validate_zero_perchain_jitter_zero_passes() -> None:
    """{'type': 'zero_perchain', 'jitter': 0} is valid (no jitter)."""
    validate_init_strategy({"type": "zero_perchain", "jitter": 0.0})


def test_validate_zero_perchain_negative_jitter_raises() -> None:
    """negative jitter raises ValueError."""
    with pytest.raises(ValueError, match="jitter must be >= 0"):
        validate_init_strategy({"type": "zero_perchain", "jitter": -0.1})


def test_validate_uniform_perchain_passes() -> None:
    """{'type': 'uniform_perchain', 'low': -1, 'high': 1} is valid."""
    validate_init_strategy({"type": "uniform_perchain", "low": -1.0, "high": 1.0})


def test_validate_uniform_perchain_missing_low_raises() -> None:
    """uniform_perchain without 'low' raises ValueError."""
    with pytest.raises(ValueError, match="low.*high"):
        validate_init_strategy({"type": "uniform_perchain", "high": 1.0})


def test_validate_uniform_perchain_missing_high_raises() -> None:
    """uniform_perchain without 'high' raises ValueError."""
    with pytest.raises(ValueError, match="low.*high"):
        validate_init_strategy({"type": "uniform_perchain", "low": -1.0})


def test_validate_uniform_perchain_inverted_bounds_raises() -> None:
    """uniform_perchain with low > high raises ValueError."""
    with pytest.raises(ValueError, match="low < high"):
        validate_init_strategy({"type": "uniform_perchain", "low": 1.0, "high": -1.0})


# ---------------------------------------------------------------------------
# Unit tests: _apply_init_strategy with per-chain types
# ---------------------------------------------------------------------------


def test_uniform_perchain_basic_shape() -> None:
    """uniform_perchain produces shape (num_chains, *original_shape)."""
    init_pos = jnp.array([1.0, 2.0, 3.0])
    strategy = {"type": "uniform_perchain", "low": -1.0, "high": 1.0}
    key = jax.random.key(0)
    num_chains = 8

    result = _apply_init_strategy(strategy, init_pos, key, num_chains=num_chains)

    assert result.shape == (num_chains, 3)


def test_uniform_perchain_multivariate_shape() -> None:
    """uniform_perchain on multi-leaf pytree produces (num_chains,) per leaf."""
    init_pos = {"x": jnp.array([1.0, 2.0]), "y": jnp.array([3.0])}
    strategy = {"type": "uniform_perchain", "low": -2.0, "high": 2.0}
    key = jax.random.key(0)
    num_chains = 5

    result = _apply_init_strategy(strategy, init_pos, key, num_chains=num_chains)

    assert isinstance(result, dict)
    assert result["x"].shape == (num_chains, 2)
    assert result["y"].shape == (num_chains, 1)


def test_uniform_perchain_values_in_range() -> None:
    """uniform_perchain draws all values in [low, high]."""
    init_pos = jnp.array([0.0] * 100)
    strategy = {"type": "uniform_perchain", "low": -5.0, "high": 10.0}
    key = jax.random.key(42)
    num_chains = 10

    result = _apply_init_strategy(strategy, init_pos, key, num_chains=num_chains)

    # All values should be in the specified range
    assert jnp.all(result >= -5.0)
    assert jnp.all(result <= 10.0)


def test_uniform_perchain_rows_distinct() -> None:
    """uniform_perchain produces distinct rows (per-chain independence)."""
    init_pos = jnp.ones(50)
    strategy = {"type": "uniform_perchain", "low": -1.0, "high": 1.0}
    key = jax.random.key(123)
    num_chains = 8

    result = _apply_init_strategy(strategy, init_pos, key, num_chains=num_chains)

    # Each row should be different (probability of coincidence is vanishingly small)
    for i in range(num_chains):
        for j in range(i + 1, num_chains):
            assert not jnp.allclose(result[i], result[j])


def test_zero_perchain_basic_shape() -> None:
    """zero_perchain produces shape (num_chains, *original_shape)."""
    init_pos = jnp.array([1.0, 2.0, 3.0])
    strategy = {"type": "zero_perchain"}
    key = jax.random.key(0)
    num_chains = 8

    result = _apply_init_strategy(strategy, init_pos, key, num_chains=num_chains)

    assert result.shape == (num_chains, 3)


def test_zero_perchain_default_jitter() -> None:
    """zero_perchain with default jitter=0.5 produces N(0, 0.5²) draws."""
    init_pos = jnp.zeros(1000)
    strategy = {"type": "zero_perchain"}  # default jitter=0.5
    key = jax.random.key(0)
    num_chains = 100

    result = _apply_init_strategy(strategy, init_pos, key, num_chains=num_chains)

    # Flatten for statistics: shape is (100, 1000), flatten to check N(0, 0.5²)
    flat = result.flatten()
    mean = jnp.mean(flat)
    std = jnp.std(flat)

    # Should be approximately N(0, 0.5²) with N=100000 samples
    assert jnp.abs(mean) < 0.02  # mean ≈ 0
    assert jnp.abs(std - 0.5) < 0.05  # std ≈ 0.5


def test_zero_perchain_custom_jitter() -> None:
    """zero_perchain with custom jitter=0.1 produces N(0, 0.1²) draws."""
    init_pos = jnp.zeros(500)
    strategy = {"type": "zero_perchain", "jitter": 0.1}
    key = jax.random.key(1)
    num_chains = 50

    result = _apply_init_strategy(strategy, init_pos, key, num_chains=num_chains)

    flat = result.flatten()
    mean = jnp.mean(flat)
    std = jnp.std(flat)

    assert jnp.abs(mean) < 0.02  # mean ≈ 0
    assert jnp.abs(std - 0.1) < 0.02  # std ≈ 0.1


def test_zero_perchain_rows_distinct() -> None:
    """zero_perchain produces distinct rows (per-chain independence)."""
    init_pos = jnp.ones(50)
    strategy = {"type": "zero_perchain", "jitter": 0.5}
    key = jax.random.key(456)
    num_chains = 8

    result = _apply_init_strategy(strategy, init_pos, key, num_chains=num_chains)

    # Each row should be different (probability of coincidence is vanishingly small)
    for i in range(num_chains):
        for j in range(i + 1, num_chains):
            assert not jnp.allclose(result[i], result[j])


def test_zero_perchain_zero_jitter_still_distinct() -> None:
    """zero_perchain with jitter=0 produces all-zeros (batch of identical rows)."""
    init_pos = jnp.ones(50)
    strategy = {"type": "zero_perchain", "jitter": 0.0}
    key = jax.random.key(789)
    num_chains = 8

    result = _apply_init_strategy(strategy, init_pos, key, num_chains=num_chains)

    # All rows should be exactly zero (no randomness)
    assert jnp.allclose(result, 0.0)


# ---------------------------------------------------------------------------
# Legacy bit-for-bit tests: old types unchanged
# ---------------------------------------------------------------------------


def test_legacy_uniform_unchanged_with_num_chains_default() -> None:
    """legacy 'uniform' behavior unchanged when num_chains=1 (default)."""
    init_pos = jnp.array([1.0, 2.0, 3.0])
    strategy = {"type": "uniform", "low": -1.0, "high": 1.0}
    key = jax.random.key(42)

    # Old call site (num_chains not passed, defaults to 1)
    result_old = _apply_init_strategy(strategy, init_pos, key)
    # New call site (num_chains=1 explicit)
    result_new = _apply_init_strategy(strategy, init_pos, key, num_chains=1)

    # Should be identical
    assert jnp.allclose(result_old, result_new)
    # And should be a single (non-batched) position
    assert result_old.shape == (3,)


def test_legacy_uniform_same_key_same_output() -> None:
    """legacy 'uniform' reproducibility: same key → same output."""
    init_pos = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    strategy = {"type": "uniform", "low": -2.5, "high": 3.5}
    key = jax.random.key(100)

    result1 = _apply_init_strategy(strategy, init_pos, key)
    result2 = _apply_init_strategy(strategy, init_pos, key)

    assert jnp.allclose(result1, result2)


def test_legacy_zero_unchanged() -> None:
    """legacy 'zero' behavior unchanged."""
    init_pos = jnp.array([1.0, 2.0, 3.0])
    strategy = {"type": "zero"}
    key = jax.random.key(0)

    result = _apply_init_strategy(strategy, init_pos, key)

    assert jnp.allclose(result, 0.0)
    assert result.shape == (3,)


def test_legacy_prior_sample_unchanged() -> None:
    """legacy 'prior_sample' behavior unchanged (identity)."""
    init_pos = jnp.array([1.0, 2.0, 3.0])
    strategy = {"type": "prior_sample"}
    key = jax.random.key(0)

    result = _apply_init_strategy(strategy, init_pos, key)

    assert jnp.allclose(result, init_pos)
    assert result.shape == init_pos.shape


# ---------------------------------------------------------------------------
# Pre-batching detection (integration with warmup's _maybe_replicate)
# ---------------------------------------------------------------------------


def test_uniform_perchain_pre_batched_passthrough() -> None:
    """uniform_perchain produces shape that _maybe_replicate will recognize."""
    from tuningfork.warmup._base import _maybe_replicate

    init_pos = jnp.array([1.0, 2.0, 3.0])
    strategy = {"type": "uniform_perchain", "low": -1.0, "high": 1.0}
    key = jax.random.key(0)
    num_chains = 4

    # Apply per-chain init
    result = _apply_init_strategy(strategy, init_pos, key, num_chains=num_chains)

    # Now replicate it (as the warmup would do)
    replicated = _maybe_replicate(result, num_chains)

    # Should be identical (recognized as pre-batched)
    assert jnp.allclose(replicated, result)
    assert replicated.shape == (num_chains, 3)


def test_zero_perchain_pre_batched_passthrough() -> None:
    """zero_perchain produces shape that _maybe_replicate will recognize."""
    from tuningfork.warmup._base import _maybe_replicate

    init_pos = jnp.array([1.0, 2.0])
    strategy = {"type": "zero_perchain", "jitter": 0.1}
    key = jax.random.key(1)
    num_chains = 8

    result = _apply_init_strategy(strategy, init_pos, key, num_chains=num_chains)
    replicated = _maybe_replicate(result, num_chains)

    # Should be identical (recognized as pre-batched)
    assert jnp.allclose(replicated, result)
    assert replicated.shape == (num_chains, 2)


def test_legacy_uniform_not_pre_batched() -> None:
    """legacy 'uniform' is not pre-batched; _maybe_replicate will replicate it."""
    from tuningfork.warmup._base import _maybe_replicate

    init_pos = jnp.array([1.0, 2.0, 3.0])
    strategy = {"type": "uniform", "low": -1.0, "high": 1.0}
    key = jax.random.key(0)
    num_chains = 4

    result = _apply_init_strategy(strategy, init_pos, key)  # default num_chains=1

    # Replicate it (as the warmup would do)
    replicated = _maybe_replicate(result, num_chains)

    # Should be (num_chains, 3) now, with broadcast
    assert replicated.shape == (num_chains, 3)
    # All rows should be identical (broadcast from single center)
    for i in range(num_chains):
        assert jnp.allclose(replicated[i], result)


# ---------------------------------------------------------------------------
# Warmup compatibility guard (fail-loud for non-ensemble warmups)
# ---------------------------------------------------------------------------


def test_perchain_uniform_pathfinder_raises() -> None:
    """uniform_perchain × pathfinder raises ValueError with clear message."""
    from tuningfork.recipes._recipe_runner import (
        _validate_init_strategy_warmup_compatibility,
    )

    strategy = {"type": "uniform_perchain", "low": -1.0, "high": 1.0}

    with pytest.raises(ValueError, match="ensemble warmups") as exc_info:
        _validate_init_strategy_warmup_compatibility(strategy, "pathfinder")

    msg = str(exc_info.value)
    assert "uniform_perchain" in msg
    assert "pathfinder" in msg
    assert "single-point" in msg
    assert "legacy" in msg


def test_perchain_zero_multipathfinder_raises() -> None:
    """zero_perchain × multipathfinder raises ValueError with clear message."""
    from tuningfork.recipes._recipe_runner import (
        _validate_init_strategy_warmup_compatibility,
    )

    strategy = {"type": "zero_perchain", "jitter": 0.5}

    with pytest.raises(ValueError, match="ensemble warmups") as exc_info:
        _validate_init_strategy_warmup_compatibility(strategy, "multipathfinder")

    msg = str(exc_info.value)
    assert "zero_perchain" in msg
    assert "multipathfinder" in msg


def test_perchain_uniform_chees_passes() -> None:
    """uniform_perchain × chees does NOT raise (compatible)."""
    from tuningfork.recipes._recipe_runner import (
        _validate_init_strategy_warmup_compatibility,
    )

    strategy = {"type": "uniform_perchain", "low": -1.0, "high": 1.0}

    # Must not raise
    _validate_init_strategy_warmup_compatibility(strategy, "chees")


def test_perchain_zero_meads_passes() -> None:
    """zero_perchain × meads does NOT raise (compatible)."""
    from tuningfork.recipes._recipe_runner import (
        _validate_init_strategy_warmup_compatibility,
    )

    strategy = {"type": "zero_perchain", "jitter": 0.2}

    # Must not raise
    _validate_init_strategy_warmup_compatibility(strategy, "meads")


def test_legacy_uniform_pathfinder_passes() -> None:
    """legacy uniform × pathfinder does NOT raise (no incompatibility)."""
    from tuningfork.recipes._recipe_runner import (
        _validate_init_strategy_warmup_compatibility,
    )

    strategy = {"type": "uniform", "low": -1.0, "high": 1.0}

    # Must not raise (legacy types are compatible with all warmups)
    _validate_init_strategy_warmup_compatibility(strategy, "pathfinder")


def test_legacy_zero_multipathfinder_passes() -> None:
    """legacy zero × multipathfinder does NOT raise (no incompatibility)."""
    from tuningfork.recipes._recipe_runner import (
        _validate_init_strategy_warmup_compatibility,
    )

    strategy = {"type": "zero"}

    # Must not raise (legacy types are compatible with all warmups)
    _validate_init_strategy_warmup_compatibility(strategy, "multipathfinder")
