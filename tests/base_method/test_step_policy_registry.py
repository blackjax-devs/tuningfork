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
"""Fast unit tests for the step-policy registry.

Tests cover:
- ``build_step_policy(None)`` → V0 library default callable
- ``build_step_policy({"kind": "uniform_int", ...})`` → correct bounds callable
- ``build_step_policy({"kind": "empirical", ...})`` → NUTS-harvested step_policy
- ``harvest_step_policy_from_chain_stats(path)`` → correct empirical spec from synthetic npz
- Deferred kinds raise ``NotImplementedError``
- Unknown kinds raise ``NotImplementedError``
- Invalid specs raise ``ValueError``

All tests are ``@pytest.mark.fast`` (no JAX compilation / chain execution,
except for the distribution-shape tests which run <100 ms).
"""

from pathlib import Path

import numpy as np
import pytest

from tuningfork.base_method._step_policy_registry import (
    build_step_policy,
    harvest_step_policy_from_chain_stats,
    harvest_step_policy_from_nis,
)

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# V0 — None (library default)
# ---------------------------------------------------------------------------


def test_none_returns_callable() -> None:
    """build_step_policy(None) returns a callable."""
    fn = build_step_policy(None)
    assert callable(fn)


def test_none_produces_randint_in_range() -> None:
    """V0 default fn(key) returns an integer in [1, 10) over many samples."""
    import jax
    import jax.numpy as jnp

    fn = build_step_policy(None)
    key = jax.random.key(0)
    keys = jax.random.split(key, 200)
    samples = jnp.array([fn(k) for k in keys])
    assert int(jnp.min(samples)) >= 1
    assert int(jnp.max(samples)) < 10


# ---------------------------------------------------------------------------
# uniform_int — V0/V1/V2 parametric path
# ---------------------------------------------------------------------------


def test_uniform_int_v0_explicit() -> None:
    """Explicit V0 spec {"kind": "uniform_int", "low": 1, "high": 10} is a callable."""
    fn = build_step_policy({"kind": "uniform_int", "low": 1, "high": 10})
    assert callable(fn)


def test_uniform_int_v0_samples_in_range() -> None:
    """V0 explicit uniform_int samples in [1, 10) over many draws."""
    import jax
    import jax.numpy as jnp

    fn = build_step_policy({"kind": "uniform_int", "low": 1, "high": 10})
    key = jax.random.key(42)
    keys = jax.random.split(key, 200)
    samples = jnp.array([fn(k) for k in keys])
    assert int(jnp.min(samples)) >= 1
    assert int(jnp.max(samples)) < 10


def test_uniform_int_v2_long_trajectory() -> None:
    """V2 spec (low=50, high=200) samples exclusively in [50, 200) over many draws."""
    import jax
    import jax.numpy as jnp

    fn = build_step_policy({"kind": "uniform_int", "low": 50, "high": 200})
    key = jax.random.key(7)
    keys = jax.random.split(key, 200)
    samples = jnp.array([fn(k) for k in keys])
    assert int(jnp.min(samples)) >= 50
    assert int(jnp.max(samples)) < 200


def test_uniform_int_v1_medium_trajectory() -> None:
    """V1 spec (low=5, high=50) returns values in [5, 50) over many draws."""
    import jax
    import jax.numpy as jnp

    fn = build_step_policy({"kind": "uniform_int", "low": 5, "high": 50})
    key = jax.random.key(13)
    keys = jax.random.split(key, 200)
    samples = jnp.array([fn(k) for k in keys])
    assert int(jnp.min(samples)) >= 5
    assert int(jnp.max(samples)) < 50


def test_uniform_int_low_equals_high_raises_value_error() -> None:
    """uniform_int with low == high raises ValueError."""
    with pytest.raises(ValueError, match="low < high"):
        build_step_policy({"kind": "uniform_int", "low": 10, "high": 10})


def test_uniform_int_low_gt_high_raises_value_error() -> None:
    """uniform_int with low > high raises ValueError."""
    with pytest.raises(ValueError, match="low < high"):
        build_step_policy({"kind": "uniform_int", "low": 20, "high": 10})


def test_uniform_int_missing_low_raises_value_error() -> None:
    """uniform_int missing 'low' key raises ValueError."""
    with pytest.raises(ValueError, match="'low' and 'high'"):
        build_step_policy({"kind": "uniform_int", "high": 10})


def test_uniform_int_missing_high_raises_value_error() -> None:
    """uniform_int missing 'high' key raises ValueError."""
    with pytest.raises(ValueError, match="'low' and 'high'"):
        build_step_policy({"kind": "uniform_int", "low": 1})


# ---------------------------------------------------------------------------
# empirical (V7) — Phase B
# ---------------------------------------------------------------------------


def test_empirical_returns_callable() -> None:
    """build_step_policy with empirical spec returns a callable."""
    spec = {"kind": "empirical", "values": [60, 80, 100], "weights": [0.3, 0.5, 0.2]}
    fn = build_step_policy(spec)
    assert callable(fn)


def test_empirical_samples_only_in_values() -> None:
    """empirical fn(key) only returns values that appear in spec['values']."""
    import jax
    import jax.numpy as jnp

    values = [60, 80, 100, 120]
    weights = [0.25, 0.25, 0.25, 0.25]
    spec = {"kind": "empirical", "values": values, "weights": weights}
    fn = build_step_policy(spec)
    key = jax.random.key(42)
    keys = jax.random.split(key, 500)
    samples = jnp.array([fn(k) for k in keys])
    values_set = set(values)
    for s in np.asarray(samples):
        assert int(s) in values_set, f"sample {s} not in values {values_set}"


def test_empirical_distribution_matches_weights() -> None:
    """empirical fn samples match input weights within Monte-Carlo tolerance.

    Uses 10k samples and checks that empirical frequencies are within
    3 * sqrt(p*(1-p)/n) of the target probabilities (roughly 3-sigma).
    """
    import jax
    import jax.numpy as jnp

    values = [10, 20, 30, 40]
    weights = [0.1, 0.4, 0.4, 0.1]
    spec = {"kind": "empirical", "values": values, "weights": weights}
    fn = build_step_policy(spec)

    n = 10_000
    key = jax.random.key(0)
    keys = jax.random.split(key, n)
    samples = np.asarray(jnp.array([fn(k) for k in keys]))

    for v, w in zip(values, weights):
        empirical_freq = np.sum(samples == v) / n
        tol = 3.0 * np.sqrt(w * (1 - w) / n)
        assert abs(empirical_freq - w) < tol, (
            f"value {v}: empirical_freq={empirical_freq:.4f} expected={w:.4f} "
            f"(3-sigma tol={tol:.4f})"
        )


def test_empirical_missing_values_raises_value_error() -> None:
    """empirical spec missing 'values' raises ValueError."""
    with pytest.raises(ValueError, match="'values' and 'weights'"):
        build_step_policy({"kind": "empirical", "weights": [1.0]})


def test_empirical_missing_weights_raises_value_error() -> None:
    """empirical spec missing 'weights' raises ValueError."""
    with pytest.raises(ValueError, match="'values' and 'weights'"):
        build_step_policy({"kind": "empirical", "values": [5]})


def test_empirical_mismatched_lengths_raises_value_error() -> None:
    """empirical spec with mismatched values/weights lengths raises ValueError."""
    with pytest.raises(ValueError, match="same length"):
        build_step_policy(
            {"kind": "empirical", "values": [1, 2, 3], "weights": [0.5, 0.5]}
        )


def test_empirical_empty_values_raises_value_error() -> None:
    """empirical spec with empty values raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        build_step_policy({"kind": "empirical", "values": [], "weights": []})


# ---------------------------------------------------------------------------
# harvest_step_policy_from_nis — Path B (raw array variant)
# ---------------------------------------------------------------------------


def test_harvest_step_policy_from_nis_basic() -> None:
    """harvest_step_policy_from_nis returns correct spec from a plain array."""
    nis = np.array([5] * 100 + [10] * 200 + [15] * 100, dtype=np.int32)
    spec = harvest_step_policy_from_nis(nis)
    assert spec["kind"] == "empirical"
    assert set(spec["values"]) == {5, 10, 15}
    assert abs(sum(spec["weights"]) - 1.0) < 1e-6
    idx5 = spec["values"].index(5)
    idx10 = spec["values"].index(10)
    idx15 = spec["values"].index(15)
    assert abs(spec["weights"][idx5] - 0.25) < 1e-6
    assert abs(spec["weights"][idx10] - 0.50) < 1e-6
    assert abs(spec["weights"][idx15] - 0.25) < 1e-6


def test_harvest_step_policy_from_nis_2d() -> None:
    """harvest_step_policy_from_nis handles 2D (num_chains, n_warmup) arrays."""
    nis = np.full((4, 1000), 87, dtype=np.int32)
    spec = harvest_step_policy_from_nis(nis)
    assert spec["kind"] == "empirical"
    assert spec["values"] == [87]
    assert abs(spec["weights"][0] - 1.0) < 1e-6


def test_harvest_step_policy_from_nis_empty_raises() -> None:
    """harvest_step_policy_from_nis raises ValueError for empty array."""
    with pytest.raises(ValueError, match="empty"):
        harvest_step_policy_from_nis(np.array([], dtype=np.int32))


def test_harvest_step_policy_from_nis_all_zero_raises() -> None:
    """harvest_step_policy_from_nis raises ValueError for all-zero NIS."""
    with pytest.raises(ValueError, match="degenerate chain"):
        harvest_step_policy_from_nis(np.zeros(100, dtype=np.int32))


# ---------------------------------------------------------------------------
# harvest_step_policy_from_chain_stats — Path A (file-based variant)
# ---------------------------------------------------------------------------


def test_harvest_step_policy_from_chain_stats_basic(tmp_path: Path) -> None:
    """harvest_step_policy_from_chain_stats returns correct kind='empirical' spec from synthetic npz."""
    # Create synthetic chain_stats.npz with known NIS distribution
    # NIS values: [5]*100 + [10]*200 + [15]*100 → weights 0.25, 0.50, 0.25
    nis = np.array([5] * 100 + [10] * 200 + [15] * 100, dtype=np.int32)
    stats_path = tmp_path / "test.chain_stats.npz"
    np.savez(str(stats_path), num_integration_steps=nis)

    spec = harvest_step_policy_from_chain_stats(stats_path)
    assert spec["kind"] == "empirical"
    assert set(spec["values"]) == {5, 10, 15}
    # Check weights sum to 1
    assert abs(sum(spec["weights"]) - 1.0) < 1e-6
    # Check individual weights
    idx5 = spec["values"].index(5)
    idx10 = spec["values"].index(10)
    idx15 = spec["values"].index(15)
    assert abs(spec["weights"][idx5] - 0.25) < 1e-6
    assert abs(spec["weights"][idx10] - 0.50) < 1e-6
    assert abs(spec["weights"][idx15] - 0.25) < 1e-6


def test_harvest_step_policy_from_chain_stats_multichain_shape(tmp_path: Path) -> None:
    """harvest_step_policy_from_chain_stats handles 2D (num_chains, n_samples) arrays via ravel."""
    # Simulate 4 chains x 1000 samples shape
    nis = np.full((4, 1000), 87, dtype=np.int32)
    nis[0, :500] = 60  # vary first chain to get two values
    stats_path = tmp_path / "mc.chain_stats.npz"
    np.savez(str(stats_path), num_integration_steps=nis)

    spec = harvest_step_policy_from_chain_stats(stats_path)
    assert spec["kind"] == "empirical"
    assert 87 in spec["values"]
    assert abs(sum(spec["weights"]) - 1.0) < 1e-6


def test_harvest_step_policy_from_chain_stats_max_values_binning(
    tmp_path: Path,
) -> None:
    """harvest_step_policy_from_chain_stats bins when distinct values exceed max_values."""
    # Create NIS distribution with 100 distinct values (1..100)
    nis = np.repeat(np.arange(1, 101, dtype=np.int32), 10)  # each value 10 times
    stats_path = tmp_path / "wide.chain_stats.npz"
    np.savez(str(stats_path), num_integration_steps=nis)

    # max_values=5 forces histogram binning
    spec = harvest_step_policy_from_chain_stats(stats_path, max_values=5)
    assert spec["kind"] == "empirical"
    assert len(spec["values"]) <= 5
    assert abs(sum(spec["weights"]) - 1.0) < 1e-6


def test_harvest_step_policy_from_chain_stats_missing_key_raises(
    tmp_path: Path,
) -> None:
    """harvest_step_policy_from_chain_stats raises KeyError when both NIS key variants missing."""
    stats_path = tmp_path / "bad.npz"
    np.savez(str(stats_path), some_other_key=np.array([1, 2, 3]))
    with pytest.raises(KeyError, match="num_integration_steps"):
        harvest_step_policy_from_chain_stats(stats_path)


def test_harvest_step_policy_from_chain_stats_n_steps_fallback(tmp_path: Path) -> None:
    """harvest_step_policy_from_chain_stats falls back to 'n_steps' key for older cache format."""
    # Older caches use 'n_steps' instead of 'num_integration_steps'
    nis = np.array([30] * 200 + [50] * 100 + [70] * 100, dtype=np.int32)
    stats_path = tmp_path / "old_format.chain_stats.npz"
    np.savez(str(stats_path), n_steps=nis)  # old key name

    spec = harvest_step_policy_from_chain_stats(stats_path)
    assert spec["kind"] == "empirical"
    assert set(spec["values"]) == {30, 50, 70}
    assert abs(sum(spec["weights"]) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Deferred kinds raise NotImplementedError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    ["log_uniform_int", "poisson", "pow2_choice"],
)
def test_deferred_kind_raises_not_implemented(kind: str) -> None:
    """Deferred kinds raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="deferred to future work"):
        build_step_policy({"kind": kind})


# ---------------------------------------------------------------------------
# Unknown kinds
# ---------------------------------------------------------------------------


def test_unknown_kind_raises_not_implemented() -> None:
    """An unrecognised kind raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        build_step_policy({"kind": "totally_unknown_kind"})
