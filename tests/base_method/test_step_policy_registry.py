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
"""Fast unit tests for the step-policy registry (Phase A scope).

Tests cover:
- ``build_step_policy(None)`` → V0 library default callable
- ``build_step_policy({"kind": "uniform_int", ...})`` → correct bounds callable
- Phase-B deferred kinds raise ``NotImplementedError``
- Unknown kinds raise ``NotImplementedError``
- Invalid ``uniform_int`` specs raise ``ValueError``

All tests are ``@pytest.mark.fast`` (no JAX compilation / chain execution).
"""

import pytest

from tuningfork.base_method._step_policy_registry import build_step_policy

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
# Phase-B deferred kinds raise NotImplementedError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    ["log_uniform_int", "poisson", "pow2_choice", "empirical"],
)
def test_deferred_kind_raises_not_implemented(kind: str) -> None:
    """Phase-B deferred kinds raise NotImplementedError with 'deferred to Phase B'."""
    with pytest.raises(NotImplementedError, match="deferred to Phase B"):
        build_step_policy({"kind": kind})


# ---------------------------------------------------------------------------
# Unknown kinds
# ---------------------------------------------------------------------------


def test_unknown_kind_raises_not_implemented() -> None:
    """An unrecognised kind raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        build_step_policy({"kind": "totally_unknown_kind"})
