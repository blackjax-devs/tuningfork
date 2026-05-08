"""Tests for bjx_bench.metrics.grad_counter.total_grad_evals.

The key empirical question (documented here for T2.5 handoff):
    Does jax.vmap over a BlackJAX-style NamedTuple-of-Arrays work
    cleanly without unwrapping?

Answer: YES.  jax.vmap maps over the *leading axis* of every pytree
leaf.  A NamedTuple is a valid JAX pytree.  So vmap(fn)(namedtuple)
delivers a NamedTuple with scalar fields to fn on each call, exactly as
expected.  No manual unpacking is needed.

Tests mirror the three grad-cost patterns in the algorithm zoo:
- HMC/NUTS: variable cost = info.num_integration_steps  → sum = 5050
- MALA/Barker: constant cost = 1                         → sum = n_samples
- RWM: zero cost                                         → sum = 0
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import pytest

from bjx_bench.metrics.grad_counter import total_grad_evals

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Fake info NamedTuple that mimics what BlackJAX produces after
# run_inference_algorithm: every field is an Array of shape (n_samples,).
# ---------------------------------------------------------------------------


class FakeHMCInfo(NamedTuple):
    """Minimal stand-in for blackjax HMCInfo with a single Array field."""

    num_integration_steps: jnp.ndarray


class FakeConstantInfo(NamedTuple):
    """Minimal stand-in for MALA/RWM info where per-step count is constant."""

    accepted: jnp.ndarray  # boolean accept flag; not used in cost but realistic


# ---------------------------------------------------------------------------
# Test: HMC-like variable cost (sum 1..100 = 5050)
# ---------------------------------------------------------------------------


class TestTotalGradEvalsHMCLike:
    """Variable cost per step — num_integration_steps changes each step."""

    def test_hmc_sum_formula(self) -> None:
        """Steps [1, 2, ..., 100] → total = 5050."""
        n_samples = 100
        infos = FakeHMCInfo(
            num_integration_steps=jnp.arange(1, n_samples + 1, dtype=jnp.int32)
        )
        result = total_grad_evals(infos, lambda i: i.num_integration_steps)
        assert result == 5050, f"Expected 5050, got {result}"

    def test_hmc_single_step(self) -> None:
        """Single step with 7 leapfrog integrations → total = 7."""
        infos = FakeHMCInfo(num_integration_steps=jnp.array([7], dtype=jnp.int32))
        result = total_grad_evals(infos, lambda i: i.num_integration_steps)
        assert result == 7

    def test_hmc_constant_steps(self) -> None:
        """All steps use the same leapfrog count → total = n × steps."""
        n_samples = 50
        n_leapfrog = 10
        infos = FakeHMCInfo(
            num_integration_steps=jnp.full((n_samples,), n_leapfrog, dtype=jnp.int32)
        )
        result = total_grad_evals(infos, lambda i: i.num_integration_steps)
        assert result == n_samples * n_leapfrog


# ---------------------------------------------------------------------------
# Test: MALA-like constant 1 grad/step
# ---------------------------------------------------------------------------


class TestTotalGradEvalsMalaLike:
    """Constant cost of 1 grad/step — return value is always n_samples."""

    def test_mala_constant_one(self) -> None:
        n_samples = 200
        infos = FakeConstantInfo(accepted=jnp.ones((n_samples,), dtype=jnp.bool_))
        result = total_grad_evals(infos, lambda i: 1)
        assert result == n_samples

    def test_mala_single_sample(self) -> None:
        infos = FakeConstantInfo(accepted=jnp.array([True]))
        result = total_grad_evals(infos, lambda i: 1)
        assert result == 1

    def test_mala_returns_python_int(self) -> None:
        """total_grad_evals must return a plain Python int, not a JAX Array."""
        infos = FakeConstantInfo(accepted=jnp.ones((10,), dtype=jnp.bool_))
        result = total_grad_evals(infos, lambda i: 1)
        assert isinstance(
            result, int
        ), f"Expected Python int, got {type(result).__name__}"


# ---------------------------------------------------------------------------
# Test: RWM-like zero grads
# ---------------------------------------------------------------------------


class TestTotalGradEvalsRWMLike:
    """Gradient-free kernel — always returns 0 regardless of chain length."""

    def test_rwm_zero(self) -> None:
        n_samples = 500
        infos = FakeConstantInfo(accepted=jnp.ones((n_samples,), dtype=jnp.bool_))
        result = total_grad_evals(infos, lambda i: 0)
        assert result == 0

    def test_rwm_zero_is_int(self) -> None:
        infos = FakeConstantInfo(accepted=jnp.ones((1,), dtype=jnp.bool_))
        result = total_grad_evals(infos, lambda i: 0)
        assert isinstance(result, int)
        assert result == 0
