"""Tests for build_logdensity_fn — the NumPyro → BlackJAX bridge."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

from bjx_bench.model._base import Posterior
from bjx_bench.model._numpyro import build_logdensity_fn

pytestmark = pytest.mark.fast


def _mvn_model():
    numpyro.sample("x", dist.Normal(jnp.zeros(3), jnp.ones(3)))


def _hierarchical_model(y, sigma):
    mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
    tau = numpyro.sample("tau", dist.HalfNormal(5.0))
    theta_raw = numpyro.sample("theta_raw", dist.Normal(jnp.zeros(len(y)), 1.0))
    theta = mu + tau * theta_raw
    numpyro.sample("y", dist.Normal(theta, sigma), obs=y)


MVN_ENTRY = Posterior(
    name="test_mvn_3",
    dim=3,
    class_="gaussian",
    numpyro_model=_mvn_model,
)

HIER_ENTRY = Posterior(
    name="test_hier",
    dim=4,
    class_="hierarchical",
    numpyro_model=_hierarchical_model,
    model_args=(jnp.array([1.0, 2.0]), jnp.array([0.5, 0.5])),
)


class TestBuildLogdensityFn:
    def test_returns_three_tuple(self):
        key = jax.random.key(0)
        result = build_logdensity_fn(key, MVN_ENTRY)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_init_position_is_dict(self):
        key = jax.random.key(0)
        init_pos, logdensity_fn, postprocess_fn = build_logdensity_fn(key, MVN_ENTRY)
        assert isinstance(init_pos, dict)
        assert "x" in init_pos

    def test_logdensity_fn_returns_finite_scalar(self):
        key = jax.random.key(1)
        init_pos, logdensity_fn, postprocess_fn = build_logdensity_fn(key, MVN_ENTRY)
        logdensity = logdensity_fn(init_pos)
        assert jnp.isfinite(logdensity), f"Expected finite logdensity, got {logdensity}"
        assert logdensity.shape == (), "Expected scalar logdensity"

    def test_logdensity_at_mode_is_maximum(self):
        """Standard normal should have maximum logdensity at x=0."""
        key = jax.random.key(2)
        _, logdensity_fn, _ = build_logdensity_fn(key, MVN_ENTRY)
        # logdensity at mode (0,0,0) should exceed logdensity at (2,2,2)
        at_mode = logdensity_fn({"x": jnp.zeros(3)})
        away_from_mode = logdensity_fn({"x": jnp.array([2.0, 2.0, 2.0])})
        assert (
            at_mode > away_from_mode
        ), f"Expected logdensity at mode > away: {at_mode:.3f} vs {away_from_mode:.3f}"

    def test_logdensity_fn_is_differentiable(self):
        key = jax.random.key(3)
        init_pos, logdensity_fn, _ = build_logdensity_fn(key, MVN_ENTRY)
        grad = jax.grad(logdensity_fn)(init_pos)
        assert isinstance(grad, dict)
        assert "x" in grad
        assert jnp.all(jnp.isfinite(grad["x"]))

    def test_postprocess_fn_callable(self):
        key = jax.random.key(4)
        init_pos, _, postprocess_fn = build_logdensity_fn(key, MVN_ENTRY)
        constrained = postprocess_fn(init_pos)
        assert isinstance(constrained, dict)

    def test_hierarchical_model_multiple_sites(self):
        key = jax.random.key(5)
        init_pos, logdensity_fn, _ = build_logdensity_fn(key, HIER_ENTRY)
        # Should have sites: mu, tau (or log_tau in unconstrained), theta_raw
        assert isinstance(init_pos, dict)
        assert "mu" in init_pos
        assert "theta_raw" in init_pos
        logdensity = logdensity_fn(init_pos)
        assert jnp.isfinite(logdensity)

    def test_logdensity_fn_jit_compatible(self):
        key = jax.random.key(6)
        init_pos, logdensity_fn, _ = build_logdensity_fn(key, MVN_ENTRY)
        jitted = jax.jit(logdensity_fn)
        result = jitted(init_pos)
        assert jnp.isfinite(result)

    def test_different_keys_give_different_init_positions(self):
        key1 = jax.random.key(0)
        key2 = jax.random.key(42)
        pos1, _, _ = build_logdensity_fn(key1, MVN_ENTRY)
        pos2, _, _ = build_logdensity_fn(key2, MVN_ENTRY)
        # Different keys should give different init positions
        assert not jnp.allclose(pos1["x"], pos2["x"])

    def test_negative_potential(self):
        """Verify logdensity_fn = -potential_fn (positive log density)."""
        key = jax.random.key(7)
        from numpyro.infer.util import initialize_model

        model_info = initialize_model(
            key,
            MVN_ENTRY.numpyro_model,
            model_args=MVN_ENTRY.model_args,
            model_kwargs=MVN_ENTRY.model_kwargs,
            dynamic_args=False,
        )
        init_pos = model_info.param_info.z
        potential_fn = model_info.potential_fn

        _, logdensity_fn, _ = build_logdensity_fn(key, MVN_ENTRY)
        # Use the same init_pos as above
        ld = logdensity_fn(init_pos)
        pot = potential_fn(init_pos)
        assert jnp.isclose(
            ld, -pot, atol=1e-5
        ), f"logdensity_fn != -potential_fn: {ld} vs {-pot}"
