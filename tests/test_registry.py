"""Tests for the bjx-bench model registry.

Covers:
- All three Phase-1 starter models are registered.
- build_logdensity_fn returns finite log-density at init position.
- Analytic samplers produce arrays of the right shape.
- Eight-schools NCP has dim==10 and reference_method==NUTS.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from bjx_bench.model import MODELS, ReferenceMethod, build_logdensity_fn


@pytest.mark.parametrize("name", ["mvn_10", "neals_funnel", "eight_schools_ncp"])
def test_all_starter_models_registered(name: str) -> None:
    """All three Phase-1 models must appear in MODELS."""
    assert name in MODELS, f"{name!r} not found in MODELS"


class TestMvn10:
    """Tests specific to the 10-D isotropic Gaussian entry."""

    def setup_method(self) -> None:
        self.entry = MODELS["mvn_10"]
        self.key = jax.random.key(0)

    def test_dim(self) -> None:
        assert self.entry.dim == 10

    def test_class(self) -> None:
        assert self.entry.class_ == "gaussian"

    def test_reference_method_analytic(self) -> None:
        assert self.entry.reference_method == ReferenceMethod.ANALYTIC

    def test_analytic_sampler_shape(self) -> None:
        n = 50
        assert self.entry.analytic_sampler is not None
        draws = self.entry.analytic_sampler(self.key, n)
        assert "x" in draws
        assert draws["x"].shape == (n, 10)

    def test_build_logdensity_fn_finite(self) -> None:
        init_pos, logdensity_fn, _ = build_logdensity_fn(self.key, self.entry)
        ld = logdensity_fn(init_pos)
        assert jnp.isfinite(ld), f"Expected finite log-density, got {ld}"


class TestNealsFunnel:
    """Tests specific to the 10-D Neal's funnel entry."""

    def setup_method(self) -> None:
        self.entry = MODELS["neals_funnel"]
        self.key = jax.random.key(1)

    def test_dim(self) -> None:
        assert self.entry.dim == 10

    def test_class(self) -> None:
        assert self.entry.class_ == "funnel"

    def test_reference_method_analytic(self) -> None:
        assert self.entry.reference_method == ReferenceMethod.ANALYTIC

    def test_analytic_sampler_shape(self) -> None:
        n = 50
        assert self.entry.analytic_sampler is not None
        draws = self.entry.analytic_sampler(self.key, n)
        assert "v" in draws
        assert "theta" in draws
        assert draws["v"].shape == (n,)
        assert draws["theta"].shape == (n, 9)

    def test_build_logdensity_fn_finite(self) -> None:
        init_pos, logdensity_fn, _ = build_logdensity_fn(self.key, self.entry)
        ld = logdensity_fn(init_pos)
        assert jnp.isfinite(ld), f"Expected finite log-density, got {ld}"


class TestEightSchoolsNCP:
    """Tests specific to the 8-Schools NCP entry."""

    def setup_method(self) -> None:
        self.entry = MODELS["eight_schools_ncp"]
        self.key = jax.random.key(2)

    def test_dim(self) -> None:
        # mu(1) + tau(1) + theta_raw(8) = 10
        assert self.entry.dim == 10

    def test_class(self) -> None:
        assert self.entry.class_ == "hierarchical"

    def test_reference_method_nuts(self) -> None:
        assert self.entry.reference_method == ReferenceMethod.NUTS

    def test_no_analytic_sampler(self) -> None:
        assert self.entry.analytic_sampler is None

    def test_posteriordb_id(self) -> None:
        assert self.entry.posteriordb_id == "8_schools-eight_schools_noncentered"

    def test_build_logdensity_fn_finite(self) -> None:
        init_pos, logdensity_fn, _ = build_logdensity_fn(self.key, self.entry)
        ld = logdensity_fn(init_pos)
        assert jnp.isfinite(ld), f"Expected finite log-density, got {ld}"

    def test_init_position_keys(self) -> None:
        init_pos, _, _ = build_logdensity_fn(self.key, self.entry)
        # NCP unconstrained sites: mu, tau (transformed), theta_raw
        assert "mu" in init_pos
        assert "tau" in init_pos
        assert "theta_raw" in init_pos
