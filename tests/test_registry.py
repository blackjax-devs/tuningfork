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
"""Tests for the bjx-bench model registry.

Covers:
- All three Phase-1 starter models are registered.
- build_logdensity_fn returns finite log-density at init position.
- Analytic samplers produce arrays of the right shape.
- Eight-schools NCP has dim==10 and reference_method==NUTS.
"""

import jax
import jax.numpy as jnp
import pytest

from tuningfork.model import MODELS, ReferenceMethod, build_logdensity_fn

pytestmark = pytest.mark.fast


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


def test_inference_namespace_imports():
    """Restructure smoke: model and inference layers
    import cleanly and expose the expected dicts.

     If this
    fails, the restructure has broken a public namespace.
    """
    from tuningfork.inference.base_method import BASE_METHODS
    from tuningfork.inference.warmup import WARMUPS, Warmup
    from tuningfork.model import MODELS

    assert isinstance(MODELS, dict)
    assert len(MODELS) >= 3  # mvn_10, neals_funnel, eight_schools_ncp
    assert isinstance(BASE_METHODS, dict)
    # Core 6 must still be present; more entries may be added later.
    core_six = {"hmc", "nuts", "mala", "barker", "rwm", "mclmc"}
    assert core_six <= set(
        BASE_METHODS.keys()
    ), f"missing core base methods: {core_six - set(BASE_METHODS.keys())}"
    assert isinstance(WARMUPS, dict)  # may be empty
    assert Warmup is not None
