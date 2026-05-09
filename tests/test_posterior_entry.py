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
"""Tests for Posterior dataclass and ReferenceMethod enum."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

from bjx_bench.model._base import Posterior, ReferenceMethod

pytestmark = pytest.mark.fast


def _dummy_model():
    numpyro.sample("x", dist.Normal(0.0, 1.0))


def _dummy_sampler(rng_key: jax.Array, n: int) -> dict[str, jax.Array]:
    return {"x": jax.random.normal(rng_key, (n,))}


class TestPosteriorConstruction:
    def test_minimal_nuts_entry(self):
        entry = Posterior(
            name="dummy_nuts",
            dim=1,
            class_="gaussian",
            numpyro_model=_dummy_model,
        )
        assert entry.name == "dummy_nuts"
        assert entry.dim == 1
        assert entry.reference_method == ReferenceMethod.NUTS
        assert entry.analytic_sampler is None

    def test_analytic_entry(self):
        entry = Posterior(
            name="dummy_analytic",
            dim=1,
            class_="gaussian",
            numpyro_model=_dummy_model,
            analytic_sampler=_dummy_sampler,
        )
        assert entry.reference_method == ReferenceMethod.ANALYTIC

    def test_frozen_dataclass_immutable(self):
        entry = Posterior(
            name="dummy",
            dim=1,
            class_="gaussian",
            numpyro_model=_dummy_model,
        )
        with pytest.raises(Exception):
            entry.name = "other"  # type: ignore[misc]

    def test_dim_zero_raises(self):
        with pytest.raises(ValueError, match="dim must be positive"):
            Posterior(
                name="bad",
                dim=0,
                class_="gaussian",
                numpyro_model=_dummy_model,
            )

    def test_dim_negative_raises(self):
        with pytest.raises(ValueError, match="dim must be positive"):
            Posterior(
                name="bad",
                dim=-5,
                class_="gaussian",
                numpyro_model=_dummy_model,
            )

    def test_non_callable_model_raises(self):
        with pytest.raises(TypeError, match="numpyro_model must be callable"):
            Posterior(
                name="bad",
                dim=1,
                class_="gaussian",
                numpyro_model="not_callable",  # type: ignore[arg-type]
            )

    def test_model_args_and_kwargs(self):
        def model_with_args(y, sigma=1.0):
            numpyro.sample("mu", dist.Normal(0.0, sigma))
            numpyro.sample("obs", dist.Normal(0.0, 1.0), obs=y)

        y_obs = jnp.array([1.0, 2.0])
        entry = Posterior(
            name="with_args",
            dim=1,
            class_="glm",
            numpyro_model=model_with_args,
            model_args=(y_obs,),
            model_kwargs={"sigma": 2.0},
        )
        assert entry.model_args[0] is y_obs
        assert entry.model_kwargs["sigma"] == 2.0

    def test_posteriordb_id_and_citations(self):
        entry = Posterior(
            name="eight_schools_ncp",
            dim=10,
            class_="hierarchical",
            numpyro_model=_dummy_model,
            posteriordb_id="8_schools-eight_schools_noncentered",
            citations=("Rubin 1981",),
        )
        assert entry.posteriordb_id == "8_schools-eight_schools_noncentered"
        assert "Rubin 1981" in entry.citations

    def test_reference_method_enum_values(self):
        assert ReferenceMethod.ANALYTIC == "analytic"
        assert ReferenceMethod.NUTS == "nuts"
