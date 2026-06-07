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
"""Tests specific to the 1600-D separable GP LGCP entry."""

import jax
import jax.numpy as jnp
import pytest

from tuningfork.model import build_logdensity_fn
from tuningfork.model.lgcp import ENTRY, Y_DATA

pytestmark = pytest.mark.fast


class TestLgcp:
    def test_dim(self) -> None:
        assert ENTRY.dim == 1600

    def test_class(self) -> None:
        assert ENTRY.class_ == "latent_gaussian"

    def test_y_data_shape(self) -> None:
        assert Y_DATA.shape == (40, 40)
        assert Y_DATA.dtype == jnp.int32

    def test_build_logdensity_fn_finite(self) -> None:
        key = jax.random.key(12345)
        init_pos, logdensity_fn, _ = build_logdensity_fn(key, ENTRY)
        ld = logdensity_fn(init_pos)
        assert jnp.isfinite(ld), f"Expected finite log-density, got {ld}"
        assert "z" in init_pos

    def test_cholesky_not_in_jaxpr(self) -> None:
        key = jax.random.key(12345)
        init_pos, logdensity_fn, _ = build_logdensity_fn(key, ENTRY)

        # Verify that cholesky calculation is not in the compiled JAXPR trace
        jaxpr = jax.make_jaxpr(logdensity_fn)(init_pos)
        jaxpr_str = str(jaxpr)
        # It shouldn't have cholesky primitive calls inside logdensity
        assert "cholesky" not in jaxpr_str.lower()
