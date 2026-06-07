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
"""Tests specific to the 500-D Rasch IRT entry."""

import jax
import jax.numpy as jnp
import pytest

from tuningfork.model import build_logdensity_fn
from tuningfork.model.irt_1pl import ENTRY, RESPONSE

pytestmark = pytest.mark.fast


class TestIrt1pl:
    def test_dim(self) -> None:
        assert ENTRY.dim == 500

    def test_class(self) -> None:
        assert ENTRY.class_ == "hierarchical"

    def test_model_args(self) -> None:
        assert ENTRY.model_args == (RESPONSE,)

    def test_response_shape(self) -> None:
        assert RESPONSE.shape == (500, 10)

    def test_build_logdensity_fn_finite(self) -> None:
        key = jax.random.key(42)
        init_pos, logdensity_fn, _ = build_logdensity_fn(key, ENTRY)
        ld = logdensity_fn(init_pos)
        assert jnp.isfinite(ld), f"Expected finite log-density, got {ld}"
        assert "theta" in init_pos
