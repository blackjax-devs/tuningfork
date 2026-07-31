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

"""Descriptor-only tests for generated warmup method families."""

import pytest

from tuningfork.recipes._warmup_protocol import (
    LAPLACE_METHOD_NAMES,
    WARMUP_SUBSTITUTE_METHOD_NAMES,
)

pytestmark = pytest.mark.fast


def test_laplace_method_names_are_complete() -> None:
    assert LAPLACE_METHOD_NAMES == frozenset(
        {"laplace_hmc", "laplace_dhmc", "laplace_mhmc", "laplace_dmhmc"}
    )


def test_warmup_substitute_method_names_include_dynamic_families() -> None:
    assert WARMUP_SUBSTITUTE_METHOD_NAMES == LAPLACE_METHOD_NAMES | {
        "dynamic_hmc",
        "dmhmc",
    }
