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
"""Unit tests for ``tuningfork.warmup._laplace_adapter``.

Tests the adapter helper that routes laplace_* base methods through
``blackjax.hmc`` for window adaptation, without touching the non-laplace path.

All tests here are pure logic / dataclass checks (no JAX trace) — fast.
"""

import inspect

import blackjax
import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.warmup._laplace_adapter import (
    LAPLACE_METHOD_NAMES,
    resolve_warmup_algorithm,
)

pytestmark = pytest.mark.fast


def test_laplace_method_names_set():
    """LAPLACE_METHOD_NAMES contains all 4 registered laplace_* variants."""
    assert "laplace_hmc" in LAPLACE_METHOD_NAMES
    assert "laplace_dhmc" in LAPLACE_METHOD_NAMES
    assert "laplace_mhmc" in LAPLACE_METHOD_NAMES
    assert "laplace_dmhmc" in LAPLACE_METHOD_NAMES
    assert len(LAPLACE_METHOD_NAMES) == 4


def test_resolve_non_laplace_returns_factory_unchanged():
    """Non-laplace base method: (base_method.factory, unchanged_kwargs) returned."""
    hmc = BASE_METHODS["hmc"]
    extra = {"num_integration_steps": 10}
    algo, kw = resolve_warmup_algorithm(hmc, extra)
    assert algo is hmc.factory
    assert kw == {"num_integration_steps": 10}


def test_resolve_non_laplace_does_not_mutate_input():
    """Non-laplace path returns a copy of extra_kwargs, not the original dict."""
    nuts = BASE_METHODS["nuts"]
    extra = {"some_key": 42}
    _, kw = resolve_warmup_algorithm(nuts, extra)
    kw["extra_key"] = "added"
    assert "extra_key" not in extra  # original untouched


@pytest.mark.parametrize(
    "method_name",
    ["laplace_hmc", "laplace_dhmc", "laplace_mhmc", "laplace_dmhmc"],
)
def test_resolve_laplace_returns_blackjax_hmc(method_name):
    """All 4 laplace_* variants resolve to blackjax.hmc as warmup algorithm."""
    method = BASE_METHODS[method_name]
    algo, _ = resolve_warmup_algorithm(method, {})
    assert algo is blackjax.hmc, f"Expected blackjax.hmc for {method_name}, got {algo}"


@pytest.mark.parametrize(
    "method_name",
    ["laplace_hmc", "laplace_dhmc", "laplace_mhmc", "laplace_dmhmc"],
)
def test_resolve_laplace_preserves_num_integration_steps(method_name):
    """Laplace path: caller-supplied num_integration_steps is preserved."""
    method = BASE_METHODS[method_name]
    algo, kw = resolve_warmup_algorithm(method, {"num_integration_steps": 7})
    assert kw["num_integration_steps"] == 7


@pytest.mark.parametrize(
    "method_name",
    ["laplace_hmc", "laplace_dhmc", "laplace_mhmc", "laplace_dmhmc"],
)
def test_resolve_laplace_default_num_integration_steps(method_name):
    """Laplace path without NIS in kwargs: default of 5 is applied."""
    method = BASE_METHODS[method_name]
    _, kw = resolve_warmup_algorithm(method, {})
    assert kw["num_integration_steps"] == 5


def test_blackjax_hmc_has_required_interface():
    """blackjax.hmc exposes .build_kernel and .init needed by window_adaptation."""
    assert hasattr(blackjax.hmc, "build_kernel"), "Missing .build_kernel"
    assert hasattr(blackjax.hmc, "init"), "Missing .init"
    # build_kernel must accept at least one parameter (integrator)
    params = inspect.signature(blackjax.hmc.build_kernel).parameters
    assert len(params) > 0, "build_kernel should accept integrator param"


def test_laplace_kwargs_only_contain_num_integration_steps():
    """Laplace path returns only num_integration_steps — no laplace-specific extras."""
    laplace_hmc = BASE_METHODS["laplace_hmc"]
    # If extra_kwargs contains non-HMC keys, they should NOT appear in result
    # (adapter strips them to avoid passing them to window_adaptation)
    extra = {"num_integration_steps": 3}
    _, kw = resolve_warmup_algorithm(laplace_hmc, extra)
    assert set(kw.keys()) == {"num_integration_steps"}
