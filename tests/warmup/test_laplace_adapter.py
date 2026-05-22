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

Tests the adapter helper that routes laplace_* / dynamic_hmc / dmhmc base methods
through ``blackjax.nuts`` for window adaptation, without touching the non-substitute
path.

All tests here are pure logic / dataclass checks (no JAX trace) — fast.
"""

import inspect

import blackjax
import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.warmup._laplace_adapter import (
    LAPLACE_METHOD_NAMES,
    WARMUP_SUBSTITUTE_METHOD_NAMES,
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


def test_warmup_substitute_method_names_set():
    """WARMUP_SUBSTITUTE_METHOD_NAMES = LAPLACE_METHOD_NAMES ∪ {dynamic_hmc, dmhmc}."""
    assert WARMUP_SUBSTITUTE_METHOD_NAMES == LAPLACE_METHOD_NAMES | {
        "dynamic_hmc",
        "dmhmc",
    }
    assert len(WARMUP_SUBSTITUTE_METHOD_NAMES) == 6


def test_resolve_non_substitute_returns_factory_unchanged():
    """Non-substitute base method: (base_method.factory, unchanged_kwargs) returned."""
    hmc = BASE_METHODS["hmc"]
    extra = {"num_integration_steps": 10}
    algo, kw = resolve_warmup_algorithm(hmc, extra)
    assert algo is hmc.factory
    assert kw == {"num_integration_steps": 10}


def test_resolve_non_substitute_does_not_mutate_input():
    """Non-substitute path returns a copy of extra_kwargs, not the original dict."""
    nuts = BASE_METHODS["nuts"]
    extra = {"some_key": 42}
    _, kw = resolve_warmup_algorithm(nuts, extra)
    kw["extra_key"] = "added"
    assert "extra_key" not in extra  # original untouched


@pytest.mark.parametrize(
    "method_name",
    [
        "laplace_hmc",
        "laplace_dhmc",
        "laplace_mhmc",
        "laplace_dmhmc",
        "dynamic_hmc",
        "dmhmc",
    ],
)
def test_resolve_substitute_returns_blackjax_nuts(method_name):
    """All 6 substitute-family methods resolve to blackjax.nuts as warmup algorithm.

    NUTS is the canonical Stan-style warmup kernel and needs no extra kernel kwargs
    at warmup time (NUTS picks its own trajectory length).  Prior to 2026-05-21 the
    substitute kernel was blackjax.hmc; the change to NUTS simplifies the substitute
    path (no num_integration_steps injection needed) and aligns with Stan convention.
    """
    method = BASE_METHODS[method_name]
    algo, _ = resolve_warmup_algorithm(method, {})
    assert (
        algo is blackjax.nuts
    ), f"Expected blackjax.nuts for {method_name}, got {algo}"


@pytest.mark.parametrize(
    "method_name",
    [
        "laplace_hmc",
        "laplace_dhmc",
        "laplace_mhmc",
        "laplace_dmhmc",
        "dynamic_hmc",
        "dmhmc",
    ],
)
def test_resolve_substitute_discards_extra_kwargs(method_name):
    """Substitute path discards extra_kwargs — NUTS needs no extra kernel kwargs.

    Even if the caller passes num_integration_steps (or any other HP), the substitute
    path returns an empty dict.  This is intentional: NUTS picks its own trajectory
    length via the no-U-turn criterion, and any HP that's specific to the downstream
    sampler (e.g. mhmc's num_integration_steps) is irrelevant at warmup time when
    NUTS is the warmup kernel.  The downstream sampler gets its HPs from the recipe
    at sample time.
    """
    method = BASE_METHODS[method_name]
    _, kw = resolve_warmup_algorithm(method, {"num_integration_steps": 7})
    assert kw == {}, f"Expected empty kwargs for {method_name}, got {kw}"


def test_blackjax_nuts_has_required_interface():
    """blackjax.nuts exposes .build_kernel and .init needed by window_adaptation."""
    assert hasattr(blackjax.nuts, "build_kernel"), "Missing .build_kernel"
    assert hasattr(blackjax.nuts, "init"), "Missing .init"
    # build_kernel must accept at least one parameter (integrator)
    params = inspect.signature(blackjax.nuts.build_kernel).parameters
    assert len(params) > 0, "build_kernel should accept integrator param"
