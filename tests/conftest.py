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
"""Pytest configuration for tuningfork tests.

Registers pytest markers for test classification:
- ``fast``: pure-logic tests that do not run a chain or warmup; use ``pytest -m fast`` for inner-loop iteration
- ``slow``: chain-running or warmup tests with JAX-compiled body (>1 s wall time)
- ``e2e``: end-to-end phase-gate tests; multiple algorithms × models (>10 s each)
- ``requires_posteriordb``: tests that need the posteriordb data cache; fail offline
- ``benchmark``: reserved for future performance benchmarks

Discipline rule: every test must be tagged with exactly one of ``fast``, ``slow``, or ``e2e``.
The ``requires_posteriordb`` marker is additive (combine with ``slow`` or ``e2e``).
"""

import sys

# Import shared fixtures so pytest discovers them globally
from tests import fixtures as _  # noqa: F401

# Python 3.11's default recursion limit (1000) is too low for some JAX pytree
# operations used by tuningfork models (e.g. lotka_volterra's logdensity_fn
# flattens a deeply nested pytree via ``jax/_src/flatten_util.py:65`` which
# blows the stack at d~30. Python 3.13's effective stack depth is higher and
# absorbs it; py3.11 needs the bump to match. Apply at conftest-load so every
# test in the session inherits.
sys.setrecursionlimit(3000)


def pytest_configure(config: object) -> None:
    """Register custom markers so pytest does not warn about unknown marks."""
    config.addinivalue_line(  # type: ignore[attr-defined]
        "markers",
        "fast: tests that don't run a chain or warmup (pure logic / dataclass / schema)",
    )
    config.addinivalue_line(  # type: ignore[attr-defined]
        "markers",
        "slow: chain-running or warmup tests; JAX-compiled body (>1 s wall time)",
    )
    config.addinivalue_line(  # type: ignore[attr-defined]
        "markers",
        "e2e: end-to-end phase-gate tests; multiple algorithms × models (>10 s each)",
    )
    config.addinivalue_line(  # type: ignore[attr-defined]
        "markers",
        "requires_posteriordb: tests that need the posteriordb data cache; fail offline",
    )
    config.addinivalue_line(  # type: ignore[attr-defined]
        "markers",
        "benchmark: reserved for future performance benchmarks",
    )
