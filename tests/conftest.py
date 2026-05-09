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
"""Pytest configuration for bjx-bench tests.

Registers the ``fast`` pytest marker used for pure-logic tests that do not
run a chain or warmup, enabling ``pytest -m fast`` for inner-loop iteration
during development.  The full ``pytest tests/`` sweep runs everything.
"""


def pytest_configure(config: object) -> None:
    """Register custom markers so pytest does not warn about unknown marks."""
    config.addinivalue_line(  # type: ignore[attr-defined]
        "markers",
        "fast: tests that don't run a chain or warmup (pure logic / dataclass / schema)",
    )
