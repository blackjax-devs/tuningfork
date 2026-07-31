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

import jax
import pytest

# Import shared fixtures so pytest discovers them globally
from tests import fixtures as _  # noqa: F401

# ---------------------------------------------------------------------------
# LFS pointer guard — shared across all test suites that load committed .npz
# ---------------------------------------------------------------------------

# Sentinel bytes at the start of every git-LFS pointer stub.
_LFS_MAGIC = b"version https://git-lfs.github.com"


def _is_lfs_pointer(path: str) -> bool:
    """Return True if *path* is an unsmudged git-LFS pointer stub.

    Committed catalog draws.npz files are LFS-tracked.  When git-LFS is not
    available (e.g. CI without ``lfs: true``), the checkout leaves a ~130-byte
    plain-text pointer file that starts with ``version https://git-lfs.github.com``.
    Passing such a file to ``np.load`` fails with an ``UnpicklingError``.
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(64).startswith(_LFS_MAGIC)
    except OSError:
        return False


@pytest.fixture(autouse=True)
def _restore_jax_x64():
    """Restore jax_enable_x64 after every test.

    Prevents in-process tests that call jax.config.update("jax_enable_x64", True)
    (e.g. requires_x64 model recipes) from leaking
    the x64 flag into subsequent tests.  Without this guard, jax.random.bits()
    returns uint64 in later tests, causing downstream uint32 seed overflow
    (>2**32-1).
    """
    before = jax.config.jax_enable_x64  # type: ignore[attr-defined]
    yield
    if jax.config.jax_enable_x64 != before:  # type: ignore[attr-defined]
        jax.config.update("jax_enable_x64", before)


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
