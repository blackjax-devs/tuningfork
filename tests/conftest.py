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
