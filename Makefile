.PHONY: install test test-fast test-slow test-e2e test-full lint clean clean-orphans tier-a tune

install:
	uv sync --group bench

# Default contributor flow: skip the heavy phase-e2e suite
test:
	$(MAKE) clean-orphans
	JAX_PLATFORM_NAME=cpu uv run pytest tests -m "not e2e" -n 2

# Inner-loop dev — only the structural / dataclass tests
test-fast:
	JAX_PLATFORM_NAME=cpu uv run pytest tests -m fast -n 2

# Slow chain-running tests (no e2e)
test-slow:
	$(MAKE) clean-orphans
	JAX_PLATFORM_NAME=cpu uv run pytest tests -m slow -n 2

# End-to-end phase gate (run before merge)
test-e2e:
	$(MAKE) clean-orphans
	JAX_PLATFORM_NAME=cpu uv run pytest tests/e2e/ -n 1

# The merge gate: everything
test-full:
	$(MAKE) clean-orphans
	JAX_PLATFORM_NAME=cpu uv run pytest tests -n 2

# META-014: kill orphan Python REPLs and stale pytest workers
clean-orphans:
	@bash tools/clean_orphans.sh

lint:
	uv run pre-commit run --all-files

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__ .venv
	find . -name "*.pyc" -delete

# Convenience entry points (Phase 6 wires up the real CLI)
tier-a:
	@echo "usage: make tier-a MODEL=<name>"
	@echo "(Phase 1 will implement: bjx-bench tier-a $(MODEL))"

tune:
	@echo "usage: make tune MODEL=<name> ALGO=<name>"
	@echo "(Phase 2 will implement: bjx-bench tune $(MODEL) $(ALGO))"

claude-perms-audit:
	uv run python tools/claude_perms_audit.py
