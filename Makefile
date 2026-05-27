.PHONY: install test test-fast test-slow test-e2e test-full lint clean clean-orphans

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
# Runs all @pytest.mark.e2e tests: recipes emit-execute tests + e2e phase CLI tests
test-e2e:
	$(MAKE) clean-orphans
	JAX_PLATFORM_NAME=cpu uv run pytest tests -m e2e -n 1

# The merge gate: everything
test-full:
	$(MAKE) clean-orphans
	JAX_PLATFORM_NAME=cpu uv run pytest tests -n 2

# Kill orphan Python REPLs and stale pytest workers (memory hygiene before sweeps).
# The script is vendored at tools/clean_orphans.sh (self-contained; no external dep).
clean-orphans:
	@bash tools/clean_orphans.sh

lint:
	uv run pre-commit run --all-files

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__ .venv
	find . -name "*.pyc" -delete
