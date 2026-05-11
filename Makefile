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
test-e2e:
	$(MAKE) clean-orphans
	JAX_PLATFORM_NAME=cpu uv run pytest tests/e2e/ -n 1

# The merge gate: everything
test-full:
	$(MAKE) clean-orphans
	JAX_PLATFORM_NAME=cpu uv run pytest tests -n 2

# Kill orphan Python REPLs and stale pytest workers (memory hygiene before sweeps).
# The script lives in claude-config (cross-repo tool); CLAUDE_CONFIG_DIR can override
# the default ~/claude-config location.
CLAUDE_CONFIG_DIR ?= $(HOME)/claude-config
clean-orphans:
	@bash $(CLAUDE_CONFIG_DIR)/tools/clean_orphans.sh

lint:
	uv run pre-commit run --all-files

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__ .venv
	find . -name "*.pyc" -delete
