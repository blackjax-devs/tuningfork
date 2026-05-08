.PHONY: install test lint clean tier-a tune

install:
	uv sync --group bench

test:
	JAX_PLATFORM_NAME=cpu uv run pytest -vv tests

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
