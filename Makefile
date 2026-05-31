.PHONY: install test test-fast test-slow test-e2e test-full lint clean clean-orphans benchmark benchmark-pr

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

# ---------------------------------------------------------------------------
# Benchmark suite (opt-in; requires bench-perf dep group)
# ---------------------------------------------------------------------------
# IMPORTANT: --benchmark-max-time=0 means "run exactly --benchmark-min-rounds=1
# time" — do NOT use a large max-time for expensive MCMC benchmarks. A 240s
# max-time causes ~240 repeated runs per cell (fine for microbenchmarks, fatal
# for MCMC: cumulative JAX recompilation + memory → native abort).

# Full benchmark: Tier 1+2, e2e + calibrated, all 12 sampler families (~15min)
# Budget: one timed run per cell (30s–1min each); D5 wall cap is per-cell, not window.
benchmark:
	$(MAKE) clean-orphans
	uv sync --group bench-perf --python 3.13
	JAX_PLATFORM_NAME=cpu uv run --python 3.13 pytest benchmarks/ \
		-m benchmark \
		--benchmark-json=benchmark_results.json \
		--benchmark-disable-gc \
		--benchmark-warmup=off \
		--benchmark-min-rounds=1 \
		--benchmark-max-time=0 \
		-v

# Tier 1 only (~60s; fast local smoke check)
benchmark-pr:
	$(MAKE) clean-orphans
	uv sync --group bench-perf --python 3.13
	JAX_PLATFORM_NAME=cpu uv run --python 3.13 pytest benchmarks/ \
		-m benchmark \
		-k "tier1 and calibrated" \
		--benchmark-json=benchmark_pr_results.json \
		--benchmark-disable-gc \
		--benchmark-warmup=off \
		--benchmark-min-rounds=1 \
		--benchmark-max-time=0 \
		-v

lint:
	uv run pre-commit run --all-files

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__ .venv
	find . -name "*.pyc" -delete
