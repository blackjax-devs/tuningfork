.PHONY: install test test-fast test-slow test-e2e test-full lint clean clean-orphans benchmark benchmark-fast benchmark-pr revalidate-w1

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

# Full nightly benchmark: fast + e2e cells (~8 min fast + 2 slow cells)
benchmark:
	$(MAKE) clean-orphans
	uv sync --group bench-perf --python 3.13
	JAX_PLATFORM_NAME=cpu uv run --python 3.13 pytest \
		benchmarks/test_fast_recipes.py benchmarks/test_e2e_recipes.py \
		-m benchmark \
		--benchmark-json=benchmark_results.json \
		--benchmark-disable-gc \
		--benchmark-warmup=off \
		--benchmark-min-rounds=1 \
		--benchmark-max-time=0 \
		-v

# Fast suite only (~8 min; 31 cells ≤60s each)
benchmark-fast:
	$(MAKE) clean-orphans
	uv sync --group bench-perf --python 3.13
	JAX_PLATFORM_NAME=cpu uv run --python 3.13 pytest benchmarks/test_fast_recipes.py \
		-m benchmark \
		--benchmark-json=benchmark_fast_results.json \
		--benchmark-disable-gc \
		--benchmark-warmup=off \
		--benchmark-min-rounds=1 \
		--benchmark-max-time=0 \
		-v

# Quick smoke (Tier 1 calibrated only, ~1 min)
benchmark-pr:
	$(MAKE) clean-orphans
	uv sync --group bench-perf --python 3.13
	JAX_PLATFORM_NAME=cpu uv run --python 3.13 pytest benchmarks/test_fast_recipes.py \
		-m benchmark \
		-k "tier1 and calibrated" \
		--benchmark-json=benchmark_pr_results.json \
		--benchmark-disable-gc \
		--benchmark-warmup=off \
		--benchmark-min-rounds=1 \
		--benchmark-max-time=0 \
		-v

# ---------------------------------------------------------------------------
# W1 catalog re-validation (runs after gate changes or catalog batch updates)
# ---------------------------------------------------------------------------
# Default: path-A-only (committed cached draws — clean signal, seconds per cell).
# Opt-in re-gen: ENABLE_REGEN=1 make revalidate-w1 (full two-stage gate on B/C cells).
revalidate-w1:
	$(MAKE) clean-orphans
	JAX_PLATFORM_NAME=cpu uv run python -m tuningfork.calibration.revalidation $(if $(ENABLE_REGEN),--regen,)

lint:
	uv run pre-commit run --all-files

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__ .venv
	find . -name "*.pyc" -delete
