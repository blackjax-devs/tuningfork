# Contributing to bjx-bench

This document outlines the conventions for developing and testing `bjx-bench`.

## Test Layout

Tests mirror the source structure under `bjx_bench/`:

```
tests/
├── inference/
│   ├── base_method/        # BaseMethod dataclass + algorithm wrappers
│   ├── warmup/             # Warmup strategy implementations
│   └── smc/                # SMC method implementations
├── models/                 # Model-specific tests (one per model)
├── recipes/                # Recipe schema and emission tests
├── metrics/                # Headline metric + diagnostics
├── tier_a/                 # Tier-A reference certification tests
├── tier_b/                 # Tier-B Bayesian optimization tests
├── reference/              # Reference cache and posteriordb cross-checks
├── numpyro/                # NumPyro integration helpers
├── e2e/                    # End-to-end phase-gate tests
│
├── test_api_pins.py        # BlackJAX upstream contract tripwires (cross-cutting)
├── test_registry.py        # BASE_METHODS + SMC_METHODS registry tests (cross-cutting)
├── conftest.py             # Marker registration
└── fixtures.py             # Shared fixtures (RNG keys, toy MVN logdensity, etc.)
```

Each folder contains an `__init__.py` for package discovery. Cross-cutting tests (contract pins, registry) stay at the root.

## Test Markers

Five markers classify tests by cost and dependency:

| Marker | Meaning | Wall Time | Default? |
|--------|---------|-----------|----------|
| `fast` | Pure logic, dataclass, schema — no JAX trace or chain. | <100 ms | Yes |
| `slow` | Runs a chain or warmup with JAX compilation. | >1 s | Yes |
| `e2e` | End-to-end phase gate; multiple algorithms × models. | >10 s | Yes |
| `requires_posteriordb` | Needs the posteriordb data cache; fails offline. | N/A | Yes |
| `benchmark` | Reserved for future perf benchmarks. | N/A | No |

**Discipline rule**: Every test must be tagged with **exactly one** of `fast`, `slow`, or `e2e`. The `requires_posteriordb` marker is additive (combine with `slow` or `e2e` if the test also needs posteriordb data).

See `tests/conftest.py` for marker registration (source of truth — markers are NOT duplicated in `pyproject.toml`).

## Common Invocations

| Intent | Command |
|--------|---------|
| Inner-loop dev (structural tests only) | `make test-fast` |
| Default contributor flow (skip e2e) | `make test` |
| Just slow tests | `make test-slow` |
| Phase gate (everything) | `make test-full` |
| End-to-end suite only | `make test-e2e` |
| Before heavy runs: kill orphan processes | `make clean-orphans` |

All `test-*` targets automatically run `make clean-orphans` first (except `test-fast` to minimize overhead). See META-014 in `/home/jp/blackjax-devs/WORKLOG.md` for details on orphan processes.

## Adding a New Test

1. **Pick the right subfolder** based on what you're testing:
   - Model correctness → `tests/models/test_<model_name>.py`
   - Warmup wrapper → `tests/inference/warmup/test_<strategy_name>.py`
   - Algorithm → `tests/inference/base_method/test_<algorithm_name>.py`
   - Recipe schema → `tests/recipes/test_schema.py`
   - Cross-cutting → `tests/test_*.py` at the root

2. **Tag with exactly one of `@pytest.mark.fast` / `@pytest.mark.slow` / `@pytest.mark.e2e`**:
   ```python
   @pytest.mark.slow
   def test_chain_convergence():
       # Your test here
   ```

   If ALL tests in a file share the same marker, use `pytestmark`:
   ```python
   pytestmark = pytest.mark.slow

   def test_one():
       ...

   def test_two():
       ...
   ```

3. **If your test needs posteriordb data**, add `@pytest.mark.requires_posteriordb` in addition:
   ```python
   @pytest.mark.slow
   @pytest.mark.requires_posteriordb
   def test_xcheck_vs_stan():
       ...
   ```

4. **Use shared fixtures from `tests/fixtures.py`** when possible:
   - `rng_key` — parametrized RNG key fixture (runs test 3× with seeds 0, 1, 42)
   - `make_rng(seed)` — manual RNG key generation
   - `mvn_5d_logdensity(position)` — canonical 5-D Gaussian logdensity
   - `mvn_5d_init()` — matching zero initialization

   Example:
   ```python
   @pytest.mark.fast
   def test_something(rng_key):
       x = mvn_5d_init()
       logp = mvn_5d_logdensity(x)
       assert logp == 0.0
   ```

5. **Run `make lint` before committing** to pass pre-commit hooks (black, isort, flake8, mypy).

## Pre-Commit and Commit Messages

See the top-level `CLAUDE.md` in `/home/jp/blackjax-devs/CLAUDE.md` for the monorepo's commit discipline. In brief:

- Run `make lint` before every commit.
- Commit frequently to demonstrate the thinking process.
- Each commit message should include a "finding / what was the error and how it was fixed" line.
- Enrich the history with trial-and-error information.

For minor test changes (adding a marker, adjusting an assertion), one-line commit messages are fine. For structural changes (adding new fixtures, reorganizing suites), include a brief explanation of why.

## Where Things Live

| Component | Module Path | Tests Location |
|-----------|-------------|-----------------|
| Algorithm wrappers (HMC, NUTS, MALA, etc.) | `bjx_bench/inference/` | `tests/inference/base_method/`, `tests/inference/warmup/` |
| Warmup strategies (Stan window, Pathfinder, etc.) | `bjx_bench/inference/warmup/` | `tests/inference/warmup/` |
| SMC variants | `bjx_bench/inference/smc/` | `tests/inference/smc/` |
| Models (MVN, funnel, horseshoe, etc.) | `bjx_bench/registry/` | `tests/models/` |
| Recipe schema and emission | `bjx_bench/recipes/` | `tests/recipes/` |
| Metrics (headline, diagnostics) | `bjx_bench/metrics/` | `tests/metrics/` |
| Tier-A certification | `bjx_bench/calibration/tier_a.py` | `tests/tier_a/` |
| Tier-B Bayesian optimization | `bjx_bench/calibration/tier_b.py` | `tests/tier_b/` |
| Reference cache + xcheck | `bjx_bench/reference/` | `tests/reference/` |

## Testing Before Commit

```bash
# Full test (default contributor flow, skips e2e)
make test

# Just the fast suite (sanity check)
make test-fast

# Full sweep including e2e (pre-merge verification)
make test-full

# Lint check
make lint
```

If a test fails, check:
1. Did it pass locally with `make test-fast` before you started?
2. Does your change break a marker contract (e.g., adding a JAX operation to a `fast` test)?
3. For timing issues: if a test now takes >1 s, consider moving it from `fast` to `slow`.
