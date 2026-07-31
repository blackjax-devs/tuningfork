# Contributing to tuningfork

This document outlines the conventions for developing and testing `tuningfork`.

## Recipe and Sampling Architecture

Changes to recipes, sampling execution, calibration, certification,
revalidation, or recipe benchmarks must follow the
[codegen-first recipe lifecycle](docs/design/codegen-first-recipes.md).

The load-bearing rules are:

- codegen is the only sampling path whose output may become catalog or
  certification evidence;
- a custom sampling script represents a missing codegen capability and must be
  replaced by typed recipe support plus a regression test;
- the generated routine must be checked against every material field in the
  recipe; and
- no recipe refactor may discard failed attempts, review history, diagnoses,
  gate evidence, provenance, or unknown legacy annotations.

## Test Layout

Tests mirror the source structure under `tuningfork/`:

```
tests/
├── base_method/            # 24 sampler wrappers
├── warmup/                 # 12 warmup wrappers
├── smc/                    # 6 SMC method wrappers
├── models/                 # Model-specific tests (one per model)
├── recipes/                # Recipe schema + emission + emit_script tests
├── metrics/                # Headline metric + diagnostics
├── reference/              # Reference cache I/O + posteriordb cross-checks
├── notebooks/              # tuningfork.catalog.inspect / render tests
├── numpyro/                # NumPyro integration helpers
├── e2e/                    # End-to-end gate tests
│
├── test_api_pins_mcmc.py   # BlackJAX MCMC contract tripwires (cross-cutting)
├── test_api_pins_warmup.py # BlackJAX warmup contract tripwires
├── test_api_pins_smc.py    # BlackJAX SMC contract tripwires
├── test_registry.py        # BASE_METHODS + SMC_METHODS registry (cross-cutting)
├── test_diagnostics.py     # ArviZ family-aware renderers (cross-cutting)
├── conftest.py             # Marker registration
└── fixtures.py             # Shared fixtures (RNG keys, toy MVN logdensity, etc.)
```

Each folder contains an `__init__.py` for package discovery. Cross-cutting tests (contract pins, registry, diagnostics) stay at the root. The `inference/` parent directory was removed in PR #10 (R3 code reorg, 2026-05-17) — base_method/warmup/smc/recipes are now flat-top under `tests/` to mirror the source layout.

## Test Markers

Five markers classify tests by cost and dependency:

| Marker | Meaning | Wall Time | Default? |
|--------|---------|-----------|----------|
| `fast` | Pure logic, dataclass, schema — no JAX trace or chain. | <100 ms | Yes |
| `slow` | Runs a chain or warmup with JAX compilation. | >1 s | Yes |
| `e2e` | End-to-end gate test; multiple algorithms × models. | >10 s | Yes |
| `requires_posteriordb` | Needs the posteriordb data cache; fails offline. | N/A | Yes |
| `benchmark` | pytest-benchmark perf-regression suite. Lives under `benchmarks/`, not `tests/`. | <240 s per cell (D5 cap) | No (`make benchmark` opt-in; weekly CI) |

**Discipline rule**: Every test must be tagged with **exactly one** of `fast`, `slow`, or `e2e`. The `requires_posteriordb` marker is additive (combine with `slow` or `e2e` if the test also needs posteriordb data).

See `tests/conftest.py` for marker registration (source of truth — markers are NOT duplicated in `pyproject.toml`).

## Common Invocations

| Intent | Command |
|--------|---------|
| Inner-loop dev (structural tests only) | `make test-fast` |
| Default contributor flow (skip e2e) | `make test` |
| Just slow tests | `make test-slow` |
| Full gate (everything) | `make test-full` |
| End-to-end suite only | `make test-e2e` |
| Before heavy runs: kill orphan processes | `make clean-orphans` |

All `test-*` targets automatically run `make clean-orphans` first (except `test-fast` to minimize overhead). See META-014 in `/home/jp/blackjax-devs/WORKLOG.md` for details on orphan processes.

## Adding a New Test

1. **Pick the right subfolder** based on what you're testing:
   - Model correctness → `tests/models/test_<model_name>.py`
   - Warmup wrapper → `tests/warmup/test_<strategy_name>.py`
   - Algorithm → `tests/base_method/test_<algorithm_name>.py`
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

Post-R3 restructure (2026-05-17), the package is split into two layers.

**Generator layer** (produces recipes):

| Component | Module Path | Tests Location |
|-----------|-------------|-----------------|
| Algorithm wrappers (HMC, NUTS, MALA, etc.) | `tuningfork/base_method/` | `tests/base_method/` |
| Warmup strategies (Stan window, Pathfinder, etc.) | `tuningfork/warmup/` | `tests/warmup/` |
| SMC variants | `tuningfork/smc/` | `tests/smc/` |
| Models (MVN, funnel, horseshoe, etc.) | `tuningfork/model/` | `tests/models/` |
| Recipe schema + generators + emit_script templates | `tuningfork/recipes/` | `tests/recipes/` |
| Metrics (headline, diagnostics) | `tuningfork/metrics/` | `tests/metrics/` |
| Reference certification + xcheck logic | `tuningfork/calibration/certify_reference.py`, `tuningfork/_cache_io.py`, `tuningfork/_posteriordb_xcheck.py` | `tests/reference/` |
| SMC runner | `tuningfork/runner/` | `tests/runner/` |

**Catalog layer** (user-facing; consumes recipes):

| Component | Module Path | Tests Location |
|-----------|-------------|-----------------|
| `load_recipe`, `summarize_recipe` | `tuningfork/catalog/inspect.py` | `tests/notebooks/test_inspect.py` |
| `load_samples`, `load_chain_stats`, `load_idata`, `samples_to_idata` | `tuningfork/catalog/render.py` | `tests/notebooks/test_render.py` |
| ArviZ family-aware diagnostic renderers | `tuningfork/catalog/diagnostics.py` | `tests/test_diagnostics.py` |
| `emit_script` (recipe → standalone `.py`) | `tuningfork/catalog/emit.py`, `tuningfork/recipes/_emit_script.py`, `tuningfork/recipes/_emit/` | `tests/recipes/test_emit_script.py` |
| Per-model artifacts (lessons.md, groundtruth.json, recipes/, reference/) | `tuningfork/catalog/<model>/` | (artifact-only; verified via parametric tests in `tests/recipes/test_schema.py`) |

**Benchmark suite** (initial release):

| Component | Module Path | Notes |
|-----------|-------------|-------|
| Recipe perf-regression benchmarks | `benchmarks/test_fast_recipes.py` | opt-in via `make benchmark`; nightly CI + per-PR (Tier 1 calibrated) |

### Running the benchmark suite

The benchmark suite requires the `bench-perf` dep group (`pytest-benchmark` + all `bench` deps):

```bash
# Install bench-perf deps
uv sync --group bench-perf

# Full nightly benchmark: Tier 1+2, e2e + calibrated (~15 min)
make benchmark

# PR-speed benchmark: Tier 1 calibrated only (~60s)
make benchmark-pr
```

### Benchmark design (D5 caps)

- **D5 budget cap**: select < 180 s, exec ≤ 240 s per cell.
- **n_samples=500**: minimum for reliable GT z-score at threshold=2.0.
- **GT-correctness gate**: each benchmark asserts `max_abs_mean_z < 2.0` vs `reference/summary.json` AFTER the timed run. This makes the benchmark a *correctness regression test*, not just timing.
- **Two modes**: `e2e` (full warmup+sample) and `calibrated` (generated execution using the canonical pinned no-warmup replay, sample only). MCLMC cells remain `e2e` where the benchmark is intended to measure tuning; Laplace cells are `e2e` because pinned replay still needs a phi-space initializer.
- **GitHub Actions**: nightly runs full Tier 1+2; per-PR runs Tier 1 calibrated only. Results stored as artifacts (90-day retention) + trend tracking via `benchmark-action/github-action-benchmark`.

### Adding a new benchmark cell

1. Emit a PASS recipe for the new (model, sampler, warmup) cell.
2. Add an entry to `_BENCH_CELLS` in `benchmarks/test_fast_recipes.py`:
   ```python
   ("tier1", "model_name", "recipe_filename.json", "e2e"),      # e2e mode
   ("tier1", "model_name", "recipe_filename.json", "calibrated"),  # calibrated (if supported)
   ```
3. Run `make benchmark-pr` locally to verify it completes within D5 budget.
4. The GT-correctness check runs automatically.

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
