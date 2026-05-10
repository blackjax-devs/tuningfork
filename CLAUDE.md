# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`bjx-bench` is a BlackJAX-native benchmark library for MCMC/VI/SMC samplers. The full design lives in `../PLAN_bjx_bench.md`. Read that first for context — the 14-model suite, 3-tier calibration protocol (Tier-A gold ref / Tier-B per-algorithm tuning / Tier-C warmup-isolated), and headline metric `min-bulk-ESS / total_grad_evals` are all specified there.

This is a **sibling repo** to `blackjax/` and `sampling-book/`, not a subdir of either. Heavy deps (Optuna, datasets, plotting) live here so `blackjax/` core stays light.

## Commands

**Package manager:** [`uv`](https://docs.astral.sh/uv/) — use `uv run` instead of activating the venv.

```bash
make install      # uv sync --group bench
make test         # run tests (default flow: skip e2e)
make test-fast    # fast / structural tests only
make test-full    # everything (merge gate)
make lint         # uv run pre-commit run --all-files
```

`pyproject.toml` mirrors `sampling-book/pyproject.toml` (same model-implementation deps) and adds:
- `optuna` — Tier-B Bayesian optimization
- `posteriordb` — Tier-A cross-check (per resolved decision: cross-check against Stan refs for shared posteriors #3 8-Schools, #6 radon, #10 IRT)
- `pytest`, `pytest-cov`, `pre-commit`

## Test Suite & Markers

Tests are organized under `tests/` mirroring the source layout in `bjx_bench/`:

```
tests/
├── inference/           # base_method, warmup, smc
├── models/              # model-specific tests
├── recipes/             # recipe schema + emission
├── metrics/             # headline metric + diagnostics
├── tier_a/, tier_b/     # certification + optimization
├── e2e/                 # end-to-end phase-gate suite
├── test_api_pins.py     # BlackJAX upstream contract (cross-cutting)
└── test_registry.py     # registry checks (cross-cutting)
```

**Five markers** (registered in `conftest.py`, source of truth):

| Marker | Meaning |
|--------|---------|
| `fast` | Pure logic / dataclass / schema (no JAX trace, <100 ms) — **inner-loop dev** |
| `slow` | Chain-running or warmup tests (JAX-compiled, >1 s) — **default suite** |
| `e2e` | End-to-end phase gate (multiple algorithms × models, >10 s) — **merge gate** |
| `requires_posteriordb` | Needs posteriordb data cache; additive (combine with `slow` or `e2e`) |
| `benchmark` | Reserved for perf benchmarks (opt-in via `-m benchmark`) |

**Discipline rule**: Every test must be tagged with exactly one of `fast`, `slow`, or `e2e`. If a test needs posteriordb, add `@pytest.mark.requires_posteriordb` as a second marker.

**Mandatory for agents**:
- Run `make clean-orphans` before any heavy test sweep. See META-014 in `/home/jp/blackjax-devs/WORKLOG.md` — orphan Python REPLs can silently consume 7+ GB.
- When adding a test, tag it with **exactly one** of `@pytest.mark.fast` / `@pytest.mark.slow` / `@pytest.mark.e2e`. Use module-level `pytestmark = pytest.mark.<marker>` if all tests in the file are the same kind.

For full contributor guidelines, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Architecture (per PLAN_bjx_bench.md)

```
bjx_bench/
├── registry/         # PosteriorEntry definitions, grouped by class
│   ├── gaussians/        # models 1, 2
│   ├── hierarchical/     # models 3, 6, 10
│   ├── pathological/     # models 4, 5, 11
│   ├── glm/              # models 7, 8, 9
│   ├── latent_gaussian/  # models 12, 14
│   └── ode/              # model 13
├── data/             # raw datasets + generation scripts
├── reference/        # Tier-A artifacts
│   ├── draws/            # *.npz (gitignored; 100k-sample chains are large)
│   ├── summaries/        # *.json (mean, std, 5%, 95%)
│   ├── adaptation/       # *.json (step_size*, IMM*, num_leapfrog*)
│   └── posteriordb_xcheck/  # discrepancy reports
├── algorithms/       # thin wrappers around BlackJAX samplers; common signature
├── warmup/           # Tier-C warmup wrappers (Stan window, MEADS, ChEES, Pathfinder, MCLMC tuning, no-op)
├── calibration/
│   ├── tier_a.py     # 1×100k NUTS, 10-chunk split-R̂ certifier
│   ├── tier_b.py     # Optuna BO loop
│   └── targets.py    # acceptance-rate / energy-error objectives
├── metrics/
│   ├── headline.py   # min-bulk-ESS / total_grad_evals
│   ├── diagnostics.py
│   ├── reference_compare.py
│   └── grad_counter.py    # logdensity_fn wrapper that counts grad evals
├── runner/           # single-cell + matrix execution; persist to sqlite/parquet
├── reporting/
└── cli.py
```

## Out of scope for v1 (per resolved decisions)

- **SGMCMC** — deferred to v2 (no MH correction; harness needs special-casing).
- **External-sampler interop** — registry stays BlackJAX-only in v1; Stan/NumPyro adapters can be added later without breaking schema.
- **Bayesian neural networks** — neuron-permutation symmetry precludes certifiable references.
- **Discrete-latent models** — require marginalization first.

## General Principles

- **Engineering Excellence**:
    - Always work on a new branch derived from `main`/`HEAD`.
    - **Enriched Commit History**: Commit frequently to demonstrate the thinking process. Each commit message must include a "finding / what was the error and how that was fixed" line so the history captures trial-and-error.
    - **Workflow Strategy**: `{implement → commit → test → fix}_loop` (NOT `{implement → test → fix}_loop → commit`). One PR contains multiple commits that break down the steps.
    - Run pre-commit before every commit: `uv run pre-commit run --all-files`.
- **Principle of Least Action**: Minimize temporary workarounds for upstream breakages. Document and monitor.
- **JAX Compatibility**: Monitor NumPyro #2174 for JAX 0.10.0.
- **GPU/CUDA**: `uv sync` installs CPU JAX. For GPU: `uv pip install "jax[cuda12]"`.

## Notebook Conventions (when narrative tutorials land in `notebooks/`)

- Only commit `.md` files (MyST/Jupytext format). Never commit `.ipynb`.
- Convert with: `jupytext notebooks/foo.md --to notebook` (for editing) and `jupytext notebooks/foo.ipynb --to myst` (before committing).

## Reference: Tier-A protocol (the load-bearing decision)

Per user direction (2026-05-07), Tier-A uses a **single long chain reshaped into chunks** rather than multi-chain × shorter:

- Default: 1 chain × 5,000 warmup × 100,000 post-warmup samples (NUTS + Stan window adaptation).
- Reshape into 10 contiguous chunks of 10,000 → rank-normalized split-R̂ (Vehtari et al. 2021).
- Certification gate: split-R̂ < 1.01, min per-chunk bulk-ESS > 400, 0 divergences, E-BFMI > 0.3.
- **Multimodal exception**: model #11 (25-mode Gaussian mixture) cannot use single-chain — uses parallel-tempered SMC + 8 well-separated cold restarts, with mode-coverage check (each of 25 modes ≥ 1% of draws).
- **Posteriordb cross-check** for #3, #6, #10: compare own marginal mean/std/5%/95% to Stan reference; tolerance |Δmean|<2 SE, |std ratio − 1|<0.05; discrepancies logged to `reference/posteriordb_xcheck/`.

## Worklog

`/home/jp/blackjax-devs/WORKLOG.md` is the shared external memory across all three repos (`blackjax/`, `sampling-book/`, `bjx-bench/`). All work on this repo is tracked under `[TASK-002]`.
