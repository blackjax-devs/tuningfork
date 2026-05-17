# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`tuningfork` is a BlackJAX-native benchmark library for MCMC / VI / SMC samplers. **Phase 5 closed at `32613f4` (2026-05-10) with the complete in-scope BlackJAX inventory wrapped**: 24 base methods × 10 warmups × 6 SMC methods, composed against a 14-model suite. The recipe generation phase is now active; the active plan is [`RECIPE_GENERATION.md`](RECIPE_GENERATION.md) (statistician-authored, 8-table colour-coded effort matrix + supersession map). The library will be open-sourced once the initial set of recipes lands.

Architecture decisions: 14-model suite, calibration protocol (certified reference draws as ground truth, BO over hyperparameters, warmup-only execution), headline metric `min-bulk-ESS / total_grad_evals`. Phase-by-phase history + frozen design snapshots (Phases 1-5 plans, audits, retrospectives) live under `../archive/bjx-bench/`.

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
- `optuna` — Bayesian optimization for hyperparameter tuning
- `posteriordb` — reference cross-check (per resolved decision: cross-check against Stan refs for shared posteriors #3 8-Schools, #6 radon, #10 IRT)
- `pytest`, `pytest-cov`, `pre-commit`

## Test Suite & Markers

Tests are organized under `tests/` mirroring the source layout in `tuningfork/`:

```
tests/
├── inference/           # base_method, warmup, smc
├── models/              # model-specific tests
├── recipes/             # recipe schema + emission
├── metrics/             # headline metric + diagnostics
├── reference/, tuning/  # reference certification + BO tuning
├── runner/              # SMC runner helpers
├── e2e/                 # end-to-end gate suite
├── test_api_pins_mcmc.py     # BlackJAX MCMC kernel contracts
├── test_api_pins_warmup.py   # BlackJAX warmup + adapter contracts
├── test_api_pins_smc.py      # BlackJAX SMC contracts
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

**Three `test_api_pins_*.py` files at root** — split by sampler family: `test_api_pins_mcmc.py` (MCMC base methods), `test_api_pins_warmup.py` (warmups + adapter contracts), `test_api_pins_smc.py` (SMC family). Append new tripwires to the right family file; do not create a single `test_api_pins.py`.

**Mandatory for agents**:
- Run `make clean-orphans` before any heavy test sweep — orphan Python REPLs can silently consume 7+ GB. The underlying script lives at `~/claude-config/tools/clean_orphans.sh` (cross-repo); override the path with `CLAUDE_CONFIG_DIR` if needed.
- When adding a test, tag it with **exactly one** of `@pytest.mark.fast` / `@pytest.mark.slow` / `@pytest.mark.e2e`. Use module-level `pytestmark = pytest.mark.<marker>` if all tests in the file are the same kind.

For full contributor guidelines, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Architecture

Source layout (top-level subpackages of `tuningfork/`). Post-R3 two-layer
split (2026-05-17): **generator layer** (wiring code that produces recipes)
vs **catalog layer** (artifacts + user-facing helpers that consume them).

```
tuningfork/
│   # ─────────── GENERATOR LAYER (produces recipes) ───────────
├── model/                     # 14 Posterior definitions
│   ├── _registry.py           # MODELS + MODELS_BY_FAMILY assembly
│   ├── _base.py, _numpyro.py  # Posterior schema + logdensity builder
│   ├── _data/                 # raw input datasets (CSV/NPZ; was tuningfork/data/)
│   ├── <model>.py × 14        # one file per model (banana, eight_schools_ncp, ...)
│   └── __init__.py            # thin re-export from _registry
├── base_method/               # 24 sampler wrappers — see ENTRIES list below
├── warmup/                    # 10 warmup wrappers — see ENTRIES list below
├── smc/                       # 6 SMC method wrappers — see ENTRIES list below
├── recipes/                   # Recipe schema + generators (CODE ONLY)
│   ├── _base.py, _instructions.py
│   ├── _generate_starter.py, _generate_groundtruth.py
│   ├── _emit_script.py        # recipe → reproduction Python script (R3.5)
│   └── _templates/            # .py.tmpl templates (string.Template substitution)
│       ├── preamble.py.tmpl       # imports + `from tuningfork.model import MODELS`
│       ├── postamble.py.tmpl, inference_loop.py.tmpl
│       ├── warmups/<warmup>.py.tmpl    # window_adaptation_diag_imm, no_warmup. R3.5b-2: pathfinder, multipathfinder, meads, chees, mclmc_tuning, adjusted_mclmc_tuning, meanfield_vi, fullrank_vi
│       └── samplers/<sampler>.py.tmpl  # nuts, hmc, dynamic_hmc, mhmc, dmhmc, ghmc, mala, barker, rwm. R3.5b-2: mclmc, adjusted_mclmc, adjusted_mclmc_dynamic, orbital_hmc, rmhmc, elliptical_slice, mgrad_gaussian, laplace_{hmc,dhmc,mhmc,dmhmc}, irmh, additive_step_random_walk, meanfield_vi, fullrank_vi
│       (no models/ subdir — model definitions imported from tuningfork.model,
│        not duplicated as templates; eliminates drift risk)
├── calibration/               # cert + Optuna BO + auto-gate
│   ├── certify_reference.py, certify_reference_analytic.py
│   ├── tune.py, statistician_gate.py
├── metrics/                   # headline + grad_counter + reference_compare
├── runner/smc.py              # SMC init/run helpers
├── _cache_io.py               # internal cache I/O (was reference/_io.py)
├── _posteriordb_xcheck.py     # internal posteriordb xcheck (was reference/_posteriordb_xcheck.py)
├── cli.py                     # CLI: tuningfork reference / warmup / tune
│
│   # ─────────── CATALOG LAYER (consumes recipes) ───────────
└── catalog/                   # USER-FACING SUBPACKAGE
    ├── __init__.py            # re-exports load_recipe, load_idata, summarize_recipe, emit_script, ...
    ├── inspect.py             # load_recipe, summarize_recipe
    ├── render.py              # load_samples, load_chain_stats, load_idata, samples_to_idata
    ├── diagnostics.py         # ArviZ family-aware diagnostic renderers
    ├── emit.py                # emit_script wrapper (R3.5; recipe → standalone .py)
    ├── notebooks/             # template + example notebooks (was repo-root notebooks/)
    │   ├── recipe_diagnostics.md   # parametrized inspection template
    │   ├── inspect_example.md      # worked example
    │   └── inspect_README.md       # user-API docs
    └── <model>/               # PER-MODEL ARTIFACTS (from R2)
        ├── groundtruth.json   # committed; canonical groundtruth recipe pin
        ├── groundtruth.imm.npz # committed; high-dim IMM sidecar (where applicable)
        ├── reference/         # committed cert artifacts
        │   ├── metadata.json, summary.json, adaptation.json, xcheck.json
        ├── recipes/           # per-cell recipes (post Recipe Phase 1+)
        │   └── low__*.json, medium__*.json, high__*.json, failed__*.json
        └── _cache/            # gitignored runtime cache
            ├── draws.npz, chain_stats.npz
            └── warmup_checkpoint/
```

The generator layer is what tuningfork **uses internally** to produce
recipes (Phase 0 groundtruth runs, Phase 1+ recipe emission, calibration,
benchmarks). The catalog layer is what an **external user** imports:
`from tuningfork.catalog import load_recipe, load_idata, summarize_recipe`
and reads the per-model artifact subdirs. A regular user never needs to
know about the generator layer.
```

User-facing template notebooks live at `tuningfork/catalog/notebooks/`
(`recipe_diagnostics.md`, `inspect_example.md`, `inspect_README.md`) —
post-R3 (2026-05-17), they sit inside the catalog subpackage next to the
recipes they operate on. As of 2026-05-12, `recipe_diagnostics` handles
NUTS/HMC only; other sampler families are deferred to Recipe Phases 2+.

### Inventory

**24 base methods** (`base_method/__init__.py:BASE_METHODS`): hmc, nuts, dynamic_hmc, mhmc, dmhmc, ghmc, mala, barker, rwm, irmh, additive_step_random_walk, mclmc, adjusted_mclmc, adjusted_mclmc_dynamic, orbital_hmc, rmhmc, elliptical_slice, mgrad_gaussian, laplace_hmc, laplace_dhmc, laplace_mhmc, laplace_dmhmc, meanfield_vi, fullrank_vi.

**10 warmups** (`warmup/__init__.py:WARMUPS`): no_warmup, window_adaptation_diag_imm, low_rank_window_adaptation, pathfinder, multipathfinder, meads, chees, mclmc_tuning, adjusted_mclmc_tuning, laps, meanfield_vi, fullrank_vi.

**6 SMC methods** (`smc/__init__.py:SMC_METHODS`): adaptive_tempered_smc, tempered_smc, partial_posteriors_smc, inner_kernel_tuning, persistent_sampling_smc, adaptive_persistent_sampling_smc.

### Specialised factories (`extra_required_kwargs`)

Some base methods need kwargs beyond the standard `(logdensity_fn, step_size, inverse_mass_matrix, ...)` shape — schema field `BaseMethod.extra_required_kwargs: tuple[str, ...]` declares them so the recipe runner can inject from `Posterior` metadata at call time. Currently:

| Method | Extra required kwargs |
|---|---|
| `mgrad_gaussian`, `elliptical_slice` | `("prior_cov", "prior_mean")` |
| `irmh` | `("proposal_distribution",)` |
| `additive_step_random_walk` | `("proposal_generator",)` |
| `laplace_hmc`, `laplace_dhmc`, `laplace_mhmc`, `laplace_dmhmc` | `("log_joint_fn", "theta_init")` |

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

## Reference protocol (the load-bearing decision)

Per user direction (2026-05-07; **amended 2026-05-11** per [`worklog/decisions/2026-05-11-phase0-reference-protocol-refinements.md`](../worklog/decisions/2026-05-11-phase0-reference-protocol-refinements.md)), the reference-draws protocol uses a **single long chain reshaped into chunks** rather than multi-chain × shorter:

- Default: 1 chain × 5,000 warmup × **40,000** post-warmup samples (NUTS + Stan window adaptation). Matches posteriordb Stan reference convention (~40k total).
- Reshape into **4** contiguous chunks of **10,000** → rank-normalized split-R̂ (Vehtari et al. 2021).
- Certification gate: split-R̂ < 1.01, min per-chunk bulk-ESS > 400, 0 divergences, E-BFMI > 0.3.
- **Per-step chain_stats** (`num_integration_steps`, `energy`, `is_divergent`, `acceptance_rate`, plus other NUTSInfo fields) persisted to `reference/<name>/chain_stats.npz` (gitignored) on every cert run — also on failure path, so failed cells leave diagnostic crumbs for the statistician.
- **Cert failure policy**: when a model fails cert at default `n_samples`, escalate to the statistician (`STATISTICIAN_BAYESIAN_WORKFLOW.md` + `STATISTICIAN_DIAGNOSTICS_RECIPE.md`). Do NOT brute-force the gate by inflating `n_samples` — `min_ess` is an absolute threshold and bumping `n` is gate-gaming, not diagnostic validation.
- **Cache invalidation policy**: do NOT pre-emptively delete cache entries when the spec changes. Existing entries remain valid for their original purpose (metadata.json records actual `num_samples`). When a downstream consumer needs different data, trigger fresh via `force_regenerate=True`; the statistician — not the engineer — has authority to mark a cached groundtruth as "needs redo" based on chain_stats pathology.
- **Multimodal exception**: model #11 (25-mode Gaussian mixture) cannot use single-chain — uses parallel-tempered SMC + 8 well-separated cold restarts, with mode-coverage check (each of 25 modes ≥ 1% of draws).
- **Posteriordb cross-check** for #3, #6, #10: compare own marginal mean/std/5%/95% to Stan reference; tolerance |Δmean|<2 SE, |std ratio − 1|<0.05; discrepancies logged to `catalog/<model>/reference/xcheck.json`.

## Worklog

`/home/jp/blackjax-devs/WORKLOG.md` is the shared external memory across all three repos (`blackjax/`, `sampling-book/`, `tuningfork/`).
