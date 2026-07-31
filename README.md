# tuningfork

A reproducible reference catalog + methodology exemplar for MCMC, VI, and SMC sampling algorithms. Calibrated hyperparameter recipes for 24 samplers × 12 warmups × 14 models, with **certified reference draws and auto-gate verdicts** — allowing principled comparisons on gradient budget and effective-sample-size metrics.

## The garden of forking paths

Borges's *Garden of Forking Paths* (1941) gave Gelman & Loken ([2013][gl2013]) a metaphor for one of the subtler problems in applied statistics: even without conscious p-hacking, the implicit multiple-comparison cost of contingent analysis choices produces results that look principled but are not reproducible. MCMC tuning has its own garden — every choice in the (warmup, sampler, step-size, mass matrix, seed, parameterization) tuple is a fork. A sampler that "works" on a model often works because the practitioner walked far enough into the garden to find a path that did, not because the path itself was principled.

`tuningfork` maps the garden. Each cell in the 24 × 10 × 14 (base methods × warmups × models) inventory is an explicit fork; every recipe records the seed, adapted parameters, and auto-gate verdict that certified it. The `Effort` taxonomy makes the cost of a fork visible — LOW means library defaults pass the auto-gate at first emit, MEDIUM means a single statistician-led workaround was required, HIGH means a full Bayesian-workflow investigation. And the auto-gate criteria (R̂ < 1.01, min bulk-ESS ≥ 400, zero divergences) are committed *before* sampling, so a recipe's verdict cannot be retroactively redefined. The canonical definitions live in `tuningfork/recipes/_base.py` (Effort enum) and `tuningfork/calibration/statistician_gate.py` (auto-gate).

[gl2013]: https://sites.stat.columbia.edu/gelman/research/unpublished/p_hacking.pdf

## Coverage

**Current scope (v1):** NUTS + HMC families + MCLMC + partial SMC recipes. ~20% of the full 24 × 12 × 14 inventory has recipes shipped; the remaining 80% is defer-scoped. Per-model artifacts are committed for the 14-model suite; recipes land incrementally as the statistical validation phase completes.

**Out of scope for v1:** SGMCMC, VI (Laplace, VI warmups), specialised samplers, multi-chain reference generation, step-policy oracle harvesting, standalone external-sampler interop. Roadmap in CONTRIBUTING.md.

## Questions

When to use tuningfork:
- *"What is the best calibrated HMC config for Neal's funnel, and how many leapfrog steps does it cost per effective sample?"*
- *"Is MCLMC actually worth it on a 500-D state-space model when both algorithms are tuned to their best?"*

Future cross-sampler comparison questions (deferred to v2, pending full-suite recipe availability):
- *"Does Pathfinder→HMC dominate Stan-window→HMC on hierarchical models, or only on well-conditioned ones?"*
- *"How does VI with full-rank covariance compare to MCMC on the sparse horseshoe?"*

## Status

**Recipe generation.** Inventory close-out (2026-05-10, `32613f4`) wrapped the BlackJAX in-scope inventory: 24 sampler kernels × 12 warmups × 6 SMC variants × 14 models. Recipe-generation prep landed 2026-05-11 (sample-quality metric in `tuningfork/metrics/reference_compare.py` + diagnostic notebook at `notebooks/recipe_diagnostics.md`). Per-cell `Recipe` artifacts that pass the auto-gate are emitted as recipe sweeps execute, and recipe coverage continues to expand.

## Suite (14 models)

| # | Name | Dim | Class |
|---|------|-----|-------|
| 1 | Standard MVN (diagonal) | 10 | Gaussian baseline |
| 2 | Ill-conditioned correlated Gaussian | 50 | Ill-conditioned (κ≈1000) |
| 3 | Eight Schools (NCP) | 10 | Hierarchical |
| 4 | Neal's Funnel | 10 | Funnel |
| 5 | Banana (Rosenbrock) | 2 | Curved manifold |
| 6 | Radon hierarchical | 390 | Hierarchical+funnel |
| 7 | Synthetic logistic regression | 3 | GLM baseline |
| 8 | German Credit logistic regression | 26 | GLM real data |
| 9 | Sparse horseshoe linear regression | 204 | Sparse / heavy-tailed |
| 10 | IRT (2PL) | 144 | Hierarchical, scale-identifiability |
| 11 | 25-mode Gaussian mixture | 2 | Multimodal |
| 12 | Stochastic volatility | 503 | Latent-Gaussian / state-space |
| 13 | Lotka–Volterra ODE inverse | 7 | Nonlinear, expensive likelihood |
| 14 | GP regression (1D) | ~200 | Latent-Gaussian (latent GPs not yet marginalised — future GP-latent marginalisation work) |

## Recipe matrix (excerpt)

The statistician-drafted recipe matrix (full version in [`RECIPE_GENERATION.md`](RECIPE_GENERATION.md)) assigns a per-cell colour verdict across the full inventory. Legend: **G** = LOW effort (library defaults pass the auto-gate at first emit), **Y** = MEDIUM (one statistician-led workaround recovers), **R** = HIGH (full Bayesian-workflow investigation) OR hard-excluded category.

The canonical baseline table — `window_adaptation_diag_imm` warmup × NUTS-family samplers — is the window-adaptation × HMC-family sweep build target:

| Warmup + Sampler | mvn_10 | ill_cond_50 | logistic_syn | eight_schools | lotka_volterra | radon | irt_2pl | german_credit | neals_funnel | gmm_25 | banana | horseshoe | gp_regression | stoch_vol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| window_adaptation_diag_imm + **nuts** | G | G | G | G | G | G | G | G | Y | R | Y | Y | G | Y |
| window_adaptation_diag_imm + **hmc** | G | G | G | G | G | Y | Y | G | Y | R | Y | Y | G | Y |
| window_adaptation_diag_imm + **mhmc** | G | G | G | G | G | Y | Y | G | Y | R | Y | Y | G | Y |
| window_adaptation_diag_imm + **mala** | G | Y | G | G | Y | Y | Y | G | R | R | Y | Y | Y | R |
| window_adaptation_diag_imm + **barker** | G | Y | G | G | Y | Y | Y | G | R | R | Y | Y | Y | R |
| window_adaptation_diag_imm + **rmhmc** | G | G | G | G | G | Y | Y | G | Y | R | Y | R | Y | R |

Gaps in Table 1 are filled by other warmup families: **MCLMC + `mclmc_tuning`** is green on `stoch_vol` (d=503) — the canonical case where NUTS `default_works=False` (the MCLMC recipe sweep target). **SMC + `adaptive_tempered`** is green on `gmm_25` — the only viable path for the 25-mode mixture, since any single-chain gradient sampler gets trapped (the SMC recipe sweep target).

### Cell-count summary

Across all 8 sub-tables (24 base methods × 12 warmups × 14 models, plus 6 SMC outer × 8 inner-kernel cells ≈ 1080 unique triples):

| Effort tier | Approx count | Description |
|---|---|---|
| 🟢 LOW (Green) | ~480 | conventional `(warmup, sampler)` pairing — library defaults pass auto-gate at first emit |
| 🟡 MEDIUM (Yellow) | ~180 | statistician investigation: seed/init/bug-fix workaround OR unconventional pairing (e.g., `window_adaptation_diag_imm + mala`) |
| 🔴 HIGH / hard-excluded (Red) | ~420 | dominated by 8 exclusion categories: multimodal × single-chain gradient, VI × pathological, Laplace × non-Gaussian-latent, `no_warmup` × high-d, MCLMC inside SMC, `rmhmc` without callable metric, `fullrank_vi` warmup at d>30, elliptical/mgrad outside Gaussian-prior models |

The full 8-table matrix, supersession map (e.g., `adaptive_tempered_smc` strictly dominates `tempered_smc`), and hard-exclusion category definitions live in [`RECIPE_GENERATION.md`](RECIPE_GENERATION.md).

## Calibration pipeline

Recipe construction draws on three building blocks under `tuningfork/calibration/`:

- **`certify_reference.py` — Gold reference draws**: 1 chain × 40 000 samples (NUTS + Stan window adaptation), reshaped into 4 chunks for rank-normalized split-R̂ (Vehtari et al. 2021). Multimodal exception for `gmm_25` (parallel-tempered SMC + multi-restart with mode-coverage check).
- **`tune.py` — Hyperparameter optimization**: Optuna BO maximizing `min-bulk-ESS / total_grad_evals`, with per-algorithm acceptance targets.
- **`statistician_gate.py` — Auto-gate**: pre-committed thresholds (R̂ < 1.01, min bulk-ESS ≥ 400, divergences = 0, `max_abs_mean_z` < 2) that every recipe must clear before emission. Thresholds are fixed before sampling — see "The garden of forking paths" above.

The `Recipe.effort` field (`tuningfork/recipes/_base.py`) records the resulting cost class: LOW (defaults pass at first emit), MEDIUM (single statistician-led workaround), HIGH (full Bayesian-workflow investigation).

## Headline metric

`primary = min_over_dimensions(bulk_ESS) / total_gradient_evaluations`

## Setup

**Prerequisite: Git LFS.** The 14-model catalog ships canonical 40k-sample
groundtruth draws (~270 MB total) as `.npz` files tracked by Git LFS at
`tuningfork/catalog/<model>/groundtruth_samples/blackjax/{draws,chain_stats}.npz`.
On a fresh clone these are text pointer stubs until you fetch the actual
binaries — `np.load` will raise a misleading `ValueError: This file contains
pickled (object) data` when handed a pointer.

```bash
# One-time per machine: install the git-lfs binary
sudo apt-get install git-lfs        # Debian / Ubuntu
sudo dnf install git-lfs            # Fedora / RHEL
brew install git-lfs                # macOS
# Releases for other platforms: https://github.com/git-lfs/git-lfs/releases

# One-time per clone: register LFS hooks + fetch the .npz blobs
git lfs install
git lfs pull
```

Verify with `file tuningfork/catalog/banana/groundtruth_samples/blackjax/draws.npz`
— expect `Zip archive data`, not `ASCII text`.

```bash
make install      # uv sync --group bench
make test         # run tests (default: skip e2e suite)
make test-fast    # inner-loop dev (fast tests only)
make test-full    # merge gate (everything)
make benchmark    # weekly perf-regression suite (D5 thresholds; opt-in)
make lint         # pre-commit
```

For GPU: `uv pip install "jax[cuda12]"` after `make install`.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for a complete guide to test markers (`fast`, `slow`, `e2e`, `benchmark`), folder layout, and adding new tests.

## Using the catalog

The user-facing API for consuming recipes is in `tuningfork.catalog`:

```python
from tuningfork.catalog import load_recipe, load_idata, summarize_recipe, emit_script

# Inspect a committed recipe + render its sample-quality diagnostics
recipe = load_recipe("tuningfork/catalog/eight_schools_ncp/groundtruth.json")
print(summarize_recipe(recipe))      # auto-renders as DataFrame in Jupyter
idata = load_idata(recipe)           # posterior + sample_stats (ArviZ-ready)

import arviz as az
az.plot_trace(idata)
az.summary(idata)

# Reproduce the recipe in a fresh environment (no tuningfork dependency in
# the inference choreography; only the model definition imports tuningfork.model)
script = emit_script(recipe, num_samples=2000)
from pathlib import Path
Path("run_eight_schools.py").write_text(script)
# $ uv run --with tuningfork --with jax --with blackjax --with numpyro \
#       python run_eight_schools.py
```

Per-model artifacts live under `tuningfork/catalog/<model>/`:

- **`lessons.md`** — distilled "what's tricky about sampling this model" knowledge ([example](tuningfork/catalog/stoch_vol/lessons.md))
- **`groundtruth.json`** — canonical long-NUTS reference recipe (for NUTS-path models) or analytic sampler config
- **`groundtruth.imm.npz`** — high-dim inverse-mass-matrix sidecar (5 high-d models: gp_regression, horseshoe, irt_2pl, radon, stoch_vol)
- **`reference/{metadata,summary,adaptation,xcheck}.json`** — committed cert artifacts (long-NUTS gold-standard run)
- **`recipes/{low,medium,high,failed}__*.json`** — per-cell recipes from the recipe-generation pipeline. 7 canonical FAILED recipes ship today documenting the [hard-exclusion categories](RECIPE_GENERATION.md#hard-exclusions) (e.g., `gmm_25/recipes/failed__nuts__window_adaptation_diag_imm.json` documents the "multimodal × single-chain gradient" exclusion). LOW/MEDIUM/HIGH recipes land as recipe sweeps execute.

## Layout

```
tuningfork/
├── tuningfork/                # the Python package
│   │   # ─── generator layer (produces recipes) ───
│   ├── model/                 # 14 NumPyro models + MODELS, MODELS_BY_FAMILY
│   │   └── _data/             # raw input datasets (CSV/NPZ); fetch via tools/
│   ├── base_method/           # 24 sampler wrappers (hmc, nuts, mclmc, ...)
│   ├── warmup/                # 12 warmup wrappers (window_adaptation_diag_imm, pathfinder, ...)
│   ├── smc/                   # declarative SMC method descriptors
│   ├── recipes/               # Recipe schema + generators + emit_script
│   │   ├── _base.py, _instructions.py
│   │   ├── _generate_starter.py, _generate_groundtruth.py
│   │   ├── _emit_script.py    # recipe → reproduction Python script — orchestrator
│   │   ├── _emit_smc_script.py # SMC recipe → generated sampling program
│   │   └── _emit/             # Python emit-functions (preamble/postamble/inference_loop/sampler/warmup)
│   ├── calibration/           # certify_reference, tune (Optuna BO), statistician_gate
│   ├── metrics/               # headline metric, grad-counter, reference_compare
│   ├── _cache_io.py           # internal cache I/O for reference artifacts
│   ├── _posteriordb_xcheck.py # posteriordb cross-check logic
│   ├── cli.py                 # tuningfork {reference, warmup, tune} subcommands
│   │
│   │   # ─── catalog layer (user-facing, consumes recipes) ───
│   └── catalog/               # USER-FACING subpackage
│       ├── inspect.py         # load_recipe, summarize_recipe
│       ├── render.py          # load_samples, load_chain_stats, load_idata, samples_to_idata
│       ├── diagnostics.py     # ArviZ family-aware diagnostic renderers
│       ├── emit.py            # emit_script (recipe → standalone .py script)
│       ├── notebooks/         # template + worked-example notebooks
│       │   ├── recipe_diagnostics.md  # parametrized inspection template
│       │   ├── inspect_example.md     # worked example
│       │   └── inspect_README.md      # docs for the catalog API
│       └── <model>/           # per-model artifacts (one dir per certified model)
│           ├── lessons.md          # distilled sampling-quirks history
│           ├── groundtruth.json    # canonical groundtruth recipe pin
│           ├── groundtruth.imm.npz # high-dim IMM sidecar (5 models)
│           ├── reference/          # committed cert artifacts
│           │   ├── metadata.json, summary.json
│           │   ├── adaptation.json (NUTS-path only)
│           │   └── xcheck.json     (posteriordb cross-check; eight_schools_ncp + radon)
│           ├── recipes/            # per-cell recipes (from recipe sweeps)
│           │   └── {low,medium,high,failed}__<sampler>__<warmup>.json
│           └── _cache/             # gitignored runtime cache
│               ├── draws.npz, chain_stats.npz
│               └── warmup_checkpoint/
├── tests/                     # source-mirroring test layout
├── benchmarks/                # pytest-benchmark perf-regression suite
│   └── test_fast_recipes.py
├── tools/                     # data fetch + generation scripts
├── .github/workflows/         # pre-commit + test-fast + test-slow + benchmark CI
├── CLAUDE.md                  # contributor / agent guide
├── RECIPE_GENERATION.md       # statistician-authored recipe matrix + plan
└── CONTRIBUTING.md            # test markers, folder layout, contribution rules
```

The package splits into two layers: the **generator** layer (`model/`,
`base_method/`, `warmup/`, `smc/`, `recipes/`, `calibration/`, `metrics/`,
`_cache_io.py`) produces recipes through generated programs; the **catalog**
layer (`tuningfork.catalog`) loads recipes, artifacts, and generated source.

## License

Apache 2.0
