# tuningfork

A BlackJAX-native benchmark library for comparing MCMC, VI, and SMC sampling algorithms — modeled after [`inference-gym`](https://pypi.org/project/inference-gym/) and [`posteriordb`](https://github.com/stan-dev/posteriordb), but designed around **calibrated, gradient-counted comparisons** over a curated 14-model suite.

## The garden of forking paths

Borges's *Garden of Forking Paths* (1941) gave Gelman & Loken ([2013][gl2013]) a metaphor for one of the subtler problems in applied statistics: even without conscious p-hacking, the implicit multiple-comparison cost of contingent analysis choices produces results that look principled but are not reproducible. MCMC tuning has its own garden — every choice in the (warmup, sampler, step-size, mass matrix, seed, parameterization) tuple is a fork. A sampler that "works" on a model often works because the practitioner walked far enough into the garden to find a path that did, not because the path itself was principled.

`tuningfork` maps the garden. Each cell in the 24 × 10 × 14 (base methods × warmups × models) inventory is an explicit fork; every recipe records the seed, adapted parameters, and auto-gate verdict that certified it. The `Effort` taxonomy makes the cost of a fork visible — LOW means library defaults pass the auto-gate at first emit, MEDIUM means a single statistician-led workaround was required, HIGH means a full Bayesian-workflow investigation. And the auto-gate criteria (R̂ < 1.01, min bulk-ESS ≥ 400, zero divergences) are committed *before* sampling, so a recipe's verdict cannot be retroactively redefined. The canonical definitions live in `tuningfork/inference/recipes/_base.py` (Effort enum) and `tuningfork/calibration/statistician_gate.py` (auto-gate).

[gl2013]: https://sites.stat.columbia.edu/gelman/research/unpublished/p_hacking.pdf

## Why

BlackJAX has 24 sampler kernels (22 MCMC + 2 VI), 10 warmup/adaptation strategies, and 6 SMC variants. None are currently benchmarked together with calibrated configurations, gradient-budget accounting, or posteriordb-style certified reference draws. `tuningfork` answers questions like:

- *"What is the best calibrated HMC config for Neal's funnel, and how many leapfrog steps does it cost per effective sample?"*
- *"Does Pathfinder→HMC dominate Stan-window→HMC on hierarchical models, or only on well-conditioned ones?"*
- *"Is MCLMC actually worth it on a 500-D state-space model when both algorithms are tuned to their best?"*

## Status

**Recipe generation phase.** Phase 5 (2026-05-10, `32613f4`) wrapped the BlackJAX in-scope inventory: 24 sampler kernels × 10 warmups × 6 SMC variants × 14 models. Recipe-generation prep landed 2026-05-11 (sample-quality metric in `tuningfork/metrics/reference_compare.py` + diagnostic notebook at `notebooks/recipe_diagnostics.md`). Recipe Phase 1 onward emits per-cell `Recipe` artifacts that pass the auto-gate. The library will be open-sourced once the initial set of recipes lands.

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
| 14 | GP regression (1D) | ~200 | Latent-Gaussian (latent GPs not yet marginalised — Recipe Phase 7 probe) |

## Recipe matrix (excerpt)

The statistician-drafted recipe matrix (full version in [`RECIPE_GENERATION.md`](RECIPE_GENERATION.md)) assigns a per-cell colour verdict across the full inventory. Legend: **G** = LOW effort (library defaults pass the auto-gate at first emit), **Y** = MEDIUM (one statistician-led workaround recovers), **R** = HIGH (full Bayesian-workflow investigation) OR hard-excluded category.

The canonical baseline table — `stan_window` warmup × NUTS-family samplers — is the Recipe Phase 1 build target:

| Warmup + Sampler | mvn_10 | ill_cond_50 | logistic_syn | eight_schools | lotka_volterra | radon | irt_2pl | german_credit | neals_funnel | gmm_25 | banana | horseshoe | gp_regression | stoch_vol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stan_window + **nuts** | G | G | G | G | G | G | G | G | Y | R | Y | Y | G | Y |
| stan_window + **hmc** | G | G | G | G | G | Y | Y | G | Y | R | Y | Y | G | Y |
| stan_window + **mhmc** | G | G | G | G | G | Y | Y | G | Y | R | Y | Y | G | Y |
| stan_window + **mala** | G | Y | G | G | Y | Y | Y | G | R | R | Y | Y | Y | R |
| stan_window + **barker** | G | Y | G | G | Y | Y | Y | G | R | R | Y | Y | Y | R |
| stan_window + **rmhmc** | G | G | G | G | G | Y | Y | G | Y | R | Y | R | Y | R |

Gaps in Table 1 are filled by other warmup families: **MCLMC + `mclmc_tuning`** is green on `stoch_vol` (d=503) — the canonical case where NUTS `default_works=False` (Recipe Phase 2 target). **SMC + `adaptive_tempered`** is green on `gmm_25` — the only viable path for the 25-mode mixture, since any single-chain gradient sampler gets trapped (Recipe Phase 5 target).

### Cell-count summary

Across all 8 sub-tables (24 base methods × 10 warmups × 14 models, plus 6 SMC outer × 8 inner-kernel cells ≈ 1080 unique triples):

| Effort tier | Approx count | Description |
|---|---|---|
| 🟢 LOW (Green) | ~480 | conventional `(warmup, sampler)` pairing — library defaults pass auto-gate at first emit |
| 🟡 MEDIUM (Yellow) | ~180 | statistician investigation: seed/init/bug-fix workaround OR unconventional pairing (e.g., `stan_window + mala`) |
| 🔴 HIGH / hard-excluded (Red) | ~420 | dominated by 8 exclusion categories: multimodal × single-chain gradient, VI × pathological, Laplace × non-Gaussian-latent, `no_warmup` × high-d, MCLMC inside SMC, `rmhmc` without callable metric, `fullrank_vi` warmup at d>30, elliptical/mgrad outside Gaussian-prior models |

The full 8-table matrix, supersession map (e.g., `adaptive_tempered_smc` strictly dominates `tempered_smc`), and hard-exclusion category definitions live in [`RECIPE_GENERATION.md`](RECIPE_GENERATION.md).

## Calibration pipeline

Recipe construction draws on three building blocks under `tuningfork/calibration/`:

- **`certify_reference.py` — Gold reference draws**: 1 chain × 100 000 samples (NUTS + Stan window adaptation), reshaped into 10 chunks for rank-normalized split-R̂ (Vehtari et al. 2021). Multimodal exception for `gmm_25` (parallel-tempered SMC + multi-restart with mode-coverage check).
- **`tune.py` — Hyperparameter optimization**: Optuna BO maximizing `min-bulk-ESS / total_grad_evals`, with per-algorithm acceptance targets.
- **`statistician_gate.py` — Auto-gate**: pre-committed thresholds (R̂ < 1.01, min bulk-ESS ≥ 400, divergences = 0, `max_abs_mean_z` < 2) that every recipe must clear before emission. Thresholds are fixed before sampling — see "The garden of forking paths" above.

The `Recipe.effort` field (`tuningfork/inference/recipes/_base.py`) records the resulting cost class: LOW (defaults pass at first emit), MEDIUM (single statistician-led workaround), HIGH (full Bayesian-workflow investigation).

## Headline metric

`primary = min_over_dimensions(bulk_ESS) / total_gradient_evaluations`

## Setup

```bash
make install      # uv sync --group bench
make test         # run tests (default: skip e2e suite)
make test-fast    # inner-loop dev (fast tests only)
make test-full    # merge gate (everything)
make lint         # pre-commit
```

For GPU: `uv pip install "jax[cuda12]"` after `make install`.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for a complete guide to test markers (`fast`, `slow`, `e2e`), folder layout, and adding new tests.

## License

Apache 2.0
