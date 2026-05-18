# tuningfork catalog — user guide

The `tuningfork.catalog` subpackage is the **consumer-side surface** of the library. If you're consuming recipes — loading them, inspecting their diagnostics, reproducing them in a fresh environment, or just looking up "what's known about sampling model X?" — this is where you start.

```python
from tuningfork.catalog import (
    load_recipe,         # Recipe JSON → Recipe dataclass
    summarize_recipe,    # Recipe → pandas DataFrame (auto-renders in Jupyter)
    load_idata,          # Recipe → ArviZ InferenceData (posterior + sample_stats)
    emit_script,         # Recipe → standalone reproduction .py script
)

recipe = load_recipe("tuningfork/catalog/eight_schools_ncp/groundtruth.json")
summarize_recipe(recipe)                 # Recipe metadata table
idata = load_idata(recipe)               # ArviZ-ready InferenceData

import arviz as az
az.plot_trace(idata)
az.summary(idata)

# Optional: reproduce the recipe in a fresh environment
script = emit_script(recipe, num_samples=2000)
from pathlib import Path
Path("run_eight_schools.py").write_text(script)
# $ uv run --with tuningfork --with jax --with blackjax --with numpyro \
#       python run_eight_schools.py
```

See [`notebooks/inspect_README.md`](notebooks/inspect_README.md) for the full API + a worked example. The repo-root [`README.md`](../../README.md) covers the broader design (generator-vs-catalog two-layer split, recipe matrix, calibration pipeline).

## Per-model layout

Each of the 14 models has a subdirectory `tuningfork/catalog/<model>/`. The contents:

| File / dir | What | Committed? |
|---|---|---|
| `lessons.md` | Distilled "what's tricky about sampling this model" knowledge | yes |
| `groundtruth.json` | Canonical long-NUTS reference recipe pin (NUTS-path models only; absent for the 5 analytic models) | yes |
| `groundtruth.imm.npz` | Diagonal inverse-mass-matrix sidecar for high-dim models (gp_regression, horseshoe, irt_2pl, radon, stoch_vol) | yes |
| `reference/metadata.json` | Cache-validity stamp (version, code-SHA, generator, num_samples, cert verdict) | yes |
| `reference/summary.json` | Per-dim posterior mean / std / 5% / 95% | yes |
| `reference/adaptation.json` | Step-size + IMM diag + num-leapfrog-median from warmup (NUTS-path only) | yes |
| `reference/xcheck.json` | Posteriordb cross-check report (only for posteriordb-shared models: eight_schools_ncp, radon) | yes |
| `recipes/{low,medium,high,failed}__<sampler>__<warmup>.json` | Per-cell recipes from the Recipe Generation Phase pipeline (R5 ships 7 canonical FAILED recipes; LOW/MEDIUM/HIGH land as Recipe Phase 1+ executes) | yes |
| `groundtruth_samples/<library>/draws.npz` | 40,000-sample groundtruth draws per sampling library (only `blackjax/` shipped initially; future-extend to `stan/`, `numpyro/`, …). 12 of 14 models ship as of 2026-05-18 — see "Groundtruth samples shipped" below. | yes (via Git LFS) |
| `groundtruth_samples/<library>/chain_stats.npz` | Per-step NUTS diagnostics (num_integration_steps, energy, is_divergent, acceptance_rate, ...) for the sampling library's run | yes (via Git LFS, NUTS-path models only) |
| `_cache/draws.npz` | Local-only working cache (`get_reference_draws(force_regenerate=True)` writes here) | **gitignored** |
| `_cache/chain_stats.npz` | Local-only working cache | **gitignored** |
| `_cache/warmup_checkpoint/` | Mid-run warmup state for resume-after-crash | **gitignored** |

## Current groundtruth status (as of 2026-05-18)

All 14 models have certified groundtruth. The 9 NUTS-path models ran a 1 chain × 5,000 warmup × 40,000 post-warmup NUTS chain with `window_adaptation_diag_imm`; the 5 analytic models drew i.i.d. samples directly from the posterior. Certification gate (the auto-gate that every groundtruth must clear before commit): **split-R̂ < 1.01, min per-chunk bulk-ESS > 400, E-BFMI > 0.3, divergence count below per-model tolerance**.

## Groundtruth samples shipped (2026-05-18)

All **14 of 14** models ship their canonical 40,000-sample groundtruth draws via Git LFS at `<model>/groundtruth_samples/blackjax/{draws,chain_stats}.npz`. Total LFS-tracked content: **~256 MB draws + ~2.5 MB chain_stats**. The 5 analytic models ship `draws.npz` only (no NUTS chain_stats); the 9 NUTS-path models ship both.

### Per-model seed + warmup policy

Different models pin different (seed, warmup) configurations — pragmatic per-model choice rather than a global lock:

- **12 models at `jax.random.key(20260517)` + `window_adaptation_diag_imm` warmup** (the catalog default): mvn_10, ill_cond_50, banana, neals_funnel, gmm_25, logistic_synthetic, eight_schools_ncp, german_credit, irt_2pl, radon, horseshoe, lotka_volterra. Re-certed cleanly during the 2026-05-17 sweep.
- **1 model at `jax.random.key(20260517)` + `multipathfinder` warmup** (custom, `n_paths=4` + broadcast init): **stoch_vol**. The AR(1) unit-root posterior has a bad-attractor mode that single-path warmups can be captured by — switching to multipathfinder (4 paths + PSIS resampling, shared broadcast init) reduces the catastrophic-capture rate from ~30 % to ~25 %. Re-certed 2026-05-18: rhat=1.0002, ESS=1612, n_div=141 (0.35 %), E-BFMI=0.92, wall=184s. A 2026-05-18 2×2 sweep tested `n_paths ∈ {4, 10}` × `init ∈ {broadcast, diverse}` at the same 8 seeds and found N=4 broadcast dominates: N=10 broadcast drops to 4/8 pass (more paths → more bad-mode weight under shared-init L-BFGS), and diverse-init crashes Pathfinder L-BFGS with NaN log-densities at 4/8 (N=4) to 7/8 (N=10) of seeds. Do not raise `n_paths` above 4 and do not enable diverse init for this model. See [`stoch_vol/lessons.md`](stoch_vol/lessons.md) and the [2026-05-18 case study](https://github.com/blackjax-devs/claude-config/blob/main/project/worklog/lessons/case-studies/stoch_vol/2026-05-18-multimodal-warmup-capture-pathfinder-rescue.md).
- **1 model at `jax.random.key(0)` + `window_adaptation_diag_imm`** (Phase 0 holdover): **gp_regression**. Not re-certed at seed=20260517 because its certification wall is ~63 h on a single CPU.

**Per-model gate override**: stoch_vol is also the only model with a relaxed `divergence_rate_tolerance` (0.5 % vs the default 0.1 %), reflecting the genuine AR(1) unit-root divergence cluster — multipathfinder warmup reduces the catastrophic-capture rate but does NOT reduce the absolute divergence count (the new cert has 141 divergences vs Phase 0's 105). See the [2026-05-12 gate-override decision](https://github.com/blackjax-devs/claude-config/blob/main/project/worklog/decisions/2026-05-12-per-model-divergence-gate-override.md).

*(Pipeline-tooling note: `reference/metadata.json::seed` records `0` for the 12 window_adaptation-warmed models because the cert pipeline doesn't currently track `rng_key` parameter values — that's a known issue to fix. The stoch_vol re-cert script wrote `seed: 20260517` directly. Use `groundtruth.json::tuning_seed` as the authoritative per-model seed pin in any case.)*

### NUTS-path groundtruth (9 models)

| Model | Dim | Class | Target accept | Step size | IMM | R̂ max | min bulk-ESS | divergences | Wall | Difficulty¹ |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|:---:|
| `logistic_synthetic` | 3 | GLM baseline | 0.80 | 0.580 | inline | 1.0000 | 4,061 | 0 | 11 s | 🟢 LOW |
| `eight_schools_ncp` | 10 | hierarchical | 0.80 | 0.397 | inline | 1.0002 | 3,651 | 1 | 19 s | 🟢 LOW |
| `german_credit` | 26 | GLM real data | 0.80 | 0.330 | inline | 1.0001 | 7,284 | 0 | 26 s | 🟢 LOW |
| `irt_2pl` | 144 | hierarchical, scale-id | 0.80 | 0.091 | sidecar | 1.0004 | 2,084 | 0 | 69 s | 🟢 LOW |
| `radon` | 390 | hierarchical+funnel | 0.80 | 0.212 | sidecar | 1.0008 | 720 | 0 | 73 s | 🟢 LOW |
| `stoch_vol`³ | 503 | latent-Gaussian / state-space | **0.99** | 0.0157 | sidecar | 1.0002 | 1,612 | 141² | 3.1 min | 🟡 MEDIUM |
| `horseshoe` | 204 | sparse heavy-tailed | **0.99** | 0.00333 | sidecar | 1.0003 | 2,543 | 0 | 6.1 min | 🟡 MEDIUM |
| `lotka_volterra` | 7 | nonlinear ODE inverse | **0.99** | 0.0358 | inline | 1.0004 | 2,363 | 0 | 8.6 min | 🟡 MEDIUM |
| `gp_regression` | 203 | latent-Gaussian (1D GP) | **0.99** | 0.00222 | sidecar | 1.0000 | 3,910 | 18 | **63 h** | 🔴 HIGH |

¹ Difficulty tier informally maps to the wall + tuning effort required to clear the gate. 🟢 LOW = `target_acceptance_rate=0.80` defaults; 🟡 MEDIUM = required `0.99` for boundary models; 🔴 HIGH = required `0.99` + sidecar IMM + statistician investigation (see `gp_regression/lessons.md`).
² stoch_vol's 141 divergences (0.35 %) are within the per-model tolerance (`Posterior.divergence_rate_tolerance = 0.005`, vs default 0.001) due to the AR(1) unit-root boundary — see [`stoch_vol/lessons.md`](stoch_vol/lessons.md). The certification passed.

³ stoch_vol uses `multipathfinder` warmup (4 paths + PSIS resampling) rather than the catalog-default `window_adaptation_diag_imm` — the AR(1) posterior is multi-modal with a unit-root attractor that single-path warmups can be captured by. Re-cert 2026-05-18 at seed=20260517. See the [2026-05-18 case study](https://github.com/blackjax-devs/claude-config/blob/main/project/worklog/lessons/case-studies/stoch_vol/2026-05-18-multimodal-warmup-capture-pathfinder-rescue.md).

### Analytic groundtruth (5 models)

These models support direct posterior sampling (closed-form or rejection-free); no NUTS run needed. The "groundtruth" is 100,000 i.i.d. draws cached at `<model>/_cache/draws.npz`. No `groundtruth.json` recipe is committed because the analytic sampler IS the canonical reference (there's no choice to pin).

| Model | Dim | Class | Method | Samples | Wall |
|---|---:|---|---|---:|---:|
| `mvn_10` | 10 | Gaussian baseline | direct multivariate normal | 100,000 | <1 s |
| `ill_cond_50` | 50 | ill-conditioned Gaussian (κ≈1000) | direct multivariate normal | 100,000 | <1 s |
| `banana` | 2 | curved-manifold (Rosenbrock) | rejection-free analytic transform | 100,000 | <1 s |
| `neals_funnel` | 10 | hierarchical funnel | rejection-free analytic transform | 100,000 | <1 s |
| `gmm_25` | 2 | 25-mode Gaussian mixture | direct sampling from each component | 100,000 | <1 s |

## Reading the table

- **R̂ max** is the maximum rank-normalised split-R̂ across all dimensions (Vehtari et al. 2021). Gate: < 1.01.
- **min bulk-ESS** is the minimum bulk effective sample size across dimensions × chunks. Gate: > 400 per chunk (with the long-chain reshaped into 4 chunks).
- **divergences** is the count of divergent NUTS transitions over the post-warmup 40,000 samples. Gate: per-model `divergence_rate_tolerance` (default 0.1% of n_samples = 40 for the standard cert; `stoch_vol`'s 105 divergences are within its raised tolerance per its AR(1) boundary characteristics).
- **Wall** is the certification run wall time (warmup + sampling on a single CPU chain). On GPU expect 5-20× speedup depending on model.
- **Difficulty** is informal — formally all rows ship as `Effort.GROUNDTRUTH` per the recipe schema. The tier here reflects the certification cost (tuning effort + wall) needed to clear the gate.

## Pointers

- **Per-model `lessons.md`** files capture the sampling-quirks history for each model — start there if you're curious about why a specific cell is the way it is.
- **`recipes/` subdirectories** will fill out as Recipe Phase 1+ emits LOW/MEDIUM/HIGH recipes per (model × warmup × sampler) cell. R5 (2026-05-17) shipped 7 canonical **FAILED** recipes documenting hard-exclusion categories — see [`gmm_25/recipes/failed__nuts__window_adaptation_diag_imm.json`](gmm_25/recipes/failed__nuts__window_adaptation_diag_imm.json) for the multimodal × single-chain-gradient exclusion example.
- **Recipe matrix** (full 24 × 10 × 14 cell-by-cell colour verdict) lives at [`../../RECIPE_GENERATION.md`](../../RECIPE_GENERATION.md).
- **Architecture** (generator-vs-catalog two-layer split) at [`../../README.md`](../../README.md) § Layout.
- **API details** (load_recipe, load_idata, emit_script, ...) at [`notebooks/inspect_README.md`](notebooks/inspect_README.md).
