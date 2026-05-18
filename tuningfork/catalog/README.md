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

12 of 14 models ship their canonical 40,000-sample draws via Git LFS at `<model>/groundtruth_samples/blackjax/{draws,chain_stats}.npz`. Total LFS-tracked content: **~122 MB draws + ~1.9 MB chain_stats**. The pinned seed for these shipped draws is `jax.random.key(20260517)` (re-cert sweep date as integer); the `reference/metadata.json` `timestamp_utc` field of `2026-05-17T...` distinguishes these from the original Phase 0 traces. *(Note: the `seed` field in `reference/metadata.json` records `0` for these — that's a stale-default in the cert pipeline that doesn't track `rng_key` parameter values; this is a known issue to be fixed in the cert tooling.)*

Two models do **not** ship draws:

- **`gp_regression`** — certification wall ≈ 63 h on a single CPU; the cert run hasn't been redone at the seed=20260517 protocol, so the committed `reference/*.json` remains at the Phase 0 `seed=0` trace. Draws can be regenerated locally via `get_reference_draws(force_regenerate=True)` (long).
- **`stoch_vol`** — known PRNG fragility (the canonical config passes at `key(0)` but fails at `key(20260517)`; see [`stoch_vol/lessons.md`](stoch_vol/lessons.md)). The committed `reference/*.json` remains at the Phase 0 `seed=0` trace; multi-seed protocol robustness is an open investigation.

### NUTS-path groundtruth (9 models)

| Model | Dim | Class | Target accept | Step size | IMM | R̂ max | min bulk-ESS | divergences | Wall | Difficulty¹ |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|:---:|
| `logistic_synthetic` | 3 | GLM baseline | 0.80 | 0.580 | inline | 1.0000 | 4,061 | 0 | 11 s | 🟢 LOW |
| `eight_schools_ncp` | 10 | hierarchical | 0.80 | 0.397 | inline | 1.0002 | 3,651 | 1 | 19 s | 🟢 LOW |
| `german_credit` | 26 | GLM real data | 0.80 | 0.330 | inline | 1.0001 | 7,284 | 0 | 26 s | 🟢 LOW |
| `irt_2pl` | 144 | hierarchical, scale-id | 0.80 | 0.091 | sidecar | 1.0004 | 2,084 | 0 | 69 s | 🟢 LOW |
| `radon` | 390 | hierarchical+funnel | 0.80 | 0.212 | sidecar | 1.0008 | 720 | 0 | 73 s | 🟢 LOW |
| `stoch_vol` | 503 | latent-Gaussian / state-space | **0.99** | 0.0138 | sidecar | 1.0006 | 1,993 | 105² | 3.1 min | 🟡 MEDIUM |
| `horseshoe` | 204 | sparse heavy-tailed | **0.99** | 0.00333 | sidecar | 1.0003 | 2,543 | 0 | 6.1 min | 🟡 MEDIUM |
| `lotka_volterra` | 7 | nonlinear ODE inverse | **0.99** | 0.0358 | inline | 1.0004 | 2,363 | 0 | 8.6 min | 🟡 MEDIUM |
| `gp_regression` | 203 | latent-Gaussian (1D GP) | **0.99** | 0.00222 | sidecar | 1.0000 | 3,910 | 18 | **63 h** | 🔴 HIGH |

¹ Difficulty tier informally maps to the wall + tuning effort required to clear the gate. 🟢 LOW = `target_acceptance_rate=0.80` defaults; 🟡 MEDIUM = required `0.99` for boundary models; 🔴 HIGH = required `0.99` + sidecar IMM + statistician investigation (see `gp_regression/lessons.md`).
² stoch_vol's 105 divergences are within the per-model tolerance (`Posterior.divergence_rate_tolerance`) due to the AR(1) unit-root boundary — see [`stoch_vol/lessons.md`](stoch_vol/lessons.md). The certification still passed.

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
