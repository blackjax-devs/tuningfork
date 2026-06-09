# Sampling lessons: neals_funnel

## TL;DR

**Structural honest null for any sampler using a global mass matrix and single global
step size.** MCLMC family (all variants) FAIL due to position-dependent varying
curvature. SMC (inner-kernel HMC) is the viable path. NUTS with diagonal IMM achieves
PASS on the standard 2-D formulation via tree-expansion.

## Canonical recipe

**SMC**: `recipes/smc__inner_kernel_tuning__hmc.json` — the only MCMC family with
a viable path via sequential tempering.

**NUTS**: low-effort `window_adaptation_diag_imm` — PASS for the 2-D standard funnel.

## Sampling quirks

### Position-dependent varying curvature (the funnel topology)
Neal's Funnel has a hyperparameter v ~ N(0,3) and latents x_i ~ N(0, exp(v)). In the
funnel mouth (large v), the latents are wide and require large step sizes. In the
funnel neck (small v), the latents are squeezed into a tiny correlated valley
requiring microscopic step sizes.

A global IMM — whether diagonal, dense, or low-rank — is an **affine transformation**.
It applies the same geometric scaling everywhere and cannot resolve position-dependent
curvature. The funnel neck forces the integrator to use a microscopic global step size
or face massive energy violations.

### MCLMC family: honest null (2026-06-08)
All MCLMC variants were tested against Neal's funnel. Results (seed stat-2026-06-08):

| Sampler | n_warmup | Max R-hat | Min ESS | Divergences | Verdict |
|---|---|---|---|---|---|
| `adjusted_mclmc` | 1,000 | 1.3889 | 8.8 | 0 | FAIL |
| `adjusted_mclmc_dynamic` | 1,000 | 1.0720 | 38.2 | 0 | FAIL |
| `adjusted_mclmc` | 10,000 | 1.0821 | 33.1 | 0 | FAIL |
| `adjusted_mclmc_dynamic` | 10,000 | 1.0481 | 96.2 | 0 | FAIL |
| `adjusted_mclmc_dynamic` | 50,000 | 1.0552 | 70.0 | 0 | FAIL |

All results archived as structured recipes in `catalog/neals_funnel/recipes/`:
- `failed__adjusted_mclmc__adjusted_mclmc_tuning.json` (10k warmup, R-hat=1.0821)
- `failed__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json` (50k warmup, R-hat=1.0552)

Full `attempted_configurations` ladders preserved verbatim in each recipe JSON.

**`adjusted_mclmc_dynamic` insight**: randomized trajectory length significantly
improves over fixed L by breaking periodic orbits in the funnel. Under 10k warmup
it outmixes NUTS (96.2 ESS vs 51.0, 0 divergences vs 16). But the strict
R-hat<1.01/ESS>100 gate still fails — the position-dependent curvature is too extreme
for any global step-size sampler to clear cleanly.

**Safety net, not sampler**: the MH correction in adjusted variants prevents
divergences (0 vs NUTS's 16 at 1k warmup) but cannot rescue the fundamental
curvature mismatch.

### Why LRD preconditioning does not help
A Low-Rank + Diagonal mass matrix is still a **constant affine transformation**. It
can flatten global rotational ill-conditioning (like ill_cond_50) but cannot adapt
to position-dependent curvature changes. neals_funnel requires a position-dependent
metric (e.g., Riemannian HMC) or sequential methods (SMC).

## Known-bad combinations

- `mclmc` (unadjusted): diverges / explodes in funnel neck. Do not use.
- `adjusted_mclmc`: fails safely (0 divergences, low ESS). Honest null.
- `adjusted_mclmc_dynamic`: fails safely with better ESS than static variant. Honest null.
- Any MCLMC variant with LRD preconditioning: not tested; not expected to improve
  (global affine transform cannot resolve local curvature).

## History

2026-06-08: MCLMC family Category C evaluation (recipes-mclmc-cat-c.md experimental
findings). Full attempted-configurations ladder committed to catalog recipes.
@statistician override: stat-2026-06-08. See `catalog/mclmc-routing-taxonomy.md` §4.

## Citations

**Synthetic baseline** — no external reference. Standard test model for sampler validation.
