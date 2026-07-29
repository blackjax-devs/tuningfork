# Sampling lessons: irt_1pl

## TL;DR

**Category A — Isotropic high-D**: d=500 NCP IRT-1PL model (J=500 students, I=10
items). Unadjusted MCLMC wins 1.70× over NUTS on ESS/grad. All three cells PASS at
LOW effort (first-emit). Confirms the O(d^{1/4}) MCLMC advantage on smooth isotropic
high-dimensional posteriors.

## Results (2026-07-29, first Category-A coverage)

| method | warmup | verdict | ESS/grad | vs NUTS | rhat | min_ESS | div | |z| |
|---|---|---|---:|---:|---:|---:|---:|---:|
| nuts | window_adaptation_diag_imm | **PASS** | 0.1239 | 1.00× | 1.0073 | 7625.7 | 0 | 3.239 |
| mclmc | mclmc_tuning | **PASS** | 0.2103 | **1.70×** | 1.0052 | 1692.5 | 0 | 3.225 |
| adjusted_mclmc_dynamic | adjusted_mclmc_tuning | **PASS** | 0.1124 | 0.91× | 1.0052 | 1841.4 | 0 | 3.060 |

MCLMC step_size ≈ 27 (range 26.3–30.7), consistent with the √d law:
1.22 × √500 ≈ 27.3. The EEVPD target (5e-4) was reached — confirming Category A
smooth geometry.

## Canonical recipes

- `recipes/low__nuts__window_adaptation_diag_imm.json` — NUTS baseline, ESS/grad=0.1239
- `recipes/low__mclmc__mclmc_tuning.json` — **best challenger**, ESS/grad=0.2103 (1.70×)
- `recipes/low__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json` — adjusted variant,
  ESS/grad=0.1124 (0.91×)

## Model geometry

IRT 1PL (Rasch model): J=500 students, I=10 items, NCP parameterization.
Unconstrained dimensionality: d = J = 500 (theta per student).

Prior: theta_j ~ N(0, 1) i.i.d. (standard normal, per NCP).
Likelihood: Bernoulli logistic, item difficulty and student ability.

The NCP parameterization makes the posterior approximately isotropic and Gaussian —
exactly Category A in the MCLMC routing taxonomy. The correlation structure is weak
(item parameters are shared but students are i.i.d.), so diagonal preconditioning
is sufficient and no rotation or LRD preconditioning is needed.

## Why MCLMC wins here

The O(d^{1/4}) scaling advantage of MCLMC materializes on this model because:
1. Geometry is smooth, isotropic, approximately Gaussian.
2. Step size follows the √d law without collapse (EEVPD hits 5e-4 target).
3. No position-dependent curvature (no funnel-like or heavy-tail structure).
4. The unadjusted variant avoids the MH rejection overhead while preserving correct
   invariant measure for smooth targets.

Adjusted MCLMC (0.91×) is slightly below NUTS because the MH correction adds
overhead without improving exploration on a target that doesn't need it.

## Note on warmup gradient cost

Per challengers-empirics.md §0.1, warmup gradients are NOT counted in the headline
metric. The adjusted_mclmc_tuning warmup typically costs ~2× its sampling budget in
gradients, which would flip the 0.91× ratio negative. The unadjusted mclmc_tuning
warmup is cheaper (~1× or less). The 1.70× MCLMC win is on sampling-only ESS/grad;
the warmup-inclusive story is tracked separately.

## Known-good combinations

All three cells are PASS at LOW effort (first-emit, seed=20260517).

## Boundary annotations

[boundary: PASS confirmed for all three cells at n_warmup=1k, n_samples=1k, num_chains=4;
first-emit result — no escalation ladder needed at LOW effort]

## History

2026-07-29: First recipes for irt_1pl. All three Category-A cells PASS at LOW effort.
Confirms the O(d^{1/4}) MCLMC scaling advantage on this model class.
See `catalog/mclmc-routing-taxonomy.md` §4 (Category A).
