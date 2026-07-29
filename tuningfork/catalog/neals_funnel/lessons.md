# Sampling lessons: neals_funnel

## TL;DR

**Structural honest null for any sampler using a global mass matrix and single global
step size.** MCLMC family (all variants) FAIL due to position-dependent varying
curvature. SMC (inner-kernel HMC) is the viable path. NUTS with diagonal IMM **FAILS
at all warmup budgets tested (1k, 3k, 10k)**; the previous claim of PASS was unverified.
The model is 10-D (v + 9 latents).
[boundary: MCLMC FAIL holds at all warmup budgets tested (1k–50k) with 0 divergences — this is a geometry-hard blocker, not warmup-limited; SMC PASS holds at the standard 10-D formulation only; NUTS+diag FAIL confirmed at 1k/3k/10k warmup (budget-invariant z worsens: 2.34->3.34->6.03); low_rank and hard_direction cells (hmc, nuts+low_rank) also fail; laplace family out_of_scope]

## Canonical recipe

**SMC**: `recipes/smc__inner_kernel_tuning__hmc.json` — the only MCMC family with
a viable path via sequential tempering.

**NUTS**: `window_adaptation_diag_imm` — **FAIL** at all warmup budgets tested (1k,
3k, 10k); see `recipes/failed__nuts__window_adaptation_diag_imm.json`. Corrected
2026-07-29: the previous claim "PASS for the 2-D standard funnel" was unverified.
The funnel's position-dependent curvature prevents any fixed step size from handling
both the neck and body; this is a geometry blocker, not a warmup-budget issue.

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
- `adjusted_mclmc` (n_warmup=10k): fails safely (0 divergences, rhat=1.08, ESS=33). Honest null.
  See `recipes/failed__adjusted_mclmc__adjusted_mclmc_tuning.json`.
  [boundary: FAIL at n_warmup=1k and 10k (both recorded); warmup-invariant geometry blocker]
- `adjusted_mclmc_dynamic` (n_warmup=50k): fails safely (rhat=1.055, ESS=70). Honest null.
  See `recipes/failed__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json`.
  [boundary: FAIL confirmed up to n_warmup=50k — the highest budget tested; R-hat improvement from 1k→50k is minimal (1.37→1.07); not tunable further]
- Any MCLMC variant with LRD preconditioning: not tested; not expected to improve
  (global affine transform cannot resolve local curvature).
- `hmc` + `window_adaptation_low_rank_imm`: **FAIL** (hard_direction). See `recipes/failed__hmc__window_adaptation_low_rank_imm.json`.
- `nuts` + `window_adaptation_low_rank_imm`: **FAIL** (hard_direction). See `recipes/failed__nuts__window_adaptation_low_rank_imm.json`.
- `nuts` + `window_adaptation_diag_imm`: **FAIL** at all budgets 1k–10k. See `recipes/failed__nuts__window_adaptation_diag_imm.json`.
  [boundary: FAIL confirmed at n_warmup=1k,3k,10k; budget-invariant geometry blocker — step_size worsens bias at higher budget (z: 2.34→3.34→6.03 as warmup 1k→3k→10k); NOT warmup-limited]
- `nuts` + `window_adaptation_diag_imm` @ `target_acceptance=0.99`: **FAIL** at 1k and 5k warmup. See `recipes/failed__nuts__window_adaptation_diag_imm__ta099.json`.
  [boundary: ta=0.99 reduces divergences (48→34) and bias (z 2.34→1.63) but worsens ESS (24.5→16.4); same geometry blocker at higher acceptance — the funnel is not an acceptance-target problem]
- Laplace family + `window_adaptation_low_rank_imm`: **FAIL** (out_of_scope). See `recipes/failed__laplace_*__window_adaptation_low_rank_imm.json`.
- `meanfield_vi` + `no_warmup`: **FAIL** (out_of_scope). See `recipes/failed__meanfield_vi__no_warmup.json`.

Recorded FAILs not discussed above: all 11 failed recipes are now covered above.

## MAMS comparison experiment (2026-07-29)

**Question**: Does NUTS converge on the funnel when given target_acceptance=0.99
(the same conservatism that MAMS uses)?

**Answer**: No. ta=0.99 does not help NUTS converge on the funnel.

| Config | n_warmup | rhat | min_ESS | div | max_z | step_size range |
|---|---|---|---|---|---|---|
| NUTS ta=0.80 | 1000 | 1.1433 | 24.5 | 48 | 2.338 | 0.077–0.166 |
| NUTS ta=0.80 | 3000 | 1.1305 | 41.5 | 3 | 3.341 | 0.042–0.123 |
| NUTS ta=0.80 | 10000 | 1.0694 | 72.8 | 9 | 6.028 | 0.040–0.155 |
| NUTS ta=0.99 | 1000 | 1.1277 | 24.5 | 34 | 1.630 | 0.003–0.074 |
| NUTS ta=0.99 | 5000 | 1.2305 | 16.4 | 3 | 1.846 | 0.002–0.032 |

At ta=0.99, the step size is 10–30× smaller than at ta=0.80 (0.003–0.074 vs
0.077–0.166). This reduces divergences (48→34) and bias (z: 2.34→1.63) because the
tiny step rarely blows up in the neck. The ESS numbers above are single-seed; an
independent 6-seed replicate reproduced the direction (median minESS 40.2 at 1k vs
24.4 at 5k) but the effect is not statistically established (Mann-Whitney p=0.41,
seed spread ~4× the claimed 1.49× effect). At R̂ > 1.13 the chains are not sampling
the same distribution, so bulk-ESS is not a well-defined quantity.

The geometry-not-acceptance-rate conclusion rests on evidence that does not depend on
ESS: (1) R̂ stays above 1.12 at all budgets; (2) divergences remain non-zero at 1k
warmup; (3) step sizes collapse 10–30× when ta rises from 0.80 to 0.99, which is
the expected mechanical response but does not resolve position-dependent curvature.

The MAMS paper compares MAMS@ta=0.99 against NUTS@ta=0.80. This experiment shows
NUTS@ta=0.99 also FAILS — the funnel is a geometry problem, not an acceptance-rate
problem. The single global step size cannot cover the position-dependent curvature
regardless of the target acceptance rate.

## History

2026-06-08: MCLMC family Category C evaluation (recipes-mclmc-cat-c.md experimental
findings). Full attempted-configurations ladder committed to catalog recipes.
Reviewed and approved 2026-06-08. See `catalog/mclmc-routing-taxonomy.md` §4.

2026-07-29: NUTS+diag IMM experiments (standard + ta=0.99). Corrected the wrong
"PASS" claim from lessons.md; added budget-invariant diagnostic ladder; ran MAMS
comparison experiment. See `failed__nuts__window_adaptation_diag_imm.json` and
`failed__nuts__window_adaptation_diag_imm__ta099.json`.

## Citations

**Synthetic baseline** — no external reference. Standard test model for sampler validation.
