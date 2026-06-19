# Sampling lessons: horseshoe

## TL;DR

NUTS PASS at LOW effort. `adjusted_mclmc` and `adjusted_mclmc_dynamic` PASS at MEDIUM
effort (10k warmup). LRD preconditioning provides **no benefit** over diagonal on this
model — the bottleneck is local sparsity funnel curvature (Cauchy tail transitions),
not global correlation. MH correction is load-bearing for stability.

## Canonical recipe

**NUTS** (default): `recipes/low__nuts__window_adaptation_diag_imm.json` — PASS.

**MCLMC family**: `adjusted_mclmc_dynamic` at 10k warmup — REVIEW (gate-clearing),
ESS=281.9. See `recipes/low__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json`.

## Sampling quirks

### Dual pathology: heavy Cauchy tails + design matrix correlations
The horseshoe model uses half-Cauchy priors for local and global shrinkage scales,
creating:
- Extremely flat, weak-gradient regions far from the sparsity funnel.
- Strong linear correlations among coefficients from the design matrix structure.

Despite this dual pathology, the **spherical ESH dynamics** of MCLMC are resilient to
the Cauchy tails — the spherical integrator does not diverge in flat regions.

### LRD preconditioning: no benefit (REVIEW, 2026-06-09)
`adjusted_mclmc_dynamic` with LRD (k=50, NUTS-pilot extraction) achieves:
- R-hat=1.0193, ESS=270.7, verdict REVIEW

This is **equivalent to** the diagonal baseline (`adjusted_mclmc_dynamic` at the same
warmup): R-hat=1.0160, ESS=281.9. LRD provides no ESS improvement.

**Scientific conclusion:** The dominant bottleneck on horseshoe is the local funnel
transition (coefficient to zero via the Cauchy shrinkage scale), not global correlation.
A global linear LRD mass matrix cannot flatten position-dependent funnel curvature.
The routing decision is: `adjusted_mclmc_dynamic` is the correct MCLMC variant for
horseshoe; LRD is unnecessary overhead.

The original experimental label "Outstanding Success!" was an overstatement. LRD
performs equivalently to the diagonal baseline — a correct REVIEW result, not an
outstanding one.

### Step-size scaling for adjusted samplers (0.55× factor)
`mclmc_find_L_and_step_size` (unadjusted warmup) adapts a large step_size optimized
for rejection-free trajectories. When running `adjusted_mclmc_dynamic`, scale the
adapted step_size by 0.55 to target ~94% acceptance probability. This is the
empirically validated scaling for horseshoe; see `test_internal_lrd_horseshoe.py`.

### Cauchy-tail resilience result (recipes/low__adjusted_mclmc__adjusted_mclmc_tuning.json)
At 10k warmup, `adjusted_mclmc` achieves R-hat=1.0186, ESS=262.3 — the MH correction
successfully stabilizes exploration of the flat Cauchy tails.

## Known-bad combinations

None documented. `adjusted_mclmc` and `adjusted_mclmc_dynamic` both pass at MEDIUM effort.

## History

2026-06-09: MCLMC LRD integration experiment (tuningfork PR #176 / blackjax PR #936).
LRD equivalence-to-baseline finding recorded; routing lesson: route horseshoe to
`adjusted_mclmc_dynamic` without LRD overhead.
See `catalog/mclmc-routing-taxonomy.md` §3 (Category C routing).
See `tests/mclmc_lrd/test_internal_lrd_horseshoe.py` for the runnable script.

## Dynamic-L Sweep (avg ladder)

Run date: 2026-06-19 | Source: sweep_dynl_variety_results.json, medians over 3 seeds

| avg | realized_avg | ESS | Rhat | 2nd-mom bias | mbias_sd | trend |
|---|---|---|---|---|---|---|
| 2 | 2.0 | 5 | 2.699 | 0.953 | 0.618 | **loud-fail** |
| 6 | 6.0 | 6 | 2.071 | 1.420 | 0.800 | loud-fail |
| 18 | 18.1 | 11 | 1.384 | 0.851 | 0.487 | loud-fail |
| 54 | 54.2 | 229 | 1.275 | 0.395 | 0.114 | **REVIEW-plateau** |
| 108 | 108.4 | 734 | 1.177 | 0.330 | 0.082 | **REVIEW-plateau** |

**Lesson:** Longer L improves monotonically (ESS 5→734, bias 0.95→0.33) but asymptotes at REVIEW tier (Rhat ~1.2).
This is a geometry-hard limit: funnel curvature is position-dependent, so a single affine preconditioning + longer L
cannot resolve it. Adaptive/position-dependent methods or reparameterization (NCP) would be needed to improve further.

See `catalog/mclmc-scaling-laws.md` §3 for generalized principles (why funnels show LONGEST-available plateaus, etc.).

## Citations

**Real-data model**: Horseshoe prior on regression coefficients; real dataset
