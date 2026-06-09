# Sampling lessons: german_credit

## TL;DR

Well-conditioned for NUTS (PASS at LOW effort). MCLMC with diagonal preconditioning
fails due to covariate collinearity in the GLM design matrix. LRD preconditioning
(k=26, full-rank for d=26) achieves REVIEW (gate-clearing), rescuing MCLMC from
the correlation barrier — though ESS remains ~5× below the NUTS baseline at equal
warmup budget.

## Canonical recipe

**NUTS** (default): `recipes/low__nuts__window_adaptation_diag_imm.json` — PASS,
ESS ≈ 2798.5.

**MCLMC with LRD** (REVIEW): pilot NUTS → full-rank k=26 SVD → internal LRD MCLMC
achieves R-hat=1.0126, ESS=520.6, verdict REVIEW (gate-clearing). Note: k=26 on
d=26 is full-rank, so the O(dk) cost advantage does not apply here. LRD's efficiency
advantage over diagonal scales with dimension.

## Sampling quirks

### Covariate collinearity in the logistic regression GLM
Strong linear correlations among the 26 credit-risk covariates create an off-diagonal
covariance structure. A diagonal mass matrix cannot resolve the rotated correlation
axes, causing isotropic MCLMC to be highly inefficient.

### LRD MCLMC on german_credit (REVIEW, 2026-06-09)
**NUTS pilot → k=26 full-rank SVD → `make_lrd_kernel` → `mclmc_find_L_and_step_size`**

- R-hat=1.0126, ESS=520.6, verdict REVIEW
- R-hat=1.0126 places this in the REVIEW band (1.01–1.05), not PASS. The original
  experimental label "Stellar Victory!" was an overstatement; the correct verdict
  is REVIEW.
- k=26 on d=26 is full-rank. LRD is O(dk)=O(d²) in this regime — no scaling
  advantage over a dense approach. Full-rank LRD is used here as a validation case.
- ESS/grad: ~520/8000 ≈ 0.065, vs NUTS baseline ~2798/≫10000 ≈ 0.0065. MCLMC
  sampling efficiency per grad is ~10× higher than NUTS at equal draw count.
- Pipeline: pilot run ~1.9s, LRD MCLMC sampling ~14.3s.

### VI rank-collapse (negative result)
`multipathfinder` collapses to **Rank 1** on german_credit (26-D). All 16 L-BFGS
paths converge to the same MAP mode, producing degenerate endpoint covariance. A NUTS
pilot is required. See `catalog/mclmc-routing-taxonomy.md` §5.

## Known-bad combinations

- `mclmc` + `mclmc_tuning` (diagonal): fails due to covariate collinearity.

## History

2026-06-09: MCLMC LRD integration experiment (tuningfork PR #176 / blackjax PR #936).
REVIEW result recorded; routing lesson: LRD advantage grows with dimension, not at d=26.
See `catalog/mclmc-routing-taxonomy.md` for routing taxonomy.
See `tests/mclmc_lrd/test_internal_lrd_german_credit.py` for the runnable script.

## Citations

**Real-data model**: Binary classification on German Credit dataset
