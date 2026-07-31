# Sampling lessons: german_credit

## TL;DR

Well-conditioned for NUTS (PASS at LOW effort). MCLMC with diagonal preconditioning
fails due to covariate collinearity in the GLM design matrix. LRD preconditioning
(k=26, full-rank for d=26) achieves REVIEW (gate-clearing), rescuing MCLMC from
the correlation barrier — though ESS remains ~5× below the NUTS baseline at equal
warmup budget.
[boundary: diagonal MCLMC FAIL due to collinearity (not warmup-limited); LRD PASS at k=8, n_warmup=2000 (pilot-free); nearest FAIL: hmc+dense_imm (n_warmup=1000), hmc+low_rank_imm (hard_direction); see recipes/failed__hmc__window_adaptation_dense_imm.json]

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
**Generated NUTS pilot → k=26 full-rank SVD → statically bound LRD kernel →
`mclmc_find_L_and_step_size`**

The recorded experiment used a direct helper named `make_lrd_kernel`; codegen now
emits the same binding inline and is the only executable route.

- R-hat=1.0126, ESS=520.6, verdict REVIEW
- R-hat=1.0126 places this in the REVIEW band (1.01–1.05), not PASS. The original
  experimental label "Stellar Victory!" was an overstatement; the correct verdict
  is REVIEW.
- k=26 on d=26 is full-rank. LRD is O(dk)=O(d²) in this regime — no scaling
  advantage over a dense approach. Full-rank LRD is used here as a validation case.
- ESS/grad: ~520/8000 ≈ 0.065 for this early k=26 REVIEW run (old estimator basis).
  Committed NUTS recipes for german_credit record 0.0432–0.2166 (see
  `low__nuts__window_adaptation_{diag,dense,low_rank}_imm.json`). The committed
  k=8 LRD recipe records headline 0.071 with basis-derived 0.071
  (min_bulk_ess 568 / 8000 grads, headline estimator); both place MCLMC-LRD
  well below the best NUTS baseline (0.2166), not 10× above it.
- Pipeline: pilot run ~1.9s, LRD MCLMC sampling ~14.3s.
  [boundary: REVIEW (not PASS) at k=26 full-rank; committed catalog artifact uses k=8, which achieves 3/3 PASS at n_warmup=2000; full-rank k=26 is documented as upper-bound validation only]

### VI rank-collapse (negative result)
`multipathfinder` collapses to **Rank 1** on german_credit (26-D). All 16 L-BFGS
paths converge to the same MAP mode, producing degenerate endpoint covariance. A NUTS
pilot is required. See `catalog/mclmc-routing-taxonomy.md` §5.

## Known-bad combinations

- `mclmc` + `mclmc_tuning` (diagonal): fails due to covariate collinearity.
  [boundary: collinearity is the geometry blocker; LRD (k=8) rescues MCLMC — see MCLMC-LRD section above]
- `hmc` + `window_adaptation_dense_imm` (n_warmup=1000): **FAIL** (non-reproducible at practical n_warmup).
  See `recipes/failed__hmc__window_adaptation_dense_imm.json`.
  [boundary: dense Welford for d=26 is marginally reproducible at n_warmup=1000; not a d>100 failure, but still fragile — prefer diag IMM for hmc on german_credit]
- `hmc` + `window_adaptation_low_rank_imm`: **FAIL** (hard_direction).
  See `recipes/failed__hmc__window_adaptation_low_rank_imm.json`.
- `laplace_*` + `window_adaptation_low_rank_imm`: **FAIL** (out_of_scope — german_credit logistic GLM lacks separable log-joint).
  See `recipes/failed__laplace_hmc__window_adaptation_low_rank_imm.json` etc.

Recorded FAILs not discussed above: all 6 failed recipes are covered above.

## Recorded LRD certification inputs (german_credit, NUTS-pilot path)

The committed artifacts are:
- `recipes/low__mclmc_lrd__mclmc_lrd_tuning.json` — golden recipe (k=8, best seed=11111)
- `recipes/low__mclmc_lrd__mclmc_lrd_tuning.imm.npz` — rank-8 LRD IMM sidecar, shape sigma=(26,)/U=(26,8)/lam=(8,)

The historical direct emitter that ran this sweep is retired. Do not repeat it;
new sampling or certification work must use the
[codegen-first recipe lifecycle](../../../docs/design/codegen-first-recipes.md).
The recorded inputs were `mclmc_lrd_tuning`, seeds 11111/22222/33333,
`n_warmup=10000`, `n_samples=1000`, and `k_rank=8`.

Certified 2026-07-29: 3/3 PASS, seeds 11111/22222/33333, gate minESS 1474–1608 (az.ess bulk),
R-hat max ~1.001 (≤ 1.005). Gate uses az.ess(method="bulk") ≥ 400. k=8, n_warmup=10000, n_samples=1000.

Note: old script-baked golden claimed minESS≈1776 (az.ess bulk basis, via auto_gate); the
pre-fix library cert measured 261–467 (blackjax/Geyer basis) on comparable chains — a
4–5× apparent gap. The chain-averaging hypothesis (D1) was refuted (adapted L≈11.2–11.9 on
all seeds); the entire story was estimator basis (D3: Geyer vs az-bulk on identical samples).
After the D3 fix, the library measures 1512–1951 on the same az-bulk basis, reproducing the
old golden's range (~1776).

**Why k=8 not k=26?** Full-rank (k=26 = d) LRD overfits the NUTS-pilot samples,
inflating λ for directions with low pilot coverage. R-hat > 1.01 at k=26 is the
signal. k=8 captures the dominant collinear axes without overfitting.

## History

2026-06-09: MCLMC LRD integration experiment (tuningfork PR #176 / blackjax PR #936).
REVIEW result recorded; routing lesson: LRD advantage grows with dimension, not at d=26.
See `catalog/mclmc-routing-taxonomy.md` for routing taxonomy.
See `tests/mclmc_lrd/test_internal_lrd_german_credit.py` for the runnable script.

## Citations

**Real-data model**: Binary classification on German Credit dataset
