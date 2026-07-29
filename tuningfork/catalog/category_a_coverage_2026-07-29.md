# Category-A coverage: first recipes for irt_1pl + lgcp

**Date**: 2026-07-29

## Summary

First recipe coverage for two Category-A (isotropic high-D) models — `irt_1pl` (d=500)
and `lgcp` (d=1600) — plus NUTS baseline and MAMS comparison experiment for
`neals_funnel` (10-D). All runs used LOW effort: n_warmup=1000, n_samples=1000,
num_chains=4, seed=20260517.

---

## neals_funnel — NUTS baseline + MAMS comparison

**Model**: 10-D Neal's funnel (v + 9 latents), geometry blocker.

### Standard cell: NUTS + window_adaptation_diag_imm

| config | n_warmup | rhat | min_ESS | div | max_z | step_size range | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| NUTS ta=0.80 | 1000 | 1.1433 | 24.5 | 48 | 2.338 | 0.077–0.166 | FAIL |
| NUTS ta=0.80 | 3000 | 1.1305 | 41.5 | 3 | 3.341 | 0.042–0.123 | FAIL |
| NUTS ta=0.80 | 10000 | 1.0694 | 72.8 | 9 | 6.028 | 0.040–0.155 | FAIL |

The funnel is a geometry problem: z-score worsens with more warmup (2.34→3.34→6.03)
because the chain becomes increasingly anchored to the funnel body, missing the neck.
This is budget-invariant geometry failure.

Artifact: `catalog/neals_funnel/recipes/failed__nuts__window_adaptation_diag_imm.json`

### MAMS comparison: NUTS @ ta=0.99

| config | n_warmup | rhat | min_ESS | div | max_z | step_size range | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| NUTS ta=0.99 | 1000 | 1.1277 | 24.5 | 34 | 1.630 | 0.003–0.074 | FAIL |
| NUTS ta=0.99 | 5000 | 1.2305 | 16.4 | 3 | 1.846 | 0.002–0.032 | FAIL |

The minESS values above are from single seeds. An independent 6-seed replicate
reproduced the direction (median minESS 40.2 at 1k vs 24.4 at 5k) but the effect is
not statistically established (Mann-Whitney p=0.41, seed spread ~4× the observed
difference). At R̂ > 1.12, the chains are not sampling the same distribution, so
bulk-ESS is not well-defined.

**The geometry-not-acceptance-rate conclusion rests on evidence that does not depend on
ESS**: (1) R̂ stays above 1.12 at all budgets; (2) divergences remain non-zero at 1k
warmup; (3) step sizes collapse 10–30× at ta=0.99 (0.003–0.074 vs 0.077–0.166 at
ta=0.80). The step-size collapse is the expected mechanical response to a higher
acceptance target, but does not resolve position-dependent curvature. NUTS at ta=0.99
fails the funnel gate for the same structural reason as ta=0.80: one global step size
cannot cover both the funnel neck and body.

Artifact: `catalog/neals_funnel/recipes/failed__nuts__window_adaptation_diag_imm__ta099.json`

---

## irt_1pl (d=500) — Category A, first recipes

**Model**: NCP IRT 1PL (Rasch model), J=500 students × I=10 items. Isotropic, smooth.

| method | warmup | verdict | ESS/grad | vs NUTS | rhat | min_ESS | div | max_z | step_size |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| nuts | window_adaptation_diag_imm | **PASS** | 0.1239 | 1.00× | 1.0073 | 7625.7 | 0 | 3.239 | — |
| mclmc | mclmc_tuning | **PASS** | 0.2103 | **1.70×** | 1.0052 | 1692.5 | 0 | 3.225 | 26.3–30.7 |
| adjusted_mclmc_dynamic | adjusted_mclmc_tuning | **PASS** | 0.1124 | 0.91× | 1.0052 | 1841.4 | 0 | 3.060 | — |

All 3 cells PASS at first emit. MCLMC step_size 26.3–30.7 is consistent with the √d
prediction from `catalog/mclmc-scaling-laws.md`: 1.22×√500=27.3 (irt_1pl was in the
original validation set; this is not an out-of-sample check for that constant).

---

## lgcp (d=1600) — Category A, first recipes

**Model**: 40×40 Log-Gaussian Cox process, squared-exponential covariance.

**Note on the model's posterior**: `lgcp.py` sets `_area = 1/1600`, giving a synthetic
dataset with approximately 2 total Poisson events across 1600 cells. The KL divergence
from posterior to prior is ~0.9 nats over 1600 dims (0.0006 nats/dim vs irt_1pl at
0.54 nats/dim); none of the 1600 dimensions are meaningfully constrained by data. The
lgcp posterior is numerically indistinguishable from its isotropic Gaussian prior —
exactly the product-measure regime where MCLMC's advantage is largest. This matters
for interpreting the speedup number below.

**Gate shape-alignment fix**: the GT summary stores `z` as `(1600,)` flat while the
sampler returns positions shaped `(40,40)`. The broadcast in `_compute_gt_compare`
crashed with a shape mismatch. Fixed in `calibration/_gate/gt_compare.py` (commit
ffad44c) with a C-order reshape; also fixed the analogous bug in `w1_realm.py`.

| method | warmup | verdict | ESS/grad | vs NUTS | rhat | min_ESS | div | max_z | step_size |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| nuts | window_adaptation_diag_imm | **PASS** | 0.06845 | 1.00× | 1.0086 | 4206.9 | 0 | 3.175 | 0.21–0.28 |
| mclmc | mclmc_tuning | **PASS** | 0.2151 | **3.14×** | 1.0047 | 1723.2 | 0 | 3.356 | 44.93–49.26 |
| adjusted_mclmc_dynamic | adjusted_mclmc_tuning | REVIEW | 0.04213 | 0.62× | 1.0117 | 819.0 | 0 | 3.441 | 23.44–25.76 |

The lgcp row is a genuine out-of-sample check of the √d step-size prediction: the
`mclmc-scaling-laws.md` constant 1.22×√1600 = 48.8 is 0.4% from the per-chain midpoint
47.1 (range 44.93–49.26). The agreement is close; "exact confirmation" overstates a
per-chain range spanning ±4.7.

---

## MCLMC speedup — consistency check

| model | d | mclmc ESS/grad | nuts ESS/grad | speedup |
|---|---:|---:|---:|---:|
| irt_1pl | 500 | 0.2103 | 0.1239 | 1.70× |
| lgcp | 1600 | 0.2151 | 0.06845 | 3.14× |

The speedup increases with d. This is consistent with a cost model where NUTS scales
worse than MCLMC in d — for example, NUTS `O(d^{1/2})` vs MCLMC `O(d^{1/4})` gives a
predicted speedup ratio of `(1600/500)^{1/4} = 1.34`; observed is 1.85×. (Under the
canonical result where both are `O(d^{1/4})`, the predicted ratio is 1.00.)

Two cautions against reading this as a scaling-law confirmation:
1. The two points are not comparable. irt_1pl has a genuine 500-D posterior (KL = 268
   nats, all 500 dims constrained). lgcp has a nearly vacuous posterior (KL = 0.9 nats,
   0/1600 dims constrained) — the lgcp target is numerically the isotropic Gaussian prior.
   The 1.70→3.14× rise is confounded with a ~300× difference in likelihood information.
2. n=2, single seed per point, two different models. This is a consistency check, not a
   scaling-law confirmation.

---

## Cost notes (lgcp cost guard check)

50-step probe: 9.8s wall → projected full run 3.3 min. Well within the 2h guard.
Actual full run: NUTS 15.4s, mclmc 8.2s, adjusted_mclmc_dynamic 7.0s.
