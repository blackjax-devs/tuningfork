# Sampling lessons: lgcp

## TL;DR

**Category A — Isotropic high-D**: d=1600 Log-Gaussian Cox process (40×40 spatial grid,
squared-exponential covariance). Unadjusted MCLMC wins **3.14×** over NUTS on ESS/grad at first
emit. All three cells ran successfully with the gate shape-alignment fix (GT summaries
store z as (1600,) flat; sampler returns (40,40) 2D grid — fixed in
`calibration/_gate/gt_compare.py`). Confirms that the O(d^{1/4}) MCLMC advantage grows
with d: 1.70× at d=500 (irt_1pl) → 3.14× at d=1600 (lgcp).

## Results (2026-07-29, first Category-A coverage)

| method | warmup | verdict | ESS/grad | vs NUTS | rhat | min_ESS | div | |z| |
|---|---|---|---:|---:|---:|---:|---:|---:|
| nuts | window_adaptation_diag_imm | **PASS** | 0.06845 | 1.00× | 1.0086 | 4206.9 | 0 | 3.175 |
| mclmc | mclmc_tuning | **PASS** | 0.2151 | **3.14×** | 1.0047 | 1723.2 | 0 | 3.356 |
| adjusted_mclmc_dynamic | adjusted_mclmc_tuning | REVIEW | 0.04213 | 0.62× | 1.0117 | 819.0 | 0 | 3.441 |

MCLMC step_size range: 44.93–49.26. The √d law predicts 1.22×√1600 = 1.22×40 = **48.8**
— directly confirmed. Adjusted MCLMC step_size 23.44–25.76 (≈ 0.5× unadjusted), consistent
with the tighter MH-correction constraint.

## Canonical recipes

- `recipes/low__nuts__window_adaptation_diag_imm.json` — NUTS baseline, ESS/grad=0.06845
- `recipes/low__mclmc__mclmc_tuning.json` — **best challenger**, ESS/grad=0.2151 (3.14×)
- `recipes/low__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json` — adjusted variant,
  ESS/grad=0.04213 (0.62×), verdict=REVIEW (rhat 1.0117 slightly above 1.01 threshold)

## Model geometry

LGCP model: 40×40 spatial grid, squared-exponential covariance kernel, Poisson intensity.
Unconstrained dimensionality: d=1600 (one latent field value per grid cell).

The model is approximately isotropic in the sense that the covariance eigenspectrum is
smooth and decays regularly. The diagonal mass matrix tuned by `mclmc_tuning` absorbs the
dominant scale differences. This is Category A in the MCLMC routing taxonomy: high-D,
weakly-correlated (or smoothly correlated with diagonal-preconditioning-tractable structure),
no hard direction of curvature. The unadjusted MCLMC's O(d^{1/4}) gradient advantage
fully materializes.

## Why MCLMC wins here (3.14×)

The 3.14× speedup is larger than irt_1pl (1.70×) because both d and the O(d^{1/4}) ratio
scale with d. At d=1600:
- 1600^{1/4} = 6.32 vs 500^{1/4} = 4.73 for irt_1pl → 34% more relative advantage.
- NUTS step count per effective sample grows with d; MCLMC's momentum persistence keeps
  it low.
- The unadjusted variant avoids the MH overhead; the squared-exponential kernel produces
  analytic (infinitely smooth) sample paths with super-exponentially decaying eigenvalues,
  so the leapfrog integrator error is small enough that the MH correction is unnecessary.

## adjusted_mclmc_dynamic — REVIEW, not PASS

The adjusted variant is REVIEW (rhat 1.0117 > 1.01) with ESS/grad 0.04213 (0.62×). Two
contributing factors:
1. MH overhead at d=1600 is non-trivial: each proposal requires one full gradient + one
   Metropolis step per leapfrog, slowing effective throughput.
2. The step_size is smaller (23–26 vs 45–49 for unadjusted) because MH correction tightens
   the error tolerance. Fewer gradient steps per effective sample.

The slight rhat elevation (1.0117) may be a stochastic fluctuation at 1k warmup; a longer
warmup might push it below 1.01. However, the headline ESS/grad is already below NUTS, so
the adjusted variant is not the recommended recipe for this model regardless of verdict.

## Note on the shape-alignment bug (gate fix)

Running lgcp recipes required a one-line fix in
`tuningfork/calibration/_gate/gt_compare.py`: the GT summary stores `z` as a flat (1600,)
array, but the sampler returns positions shaped (40,40) per draw. The `denom` broadcast
in `_compute_gt_compare` crashed with shape mismatch `(40,40) vs (1600,)`.

Fix: after loading `gt_mean`/`gt_std`/`between_chain_se`/`gt_bulk_ess` from the GT
summaries, reshape them to `sample_mean.shape` when sizes agree but shapes differ. This
generalises to any future model with structured event shapes stored flat in ground truth.

## Known-good combinations

| combination | verdict | ESS/grad | notes |
|---|---|---:|---|
| nuts + window_adaptation_diag_imm | PASS | 0.06845 | solid baseline |
| mclmc + mclmc_tuning | PASS | 0.2151 | **recommended** |

## Boundary annotations

[boundary: PASS for nuts and mclmc; REVIEW for adjusted_mclmc_dynamic at n_warmup=1k,
n_samples=1k, num_chains=4; first-emit result — no escalation ladder needed for PASS cells]

## History

2026-07-29: First recipes for lgcp. All three Category-A cells run. Gate shape-alignment
fix needed first (GT flat-vs-structured shape mismatch for 2D grid params).
See `catalog/mclmc-routing-taxonomy.md` §4 (Category A).
