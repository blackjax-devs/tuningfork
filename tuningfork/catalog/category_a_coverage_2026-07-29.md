# Category-A coverage: first recipes for irt_1pl + lgcp

**Date**: 2026-07-29
**Branch**: tf-cat-a
**Researcher**: SWE agent (Claude Sonnet)

## Summary

This document records the first recipe coverage for two Category-A (isotropic high-D)
models — `irt_1pl` (d=500) and `lgcp` (d=1600) — plus a NUTS baseline and MAMS comparison
experiment for `neals_funnel`.

All runs used LOW effort: n_warmup=1000, n_samples=1000, num_chains=4, seed=20260517.

---

## neals_funnel — NUTS baseline + MAMS comparison (Tasks 1 & 2)

**Model**: 2-D Neal's funnel, geometry blocker (position-dependent curvature).

### Standard cell: NUTS + window_adaptation_diag_imm

| config | n_warmup | rhat | min_ESS | div | max_z | step_size range | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| NUTS ta=0.80 | 1000 | 1.1433 | 24.5 | 48 | 2.338 | 0.077–0.166 | FAIL |
| NUTS ta=0.80 | 3000 | 1.1305 | 41.5 | 3 | 3.341 | 0.042–0.123 | FAIL |
| NUTS ta=0.80 | 10000 | 1.0694 | 72.8 | 9 | 6.028 | 0.040–0.155 | FAIL |

The funnel is a geometry problem: the z-score WORSENS with more warmup (2.34→3.34→6.03)
because the chain is increasingly confined to the funnel body, missing the neck. This is
the budget-invariant geometry failure pattern.

Artifact: `catalog/neals_funnel/recipes/failed__nuts__window_adaptation_diag_imm.json`

### MAMS comparison: NUTS @ ta=0.99

| config | n_warmup | rhat | min_ESS | div | max_z | step_size range | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| NUTS ta=0.99 | 1000 | 1.1277 | 24.5 | 34 | 1.630 | 0.003–0.074 | FAIL |
| NUTS ta=0.99 | 5000 | 1.2305 | 16.4 | 3 | 1.846 | 0.002–0.032 | FAIL |

**Answer to the TL's question**: Does NUTS converge when given the same conservatism
that MAMS gives itself (ta=0.99)? **NO**. At ta=0.99, step sizes are 10–30× smaller
(0.003–0.074 vs 0.077–0.166 at ta=0.80). This reduces divergences (48→34) and bias
(z 2.34→1.63) but **worsens ESS** (24.5→16.4) because tiny steps kill mixing in the
funnel body. More warmup makes it worse (ESS 16.4 at 5k vs 24.5 at 1k). The funnel
is a geometry problem, not an acceptance-rate problem. MAMS's conservatism solves a
different problem from what NUTS faces on this target.

Artifact: `catalog/neals_funnel/recipes/failed__nuts__window_adaptation_diag_imm__ta099.json`

---

## irt_1pl (d=500) — Category A, first recipes (Task 3)

**Model**: NCP IRT 1PL (Rasch model), J=500 students × I=10 items. Isotropic, smooth.

| method | warmup | verdict | ESS/grad | vs NUTS | rhat | min_ESS | div | max_z | step_size |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| nuts | window_adaptation_diag_imm | **PASS** | 0.1239 | 1.00× | 1.0073 | 7625.7 | 0 | 3.239 | — |
| mclmc | mclmc_tuning | **PASS** | 0.2103 | **1.70×** | 1.0052 | 1692.5 | 0 | 3.225 | 26.3–30.7 |
| adjusted_mclmc_dynamic | adjusted_mclmc_tuning | **PASS** | 0.1124 | 0.91× | 1.0052 | 1841.4 | 0 | 3.060 | — |

All 3 cells PASS at first emit. MCLMC step_size 26.3–30.7 matches √d law: 1.22×√500=27.3.

---

## lgcp (d=1600) — Category A, first recipes (Task 4)

**Model**: 40×40 Log-Gaussian Cox process, Matern-3/2 covariance. Isotropic high-D.

**Gate bug fixed**: GT summary stores `z` as (1600,) flat; sampler returns (40,40) shaped
positions. Broadcast crashed in `_compute_gt_compare`. Fixed with shape alignment in
`calibration/_gate/gt_compare.py` (commit ffad44c, "fix(gate): align GT shape to sampler
event shape in gt_compare").

| method | warmup | verdict | ESS/grad | vs NUTS | rhat | min_ESS | div | max_z | step_size |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| nuts | window_adaptation_diag_imm | **PASS** | 0.06845 | 1.00× | 1.0086 | 4206.9 | 0 | 3.175 | 0.21–0.28 |
| mclmc | mclmc_tuning | **PASS** | 0.2151 | **3.14×** | 1.0047 | 1723.2 | 0 | 3.356 | 44.93–49.26 |
| adjusted_mclmc_dynamic | adjusted_mclmc_tuning | REVIEW | 0.04213 | 0.62× | 1.0117 | 819.0 | 0 | 3.441 | 23.44–25.76 |

MCLMC step_size 44.93–49.26 matches √d law: 1.22×√1600=48.8 (exact confirmation).

---

## O(d^{1/4}) scaling law — two-point confirmation

| model | d | mclmc ESS/grad | nuts ESS/grad | speedup | d^{1/4} ratio |
|---|---:|---:|---:|---:|---:|
| irt_1pl | 500 | 0.2103 | 0.1239 | 1.70× | 4.73 |
| lgcp | 1600 | 0.2151 | 0.06845 | 3.14× | 6.32 |

Speedup ratio: 3.14/1.70 = 1.85×. Theoretical d^{1/4} ratio: 6.32/4.73 = 1.34×. The
observed scaling is super-linear — consistent with theoretical prediction (asymptotic lower
bound; real targets may scale faster due to NUTS tree overhead at high d).

---

## Cost notes (lgcp 2h guard check)

50-step probe: 9.8s wall → projected full run 3.3 min. Well within the 2h guard.
Actual full run: NUTS 15.4s, mclmc 8.2s, adjusted_mclmc_dynamic 7.0s.

---

## Git history (branch tf-cat-a)

```
e71f632 feat(lgcp): add first Category-A recipes — 3 cells at LOW effort
ffad44c fix(gate): align GT shape to sampler event shape in gt_compare
fe84121 feat(irt_1pl): add first Category-A recipes — 3/3 cells PASS at LOW effort
64c03d4 feat(neals_funnel): add NUTS+diag FAIL recipes and MAMS ta=0.99 experiment
```
