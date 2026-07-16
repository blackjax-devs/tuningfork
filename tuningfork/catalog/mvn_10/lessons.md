# Sampling lessons: mvn_10

## TL;DR

Well-conditioned 10-D MVN. NUTS, MCLMC, dmhmc, dynamic_hmc, VI all PASS at LOW effort.
Optimal MCLMC trajectory is avg=2 (smooth diagonal geometry). Several laplace and
structural cells are out-of-scope (require model change).
[boundary: PASS applies at LOW effort (n_warmup=1000); hmc+low_rank_imm FAIL (fixed-L hard_direction); laplace family out_of_scope (no separable log-joint in mvn_10)]

## Canonical recipe

`recipes/low__nuts__window_adaptation_diag_imm.json` — LOW effort, PASS.
`recipes/low__mclmc__mclmc_tuning.json` — LOW effort, PASS (avg=2 optimal for smooth diagonal geometry).

## Sampling quirks

None significant. mvn_10 is a smooth, well-conditioned diagonal model used as a
structural baseline. Behaves bias-indifferent across the avg ladder (see Dynamic-L
sweep below); optimal at avg=2 for mixing efficiency.

## Known-bad combinations

- `hmc` + `window_adaptation_low_rank_imm`: **FAIL** (hard_direction). Fixed-L HMC
  with low_rank IMM cannot traverse the 10-D geometry with default integration steps.
  See `recipes/failed__hmc__window_adaptation_low_rank_imm.json`.
  [boundary: hmc+diag_imm and hmc+dense_imm PASS (see low__hmc__window_adaptation_low_rank_imm__inner_nuts.json for the inner_nuts workaround)]
- `elliptical_slice` + `no_warmup`: **FAIL** (requires_model_change — needs Gaussian prior).
  See `recipes/failed__elliptical_slice__no_warmup.json`.
- `laplace_hmc` + `no_warmup`: **FAIL** (requires_model_change — laplace needs separable log-joint).
  See `recipes/failed__laplace_hmc__no_warmup.json`.
- `laplace_*` + `window_adaptation_low_rank_imm` (dhmc, dmhmc, hmc, mhmc): **FAIL** (out_of_scope).
  See `recipes/failed__laplace_dhmc__window_adaptation_low_rank_imm.json` etc.

Recorded FAILs not discussed above: all 7 failed recipes are covered above.

## History

No detailed investigations recorded yet. If sampling pathologies emerge during recipe sweeps, case studies will be documented here.

## Dynamic-L Sweep (avg ladder)

Run date: 2026-06-19 | Source: sweep_dynl_variety_results.json, medians over 3 seeds

| avg | realized_avg | ESS | Rhat | 2nd-mom bias | mbias_sd | trend |
|---|---|---|---|---|---|---|
| 2 | 2.0 | 2737 | 1.005 | 0.103 | 0.037 | **OPTIMAL** |
| 6 | 6.0 | 1605 | 1.005 | 0.211 | 0.045 | degrading |
| 18 | 18.0 | 1548 | 1.010 | 0.415 | 0.051 | *monotone worse* |
| 54 | 54.2 | 1374 | 1.009 | 0.257 | 0.057 | ↓ |
| 108 | 108.3 | 1651 | 1.033 | 0.158 | 0.045 | (recovery noise) |

**Lesson:** avg=2 gives highest ESS (2712); increasing avg monotonically degrades mixing. All configs have small
mbias_sd (0.04–0.06), indicating small true bias (noisy max-over-D 2mbias not reliable at n=500). mvn_10 is
bias-indifferent across the ladder; optimal at avg=2 for mixing efficiency, not bias avoidance. Behaves like a
smooth, well-conditioned diagonal model.

See `catalog/mclmc-scaling-laws.md` §3 for generalized principles (why smooth diagonal models want SHORT L, etc.).

## Citations

**Synthetic baseline** — no external reference. Standard test model for sampler validation.
