# Sampling lessons: irt_2pl

## TL;DR

144-D IRT model. NUTS and dynamic_hmc PASS at LOW effort with diag or low_rank IMM.
**Dense IMM is an honest FAIL at d=144, n_warmup=2000** — the Welford estimator is
severely underdetermined (0.77 samples per covariance parameter). Do not transfer
"dense IMM works on ill_cond_50 (d=50)" claims to irt_2pl (d=144): dimension makes
the critical difference.
[boundary: diag/low_rank IMM PASS at n_warmup=1000; dense IMM FAIL at n_warmup=2000 (d=144); nearest FAIL: dynamic_hmc+dense_imm (rhat=1.57, ESS=7, n_warmup=2000, see recipes/failed__dynamic_hmc__window_adaptation_dense_imm.json)]

## Canonical recipe

`recipes/low__nuts__window_adaptation_diag_imm.json` — LOW effort, PASS.
`recipes/low__dynamic_hmc__window_adaptation_diag_imm.json` — LOW effort, PASS.

## Sampling quirks

### Dense IMM Welford failure at d=144 (confirmed 2026-05-30)
At d=144, the full covariance matrix has 10,296 independent parameters. With 4 chains
× 2,000 warmup steps = 8,000 total warmup draws, the Welford estimator has ~0.77
samples per parameter — severely underdetermined. One chain adapts step_size=0
(near-singular Cholesky), causing complete chain stagnation. rhat=1.57, ESS=7.1.

The revised dense IMM ceiling based on this failure: d≈80–100 at n_warmup=2000.
For irt_2pl (d=144), use `window_adaptation_diag_imm` or `window_adaptation_low_rank_imm`.

To use dense IMM at d=144, n_warmup would need to exceed ~10,000 to give the Welford
estimator adequate coverage. This was not tested.

### diag and low_rank IMM: PASS at LOW effort
`window_adaptation_diag_imm` (n_warmup=1000) and `window_adaptation_low_rank_imm`
PASS cleanly for NUTS, dynamic_hmc, and hmc (inner_nuts) variants.
[boundary: holds at n_warmup=1000; dmhmc+diag_imm FAIL at n_warmup=2000 due to V7 oracle miscalibration issue — use medium__dmhmc__window_adaptation_diag_imm__policy_v2-long.json instead]

## Known-bad combinations

- `dynamic_hmc` + `window_adaptation_dense_imm` (n_warmup=2000): **FAIL** (rhat=1.57, ESS=7.1).
  Welford estimator underdetermined at d=144. See `recipes/failed__dynamic_hmc__window_adaptation_dense_imm.json`.
  [⚠ boundary: dense IMM FAILS on irt_2pl's OWN recipes — "dense IMM handles ill-conditioning" does not transfer here at n_warmup=2000]
- `dmhmc` + `window_adaptation_dense_imm`: **FAIL** (extrapolated from dynamic_hmc F1; same root cause).
  See `recipes/failed__dmhmc__window_adaptation_dense_imm.json`.
- `dmhmc` + `window_adaptation_diag_imm` (n_warmup=2000, V7 oracle policy): **FAIL** (rhat=1.32, ESS=9.9).
  V7 auto-oracle miscalibrated for irt_2pl at ta=0.8. Use medium__dmhmc__window_adaptation_diag_imm__policy_v2-long.json.
  See `recipes/failed__dmhmc__window_adaptation_diag_imm.json`.

Recorded FAILs not discussed above: none — all three failed recipes are covered above.

## History

No detailed investigations recorded yet. If sampling pathologies emerge during recipe sweeps, case studies will be documented here.
2026-05-30: dense IMM failure recorded at d=144, n_warmup=2000 (statistician-id stat-2026-05-30).

## Citations

**Posteriordb reference**: [posteriordb.org #10 (irt_2pl)](https://posteriordb.org/)

**Stan reference**: Stan Case Studies
