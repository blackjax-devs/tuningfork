# Sampling lessons: lotka_volterra

## TL;DR

Stiff ODE posterior with bimodal structure. Dense and low_rank IMM PASS for NUTS/dmhmc/dynamic_hmc/hmc at LOW effort. **Diag IMM FAILS for dynamic_hmc and dmhmc** at LOW effort and even MEDIUM; the stiff ODE geometry requires off-diagonal mass matrix structure. `mhmc` is structurally unsuitable (step_size collapses). MCLMC variants FAIL (warmup hang). VI is out_of_scope.
[boundary: dense/low_rank IMM PASS holds at LOW n_warmup=1000; diag IMM FAIL confirmed across multiple step policies; nearest FAIL: dynamic_hmc+diag_imm (see recipes/failed__dynamic_hmc__window_adaptation_diag_imm.json); dense IMM PASS for nuts (see recipes/failed__nuts__window_adaptation_dense_imm.json — this one actually FAILS too, see below)]

## Canonical recipe

`recipes/low__nuts__window_adaptation_low_rank_imm.json` — LOW effort, PASS.
`recipes/low__dynamic_hmc__window_adaptation_dense_imm.json` — LOW effort, PASS.
`recipes/low__mclmc__mclmc_tuning.json` — LOW effort, PASS.

## Sampling quirks

### Diag IMM insufficient for stiff ODE geometry
`dynamic_hmc` and `dmhmc` with `window_adaptation_diag_imm` FAIL even with MEDIUM
step policies (v1-medium, v7-empirical-oracle). The bimodal ODE posterior has
off-diagonal covariance that the diagonal IMM cannot capture.
[boundary: diag IMM FAIL is policy-invariant — same failure at LOW (default), v1-medium, and v7-empirical-oracle; use dense or low_rank IMM]

### mhmc: structurally unsuitable
`mhmc` with any IMM (diag, dense, low_rank) fails because step_size adaptation
collapses near the near-degenerate ODE likelihood boundary.
[boundary: mhmc FAIL is IMM-invariant; see failed__mhmc__window_adaptation_{dense,diag,low_rank}_imm.json]

### MCLMC variants: warmup hang
`adjusted_mclmc` and `adjusted_mclmc_dynamic` with `adjusted_mclmc_tuning` hang
during warmup (non-terminating warmup loop) due to the stiff ODE gradient landscape.
[boundary: warmup hang at any budget; not tunable; do not use MCLMC on lotka_volterra without a custom warmup]

### NUTS + dense IMM: FAIL at LOW (specific cell)
`nuts` + `window_adaptation_dense_imm` at LOW effort FAILS (hard_direction).
NUTS + `window_adaptation_low_rank_imm` PASSES at LOW effort.
[boundary: dense IMM fails for NUTS at default LOW; low_rank IMM PASS for NUTS; but dense IMM PASS for dynamic_hmc and dmhmc — applies at DIFFERENT configurations]

## Known-bad combinations

- `dynamic_hmc` + `window_adaptation_diag_imm` (any step policy): **FAIL**. See `recipes/failed__dynamic_hmc__window_adaptation_diag_imm.json`, `failed__dynamic_hmc__window_adaptation_diag_imm__policy_v1-medium.json`, `failed__dynamic_hmc__window_adaptation_diag_imm__policy_v7-empirical-oracle.json`.
- `dmhmc` + `window_adaptation_diag_imm` (any step policy): **FAIL**. See `recipes/failed__dmhmc__window_adaptation_diag_imm.json`, `failed__dmhmc__window_adaptation_diag_imm__policy_v1-medium.json`, `failed__dmhmc__window_adaptation_diag_imm__policy_v7-empirical-oracle.json`.
- `mhmc` + `window_adaptation_dense_imm`: **FAIL** (step_size collapse). See `recipes/failed__mhmc__window_adaptation_dense_imm.json`.
- `mhmc` + `window_adaptation_diag_imm`: **FAIL**. See `recipes/failed__mhmc__window_adaptation_diag_imm.json`.
- `mhmc` + `window_adaptation_low_rank_imm`: **FAIL**. See `recipes/failed__mhmc__window_adaptation_low_rank_imm.json`.
- `adjusted_mclmc` + `adjusted_mclmc_tuning`: **FAIL** (warmup hang). See `recipes/failed__adjusted_mclmc__adjusted_mclmc_tuning.json`.
- `adjusted_mclmc_dynamic` + `adjusted_mclmc_tuning`: **FAIL** (warmup hang). See `recipes/failed__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json`.
- `nuts` + `window_adaptation_dense_imm` (LOW): **FAIL** (hard_direction). See `recipes/failed__nuts__window_adaptation_dense_imm.json`.
- `fullrank_vi` + `no_warmup`: **FAIL** (out_of_scope). See `recipes/failed__fullrank_vi__no_warmup.json`.
- `meanfield_vi` + `no_warmup`: **FAIL** (out_of_scope). See `recipes/failed__meanfield_vi__no_warmup.json`.

Recorded FAILs not discussed above: all 14 failed recipes are covered above.

## History

No detailed investigations recorded yet. If sampling pathologies emerge during recipe sweeps execute, case studies will be logged to `worklog/lessons/case-studies/lotka_volterra/`.

## Citations

**Real-data model**: Lotka-Volterra predator-prey ODE with real population data
