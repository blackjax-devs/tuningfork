# Sampling lessons: logistic_synthetic

## TL;DR

Well-conditioned logistic regression. NUTS, MCLMC, dmhmc, dynamic_hmc, hmc, VI all PASS
at LOW effort. Laplace family cells with `window_adaptation_low_rank_imm` are out_of_scope
(no separable log-joint structure in the synthetic logistic model).
[boundary: laplace+low_rank_imm FAIL is out_of_scope, not a sampler failure; diag and dense IMM PASS for all HMC variants at LOW effort]

## Canonical recipe

`recipes/low__nuts__window_adaptation_diag_imm.json` — LOW effort, PASS.
`recipes/low__mclmc__mclmc_tuning.json` — LOW effort, PASS.

## Sampling quirks

None significant. logistic_synthetic is a well-conditioned baseline for logistic regression.

## Known-bad combinations

- `laplace_hmc` + `window_adaptation_low_rank_imm`: **FAIL** (out_of_scope — logistic_synthetic lacks the separable phi/theta log-joint required by laplace kernels).
  See `recipes/failed__laplace_hmc__window_adaptation_low_rank_imm.json`.
- `laplace_dhmc` + `window_adaptation_low_rank_imm`: **FAIL** (out_of_scope, same reason).
  See `recipes/failed__laplace_dhmc__window_adaptation_low_rank_imm.json`.
- `laplace_dmhmc` + `window_adaptation_low_rank_imm`: **FAIL** (out_of_scope).
  See `recipes/failed__laplace_dmhmc__window_adaptation_low_rank_imm.json`.
- `laplace_mhmc` + `window_adaptation_low_rank_imm`: **FAIL** (out_of_scope).
  See `recipes/failed__laplace_mhmc__window_adaptation_low_rank_imm.json`.
  [boundary: these are structural out_of_scope failures, not tunable — do not expect PASS at any n_warmup for laplace family on this model]

Recorded FAILs not discussed above: all 4 failed recipes are covered above.

## History

No detailed investigations recorded yet. If sampling pathologies emerge during recipe sweeps execute, case studies will be logged to `worklog/lessons/case-studies/logistic_synthetic/`.

## Citations

**Synthetic baseline** — no external reference. Standard test model for sampler validation.
