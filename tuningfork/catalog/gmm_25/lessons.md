# Sampling lessons: gmm_25

## TL;DR

25-mode Gaussian mixture. **Standard MCMC (NUTS, hmc) fails** due to multimodality —
single-chain samplers cannot explore all 25 modes. SMC is the viable path: both
`adaptive_tempered_smc__rwm` and `inner_kernel_tuning__hmc` PASS.
[boundary: SMC PASS holds at the standard 25-mode synthetic GMM; MCMC cells (nuts+diag, nuts+low_rank, hmc+low_rank) are requires_alt_sampler or out_of_scope; laplace family out_of_scope]

## Canonical recipe

`recipes/smc__adaptive_tempered_smc__rwm.json` — PASS.
`recipes/smc__inner_kernel_tuning__hmc.json` — PASS.

## Sampling quirks

### Multimodal geometry requires SMC
25 equally-weighted Gaussian components. Standard single-chain MCMC (NUTS, HMC)
cannot reliably traverse between 25 separated modes. SMC with sequential tempering
explores the full mixture by building up from the prior.
[boundary: multimodal failure is model-structural, not tunable via n_warmup or mass matrix; requires_alt_sampler for single-chain methods]

## Known-bad combinations

- `nuts` + `window_adaptation_diag_imm`: **FAIL** (out_of_scope — multimodal requires alt sampler). See `recipes/failed__nuts__window_adaptation_diag_imm.json`.
- `nuts` + `window_adaptation_low_rank_imm`: **FAIL** (requires_alt_sampler). See `recipes/failed__nuts__window_adaptation_low_rank_imm.json`.
- `hmc` + `window_adaptation_low_rank_imm`: **FAIL** (requires_alt_sampler). See `recipes/failed__hmc__window_adaptation_low_rank_imm.json`.
- Laplace family + `window_adaptation_low_rank_imm`: **FAIL** (out_of_scope). See `recipes/failed__laplace_*__window_adaptation_low_rank_imm.json`.
  [boundary: all MCMC failures are structural (multimodal geometry); not tunable; use SMC path]

Recorded FAILs not discussed above: all 7 failed recipes are now documented above.

## History

No detailed investigations recorded yet. If sampling pathologies emerge during recipe sweeps, case studies will be documented here.

## Citations

**Synthetic baseline** — no external reference. Standard test model for sampler validation.
