# Sampling lessons: eight_schools_ncp

## TL;DR

Well-conditioned post-NCP geometry; library defaults pass at LOW. Notable supersession: **`laplace_dhmc` and `laplace_dmhmc` dominate `laplace_hmc` and `laplace_mhmc`** on this model — prefer the dynamic-L variants for any laplace recipe.

## Canonical recipe

`low__nuts__window_adaptation_diag_imm.json` is the canonical LOW recipe (4 chains × 1000 samples, ta=0.8, gate PASS).

## Sampling quirks

- Standard d=10 hierarchical model with phi=(mu, tau) hyperparameters + theta_raw (d=8) NCP innovations. Window adaptation captures the modest scale anisotropy via diagonal IMM.
- `phi` ↔ `theta_raw` correlation is mild post-NCP; dense IMM doesn't materially improve over diag.

## Known-bad combinations

- **`laplace_hmc` + `window_adaptation_dense_imm`** (REVIEW): fixed-L HMC on the marginal log-density under a dense 2×2 IMM under-mixes; min_bulk_ESS=241 misses the 400 gate. The `laplace_dhmc` (dynamic-L) variant of the same cell PASSes cleanly. Prefer dhmc/dmhmc for laplace cells whenever the trajectory-length distribution matters — confirmed by the wadapt-hmc-sweep Phase 3c run (`worklog/threads/wadapt-hmc-sweep.md` § 11 surprise #5).
- **fixed-L `hmc` + dense IMM** on this model: 16 divergences at ta=0.8 (the default `num_integration_steps` doesn't match the resonance of the dense-IMM-transformed Hamiltonian). Use NUTS or dynamic_hmc to side-step.

## Laplace supersession claim (wadapt-hmc-sweep Phase 3c, 2026-05-19)

| Cell | Effort | ESS | Note |
|---|---|---|---|
| W1 × laplace_hmc | LOW | 804 | passes but ~4× lower ESS than laplace_dhmc |
| W1 × laplace_dhmc | LOW | 3080 | **preferred** |
| W1 × laplace_mhmc | LOW | passes | similar ESS profile to dhmc |
| W1 × laplace_dmhmc | LOW | passes | **preferred (multinomial proposal)** |
| W2 × laplace_hmc | REVIEW | 241 | misses gate; supersession-failed |
| W2 × laplace_dhmc | LOW | passes | |

The empirical pattern: dynamic-L laplace variants (`laplace_dhmc`, `laplace_dmhmc`) sample the same posterior at materially higher ESS/grad than their fixed-L counterparts (`laplace_hmc`, `laplace_mhmc`) on this model. Cite when recommending a laplace recipe.

## History

- 2026-05-19 (Phase 3c): full laplace_* × {diag, dense} sweep run. 7/8 cells PASS; W2×laplace_hmc REVIEW surfaces the supersession claim above.
- 2026-05-18 (Phase 2a): laplace-marginal warmup pathway verified end-to-end via `_laplace_adapter.resolve_warmup_algorithm` substituting `blackjax.hmc` at warmup time. E2E test at `tests/inference/warmup/test_laplace_e2e_verify.py` PASSes (n_warmup=500, n_samples=1000, IMM (2,2) phi-dimensional, 0 divergence rate).

## Citations

**Posteriordb reference**: [posteriordb.org #3 (eight_schools_centeredncp)](https://posteriordb.org/)

**Stan reference**: Stan User's Guide § 1.2
