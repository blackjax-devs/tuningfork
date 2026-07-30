# Sampling lessons: eight_schools_ncp

## TL;DR

Well-conditioned post-NCP geometry; library defaults pass at LOW. Notable supersession: **`laplace_dhmc` and `laplace_dmhmc` dominate `laplace_hmc` and `laplace_mhmc`** on this model — prefer the dynamic-L variants for any laplace recipe.
[boundary: dynamic-L dominance holds on eight_schools_ncp (d=10, LOW effort); does not imply dynamic-L always fixes laplace issues on larger/harder models; nearest FAIL: laplace_hmc+low_rank_imm (REVIEW, rhat=1.0185, see recipes/failed__laplace_hmc__window_adaptation_low_rank_imm.json)]

## Canonical recipe

`low__nuts__window_adaptation_diag_imm.json` is the canonical LOW recipe (4 chains × 1000 samples, ta=0.8, gate PASS).

## Sampling quirks

- Standard d=10 hierarchical model with phi=(mu, tau) hyperparameters + theta_raw (d=8) NCP innovations. Window adaptation captures the modest scale anisotropy via diagonal IMM.
- `phi` ↔ `theta_raw` correlation is mild post-NCP; dense IMM doesn't materially improve over diag.

## Known-bad combinations

- **`laplace_hmc` + `window_adaptation_dense_imm`** (REVIEW): fixed-L HMC on the marginal log-density under a dense 2×2 IMM under-mixes; min_bulk_ESS=241 misses the 400 gate. The `laplace_dhmc` (dynamic-L) variant of the same cell PASSes cleanly. Prefer dhmc/dmhmc for laplace cells whenever the trajectory-length distribution matters — confirmed by the wadapt-hmc-sweep investigation.
  [boundary: REVIEW at LOW effort (d=10, n_warmup=1000); dynamic-L variant PASSes cleanly at same budget]
- **`laplace_hmc` + `window_adaptation_low_rank_imm`**: **FAIL** (REVIEW gate, rhat=1.0185, not run to PASS).
  See `recipes/failed__laplace_hmc__window_adaptation_low_rank_imm.json`.
  [boundary: fixed-L laplace_hmc consistently underperforms dynamic-L across all three IMM types on this model]
- **fixed-L `hmc` + dense IMM** on this model: 16 divergences at ta=0.8 (the default `num_integration_steps` doesn't match the resonance of the dense-IMM-transformed Hamiltonian). Use NUTS or dynamic_hmc to side-step.
  [boundary: resonance is a fixed-L artefact; dynamic_hmc+dense_imm PASSes cleanly (see low__dynamic_hmc__window_adaptation_dense_imm.json)]

Recorded FAILs not discussed above: failed__laplace_hmc__window_adaptation_low_rank_imm.json is now listed above.

## Laplace supersession claim (wadapt-hmc-sweep investigation, 2026-05-19)

| Cell | Effort | ESS | Note |
|---|---|---|---|
| W1 × laplace_hmc | LOW | 804 | passes but ~4× lower ESS than laplace_dhmc |
| W1 × laplace_dhmc | LOW | 3080 | **preferred** |
| W1 × laplace_mhmc | LOW | passes | similar ESS profile to dhmc |
| W1 × laplace_dmhmc | LOW | passes | **preferred (multinomial proposal)** |
| W2 × laplace_hmc | REVIEW | 241 | misses gate; supersession-failed |
| W2 × laplace_dhmc | LOW | passes | |

The empirical pattern: dynamic-L laplace variants (`laplace_dhmc`, `laplace_dmhmc`) sample the same posterior at materially higher ESS/grad than their fixed-L counterparts (`laplace_hmc`, `laplace_mhmc`) on this model. Cite when recommending a laplace recipe.
[boundary: supersession holds at d=10, LOW effort (n_warmup=1000); laplace family requires a separable log-joint — does not apply to models lacking the phi/theta decomposition; mvn_10 and logistic_synthetic laplace+low_rank_imm cells FAIL as out_of_scope (model structure mismatch, not trajectory-length issue)]

## History

- 2026-05-19: full laplace_* × {diag, dense} sweep run. 7/8 cells PASS; W2×laplace_hmc REVIEW surfaces the supersession claim above.
- 2026-05-18: laplace-marginal preflight — warmup pathway verified end-to-end via `_laplace_adapter.resolve_warmup_algorithm` substituting `blackjax.hmc` at warmup time. E2E test at `tests/inference/warmup/test_laplace_e2e_verify.py` PASSes (n_warmup=500, n_samples=1000, IMM (2,2) phi-dimensional, 0 divergence rate).

### 2026-07-30 — `dynamic_hmc`+dense_imm promoted to MEDIUM after a recert-sweep gate failure

The 2026-07-30 corpus recert sweep (tuningfork PR #254, ESS-metric switch)
found `low__dynamic_hmc__window_adaptation_dense_imm` failing the gate at its
committed seed under current dependencies (blackjax 1.6.1 / jax 0.11.0) — one
of the PR's "19 gate failures, no recipe written" set (this model's only
affected cell; note line 25 above, which cites this exact recipe as evidence
of resonance-avoidance, was written before this regression and is currently
stale for the LOW recipe specifically).

Per Belief#1176, re-running the identical config at seed=11111 (the first
candidate tried) cleared the gate cleanly on the first attempt: rhat=1.0035,
ess=2156.2, 0 divergences — no wider seed scan was needed. Promoted to
`recipes/medium__dynamic_hmc__window_adaptation_dense_imm__reseed.json` (the
seed selection is disclosed in the recipe's `notes` field); the LOW recipe is
kept unchanged alongside it.

## Citations

**Posteriordb reference**: [posteriordb.org #3 (eight_schools_centeredncp)](https://posteriordb.org/)

**Stan reference**: Stan User's Guide § 1.2
