# Sampling lessons: ill_cond_50

## TL;DR

**Rotational ill-conditioning (κ=1000) is the defining geometry.** Diagonal mass
matrices — whether NUTS or MCLMC — require careful treatment. NUTS with
`window_adaptation_diag_imm` passes at LOW effort (the diagonal IMM effectively
rescales the rotated axes during adaptation). Standard diagonal `mclmc` is an
**honest FAIL** at any warmup budget; the diagonal mass matrix cannot capture the
rotated correlation axes. The only viable MCLMC path is **LRD preconditioning**
(k=40, NUTS-pilot extraction).
[boundary: NUTS+diag_imm PASS holds at n_warmup=1000; LRD PASS holds at k=40, n_warmup=1000; dense IMM PASS for NUTS/dmhmc/dynamic_hmc at n_warmup=1000 but FAIL for fixed-L hmc (resonance); nearest FAIL: hmc+dense_imm (see recipes/failed__hmc__window_adaptation_dense_imm.json)]

## Canonical recipe

**NUTS** (default): `recipes/low__nuts__window_adaptation_diag_imm.json` — PASS,
headline ESS/grad ≈ 0.0065.

**MCLMC with LRD** (certified): `recipes/low__mclmc_lrd__mclmc_lrd_tuning.json` — PASS,
headline ESS/grad ≈ 0.240 (414× over diagonal MCLMC baseline; vs best NUTS 0.137, ratio 1.75×).

## Sampling quirks

### Rotational ill-conditioning (κ=1000)
The covariance matrix is Σ = U Λ Uᵀ with eigenvalues logarithmically spaced from
1 to 1000 and U a fixed random orthogonal matrix. The principal axes are **rotated
relative to the coordinate axes**, so any diagonal mass matrix is misaligned.

- NUTS `window_adaptation_diag_imm`: the adaptive diagonal IMM finds a diagonal
  approximation to the rotated geometry during warmup, which is sufficient for PASS.
  [boundary: PASS at n_warmup=1000; dense/low_rank IMM also PASS for NUTS at n_warmup=1000]
- Diagonal MCLMC (`mclmc_tuning`): fails catastrophically at all warmup budgets.
  The trajectory length L cannot compensate for the rotational mismatch.
  Plateau confirmed at `n_warmup=100k`: R-hat oscillates 1.05–1.07, ESS≈135.
  See `recipes/failed__mclmc__mclmc_tuning.json` for the full attempted-configurations
  ladder.
  [boundary: FAIL confirmed up to n_warmup=100k; no warmup budget recovers this — geometry is the blocker]

### LRD MCLMC pipeline (ill_cond_50, k=40)
**Generated NUTS pilot → SVD extraction → statically bound LRD kernel →
`mclmc_find_L_and_step_size`**

1. Emit and run a 1000-step diagonal NUTS pilot
2. Compute empirical σ and top-40 eigenvectors/eigenvalues via emitted SVD extraction
3. Construct `LowRankInverseMassMatrix(sigma, U, lam)`
4. Emit a statically bound LRD kernel and run
   `mclmc_find_L_and_step_size(diagonal_preconditioning=False)`

The recorded experiment used direct helpers named `run_pilot_nuts`,
`extract_lrd_from_samples`, and `make_lrd_kernel`; codegen now emits the same steps
inline and is the only executable route.

Result: R-hat=1.0039, ESS=1993.3, ESS/grad=0.2492, PASS (statistician independent
run, seed=98765). Multi-seed hardening at seeds 11111/22222/33333 all PASS
(ESS 1944–2030). 426× ESS/grad improvement over the diagonal MCLMC baseline.
[boundary: PASS holds at k=40, n_warmup=1000 (pilot), avg=2; FAIL at k=10/20 (see integrator ladder); nearest FAIL: k=20 is REVIEW, k=10 is FAIL; this is the lowest-headroom PASS in the catalog]

**Headroom note (k=40 truncation).** This certified PASS is the lowest-headroom PASS in the catalog. Rank k=40 captures ~92% of the Frobenius norm of Σ, leaving the lowest-eigenvalue rotated axes under-preconditioned — so visually subpar mixing along the stiff axes is consistent with the truncation, **not** a regression. The integrator ladder below shows the gap explicitly: k=40 internal LRD minESS 2079 vs dense Cholesky oracle minESS 2244. The lever to close it is **richer preconditioning (higher LRD rank k)**, NOT longer trajectory length — ill_cond_50 is the geometry-opposite case that wants SHORT L (avg=2). Quantifying the k=50/60→dense headroom is the open #22 Lever-2 probe.

### Integrator ladder (LRD geometry discovery)
The following ladder was validated during the LRD integration experiment
(see `tests/mclmc_lrd/` for runnable scripts):

| Strategy | Rank k | Max R-hat | Min ESS | Verdict |
|---|---|---|---|---|
| Diagonal MCLMC | — | 1.4461 | 8.0 | FAIL |
| External LRD (oracle) | k=10 | 1.0819 | 48.6 | FAIL |
| External LRD (oracle) | k=20 | 1.0201 | 436.1 | REVIEW |
| External LRD (oracle) | k=40 | 1.0038 | 1977.8 | PASS |
| Adaptive LRD (NUTS pilot) | k=40 | 1.0034 | 1776.9 | PASS |
| **Internal LRD** (production) | **k=40** | **1.0030** | **2079.5** | **PASS** |
| Dense Cholesky (oracle) | full | 1.0027 | 2244.0 | PASS |

Clean rank progression: k≥20 resolves enough of the κ=1000 spectrum to pass.
k=40 captures ~92% of the Frobenius norm of Σ.

### VI rank-collapse (negative result)
`multipathfinder` (16 paths, 1000 samples) collapses to **Rank 6** on ill_cond_50.
The L-BFGS history at MAP convergence cannot capture the global elongated typical
set. A NUTS pilot run is the minimum viable geometry-discovery step. See
`tests/mclmc_lrd/test_multipathfinder_lrd.py` and `catalog/mclmc-routing-taxonomy.md` §5.

## Known-bad combinations

- `mclmc` + `mclmc_tuning` (any n_warmup): **FAIL** (honest null at κ=1000).
  See `recipes/failed__mclmc__mclmc_tuning.json`.
- `adjusted_mclmc` + `adjusted_mclmc_tuning`: **FAIL**.
  See `recipes/failed__adjusted_mclmc__adjusted_mclmc_tuning.json`.
- `adjusted_mclmc_dynamic` + `adjusted_mclmc_tuning`: **FAIL**.
  See `recipes/failed__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json`.
- `hmc` + any IMM (dense, diag, low_rank): **FAIL** (fixed-L resonance trap: L×ε≈2π kills mixing).
  See `recipes/failed__hmc__window_adaptation_dense_imm.json`, `failed__hmc__window_adaptation_diag_imm.json`, `failed__hmc__window_adaptation_low_rank_imm.json`.
- `dynamic_hmc` + `window_adaptation_diag_imm`: **FAIL** (diag IMM misaligned to rotated axes).
  See `recipes/failed__dynamic_hmc__window_adaptation_diag_imm.json`.
- `dmhmc` + `window_adaptation_diag_imm`: **FAIL** (same root cause as dynamic_hmc+diag).
  See `recipes/failed__dmhmc__window_adaptation_diag_imm.json`.
- `laplace_*` + `window_adaptation_low_rank_imm`: **FAIL** (all four laplace variants).
  See `recipes/failed__laplace_hmc__window_adaptation_low_rank_imm.json` etc.

Recorded FAILs not discussed above: failed__dmhmc__window_adaptation_dense_imm.json (old Phase 3c/4 attempt; a later n_warmup=1000 attempt PASSes as low__dmhmc__window_adaptation_dense_imm.json), failed__dynamic_hmc__window_adaptation_dense_imm.json (old attempt; later PASS exists as low__dynamic_hmc__window_adaptation_dense_imm.json).

## Recorded LRD certification inputs (ill_cond_50, pilot-path calibration)

The committed artifacts are:
- `recipes/low__mclmc_lrd__mclmc_lrd_tuning.json` — golden recipe (k=40, best seed=77777)
- `recipes/low__mclmc_lrd__mclmc_lrd_tuning.imm.npz` — rank-40 LRD IMM sidecar (NUTS-pilot path)

The historical direct emitter that ran this sweep is retired. Do not repeat it;
new sampling or certification work must use the
[codegen-first recipe lifecycle](../../../docs/design/codegen-first-recipes.md).
The recorded inputs were `mclmc_lrd_tuning`, seeds 77777/88888/99999,
`n_warmup=1000`, `n_samples=1000`, `k_rank=40`,
`pilot_n_warmup=10000`, and `pilot_n_samples=10000`.

Certified 2026-07-29 (PR #253): 3/3 PASS, seeds 77777/88888/99999, gate minESS 1917/1726/1782,
headline_metric (best seed) = 0.23958, R-hat max ~1.003. k=40, n_warmup=1000, pilot 10k.
Note: headline fell from ~0.247 (pre-fix) to 0.240 after PR #253 corrected the
headline_basis to use the headline (effective_sample_size) ESS rather than the gate
(ess_bulk) ESS. The ratio vs best NUTS (dense IMM, headline 0.137) is now 1.75×, not
the previously implied ~1.80×.

**Seed-sensitivity note (PR #253 investigation):** During re-emission an initial attempt
with wrong parameters (k_rank=8, cert_seeds=11111/22222/33333, no pilot) produced 0/3
PASS with gate minESS 24–45 and rhat 1.09–1.14, an 80× swing vs the successful run.
The dramatic difference was entirely due to wrong parameters (the k_rank=8 got clamped
to 1–2 by the rank-safety check, and the absent pilot left the LRD IMM poorly
initialised). With the correct parameters (k_rank=40, pilot 10k) all 3 seeds PASS.
This is parameter sensitivity, not fundamental seed fragility of the certified result.

**Why pilot and not oracle for the catalog artifact?** The oracle COV path (decompose
`ill_cond_50.COV` directly) is the upper bound (ESS/grad≈0.249). The pilot path is
portable to any model and is the standard library path. The pilot-path golden passes
at headline ESS/grad≈0.214–0.240 (3/3 seeds, best seed 77777).
The oracle 0.249 is a reference ceiling, not the committed artifact.

## History

2026-06-09: MCLMC LRD integration experiment (tuningfork PR #176 / blackjax PR #936).
Full integrator ladder validated; internal LRD certified PASS at 426× ESS/grad.
See `catalog/mclmc-routing-taxonomy.md` for routing taxonomy and scientific context.

### 2026-07-30 — 2 LOW cells promoted to MEDIUM after a recert-sweep gate failure

The 2026-07-30 corpus recert sweep (tuningfork PR #254, ESS-metric switch) found
`low__dynamic_hmc__window_adaptation_dense_imm` and
`low__mhmc__window_adaptation_dense_imm__inner_nuts` failing the gate at their
committed seed under current dependencies (blackjax 1.6.1 / jax 0.11.0) — both
are in the PR's "19 gate failures, no recipe written" set. Per Belief#1176
(seed selection is permitted at MEDIUM tier provided it is disclosed and
independently gate-verified with no relaxation), a 3-seed scan of each cell's
exact committed configuration (same step_policy / target_acceptance /
warmup_inner_kernel, only the seed varies) found:

| cell | seed=11111 | seed=22222 | seed=33333 |
|---|---|---|---|
| `dynamic_hmc` + `window_adaptation_dense_imm` | REVIEW (rhat=1.0200, ess=236.7) | REVIEW (rhat=1.0339, ess=105.5) | **PASS** (rhat=1.0061, ess=736.3) |
| `mhmc` + `window_adaptation_dense_imm` (`inner_nuts`) | FAIL (rhat=1.2890, ess=11.6) | FAIL (rhat=1.3977, ess=11.9) | **PASS** (rhat=1.0020, ess=3379.3) |

1 of 3 clean PASS for each cell. Both are promoted to MEDIUM at seed=33333:
`recipes/medium__dynamic_hmc__window_adaptation_dense_imm__reseed.json` and
`recipes/medium__mhmc__window_adaptation_dense_imm__inner_nuts__reseed.json`
(each recipe's `notes` field carries the same disclosure as this entry). The
LOW recipes stay committed unchanged (they still record the historical PASS at
their original seed and dependency stack, now unreproducible under current
dependencies — same "LOW unstable across seeds" pattern documented for
`lotka_volterra`).

## Dynamic-L Sweep (avg ladder)

Run date: 2026-06-19 | Source: sweep_dynl_variety_results.json, medians over 3 seeds

| avg | realized_avg | ESS | Rhat | 2nd-mom bias | mbias_sd | trend |
|---|---|---|---|---|---|---|
| 2 | 2.0 | 1928 | 1.014 | 0.177 | 0.055 | **OPTIMAL** |
| 6 | 6.0 | 1675 | 1.011 | 0.179 | 0.075 | degrading |
| 18 | 18.0 | 1241 | 1.011 | 0.511 | 0.070 | *monotone worse* |
| 54 | 54.2 | 559 | 1.013 | 0.590 | 0.092 | ↓ |
| 108 | 108.3 | 382 | 1.015 | 1.008 | 0.145 | **worst** |

**Lesson:** avg=2 is optimal. Every step increasing avg monotonically degrades ESS and inflates 2nd-moment bias
(0.183 → 1.016), both bias and mbias_sd rising systematically across seeds. Longer trajectories overshoot the
rotated-but-not-funnel geometry; the avg=2 default is tuned correctly for LRD-MCLMC on this model.
[boundary: avg=2 optimality holds for LRD-MCLMC (k=40) on ill_cond_50; banana geometry-opposite: requires avg≥18 to PASS; do not transfer this rule to curved/funnel models]

See `catalog/mclmc-scaling-laws.md` §3 for generalized principles (why stiff/rotated models want SHORT L, etc.).

## Citations

**Synthetic baseline** — no external reference. Standard test model for sampler validation.
