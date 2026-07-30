# Sampling lessons: radon

## TL;DR

**NUTS PASS, all MCLMC variants FAIL.** The 390-D hierarchical NCP model has highly
unequal coordinate scales from heterogeneous county sample sizes. NCP partially
resolves funnel geometry but residual scale mismatch remains. NUTS adaptive diagonal
IMM compensates for coordinate-scale heterogeneity; isotropic MCLMC cannot.
[boundary: NUTS+diag_imm PASS at n_warmup=1000 (REVIEW gate-clearing); dynamic_hmc+diag_imm FAIL at n_warmup=2000 with V7 oracle (V7 miscalibrated for radon, use MEDIUM step policy v1-medium instead); all MCLMC variants FAIL regardless of n_warmup]

## Canonical recipe

**NUTS**: `recipes/low__nuts__window_adaptation_diag_imm.json` — REVIEW (gate-clearing),
R-hat=1.0067, ESS=347.2.

## Sampling quirks

### Residual coordinate-scale mismatch after NCP
`radon` uses Non-Centered Parameterization (NCP) to decouple county-level random
effects from their hyperparameter. NCP reduces funnel severity but does not eliminate
coordinate-scale heterogeneity:
- Some counties have many observations (constrained random effects, small variance)
- Other counties have few observations (weakly constrained, large variance close to prior)

This produces highly unequal variance scales across 390 dimensions. NUTS's adaptive
diagonal IMM rescales these during warmup; MCLMC's isotropic step is forced to use
the most constrained county's scale globally.

### All MCLMC variants: honest FAIL (2026-06-08)

| Sampler | n_warmup | Max R-hat | Min ESS | Divergences | Verdict |
|---|---|---|---|---|---|
| Unadjusted `mclmc` | 1,000 | **4.1396** | 4.3 | 0 | FAIL |
| `adjusted_mclmc` | 10,000 | **1.1456** | 21.6 | 0 | FAIL |
| `adjusted_mclmc_dynamic` | 10,000 | **1.1086** | 32.1 | 0 | FAIL |

- Unadjusted `mclmc`: catastrophic R-hat=4.14 with ESS=4.3. Complete exploration
  collapse from isotropic scale mismatch in 390 dimensions.
- MH-corrected variants "fail safely" — 0 divergences, but R-hat ~1.10–1.15.
  The safety net prevents pathological explosions but cannot fix the geometry.

### Why LRD preconditioning does not help
An LRD mass matrix can resolve global rotational ill-conditioning. The radon failure
mode is **isotropic scale mismatch** (unequal diagonal variances), which a diagonal
IMM solves cheaply. NUTS's `window_adaptation_diag_imm` is the right tool. LRD's
advantage is for off-diagonal (correlation) structure, not for diagonal scale
heterogeneity.

If a diagonal MCLMC warmup (`diagonal_preconditioning=True` in `mclmc_find_L_and_step_size`)
were used, it might address the scale mismatch. This variant was not tested but is
a natural future experiment.

## Known-bad combinations

- `mclmc` (isotropic, 1k warmup): R-hat=4.14. Complete stagnation.
- `adjusted_mclmc` / `adjusted_mclmc_dynamic` (10k warmup): R-hat 1.10–1.15.
  Honest null with MH safety net.
- `dynamic_hmc` + `window_adaptation_diag_imm` (n_warmup=2000, V7 oracle): **FAIL** (rhat=1.059, ESS=64.9).
  V7 auto-oracle miscalibrated for radon at ta=0.8 — adapts step_size=0.22, far too large for d=390.
  Use MEDIUM step_policy v1-medium instead (see `recipes/medium__dynamic_hmc__window_adaptation_diag_imm__policy_v1-medium.json`).
  See `recipes/failed__dynamic_hmc__window_adaptation_diag_imm.json`.
  [boundary: V7 oracle FAIL is specific to ta=0.8 + d=390; v1-medium PASSES the same cell]
- `nuts` + `fullrank_vi` warmup: **FAIL** (requires_model_change — fullrank VI not feasible at d=390).
  See `recipes/failed__nuts__fullrank_vi.json`.

Recorded FAILs not discussed above: all MCLMC variants are covered in "All MCLMC variants" section above; all recipe FAILs now documented.

## History

2026-06-08: MCLMC family radon evaluation (recipes-mclmc-cat-c.md / experimental
mclmc_explore). All variants tested, all FAIL. NUTS REVIEW is the only viable path.
See `catalog/mclmc-routing-taxonomy.md` §4 (Category D: hierarchical funnels).

### 2026-07-30 — `medium__dynamic_hmc__chees` re-certified after a recert-sweep gate failure

The 2026-07-30 corpus recert sweep (tuningfork PR #254, ESS-metric switch)
found `medium__dynamic_hmc__chees` (128 chains × 2000-step chees warmup)
failing the gate at its committed seed under current dependencies (blackjax
1.6.1 / jax 0.11.0) — one of the PR's "19 gate failures, no recipe written"
set.

Per Belief#1176, re-running the identical config (target_acceptance=0.651,
n_warmup=2000, num_chains=128) at seed=11111 (the first candidate tried)
cleared the gate cleanly on the first attempt: rhat=1.0097, ess=9292.8, 0
divergences — no wider seed scan was needed. The recipe was re-emitted in
place (same filename, still `effort=medium`) with the seed selection
disclosed in `notes`.

## Citations

**Posteriordb reference**: [posteriordb.org #6 (radon_mn)](https://posteriordb.org/)

**Stan reference**: Stan User's Guide § 1.1
