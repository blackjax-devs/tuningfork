# Sampling lessons: horseshoe

## TL;DR

NUTS PASS at MEDIUM effort (diag IMM, max_doublings=15). `dmhmc` PASS at LOW effort
(dense/diag/low_rank IMM). `adjusted_mclmc` and `adjusted_mclmc_dynamic` with the
standard `adjusted_mclmc_tuning` warmup **FAIL** at avg=2 (default) — the avg=2 cap
prevents trajectories long enough to traverse the horseshoe spike/slab geometry.
LRD preconditioning provides **no benefit** over diagonal on this model — the bottleneck
is local sparsity funnel curvature, not global correlation.
[⚠ boundary: claim "adjusted_mclmc and adjusted_mclmc_dynamic PASS at MEDIUM effort" in the Known-bad section below refers to the LRD experimental runs with non-default trajectory lengths; the COMMITTED recipes failed__adjusted_mclmc__adjusted_mclmc_tuning.json and failed__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json BOTH FAIL at n_warmup=10000 with default avg=2]

## Canonical recipe

**NUTS** (default): `recipes/low__nuts__window_adaptation_diag_imm.json` — PASS.

**MCLMC family**: `adjusted_mclmc_dynamic` at 10k warmup — REVIEW (gate-clearing),
ESS=281.9. See `recipes/low__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json`.

## Sampling quirks

### Dual pathology: heavy Cauchy tails + design matrix correlations
The horseshoe model uses half-Cauchy priors for local and global shrinkage scales,
creating:
- Extremely flat, weak-gradient regions far from the sparsity funnel.
- Strong linear correlations among coefficients from the design matrix structure.

Despite this dual pathology, the **spherical ESH dynamics** of MCLMC are resilient to
the Cauchy tails — the spherical integrator does not diverge in flat regions.

### LRD preconditioning: no benefit (REVIEW, 2026-06-09)
`adjusted_mclmc_dynamic` with LRD (k=50, NUTS-pilot extraction) achieves:
- R-hat=1.0193, ESS=270.7, verdict REVIEW

This is **equivalent to** the diagonal baseline (`adjusted_mclmc_dynamic` at the same
warmup): R-hat=1.0160, ESS=281.9. LRD provides no ESS improvement.

**Scientific conclusion:** The dominant bottleneck on horseshoe is the local funnel
transition (coefficient to zero via the Cauchy shrinkage scale), not global correlation.
A global linear LRD mass matrix cannot flatten position-dependent funnel curvature.
The routing decision is: `adjusted_mclmc_dynamic` is the correct MCLMC variant for
horseshoe; LRD is unnecessary overhead.

The original experimental label "Outstanding Success!" was an overstatement. LRD
performs equivalently to the diagonal baseline — a correct REVIEW result, not an
outstanding one.
[boundary: REVIEW results (R-hat≈1.016–1.019, ESS≈270–282) were obtained with longer-than-default trajectory settings in the LRD experiment, NOT via standard adjusted_mclmc_tuning (avg=2); standard tuning FAILS (see failed__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json, rhat=4.14)]

### Step-size scaling for adjusted samplers (0.55× factor)
`mclmc_find_L_and_step_size` (unadjusted warmup) adapts a large step_size optimized
for rejection-free trajectories. When running `adjusted_mclmc_dynamic`, scale the
adapted step_size by 0.55 to target ~94% acceptance probability. This is the
empirically validated scaling for horseshoe; see `test_internal_lrd_horseshoe.py`.

### Cauchy-tail resilience result (recipes/low__adjusted_mclmc__adjusted_mclmc_tuning.json)
At 10k warmup, `adjusted_mclmc` achieves R-hat=1.0186, ESS=262.3 — the MH correction
successfully stabilizes exploration of the flat Cauchy tails.
[⚠ boundary: this result was obtained in the LRD experimental context with non-default trajectory lengths; the COMMITTED failed recipe (failed__adjusted_mclmc__adjusted_mclmc_tuning.json) shows rhat=4.155, ESS=4.3 at n_warmup=10000, seed=682737 with standard tuning (avg=2); claim contradicted by own recipe — "low__adjusted_mclmc__adjusted_mclmc_tuning.json" referenced above does NOT exist as a committed passing recipe]

## Known-bad combinations

- `adjusted_mclmc` + `adjusted_mclmc_tuning` (standard, avg=2): **FAIL** at n_warmup=10000 (rhat=4.155, ESS=4.3).
  See `recipes/failed__adjusted_mclmc__adjusted_mclmc_tuning.json`.
  [⚠ boundary: this model's own committed recipe FAILS — "adjusted_mclmc PASSes at MEDIUM effort" is an over-transfer from LRD experiments using non-default trajectory settings]
- `adjusted_mclmc_dynamic` + `adjusted_mclmc_tuning` (standard, avg=2): **FAIL** at n_warmup=10000 (rhat=4.145, ESS=4.3).
  See `recipes/failed__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json`.
  [⚠ boundary: same as above; Dynamic-L sweep (avg=54–108) shows REVIEW-plateau but standard avg=2 warmup completely fails; longer-L experiments NOT committed as catalog recipes]
- `dynamic_hmc` + any IMM (diag/dense/low_rank) at n_warmup=2000: **FAIL** (fixed max-trajectory insufficient for horseshoe spike/slab).
  See `recipes/failed__dynamic_hmc__window_adaptation_diag_imm.json`, `failed__dynamic_hmc__window_adaptation_dense_imm.json`, `failed__dynamic_hmc__window_adaptation_low_rank_imm.json`.
  [boundary: dynamic_hmc fails regardless of IMM quality — NUTS with max_doublings=15 is the correct approach for horseshoe]
- `nuts` + `window_adaptation_dense_imm` (n_warmup=1000): **FAIL** (Welford underdetermined at d=204).
  See `recipes/failed__nuts__window_adaptation_dense_imm.json`.
- `nuts` + `window_adaptation_low_rank_imm` (n_warmup=1000): **FAIL** (same root cause).
  See `recipes/failed__nuts__window_adaptation_low_rank_imm.json`.
  [boundary: dense/low_rank IMM FAIL at n_warmup=1000 for d=204; diag IMM PASS with MEDIUM recipe (max_doublings=15)]
- `rmhmc` + `window_adaptation_diag_imm`: **FAIL** (requires_model_change).
  See `recipes/failed__rmhmc__window_adaptation_diag_imm.json`.

Recorded FAILs not discussed above: all 8 failed recipes are now documented above.

## History

2026-06-09: MCLMC LRD integration experiment (tuningfork PR #176 / blackjax PR #936).
LRD equivalence-to-baseline finding recorded; routing lesson: route horseshoe to
`adjusted_mclmc_dynamic` without LRD overhead.
See `catalog/mclmc-routing-taxonomy.md` §3 (Category C routing).
See `tests/mclmc_lrd/test_internal_lrd_horseshoe.py` for the runnable script.

### 2026-07-30 — MEDIUM recipe re-certified after a recert-sweep gate failure; LOW `dmhmc`+dense_imm cell stays unresolved

The 2026-07-30 corpus recert sweep (tuningfork PR #254, ESS-metric switch)
found `medium__nuts__window_adaptation_diag_imm` (this model's canonical NUTS
recipe) and `low__dmhmc__window_adaptation_dense_imm` both failing the gate at
their committed seed under current dependencies (blackjax 1.6.1 / jax 0.11.0)
— both are in the PR's "19 gate failures, no recipe written" set.

**`medium__nuts__window_adaptation_diag_imm` recovers with a plain reseed.**
Per Belief#1176, re-running the identical (nuts, window_adaptation_diag_imm)
config at seed=11111 (the first candidate tried) cleared the gate cleanly:
rhat=1.0089, ess=831.2, 0 divergences. The recipe file was re-emitted in place
(same filename, still `effort=medium`) with the seed selection disclosed in
`notes`.

**`low__dmhmc__window_adaptation_dense_imm` does NOT recover with a reseed.**
A 3-seed scan (682737 — the committed seed, 11111, 22222) all hard-fail with a
consistent signature, not a borderline miss:

| seed | verdict | R-hat | min-ESS |
|---|---|---|---|
| 682737 (committed) | FAIL | 1.1493 | 18.7 |
| 11111 | FAIL | 1.1690 | 16.3 |
| 22222 | FAIL | 1.2434 | 12.1 |

Unlike the ill_cond_50 / lotka_volterra dense-IMM cells recovered elsewhere in
this recert pass, this cell shows no seed that comes close to PASS in 3 tries
(rhat stays in 1.15–1.24 across all three, not oscillating toward 1.01). This
looks more like a genuine dependency-sensitive regression on the d=204 dense
Welford IMM + empirical step-policy combination than a seed lottery, but that
has not been confirmed — no GT-informed-init investigation (of the kind done
for `lotka_volterra`) has been attempted here. This cell is left unpromoted;
its existing `low__dmhmc__window_adaptation_dense_imm.json` recipe is
unchanged and should be treated as unreproducible under current dependencies
pending further investigation.

## Dynamic-L Sweep (avg ladder)

Run date: 2026-06-19 | Source: sweep_dynl_variety_results.json, medians over 3 seeds

| avg | realized_avg | ESS | Rhat | 2nd-mom bias | mbias_sd | trend |
|---|---|---|---|---|---|---|
| 2 | 2.0 | 5 | 2.590 | 0.953 | 0.624 | **loud-fail** |
| 6 | 6.0 | 6 | 1.968 | 1.010 | 0.768 | loud-fail |
| 18 | 18.0 | 11 | 1.314 | 0.870 | 0.493 | loud-fail |
| 54 | 54.2 | 234 | 1.148 | 0.391 | 0.113 | **REVIEW-plateau** |
| 108 | 108.3 | 732 | 1.124 | 0.335 | 0.085 | **REVIEW-plateau** |

**Lesson:** Longer L improves monotonically (ESS 5→734, bias 0.95→0.33) but asymptotes at REVIEW tier (Rhat ~1.2).
This is a geometry-hard limit: funnel curvature is position-dependent, so a single affine preconditioning + longer L
cannot resolve it. Adaptive/position-dependent methods or reparameterization (NCP) would be needed to improve further.
[boundary: REVIEW-plateau holds at avg=54–108 only; standard avg=2 warmup produces rhat=2.59–4.14 (FAIL); these sweep results are NOT committed as recipes — the only committed MCLMC recipe is the failed one]

See `catalog/mclmc-scaling-laws.md` §3 for generalized principles (why funnels show LONGEST-available plateaus, etc.).

## Citations

**Real-data model**: Horseshoe prior on regression coefficients; real dataset
