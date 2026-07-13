# Multichain GT Migration — Coherence Screen

**Date:** 2026-07-13
**Branch:** feat/multichain-gt-migration
**Analyst:** swe-gt-migration
**Status:** COMPLETE (re-emissions pending)

---

## Purpose

Verify that swapping the ground-truth from single-chain (40k draws, legacy SE) to
10-chain multichain (100k draws, between-chain SE) does not silently flip committed
PASS recipes to FAIL under the tighter gate.

---

## Method

### Step 1 — Coherence test

For each model with new GT (summary_v2.json vs reference/summary.json):

```
C_d = |μ_old_d - μ_new_d| / sqrt(se_old_d² + se_new_d²)
```

where `se_old = gt_std_old / sqrt(n_old)` (legacy formula) and
`se_new = max(between_chain_se, gt_std_new / sqrt(min(bulk_ess, n_total)))`.

Under the null (both GTs estimating the same truth):
`C_d ~ |N(0, 1)|` per dimension.

Expected maximum over D dims: `null_E = sqrt(2·ln(2D))`.
95th-percentile noise band: `p95 = null_E + 0.6`.
Coherent if `ratio = C_max / null_E ≤ 1.2`.

### Step 2 — Ratio-dominant re-screen

For coherent models: `worst_z = z_committed × max_d(R_d) + slack`
where `R_d = se_old_d / se_new_d` (SE tightening ratio per dim).
Slack ∈ {0.0, 0.5, 1.0} — noise contribution from the coherent-noise floor.
Non-coherent models: same formula but max_delta_disagree added for real-disagreement dims.

FLAG_FAIL if `worst_z ≥ 4.0`. FLAG_REVIEW if `worst_z ∈ [pass_hi, 4.0)`.

### Step 3 — Cross-reference exact test

The 26-recipe exact test ran old GT vs new GT on identical cached draws.
Any recipe adjudicated by exact test supersedes the bound.

---

## Step 1 Results — Coherence Table

| Model | D | null_E | C_max | ratio | coherent | max_R | argmax_C dim |
|-------|---|--------|-------|-------|----------|-------|-------------|
| banana | 2 | 1.665 | 0.974 | 0.585 | OK | 1.406 | x2[0] |
| eight_schools_ncp | 10 | 2.448 | 1.858 | 0.759 | OK | 1.589 | theta_raw[4] |
| german_credit | 26 | 2.811 | 1.501 | 0.534 | OK | 1.604 | beta[12] |
| gmm_25 | 2 | 1.665 | 1.228 | 0.738 | OK | 1.583 | x[0] |
| horseshoe | 204 | 3.467 | 3.010 | 0.868 | OK | 1.536 | lambda_[87] |
| ill_cond_50 | 50 | 3.035 | 2.552 | 0.841 | OK | 1.601 | x[46] |
| irt_2pl | 144 | 3.365 | 2.842 | 0.844 | OK | 1.606 | theta_raw[78] |
| logistic_synthetic | 3 | 1.893 | 1.962 | 1.037 | OK | 0.961 | beta[2] |
| mvn_10 | 10 | 2.448 | 3.192 | 1.304 | !! | 1.585 | x[3] |
| neals_funnel | 10 | 2.448 | 2.073 | 0.847 | OK | 3.810 | theta dims |
| radon | 390 | 3.649 | 2.759 | 0.756 | OK | 1.612 | alpha_raw[93] |
| gp_regression | N/A | — | — | — | legacy (no GT change) | — | — |
| lotka_volterra | N/A | — | — | — | legacy (no GT change) | — | — |
| stoch_vol | N/A | — | — | — | legacy (no GT change) | — | — |

### mvn_10 — Coherence ruling: FALSE ALARM (TL-ratified 2026-07-13)

mvn_10's ratio=1.304 marginally exceeds the 1.2 threshold due to a single dim
(x[3], C_d=3.192 vs p95=3.048 — only 5% over the noise band). With D=10, the
false-alarm probability of at least one dim exceeding p95 by chance is ~40%
(1-(0.95)^10). Δμ/σ_posterior = 0.019 (1.9% shift) — physically negligible.

**Structural clincher:** Both old GT (`reference/metadata.json`: `"generator": "analytic"`,
40k draws) and new GT (`summary_v2.json`: `"generator": "analytic_iid"`, 10×10k
draws) are i.i.d. exact samples from the analytic MVN posterior. The mean deltas are
DEFINITIONALLY pure Monte Carlo noise — no real posterior disagreement is possible
since both GTs are unbiased estimates of the same exact analytic truth with different
random seeds.

**Ruling:** mvn_10 = COHERENT. All mvn_10 recipes treated under the ratio-dominant
bound (worst_z = z_committed × max_R + slack) not the non-coherent form.

### neals_funnel — max_R=3.81 diagnosis

The outlier max_R is driven entirely by the posterior std changing between old and
new GT, NOT by mean disagreement:

```
theta[5]: old_std=23.72, new_std=9.83
R = (23.72 / 9.83) × sqrt(100000 / 40000) = 2.41 × 1.581 = 3.81
```

The old single-chain run (40k draws) overestimated sigma for theta[5] by 2.4×
— characteristic of Neal's funnel heavy-tail sampling where a single chain may
drift into the high-variance region and inflate std estimates. C_d at theta[5] =
1.038 vs null_E=2.448 → C/null=0.424 (noise-level). The old GT SE for these
dims was understated by 3.8× vs the honest multichain estimate. Under the new GT,
the gate correctly tightens for funnel dims. Coherent PASS.

---

## Step 2 Results — Refined Screen

FLAG counts over 138 committed PASS recipes:

| slack | FLAG_FAIL total | FLAG_FAIL non-mvn_10 | FLAG_REVIEW |
|-------|-----------------|----------------------|-------------|
| 0.0   | 29              | 7                    | 39          |
| 0.5   | 33              | 11                   | 51          |
| 1.0   | 45              | 23                   | 54          |

Note: all 22 mvn_10 PASS recipes flag at every slack level under the non-coherent
bound (max_delta_disagree ≈ 5.98 dominates). Under the COHERENT ruling above,
mvn_10 SAFE count at slack=0: 20 of 22 (2 still FLAG_FAIL as coherent: the two
medium HMC recipes with z_committed=3.4/3.2 × max_R=1.585).

### Non-mvn_10 FLAG_FAIL at slack=0.0 (7 recipes, all coherent, worst_z = z×max_R):

| Recipe | z_committed | max_R | worst_z |
|--------|-------------|-------|---------|
| banana/medium__adjusted_mclmc_dynamic__adjusted_mclmc_tuning | 3.736 | 1.406 | 5.252 |
| german_credit/medium__hmc__window_adaptation_diag_imm | 3.268 | 1.604 | 5.242 |
| radon/medium__dynamic_hmc__chees | 3.009 | 1.612 | 4.851 |
| eight_schools_ncp/low__hmc__window_adaptation_low_rank_imm | 2.661 | 1.589 | 4.228 |
| eight_schools_ncp/low__dmhmc__window_adaptation_dense_imm | 2.565 | 1.589 | 4.075 |
| german_credit/low__dynamic_hmc__chees | 2.508 | 1.604 | 4.023 |
| ill_cond_50/medium__dmhmc__window_adaptation_diag_imm__policy_v7-empirical-oracle | 2.502 | 1.601 | 4.006 |

Plus 2 mvn_10 medium HMC recipes: worst_z = 5.43 / 5.12 under coherent bound.

**Total FLAG_FAIL at slack=0 after mvn_10 coherence ruling: 9 recipes.**

---

## Step 3 Results — Exact-test Overlap

At all slack levels: **0 of the 9 FLAG_FAIL recipes were in the 26-recipe exact test.**

The one PASS→REVIEW case from the exact test
(`eight_schools_ncp/low__laplace_dmhmc__window_adaptation_dense_imm`, z: 2.205→2.366)
is NOT in any FLAG_FAIL list (worst_z ≈ 3.5 at slack=0, correctly FLAG_REVIEW).
The exact result (REVIEW, not FAIL) is consistent with the bound being conservative.

---

## Disposition

Per TL directive 2026-07-13:
- **9 FLAG_FAIL recipes → targeted re-emission** on this box (CPU) and colossus (GPU, radon).
  Re-emissions use the same (model, warmup, sampler) config + committed n_warmup/n_samples.
  Results recorded in `experiments/gt_migration_reemit_results.json`.
- **LFS upload** (13 draws.npz) waits on: stoch_vol reseed result + TL go.
- GT itself is validated (coherence 11/11 + posteriordb cross-check + max |Δμ/σ| ≤ 0.019).
  The 9 flags are recipe-verdict questions, not GT-quality questions.

Re-emission results: see `experiments/gt_migration_reemit_results.json`.
