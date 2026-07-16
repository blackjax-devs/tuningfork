# BLOCKED — W1 Full-Catalog Re-Validation

**Date:** 2026-07-16
**Branch:** `chore/w1-full-catalog-revalidation`
**Status:** STOP — awaiting TL/statistician diagnosis

---

## What was found

The 3-pass full-catalog W1 sweep (B=5000, PCG64 seed=42, 11/89 tail-ESS gate) completed:

- **127 OK / 6 SKIP / 0 ERROR**
- **1 genuine FLIP** (path A = cached draws — not a resampling artifact):

| Cell | max_w1σ | floor | % over |
|------|---------|-------|--------|
| `german_credit/medium__hmc__window_adaptation_diag_imm` | 0.0864 | 0.0708 | +22% |

Step 4 (irt_2pl×chees, nc=16, n_warmup=500, n_samples=1000): **W1=PASS** (max_w1σ=0.0283 vs floor=0.0563).

---

## What was tried (3 passes)

**Pass 1 (B=500):** Job killed at cell 126/133. Resume logic preserved checkpoint.

**Pass 2 (B=500):** Completed all 133 sweep cells. Uncovered and fixed harness classification bugs:
1. CHEES/MEADS skip_warmup=True fails (non-serialisable callables) → path C or SK
2. Sidecar IMM new-schema mismatch → check both old+new schema; path C
3. Laplace without cache → fresh run diverges from prior → SK (false-flip prevention)
4. VI-warmup without cache → seed-sensitive ADVI → SK (false-flip prevention)

After fixes: borderline irt_2pl/mhmc flip (3% at B=500) identified as candidate noise.

**Pass 3 (B=5000):** All cells re-run at canonical gate B. Result: irt_2pl/mhmc resolved PASS (B=500 noise confirmed). german_credit/hmc confirmed FAIL (22% above floor at B=5000).

---

## Root cause hypothesis for the flip

`german_credit/medium__hmc__window_adaptation_diag_imm`:
- Path A: uses committed cached draws (existing artifact, not a fresh re-run)
- The flip is therefore about the quality of the HMC draws themselves, not about resampling variance or harness bugs
- D=26 logistic regression posterior; diagonal mass matrix may not capture the posterior geometry well
- Candidate causes:
  1. Step size too large at "medium" effort → systematic HMC discretization bias in at least 1 of 26 dims
  2. Diagonal IMM inadequate for german_credit's posterior (correlated dimensions)
  3. The specific cached draw set happened to be a poor seed (unlikely at B=5000 — this is the gate confirming real bias)

---

## What to try next (TL decision required)

Per task brief: "STOP (any flip → report to TL, no remediation)." Awaiting TL diagnosis:

1. **Inspect the flip dims**: which of the 26 german_credit dimensions drives max_w1σ=0.0864? If it's a tail dimension or a correlated pair, that confirms the IMM hypothesis.
2. **Re-run at medium×dense_imm**: does `german_credit/medium__hmc__window_adaptation_dense_imm` (if it exists) pass? That would isolate the diagonal IMM issue.
3. **Check n_steps**: HMC at medium effort — what is the stored step_size / n_steps? If trajectory length is short → poor mixing.
4. **A/B #18 routing**: TL had already flagged a stat A/B diagnosis on the german_credit flip; this 3-pass B=5000 result is the definitive input (real discrepancy, not B=500 noise).

---

## Files

- `experiments/w1_full_catalog_revalidation.py` — driver (force-added)
- `experiments/w1_full_catalog_revalidation_results.json` — results artifact (force-added)
- Phase-1 thread file — appended revalidation record

Branch pushed: `origin/chore/w1-full-catalog-revalidation` (8 commits).
