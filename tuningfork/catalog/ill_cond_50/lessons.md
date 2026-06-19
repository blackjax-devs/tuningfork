# Sampling lessons: ill_cond_50

## TL;DR

**Rotational ill-conditioning (κ=1000) is the defining geometry.** Diagonal mass
matrices — whether NUTS or MCLMC — require careful treatment. NUTS with
`window_adaptation_diag_imm` passes at LOW effort (the diagonal IMM effectively
rescales the rotated axes during adaptation). Standard diagonal `mclmc` is an
**honest FAIL** at any warmup budget; the diagonal mass matrix cannot capture the
rotated correlation axes. The only viable MCLMC path is **LRD preconditioning**
(k=40, NUTS-pilot extraction).

## Canonical recipe

**NUTS** (default): `recipes/low__nuts__window_adaptation_diag_imm.json` — PASS,
headline ESS/grad ≈ 0.0065.

**MCLMC with LRD** (certified): `recipes/low__mclmc_lrd__mclmc_lrd_tuning.json` — PASS,
headline ESS/grad ≈ 0.249 (426× over diagonal MCLMC baseline).

## Sampling quirks

### Rotational ill-conditioning (κ=1000)
The covariance matrix is Σ = U Λ Uᵀ with eigenvalues logarithmically spaced from
1 to 1000 and U a fixed random orthogonal matrix. The principal axes are **rotated
relative to the coordinate axes**, so any diagonal mass matrix is misaligned.

- NUTS `window_adaptation_diag_imm`: the adaptive diagonal IMM finds a diagonal
  approximation to the rotated geometry during warmup, which is sufficient for PASS.
- Diagonal MCLMC (`mclmc_tuning`): fails catastrophically at all warmup budgets.
  The trajectory length L cannot compensate for the rotational mismatch.
  Plateau confirmed at `n_warmup=100k`: R-hat oscillates 1.05–1.07, ESS≈135.
  See `recipes/failed__mclmc__mclmc_tuning.json` for the full attempted-configurations
  ladder.

### LRD MCLMC pipeline (ill_cond_50, k=40)
**NUTS pilot → SVD extraction → `make_lrd_kernel` → `mclmc_find_L_and_step_size`**

1. Run 1000-step diagonal NUTS pilot (`run_pilot_nuts`)
2. Compute empirical σ and top-40 eigenvectors/eigenvalues via SVD
   (`extract_lrd_from_samples`)
3. Construct `LowRankInverseMassMatrix(sigma, U, lam)`
4. Bind with `make_lrd_kernel` and run `mclmc_find_L_and_step_size(diagonal_preconditioning=False)`

Result: R-hat=1.0039, ESS=1993.3, ESS/grad=0.2492, PASS (statistician independent
run, seed=98765). Multi-seed hardening at seeds 11111/22222/33333 all PASS
(ESS 1944–2030). 426× ESS/grad improvement over the diagonal MCLMC baseline.

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

## Recipe regen (ill_cond_50 LRD, pilot-path calibration)

The committed artifacts are:
- `recipes/low__mclmc_lrd__mclmc_lrd_tuning.json` — golden recipe (step_size≈7.883, L≈5.628, k=40, best seed=99999)
- `recipes/low__mclmc_lrd__mclmc_lrd_tuning.imm.npz` — rank-40 LRD IMM sidecar (NUTS-pilot path)

**Standard regen command** (re-runs NUTS pilot + 3-seed cert sweep, deterministic):

```bash
uv run python -m tuningfork.recipes._generate_starter \
    --warmup mclmc_lrd_tuning --only ill_cond_50 \
    --calibrate --cert-seeds 77777 88888 99999
```

Certified 2026-06-10: 3/3 PASS, seeds 77777/88888/99999, minESS 1607/1604/1787 (az.ess bulk basis, Geyer comparison: 1587/1599/1779),
R-hat ≤ 1.0031 (max 1.0030, 1.0026, 1.0031). Gate uses az.ess(method="bulk") ≥ 400 (auto_gate basis). k=40, n_warmup=1000.

**Why pilot and not oracle for the catalog artifact?** The oracle COV path (decompose
`ill_cond_50.COV` directly) is the upper bound (ESS/grad≈0.249). The pilot path is
portable to any model and is the standard library path. Both are documented in
`attempted_configurations`. The pilot-path golden passed at ESS/grad≈0.198–0.222 (3/3 seeds, best seed 99999).
The oracle 0.2492 is a reference ceiling in the thread file, not the committed artifact.

## History

2026-06-09: MCLMC LRD integration experiment (tuningfork PR #176 / blackjax PR #936).
Full integrator ladder validated; internal LRD certified PASS at 426× ESS/grad.
See `catalog/mclmc-routing-taxonomy.md` for routing taxonomy and scientific context.

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

See `catalog/mclmc-scaling-laws.md` §3 for generalized principles (why stiff/rotated models want SHORT L, etc.).

## Citations

**Synthetic baseline** — no external reference. Standard test model for sampler validation.
