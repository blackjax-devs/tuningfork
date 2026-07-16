# Sampling lessons: banana

## TL;DR

Curved banana geometry. **NUTS/dynamic_hmc with diag IMM require MEDIUM effort (step_policy
v1-medium) to PASS; LOW-effort defaults FAIL.** `adjusted_mclmc_dynamic` PASSes at MEDIUM
effort (n_warmup=5000, avg≥18). Most LOW-effort cells fail. Laplace family is out_of_scope.
[boundary: adjusted_mclmc_dynamic PASS holds at n_warmup=5000 MEDIUM; adjusted_mclmc (static) FAIL at n_warmup=10000 (avg=2 cap); nearest FAILs: nuts+diag_imm, nuts+dense_imm, nuts+low_rank_imm at LOW effort (see recipes/failed__nuts__*.json)]

## Canonical recipe

`recipes/medium__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json` — MEDIUM effort, PASS.
`recipes/medium__dynamic_hmc__window_adaptation_diag_imm__policy_v1-medium.json` — MEDIUM effort, PASS.

## Sampling quirks

### Curved geometry requires long trajectories
banana's curved posterior requires trajectory length avg≥18 for MCLMC to reach
sufficient mixing. At avg=2 (default), chains cannot traverse the banana shape.

### adjusted_mclmc (static L): FAIL at any standard warmup budget
`adjusted_mclmc` with `adjusted_mclmc_tuning` (n_warmup=10000): rhat=1.059, ESS=61.9,
bias clean (max_z=0.942). The avg=2 cap limits trajectory length and chains cannot mix.
This is warmup-invariant: same rhat at n_warmup=1k and 10k.
[boundary: FAIL confirmed at n_warmup=10k; warmup-invariant (geometry blocker, not warmup budget); see recipes/failed__adjusted_mclmc__adjusted_mclmc_tuning.json]

### adjusted_mclmc_dynamic: PASS at MEDIUM (avg≥18)
With dynamic trajectory (avg=18–54 from Dynamic-L sweep), bias cleans up and ESS rises.
The committed MEDIUM recipe targets avg~18 at n_warmup=5000.
[boundary: PASS at n_warmup=5000, avg≈18; FAIL at avg=2 (same rhat as static); do not use adjusted_mclmc (non-dynamic) on banana]

### NUTS and dynamic_hmc: FAIL at LOW effort
NUTS with diag, dense, or low_rank IMM all FAIL at LOW effort (default n_warmup=1000).
MEDIUM step_policy (v1-medium) rescues dynamic_hmc+diag_imm.
[boundary: LOW-effort NUTS fails; MEDIUM step_policy PASS; dense and low_rank IMM FAIL even at MEDIUM (see failed__nuts__window_adaptation_dense_imm.json, failed__nuts__window_adaptation_low_rank_imm.json)]

## Known-bad combinations

- `nuts` + `window_adaptation_diag_imm` (LOW effort): **FAIL**. See `recipes/failed__nuts__window_adaptation_diag_imm.json`.
- `nuts` + `window_adaptation_dense_imm`: **FAIL**. See `recipes/failed__nuts__window_adaptation_dense_imm.json`.
- `nuts` + `window_adaptation_low_rank_imm`: **FAIL**. See `recipes/failed__nuts__window_adaptation_low_rank_imm.json`.
- `dynamic_hmc` + `window_adaptation_diag_imm` (LOW effort): **FAIL**. See `recipes/failed__dynamic_hmc__window_adaptation_diag_imm.json`.
- `dynamic_hmc` + `window_adaptation_dense_imm`: **FAIL**. See `recipes/failed__dynamic_hmc__window_adaptation_dense_imm.json`.
- `dmhmc` + `window_adaptation_diag_imm` (LOW effort): **FAIL**. See `recipes/failed__dmhmc__window_adaptation_diag_imm.json`.
- `dmhmc` + `window_adaptation_dense_imm`: **FAIL**. See `recipes/failed__dmhmc__window_adaptation_dense_imm.json`.
- `adjusted_mclmc` + `adjusted_mclmc_tuning` (any budget): **FAIL** (avg=2 cap is warmup-invariant blocker). See `recipes/failed__adjusted_mclmc__adjusted_mclmc_tuning.json`.
- `hmc` + `window_adaptation_low_rank_imm`: **FAIL** (hard_direction). See `recipes/failed__hmc__window_adaptation_low_rank_imm.json`.
- Laplace family + `window_adaptation_low_rank_imm`: **FAIL** (out_of_scope). See `recipes/failed__laplace_*__window_adaptation_low_rank_imm.json`.

Recorded FAILs not discussed above: all 13 failed recipes are covered above.

## History

No detailed investigations recorded yet. If sampling pathologies emerge during recipe sweeps, case studies will be documented here.

## Dynamic-L Sweep (avg ladder)

Run date: 2026-06-19 | Source: sweep_dynl_variety_results.json, medians over 3 seeds

| avg | realized_avg | ESS | Rhat | 2nd-mom bias | mbias_sd | acceptance | verdict |
|---|---|---|---|---|---|---|---|
| 2 | 2.0 | 40 | 1.095 | 0.275 | 0.189 | 0.988 | **loud-fail** |
| 6 | 6.0 | 214 | 1.014 | 0.214 | 0.077 | 0.966 | **loud-fail** |
| 18 | 18.0 | 703 | 1.003 | 0.086 | 0.035 | 0.914 | **PASS** |
| 54 | 54.2 | 1023 | 1.002 | 0.107 | 0.063 | 0.817 | **borderline (in-window)** |
| 108 | 108.3 | 1033 | 1.002 | 0.112 | 0.030 | 0.726 | **overshoot onset** |

**Lesson:** Clean PASS window avg=18–54 (Rhat ~1.003, bias <0.1, ESS efficient). At avg=108, acceptance erodes
and ESS stops climbing (diminishing returns / trajectory saturation), not a silent bias. A new medium__ `adjusted_mclmc`
recipe targets avg~18 and offers an MCLMC alternative to NUTS in this efficiency window (see catalog recipes).

See `catalog/mclmc-scaling-laws.md` §3 for generalized principles (geometry-opposite optima, why bigger L is not always
better, etc.).

## Citations

**Synthetic baseline** — no external reference. Standard test model for sampler validation.
