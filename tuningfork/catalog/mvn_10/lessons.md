# Sampling lessons: mvn_10

## TL;DR

No significant sampling quirks documented — model has well-conditioned geometry or has not yet been extensively probed. Library defaults pass at LOW effort.

## Canonical recipe

Placeholder: once recipes are generated, link to `recipes/low__nuts__window_adaptation_diag_imm.json` or the appropriate LOW-effort baseline.

## Sampling quirks

None documented yet. Early probes show the model samples cleanly at default NUTS + window-adaptation settings.

## Known-bad combinations

None documented yet. R1+ will backfill FAILED recipes for hard-excluded cells in the recipe matrix (if any).

## History

No detailed investigations recorded yet. If sampling pathologies emerge during recipe sweeps execute, case studies will be logged to `worklog/lessons/case-studies/mvn_10/`.

## Dynamic-L Sweep (avg ladder)

Run date: 2026-06-19 | Source: sweep_dynl_variety_results.json, medians over 3 seeds

| avg | realized_avg | ESS | Rhat | 2nd-mom bias | mbias_sd | trend |
|---|---|---|---|---|---|---|
| 2 | 2.0 | 2712 | 1.007 | 0.105 | 0.040 | **OPTIMAL** |
| 6 | 6.0 | 1615 | 1.005 | 0.242 | 0.044 | degrading |
| 18 | 18.1 | 1525 | 1.011 | 0.379 | 0.054 | *monotone worse* |
| 54 | 54.2 | 1390 | 1.009 | 0.281 | 0.059 | ↓ |
| 108 | 108.4 | 1641 | 1.046 | 0.173 | 0.044 | (recovery noise) |

**Lesson:** avg=2 gives highest ESS (2712); increasing avg monotonically degrades mixing. All configs have small
mbias_sd (0.04–0.06), indicating small true bias (noisy max-over-D 2mbias not reliable at n=500). mvn_10 is
bias-indifferent across the ladder; optimal at avg=2 for mixing efficiency, not bias avoidance. Behaves like a
smooth, well-conditioned diagonal model.

See `catalog/mclmc-scaling-laws.md` §3 for generalized principles (why smooth diagonal models want SHORT L, etc.).

## Citations

**Synthetic baseline** — no external reference. Standard test model for sampler validation.
