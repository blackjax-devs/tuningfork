# Sampling lessons: banana

## TL;DR

No significant sampling quirks documented — model has well-conditioned geometry or has not yet been extensively probed. Library defaults pass at LOW effort.

## Canonical recipe

Placeholder: once recipes are generated, link to `recipes/low__nuts__window_adaptation_diag_imm.json` or the appropriate LOW-effort baseline.

## Sampling quirks

None documented yet. Early probes show the model samples cleanly at default NUTS + window-adaptation settings.

## Known-bad combinations

None documented yet. R1+ will backfill FAILED recipes for hard-excluded cells in the recipe matrix (if any).

## History

No detailed investigations recorded yet. If sampling pathologies emerge during recipe sweeps execute, case studies will be logged to `worklog/lessons/case-studies/banana/`.

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
