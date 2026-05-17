# Sampling lessons: stoch_vol

## TL;DR

AR(1) unit-root geometry causes divergence clusters at high persistence; requires per-model gate tolerance (0.5%) despite correct R̂ and ESS.

## Canonical recipe

low__nuts__window_adaptation_diag_imm (n_warmup=5000, target_acceptance=0.99; account for 0.5% divergence rate in gate)

## Sampling quirks

503-D NCP recursive AR(1) stochastic volatility with Uniform(-1,1) phi prior. Divergences cluster at extreme phi ≈ 0.9999 (unit-root boundary) where the stationary-distribution initialization `h[0] = mu + (sigma / sqrt(1 - phi²)) * h_std[0]` diverges. Diagonal-IMM condition number ≫1× due to boundary-element variance amplification (interior vs boundary elements experience different AR(1) persistence scaling), but this is downstream symptom, not primary cause. Tightening the prior (Beta(20,1.5)-shifted phi) tripled divergences by pulling the posterior bulk closer to the difficult region.

## Known-bad combinations

None documented yet. FAILED recipes will be backfilled in R5 once the recipe matrix is populated.

## History

The following case studies document the investigation path and distilled lessons:

- [2026-05-11-diagonal-imm-condition-ar1-boundary.md](worklog/lessons/case-studies/stoch_vol/2026-05-11-diagonal-imm-condition-ar1-boundary.md) — IMM condition-number dissection; shows the signal is real but downstream
- [2026-05-12-ar1-unit-root-divergence-cluster.md](worklog/lessons/case-studies/stoch_vol/2026-05-12-ar1-unit-root-divergence-cluster.md) — cluster analysis showing 92%/78% tail concentration at mu/phi extremes; identifies unit-root as primary driver
- [2026-05-12-failed-prior-swap-beta-shifted-phi.md](worklog/lessons/case-studies/stoch_vol/2026-05-12-failed-prior-swap-beta-shifted-phi.md) — hypothesis (tighten prior to suppress phi→1 tail) tested and failed; correct diagnosis ≠ straightforward fix

See `worklog/lessons/case-studies/stoch_vol/README.md` for the index and quick summary.

## Citations

Stan User's Guide § 2.5 (NCP AR(1) form + prior recommendations); Kim, Shephard, Chib 1998 (original KSC model)
