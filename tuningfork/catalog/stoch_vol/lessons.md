# Sampling lessons: stoch_vol

## TL;DR

AR(1) unit-root geometry causes divergence clusters at high persistence; requires per-model gate tolerance (0.5%) despite correct R̂ and ESS.

## Canonical recipe

low__nuts__window_adaptation_diag_imm (n_warmup=5000, target_acceptance=0.99; account for 0.5% divergence rate in gate)

## Sampling quirks

503-D NCP recursive AR(1) stochastic volatility with Uniform(-1,1) phi prior. Divergences cluster at extreme phi ≈ 0.9999 (unit-root boundary) where the stationary-distribution initialization `h[0] = mu + (sigma / sqrt(1 - phi²)) * h_std[0]` diverges. Diagonal-IMM condition number ≫1× due to boundary-element variance amplification (interior vs boundary elements experience different AR(1) persistence scaling), but this is downstream symptom, not primary cause. Tightening the prior (Beta(20,1.5)-shifted phi) tripled divergences by pulling the posterior bulk closer to the difficult region.

**⚠️ Multi-mode warmup capture (re-framed 2026-05-18)**: the 2026-05-17 "seed=20260517 fails" finding was real but mis-framed as PRNG fragility. A 2026-05-18 multi-seed sweep at the recert config (depth=10, ta=0.99) shows **44 % gate failure rate across 7 seeds** (30 % catastrophic + 14 % borderline). The catastrophic mode: warmup adaptation gets captured by the AR(1) unit-root attractor within the first ~50 steps; step-size collapses to ~10⁻⁶ (matching the local geometry at phi ≈ 0.9999); the post-warmup chain is stuck. The chain inits in the bulk normally — the capture happens during warmup, regardless of init. **Pathfinder→NUTS init RESCUES the catastrophic seed=20260517 trace** (4000-draw quick test passes the gate with n_div=0.42 %, R̂≈1.003 across mu/phi/sigma). **Recommended fix (pending re-cert)**: switch this model's warmup from `window_adaptation_diag_imm` to `pathfinder` (or `multipathfinder`). Full diagnostics + reproduction at [`worklog/lessons/case-studies/stoch_vol/2026-05-18-multimodal-warmup-capture-pathfinder-rescue.md`](worklog/lessons/case-studies/stoch_vol/2026-05-18-multimodal-warmup-capture-pathfinder-rescue.md); the original "PRNG fragility" framing is preserved at [`2026-05-17-prng-fragility-recert.md`](worklog/lessons/case-studies/stoch_vol/2026-05-17-prng-fragility-recert.md) for context. Currently the shipped `groundtruth_samples/blackjax/draws.npz` for stoch_vol is the Phase 0 trace at `tuning_seed=42` (which passed the cert under window_adaptation_diag_imm by luck of the 3/7 PASS distribution).

## Known-bad combinations

None documented yet. FAILED recipes will be backfilled in R5 once the recipe matrix is populated.

## History

The following case studies document the investigation path and distilled lessons:

- [2026-05-11-diagonal-imm-condition-ar1-boundary.md](worklog/lessons/case-studies/stoch_vol/2026-05-11-diagonal-imm-condition-ar1-boundary.md) — IMM condition-number dissection; shows the signal is real but downstream
- [2026-05-12-ar1-unit-root-divergence-cluster.md](worklog/lessons/case-studies/stoch_vol/2026-05-12-ar1-unit-root-divergence-cluster.md) — cluster analysis showing 92%/78% tail concentration at mu/phi extremes; identifies unit-root as primary driver
- [2026-05-12-failed-prior-swap-beta-shifted-phi.md](worklog/lessons/case-studies/stoch_vol/2026-05-12-failed-prior-swap-beta-shifted-phi.md) — hypothesis (tighten prior to suppress phi→1 tail) tested and failed; correct diagnosis ≠ straightforward fix
- [2026-05-17-prng-fragility-recert.md](worklog/lessons/case-studies/stoch_vol/2026-05-17-prng-fragility-recert.md) — re-cert at seed=20260517 fails catastrophically (R̂≈5, ESS≈0.5); original "PRNG fragility" framing; **superseded by 2026-05-18**
- [2026-05-18-multimodal-warmup-capture-pathfinder-rescue.md](worklog/lessons/case-studies/stoch_vol/2026-05-18-multimodal-warmup-capture-pathfinder-rescue.md) — 7-seed sweep at recert config reveals 44 % gate-failure rate; warmup-adaptation capture by unit-root attractor is the mechanism; **Pathfinder→NUTS rescues the failing seed**; TRIVIAL recommendation to switch catalog warmup

See `worklog/lessons/case-studies/stoch_vol/README.md` for the index and quick summary.

## Citations

Stan User's Guide § 2.5 (NCP AR(1) form + prior recommendations); Kim, Shephard, Chib 1998 (original KSC model)
