# Sampling lessons: stoch_vol

## TL;DR

AR(1) unit-root geometry once caused divergence clusters at high persistence; a 2026-05-18 weakly-informative prior revision (Beta(4,4) phi factor + Normal(0,5) mu, PR #27) shifted the posterior bulk away from the boundary and drove cert divergences to 0 under bare `window_adaptation_diag_imm` (the catalog default). The per-model gate tolerance of 0.5 % is retained as a structural guard-rail (load-bearing under the original Uniform/Cauchy priors).

## Canonical recipe

**groundtruth__nuts__window_adaptation_diag_imm** (n_warmup=5000, target_acceptance=0.99, max_num_doublings=15) on the **post-PR-#27 model** (Beta(4,4) phi factor + Normal(0,5) mu). Cert verdict at seed=20260517: rhat_max=1.0001, min_bulk_ESS=2872, n_div=0 (0.00 %), E-BFMI=0.89, wall ≈ 99 s. The 0.5 % divergence-rate tolerance stays as guard-rail despite the current 0-divergence cert.

stoch_vol uses the catalog-default `window_adaptation_diag_imm` warmup, same as the other 12 NUTS-path catalog models. **This is a simplification landed 2026-05-19 (this commit)**: the previous pin was `multipathfinder` (4 paths + PSIS resampling, broadcast init), inherited from the PR #25 cert (2026-05-18) which used `multipathfinder → window_adaptation_diag_imm` as a two-stage pipeline to dodge a multi-mode warmup-capture pathology measured under the *original* Uniform(-1,1) phi / Cauchy(0,10) mu priors. The subsequent PR #27 prior revision (Beta(4,4) phi factor + Normal(0,5) mu) shifted the posterior bulk from phi_con≈0.987 to ≈0.961 — far enough from the unit-root attractor that bare `window_adaptation_diag_imm` no longer gets captured. A 2026-05-19 re-cert at seed=20260517 under the new priors with bare `window_adaptation_diag_imm` lands rhat=1.0001, ESS=2872, n_div=0, E-BFMI=0.89, wall=99 s — slightly *better* than the previous two-stage cert (rhat=1.0002, ESS=3197, n_div=0, wall=142 s). The multipathfinder pre-stage is no longer load-bearing.

## Sampling quirks

503-D NCP recursive AR(1) stochastic volatility. **Current model (post-PR-#27, 2026-05-18)**: weakly-informative prior with Uniform(-1,1) phi base × `numpyro.factor("phi_beta44_factor", Beta(4,4).log_prob((phi+1)/2))` (equivalent to phi_01 ~ Beta(4,4), phi = 2·phi_01−1; preserves "phi" as the unconstrained site name to keep `draws.npz` schema-stable while avoiding the `TransformedDistribution(Beta, AffineTransform)` NaN-gradient bug in NumPyro), plus mu ~ Normal(0, 5) replacing Cauchy(0, 10), plus sigma ~ HalfCauchy(5) unchanged. Posterior phi_con bulk shifted to 0.961 (from 0.987 under Uniform).

**Historical context (pre-PR-#27)**: under the original Uniform(-1,1) phi / Cauchy(0,10) mu priors, divergences clustered at extreme phi ≈ 0.9999 (unit-root boundary) where the stationary-distribution initialization `h[0] = mu + (sigma / sqrt(1 - phi²)) * h_std[0]` diverges. Diagonal-IMM condition number ≫1× due to boundary-element variance amplification (interior vs boundary elements experience different AR(1) persistence scaling), but this is downstream symptom, not primary cause. An early prior-tightening attempt (Beta(20, 1.5)-shifted phi) tripled divergences by pulling the posterior bulk closer to the difficult region; the eventual Beta(4, 4) symmetric prior succeeded by pulling the bulk AWAY from the boundary (toward phi_con = 0, std ≈ 0.30 in constrained space). See [2026-05-18 weakly-informative-prior case study](../../../../../worklog/lessons/case-studies/stoch_vol/2026-05-18-weakly-inf-prior-beta44-recert.md) for the full prior-sensitivity trial.

**Historical multi-mode warmup-capture (pre-PR-#27, no longer a concern)**: under the original Uniform/Cauchy priors the AR(1) posterior had a bad-attractor mode at the unit-root tail (phi ≈ 0.9999) that single-stage `window_adaptation_diag_imm` (init from NumPyro's default `init_to_uniform`) could be captured by during the first ~50 warmup steps (step_size collapses to ~10⁻⁶, post-warmup chain stuck). A 2026-05-18 multi-seed sweep showed **44 % gate failure rate under bare `window_adaptation_diag_imm`** at the recert seeds vs **25 % under the `multipathfinder → window_adaptation_diag_imm` two-stage pipeline** with 4 paths + PSIS resampling. The PR #27 Beta(4,4) prior revision then shifted the bulk away from the unit-root tail entirely, neutralising the capture pathology in the model itself. A 2026-05-19 re-cert at seed=20260517 under the new priors with bare `window_adaptation_diag_imm` confirmed this: 0 divergences, 0 warmup-capture failures, cleanest cert at lower wall than the two-stage equivalent. The multipathfinder pre-stage retired 2026-05-19. Full diagnostics at [`2026-05-18 multimodal-warmup case study`](../../../../../worklog/lessons/case-studies/stoch_vol/2026-05-18-multimodal-warmup-capture-pathfinder-rescue.md); the original "PRNG fragility" framing is preserved for context at [`2026-05-17-prng-fragility-recert.md`](../../../../../worklog/lessons/case-studies/stoch_vol/2026-05-17-prng-fragility-recert.md).

**Historical multipathfinder config — what was tried, what was retired (2026-05-18 2×2 sweep; obsoleted 2026-05-19)**. A 2026-05-18 2×2 sweep at 8 seeds tested {N_PATHS=4, N_PATHS=10} × {broadcast init, diverse init} when multipathfinder was still the pinned pre-stage. Results:

| Config | Pass rate | Notes |
|---|---:|---|
| **N=4, broadcast (then-pin)** | **6/8 (75 %)** | best in that sweep |
| N=10, broadcast | 4/8 (50 %) | more paths → more bad-mode weight under shared init |
| N=4, diverse | 3/8 (37.5 %) | 4 of 8 seeds crash Pathfinder L-BFGS with NaN |
| N=10, diverse | 0/8 | 7 of 8 seeds crash Pathfinder L-BFGS with NaN |

Conclusions had been baked into the multipathfinder pin: do NOT raise n_paths above 4 (empirically harms pass rate) and do NOT diverse-init (crashes Pathfinder's quasi-Newton step on this model's heavy-tail geometry). These conclusions are now of archival value only — the PR #27 prior revision retired the entire multipathfinder pre-stage. The L-BFGS heavy-tail-overflow finding remains a useful Pathfinder-on-AR(1) caveat if anyone tries to add multipathfinder back in for a different reason. Raw 2×2 data at `worklog/lessons/case-studies/stoch_vol/2026-05-18-multimodal-warmup-capture-pathfinder-rescue.md` § "2×2 follow-up sweep".

**Historical init-range sweep null result (2026-05-18; obsoleted 2026-05-19)**. The natural follow-up — "would a *narrower* init range rescue diverse-init multipathfinder?" — was tested across 7 variants on the same 8 seeds (additive jitter on the broadcast init at σ ∈ {0.01, 0.03, 0.05, 0.1, 0.3}; clamped diverse with `phi_unc ∈ [−2, +2]` only; clamped diverse with full `(mu, phi, sigma)` bracketing). No variant cleared "≥6/8 pass AND zero crashes". Even σ=0.01 jitter (1 % of NumPyro's `init_to_uniform(radius=2)`) crashed ≥1 seed via float-overflow in the AR(1) recursion. Detailed forensics at [`worklog/lessons/case-studies/stoch_vol/2026-05-18-init-range-sweep-no-winner.md`](worklog/lessons/case-studies/stoch_vol/2026-05-18-init-range-sweep-no-winner.md). Like the 2×2 sweep, this result is now archival — bare `window_adaptation_diag_imm` post-PR-#27 sidesteps the question entirely.

## Known-bad combinations

None documented yet. FAILED recipes will be backfilled in R5 once the recipe matrix is populated.

## History

The following case studies document the investigation path and distilled lessons:

- [2026-05-11-diagonal-imm-condition-ar1-boundary.md](worklog/lessons/case-studies/stoch_vol/2026-05-11-diagonal-imm-condition-ar1-boundary.md) — IMM condition-number dissection; shows the signal is real but downstream
- [2026-05-12-ar1-unit-root-divergence-cluster.md](worklog/lessons/case-studies/stoch_vol/2026-05-12-ar1-unit-root-divergence-cluster.md) — cluster analysis showing 92%/78% tail concentration at mu/phi extremes; identifies unit-root as primary driver
- [2026-05-12-failed-prior-swap-beta-shifted-phi.md](worklog/lessons/case-studies/stoch_vol/2026-05-12-failed-prior-swap-beta-shifted-phi.md) — hypothesis (tighten prior to suppress phi→1 tail) tested and failed; correct diagnosis ≠ straightforward fix
- [2026-05-17-prng-fragility-recert.md](worklog/lessons/case-studies/stoch_vol/2026-05-17-prng-fragility-recert.md) — re-cert at seed=20260517 fails catastrophically (R̂≈5, ESS≈0.5); original "PRNG fragility" framing; **superseded by 2026-05-18**
- [2026-05-18-multimodal-warmup-capture-pathfinder-rescue.md](worklog/lessons/case-studies/stoch_vol/2026-05-18-multimodal-warmup-capture-pathfinder-rescue.md) — 7-seed sweep at recert config under the *original* Uniform/Cauchy priors reveals 44 % gate-failure rate; warmup-adaptation capture by unit-root attractor is the mechanism; **Pathfinder→NUTS rescues the failing seed**; led to the multipathfinder pin (subsequently retired 2026-05-19 once PR #27 priors made the attractor unreachable)
- [2026-05-18-init-range-sweep-no-winner.md](worklog/lessons/case-studies/stoch_vol/2026-05-18-init-range-sweep-no-winner.md) — 7-variant init-range sweep; **no winner** for diverse-init multipathfinder; archival now that the multipathfinder pre-stage is retired
- [2026-05-18-weakly-inf-prior-beta44-recert.md](worklog/lessons/case-studies/stoch_vol/2026-05-18-weakly-inf-prior-beta44-recert.md) — PR #27 prior revision: Beta(4,4) factor on phi + Normal(0,5) on mu. Trial-level divergences 1.22 % → 0.03 %. The bulk-shift is what enabled the 2026-05-19 simplification back to bare `window_adaptation_diag_imm`

See `worklog/lessons/case-studies/stoch_vol/README.md` for the index and quick summary.

## Citations

Stan User's Guide § 2.5 (NCP AR(1) form + prior recommendations); Kim, Shephard, Chib 1998 (original KSC model)
