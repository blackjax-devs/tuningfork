# Sampling lessons: stoch_vol

## TL;DR

AR(1) unit-root geometry once caused divergence clusters at high persistence; a 2026-05-18 weakly-informative prior revision (Beta(4,4) phi factor + Normal(0,5) mu, PR #27) shifted the posterior bulk away from the boundary and drove cert divergences to 0. The per-model gate tolerance of 0.5 % is retained as a structural guard-rail (load-bearing under the original Uniform/Cauchy priors).

## Canonical recipe

**groundtruth__nuts__multipathfinder** (n_warmup=5000, n_paths=4, target_acceptance=0.99, max_num_doublings=15) on the **post-PR-#27 model** (Beta(4,4) phi factor + Normal(0,5) mu). Cert verdict at seed=20260517: rhat_max=1.0002, min_bulk_ESS=3197, n_div=0 (0.00 %), E-BFMI=0.88, wall ≈ 142 s. The 0.5 % divergence-rate tolerance stays as guard-rail despite the current 0-divergence cert.

The canonical warmup for stoch_vol is **`multipathfinder`** rather than the catalog-default `window_adaptation_diag_imm` — this is the only model in the 14-model catalog with a non-default warmup. Rationale: the AR(1) posterior is multi-modal with a bad attractor at the unit-root tail (phi ≈ 0.9999), and single-path window_adaptation can be captured by that mode during the first ~50 warmup steps (causing step_size to collapse to ~10⁻⁶ and the post-warmup chain to be stuck). Multipathfinder runs 4 independent Pathfinder fits and PSIS-resamples for the init position — landing in the bulk mode ~75 % of the time (vs ~57 % for window_adaptation). See the 2026-05-18 case-study for the multi-seed comparison.

## Sampling quirks

503-D NCP recursive AR(1) stochastic volatility. **Current model (post-PR-#27, 2026-05-18)**: weakly-informative prior with Uniform(-1,1) phi base × `numpyro.factor("phi_beta44_factor", Beta(4,4).log_prob((phi+1)/2))` (equivalent to phi_01 ~ Beta(4,4), phi = 2·phi_01−1; preserves "phi" as the unconstrained site name to keep `draws.npz` schema-stable while avoiding the `TransformedDistribution(Beta, AffineTransform)` NaN-gradient bug in NumPyro), plus mu ~ Normal(0, 5) replacing Cauchy(0, 10), plus sigma ~ HalfCauchy(5) unchanged. Posterior phi_con bulk shifted to 0.961 (from 0.987 under Uniform).

**Historical context (pre-PR-#27)**: under the original Uniform(-1,1) phi / Cauchy(0,10) mu priors, divergences clustered at extreme phi ≈ 0.9999 (unit-root boundary) where the stationary-distribution initialization `h[0] = mu + (sigma / sqrt(1 - phi²)) * h_std[0]` diverges. Diagonal-IMM condition number ≫1× due to boundary-element variance amplification (interior vs boundary elements experience different AR(1) persistence scaling), but this is downstream symptom, not primary cause. An early prior-tightening attempt (Beta(20, 1.5)-shifted phi) tripled divergences by pulling the posterior bulk closer to the difficult region; the eventual Beta(4, 4) symmetric prior succeeded by pulling the bulk AWAY from the boundary (toward phi_con = 0, std ≈ 0.30 in constrained space). See [2026-05-18 weakly-informative-prior case study](../../../../../worklog/lessons/case-studies/stoch_vol/2026-05-18-weakly-inf-prior-beta44-recert.md) for the full prior-sensitivity trial.

**⚠️ Multi-mode warmup capture (2026-05-18 finding, addressed by switching to multipathfinder)**: under the original priors the AR(1) posterior had a bad-attractor mode at the unit-root tail (phi ≈ 0.9999) that single-path warmups could be captured by during the first ~50 warmup steps (step_size collapses to ~10⁻⁶, post-warmup chain stuck). A 2026-05-18 multi-seed sweep showed **44 % gate failure rate under `window_adaptation_diag_imm`** (vs **25 % under `multipathfinder`** with 4 paths + PSIS resampling). The catalog's canonical warmup for stoch_vol became `multipathfinder` for this reason. Subsequently the PR-#27 Beta(4,4) prior revision shifted the bulk away from the unit-root tail entirely, so the multi-mode-capture risk is also substantially attenuated under the current model — but `multipathfinder` remains the pinned warmup as belt-and-braces (the AR(1) geometry's residual non-Gaussian shape still benefits from PSIS-resampled init). Post-PR-#27 re-cert at seed=20260517: rhat=1.0002, ESS=3197, n_div=0, wall=142s — see [`reference/metadata.json`](reference/metadata.json). Multi-seed cert validation (rather than single-seed point-pin) is the appropriate cert protocol if you regenerate locally — `force_regenerate=True` may not reproduce the bulk-mode landing on every run. Full diagnostics at [`2026-05-18 multimodal-warmup case study`](../../../../../worklog/lessons/case-studies/stoch_vol/2026-05-18-multimodal-warmup-capture-pathfinder-rescue.md); original "PRNG fragility" framing preserved for context at [`2026-05-17-prng-fragility-recert.md`](../../../../../worklog/lessons/case-studies/stoch_vol/2026-05-17-prng-fragility-recert.md).

**Multipathfinder config — what was tried, what was kept (2026-05-18 2×2 sweep)**. A follow-up 2×2 sweep at the same 8 seeds tested {N_PATHS=4, N_PATHS=10} × {broadcast init, diverse init}:

| Config | Pass rate | Notes |
|---|---:|---|
| **N=4, broadcast (committed pin)** | **6/8 (75 %)** | best |
| N=10, broadcast | 4/8 (50 %) | more paths → more bad-mode weight under shared init |
| N=4, diverse | 3/8 (37.5 %) | 4 of 8 seeds crash Pathfinder L-BFGS with NaN |
| N=10, diverse | 0/8 | 7 of 8 seeds crash Pathfinder L-BFGS with NaN |

Conclusions baked into the canonical pin: **do NOT raise n_paths above 4** (empirically harms pass rate) and **do NOT diverse-init** (crashes Pathfinder's quasi-Newton step on this model's heavy-tail geometry). Raw 2×2 data at `worklog/lessons/case-studies/stoch_vol/2026-05-18-multimodal-warmup-capture-pathfinder-rescue.md` § "2×2 follow-up sweep".

**Init-range sweep follow-up (2026-05-18 statistician null result)**. The natural next question — "would a *narrower* init range rescue diverse-init?" — was tested across 7 variants on the same 8 seeds (additive jitter on the broadcast init at σ ∈ {0.01, 0.03, 0.05, 0.1, 0.3}; clamped diverse with `phi_unc ∈ [−2, +2]` only; clamped diverse with full `(mu, phi, sigma)` bracketing). The criterion was "winner = ≥6/8 pass AND zero crashes across the 8 seeds". **No variant clears the bar.** Even σ=0.01 jitter (1 % of NumPyro's default `init_to_uniform(radius=2)` range) crashes ≥1 seed via float-overflow in the AR(1) recursion. The crash mechanism is downstream of `sigma_unc + h_raw` heavy-tail compounding through `Normal(0, exp(h/2))`: any per-path variance pushes some L-BFGS iterates into the overflow regime where the gradient flips sign and the quasi-Newton method runs away to magnitudes ~1e7 (unconstrained). Detailed forensics + 7-variant table at [`worklog/lessons/case-studies/stoch_vol/2026-05-18-init-range-sweep-no-winner.md`](worklog/lessons/case-studies/stoch_vol/2026-05-18-init-range-sweep-no-winner.md).

## Known-bad combinations

None documented yet. FAILED recipes will be backfilled in R5 once the recipe matrix is populated.

## History

The following case studies document the investigation path and distilled lessons:

- [2026-05-11-diagonal-imm-condition-ar1-boundary.md](worklog/lessons/case-studies/stoch_vol/2026-05-11-diagonal-imm-condition-ar1-boundary.md) — IMM condition-number dissection; shows the signal is real but downstream
- [2026-05-12-ar1-unit-root-divergence-cluster.md](worklog/lessons/case-studies/stoch_vol/2026-05-12-ar1-unit-root-divergence-cluster.md) — cluster analysis showing 92%/78% tail concentration at mu/phi extremes; identifies unit-root as primary driver
- [2026-05-12-failed-prior-swap-beta-shifted-phi.md](worklog/lessons/case-studies/stoch_vol/2026-05-12-failed-prior-swap-beta-shifted-phi.md) — hypothesis (tighten prior to suppress phi→1 tail) tested and failed; correct diagnosis ≠ straightforward fix
- [2026-05-17-prng-fragility-recert.md](worklog/lessons/case-studies/stoch_vol/2026-05-17-prng-fragility-recert.md) — re-cert at seed=20260517 fails catastrophically (R̂≈5, ESS≈0.5); original "PRNG fragility" framing; **superseded by 2026-05-18**
- [2026-05-18-multimodal-warmup-capture-pathfinder-rescue.md](worklog/lessons/case-studies/stoch_vol/2026-05-18-multimodal-warmup-capture-pathfinder-rescue.md) — 7-seed sweep at recert config reveals 44 % gate-failure rate; warmup-adaptation capture by unit-root attractor is the mechanism; **Pathfinder→NUTS rescues the failing seed**; TRIVIAL recommendation to switch catalog warmup
- [2026-05-18-init-range-sweep-no-winner.md](worklog/lessons/case-studies/stoch_vol/2026-05-18-init-range-sweep-no-winner.md) — 7-variant init-range sweep (jitter σ ∈ {0.01, 0.03, 0.05, 0.1, 0.3}; clamped diverse {phi-only, mu+phi+sigma}) tests whether a narrower init range rescues diverse-init for multipathfinder. **No winner** across 8 seeds; AR(1) heavy-tail geometry makes L-BFGS step-randomization fragile under any per-path init variance. Confirms the N=4 broadcast pin is the structural best

See `worklog/lessons/case-studies/stoch_vol/README.md` for the index and quick summary.

## Citations

Stan User's Guide § 2.5 (NCP AR(1) form + prior recommendations); Kim, Shephard, Chib 1998 (original KSC model)
