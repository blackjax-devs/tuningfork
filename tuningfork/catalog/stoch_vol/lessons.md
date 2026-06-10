# Sampling lessons: stoch_vol

## TL;DR

**Prior divergence:** stoch_vol uses a **deliberately modified prior (Beta(4,4) phi factor + Normal(0,5) mu)** divergent from posteriordb's canonical (Uniform(-1,1) phi / Cauchy(0,10) mu). Ground-truth draws are certified via tuningfork's own NUTS reference (seed=20260517, cert wall ≈99 s). **We explicitly do NOT claim posteriordb agreement** for stoch_vol — the prior revision is an intentional departure justified by convergence geometry on the unit-root boundary. See § "Sampling quirks" for the full prior-sensitivity rationale.

AR(1) unit-root geometry once caused divergence clusters at high persistence; the 2026-05-18 weakly-informative prior revision shifted the posterior bulk away from the boundary and drove cert divergences to 0 under bare `window_adaptation_diag_imm` (the catalog default). The per-model `divergence_rate_tolerance` override (0.005) was removed 2026-05-19 — the current cert is well under the global default 0.001 (40 divergences allowed in 40k; current cert has 0).

## Canonical recipe

**groundtruth__nuts__window_adaptation_diag_imm** (n_warmup=5000, target_acceptance=0.99, max_num_doublings=15) on the **post-PR-#27 model** (Beta(4,4) phi factor + Normal(0,5) mu). Cert verdict at seed=20260517: rhat_max=1.0001, min_bulk_ESS=2872, n_div=0 (0.00 %), E-BFMI=0.89, wall ≈ 99 s. Passes the global default `divergence_rate_tolerance = 0.001` (= 40 divs allowed in 40k) without the previous per-model override.

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

## MCLMC LRD experiment (2026-06-09)

Two MCLMC LRD variants were stress-tested on stoch_vol (503-D). Both results are
important and must be preserved together; they illuminate the prior-centering
preservation principle.

| Variant | Rank k | Max R-hat | Min ESS | Verdict |
|---|---|---|---|---|
| External LRD (whitening, mean-centred) | k=50 | **2.0536** | 5.4 | **FAIL** |
| Internal LRD (native integrator) | k=50 | **1.0498** | 156.1 | **REVIEW** |

### External whitening: FAIL (R-hat=2.0536)
External coordinate-whitening applies the transformation x = L_LR(y) + mean, where
`mean` is the posterior mean estimated from the NUTS pilot samples. This translation
completely breaks the prior-centered AR(1) structure: the latents h_t are defined
natively under h_t ~ N(φ h_{t-1}, σ²), and shifting them by their posterior mean
disrupts the hierarchical prior coupling. The result is immediate step-size collapse
and non-convergence.

This is a crucial negative result: **external coordinate-whitening is structurally
incompatible with prior-centered hierarchical models**. See
`tests/mclmc_lrd/test_adaptive_lrd_stoch_vol.py`.

### Internal LRD: REVIEW (R-hat=1.0498, partial improvement)
The internal LRD integrator rescales and rotates **momentum/velocity vectors** only —
it does NOT translate position coordinates. This preserves the prior-centered
AR(1) structure, allowing the sampler to operate in the original coordinate frame
while preconditioning the strong linear AR(1) correlations in momentum space.

Result: partial improvement over diagonal MCLMC (baseline would FAIL; REVIEW is an
improvement). However, non-linear varying curvature from the hierarchical funnel
geometry limits mixing: the single global LRD mass matrix cannot adapt to the
position-dependent curvature of the AR(1) funnel neck. REVIEW is the expected ceiling.

The prior doc label "Successful Rescue!" overstated the result. Correct characterization:
**partial improvement over diagonal; funnel geometry limits mixing.**

### Prior-centering preservation principle
Internal LRD momentum-only scaling preserves prior-centering; external position
translation destroys it. For any hierarchical model with prior-centered parameterization
(NCP), use internal LRD only. Even then, expect REVIEW (not PASS) when the
hierarchical funnel geometry is the dominant bottleneck.

### Routing rule for stoch_vol
Route to **NUTS** (`window_adaptation_diag_imm`, PASS). MCLMC family is not viable
as the primary sampler on stoch_vol due to the AR(1) funnel geometry.

## Known-bad combinations

- External LRD coordinate-whitening on NCP models: **FAIL** (breaks prior-centering).
  Fundamental incompatibility, not a tuning issue.

## Recipe regen (stoch_vol LRD flat-init, NUTS-pilot path)

**STATUS: DEFERRED** — LRD calibration track stopped per mission fallback (2026-06-10).
Phase (c) Track 2: 0/3 cert seeds ERROR (mixed-rank pytree crash in `_run_cert_seed`
post-sampling R-hat/ESS aggregation step). See `worklog/threads/feat-mclmc-lrd-integrator.md` § Phase (c) Track 2 VERDICT
and this catalog's `lrd_track2_failure_analysis_2026-06-09.md`.
`scripts/calibrate_stoch_vol_lrd.py` is the sole provenance for the committed artifacts
until @user authorises a retry.

The committed artifacts are:
- `recipes/low__mclmc_lrd__mclmc_lrd_tuning_flatinit.json` — golden recipe (k=30, 2-seed REVIEW, script-baked)
- `recipes/low__mclmc_lrd__mclmc_lrd_tuning_flatinit.imm.npz` — rank-30 LRD IMM sidecar (from seed=99 pilot)

**Note:** This is the flat-init NCP variant, NOT the registered `stoch_vol` model.
The registered model uses stationary init (`h[0] = mu + (sigma/sqrt(1-phi^2)) * h_std[0]`);
the flat-init variant uses `h[0] = mu + sigma * h_std[0]` to reduce phi coupling.

**To regenerate** (once @user authorises retry and mixed-rank fix is on main):
Re-run `scripts/calibrate_stoch_vol_lrd.py` with the same parameters (flat-init variant —
the sole provenance for the committed artifacts). For the standard registered `stoch_vol`
model, the generator command is:
```bash
uv run python -m tuningfork.recipes._generate_starter \
    --warmup mclmc_lrd_tuning --only stoch_vol \
    --calibrate --cert-seeds <seeds> --n-warmup 3000 --n-samples 2000 --k-rank 30
```
Statistician-approved config (2026-06-09):
- k=30 (not full-rank k=50 — higher rank degrades R-hat on the funnel neck)
- n_warmup=3000, n_samples=2000, num_chains=4
- NUTS pilot: pilot_n_warmup=1000, pilot_n_samples=1000 (single chain)
- Seeds [42, 99] (two-seed protocol)
- Expected verdict: REVIEW (R-hat 1.01–1.05; funnel geometry limits mixing)

The standard `mclmc_lrd_tuning` warmup path via the generator requires model
registration. Until then, regeneration requires `scripts/calibrate_stoch_vol_lrd.py`.

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
