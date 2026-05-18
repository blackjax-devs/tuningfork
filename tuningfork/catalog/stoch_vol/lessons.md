# Sampling lessons: stoch_vol

## TL;DR

AR(1) unit-root geometry causes divergence clusters at high persistence; requires per-model gate tolerance (0.5%) despite correct R̂ and ESS.

## Canonical recipe

**groundtruth__nuts__multipathfinder** (n_warmup=5000, n_paths=4, target_acceptance=0.99, max_num_doublings=15; account for 0.5% divergence rate in gate). Cert verdict at seed=20260517: rhat_max=1.00023, min_bulk_ESS=1612, n_div=141 (0.35%), E-BFMI=0.92, wall ≈ 184 s.

The canonical warmup for stoch_vol is **`multipathfinder`** rather than the catalog-default `window_adaptation_diag_imm` — this is the only model in the 14-model catalog with a non-default warmup. Rationale: the AR(1) posterior is multi-modal with a bad attractor at the unit-root tail (phi ≈ 0.9999), and single-path window_adaptation can be captured by that mode during the first ~50 warmup steps (causing step_size to collapse to ~10⁻⁶ and the post-warmup chain to be stuck). Multipathfinder runs 4 independent Pathfinder fits and PSIS-resamples for the init position — landing in the bulk mode ~75 % of the time (vs ~57 % for window_adaptation). See the 2026-05-18 case-study for the multi-seed comparison.

## Sampling quirks

503-D NCP recursive AR(1) stochastic volatility with Uniform(-1,1) phi prior. Divergences cluster at extreme phi ≈ 0.9999 (unit-root boundary) where the stationary-distribution initialization `h[0] = mu + (sigma / sqrt(1 - phi²)) * h_std[0]` diverges. Diagonal-IMM condition number ≫1× due to boundary-element variance amplification (interior vs boundary elements experience different AR(1) persistence scaling), but this is downstream symptom, not primary cause. Tightening the prior (Beta(20,1.5)-shifted phi) tripled divergences by pulling the posterior bulk closer to the difficult region.

**⚠️ Multi-mode warmup capture (2026-05-18 finding, addressed by switching to multipathfinder)**: the AR(1) posterior has a bad-attractor mode at the unit-root tail (phi ≈ 0.9999) that single-path warmups can be captured by during the first ~50 warmup steps (step_size collapses to ~10⁻⁶, post-warmup chain stuck). A 2026-05-18 multi-seed sweep showed **44 % gate failure rate under `window_adaptation_diag_imm`** (vs **25 % under `multipathfinder`** with 4 paths + PSIS resampling). The catalog's canonical warmup for stoch_vol is now `multipathfinder` (re-certed at seed=20260517 with cert verdict rhat=1.0002, ESS=1612, n_div=141, wall=184s — see metadata.json). The ~25 % residual multipathfinder failure rate is acceptable for this model: pareto-k > 1 (PSIS unreliable for 503-D multimodal) is expected, so document and move on. Multi-seed cert validation (rather than single-seed point-pin) is the appropriate cert protocol if you regenerate locally — `force_regenerate=True` may not reproduce the bulk-mode landing on every run. Full diagnostics at [`worklog/lessons/case-studies/stoch_vol/2026-05-18-multimodal-warmup-capture-pathfinder-rescue.md`](worklog/lessons/case-studies/stoch_vol/2026-05-18-multimodal-warmup-capture-pathfinder-rescue.md); original "PRNG fragility" framing preserved for context at [`2026-05-17-prng-fragility-recert.md`](worklog/lessons/case-studies/stoch_vol/2026-05-17-prng-fragility-recert.md).

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
