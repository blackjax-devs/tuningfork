# Sampling lessons: lotka_volterra

## TL;DR

**The in-scope LOW-effort PASS verdicts below are not reproducible and should be
treated as provisional** — see the 2026-07-29 entry under History. The posterior
has an absorbing secondary mode ~420 nats behind a barrier, and every LOW-effort
recipe run starts all chains from a single hostile `init_to_uniform` point, so
each run is a lottery over which chains fall in. The same cell passes at some
seeds and fails at others under both the current and the original dependency
stack. **7 of the 8 affected cells have a reliable MEDIUM-effort alternative**
using a GT-informed `init_strategy` + a disclosed, seed-selected seed — see the
2026-07-30 entry under History and prefer the `medium__*__gt_informed_init.json`
recipes over the corresponding `low__*` ones. `hmc + window_adaptation_dense_imm
+ inner_nuts` is the one exception that stays unresolved at LOW.

Stiff ODE posterior with bimodal structure. Dense and low_rank IMM PASS for NUTS/dmhmc/dynamic_hmc/hmc at LOW effort. **Diag IMM FAILS for dynamic_hmc and dmhmc** at LOW effort and even MEDIUM; the stiff ODE geometry requires off-diagonal mass matrix structure. `mhmc` is structurally unsuitable (step_size collapses). MCLMC variants FAIL (warmup hang). VI is out_of_scope.
[boundary: dense/low_rank IMM PASS holds at LOW n_warmup=1000; diag IMM FAIL confirmed across multiple step policies; nearest FAIL: dynamic_hmc+diag_imm (see recipes/failed__dynamic_hmc__window_adaptation_diag_imm.json); dense IMM PASS for nuts (see recipes/failed__nuts__window_adaptation_dense_imm.json — this one actually FAILS too, see below)]

## Canonical recipe

`recipes/low__nuts__window_adaptation_low_rank_imm.json` — LOW effort, PASS.
`recipes/low__dynamic_hmc__window_adaptation_dense_imm.json` — LOW effort, PASS.
`recipes/low__mclmc__mclmc_tuning.json` — LOW effort, PASS.

**MEDIUM-tier, GT-informed-init recipes (2026-07-30, see History below) — prefer
these over the LOW recipes above where both exist**, since the LOW recipes are
the unreliable single-broadcast-init lottery described in the TL;DR:
`recipes/medium__dmhmc__window_adaptation_dense_imm__gt_informed_init.json`,
`recipes/medium__dmhmc__window_adaptation_low_rank_imm__gt_informed_init.json`,
`recipes/medium__dynamic_hmc__window_adaptation_dense_imm__gt_informed_init.json`,
`recipes/medium__dynamic_hmc__window_adaptation_low_rank_imm__gt_informed_init.json`,
`recipes/medium__hmc__window_adaptation_diag_imm__inner_nuts__gt_informed_init.json`,
`recipes/medium__hmc__window_adaptation_low_rank_imm__inner_nuts__gt_informed_init.json`,
`recipes/medium__nuts__window_adaptation_low_rank_imm__gt_informed_init.json`.

## Sampling quirks

### Diag IMM insufficient for stiff ODE geometry
`dynamic_hmc` and `dmhmc` with `window_adaptation_diag_imm` FAIL even with MEDIUM
step policies (v1-medium, v7-empirical-oracle). The bimodal ODE posterior has
off-diagonal covariance that the diagonal IMM cannot capture.
[boundary: diag IMM FAIL is policy-invariant — same failure at LOW (default), v1-medium, and v7-empirical-oracle; use dense or low_rank IMM]

### mhmc: structurally unsuitable
`mhmc` with any IMM (diag, dense, low_rank) fails because step_size adaptation
collapses near the near-degenerate ODE likelihood boundary.
[boundary: mhmc FAIL is IMM-invariant; see failed__mhmc__window_adaptation_{dense,diag,low_rank}_imm.json]

### MCLMC variants: warmup hang
`adjusted_mclmc` and `adjusted_mclmc_dynamic` with `adjusted_mclmc_tuning` hang
during warmup (non-terminating warmup loop) due to the stiff ODE gradient landscape.
[boundary: warmup hang at any budget; not tunable; do not use MCLMC on lotka_volterra without a custom warmup]

### NUTS + dense IMM: FAIL at LOW (specific cell)
`nuts` + `window_adaptation_dense_imm` at LOW effort FAILS (hard_direction).
NUTS + `window_adaptation_low_rank_imm` PASSES at LOW effort.
[boundary: dense IMM fails for NUTS at default LOW; low_rank IMM PASS for NUTS; but dense IMM PASS for dynamic_hmc and dmhmc — applies at DIFFERENT configurations]

## Known-bad combinations

- `dynamic_hmc` + `window_adaptation_diag_imm` (any step policy): **FAIL**. See `recipes/failed__dynamic_hmc__window_adaptation_diag_imm.json`, `failed__dynamic_hmc__window_adaptation_diag_imm__policy_v1-medium.json`, `failed__dynamic_hmc__window_adaptation_diag_imm__policy_v7-empirical-oracle.json`.
- `dmhmc` + `window_adaptation_diag_imm` (any step policy): **FAIL**. See `recipes/failed__dmhmc__window_adaptation_diag_imm.json`, `failed__dmhmc__window_adaptation_diag_imm__policy_v1-medium.json`, `failed__dmhmc__window_adaptation_diag_imm__policy_v7-empirical-oracle.json`.
- `mhmc` + `window_adaptation_dense_imm`: **FAIL** (step_size collapse). See `recipes/failed__mhmc__window_adaptation_dense_imm.json`.
- `mhmc` + `window_adaptation_diag_imm`: **FAIL**. See `recipes/failed__mhmc__window_adaptation_diag_imm.json`.
- `mhmc` + `window_adaptation_low_rank_imm`: **FAIL**. See `recipes/failed__mhmc__window_adaptation_low_rank_imm.json`.
- `adjusted_mclmc` + `adjusted_mclmc_tuning`: **FAIL** (warmup hang). See `recipes/failed__adjusted_mclmc__adjusted_mclmc_tuning.json`.
- `adjusted_mclmc_dynamic` + `adjusted_mclmc_tuning`: **FAIL** (warmup hang). See `recipes/failed__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json`.
- `nuts` + `window_adaptation_dense_imm` (LOW): **FAIL** (hard_direction). See `recipes/failed__nuts__window_adaptation_dense_imm.json`.
- `fullrank_vi` + `no_warmup`: **FAIL** (out_of_scope). See `recipes/failed__fullrank_vi__no_warmup.json`.
- `meanfield_vi` + `no_warmup`: **FAIL** (out_of_scope). See `recipes/failed__meanfield_vi__no_warmup.json`.

Recorded FAILs not discussed above: all 14 failed recipes are covered above.

## History

### 2026-07-29 — the in-scope cells are a warmup lottery, not a dependency regression

A corpus re-emission under blackjax 1.6.1 / jax 0.11.0 found 11 of the 12
in-scope lotka_volterra cells failing, against committed baselines recorded at
blackjax 1.6.dev84 / jax 0.10.0. The failure was initially read as a dependency
regression in the ODE integration path. It is not. Both the jax hypothesis and
the ODE hypothesis were tested directly and refuted, and the underlying cause is
a property of this posterior plus the way recipe runs start their chains.

**The model computation did not change.** Log-density and gradient were evaluated
at 10 fixed unconstrained positions, plus the raw `_solve_lv` outputs, under two
stacks differing only in jax (0.11.0 vs 0.10.0, with numpyro 0.21.0 / probdiffeq
0.9.2 / numpy 2.4.6 held fixed). 24 of 34 compared arrays are bit-for-bit
identical, including every `u_mean` / `u_std` array from the ODE solve and the
log-density at both the prior centre and the synthetic truth. The 10 that differ
are gradients and two log-densities, differing by at most 72 ULP (max relative
difference 1.5e-14) — floating-point reassociation, not a behaviour change. The
same probe re-run twice on one stack is 34/34 identical, so those ULP differences
are attributable to the version and not to run-to-run noise. The numpyro init
position and the raw uniform RNG stream are also bit-identical across the two
versions.

**The posterior has an absorbing decoy mode.** Walking a straight line in
unconstrained space from the certified reference mean to the mean of a failing
chain: log-density is -65.8 at the reference mode, -236.5 at the decoy, and dips
to -656.3 at the midpoint. The decoy therefore sits ~171 nats below the true mode
(negligible posterior mass, correctly excluded from the reference) behind a
barrier ~420 nats deep. No HMC chain crosses that in a 1000-step warmup, so a
chain that lands there is trapped for the life of the run. The decoy is the
classic ODE-inverse failure: `sigma_obs` = 3.86 instead of 0.57, i.e. a bad
trajectory fit absorbed into inflated observation noise.

**Two design facts make landing there a coin flip.** The emit path starts every
chain from ONE init position broadcast to all chains with no jitter, so chains
are distinguished only by their warmup keys. That single init comes from
numpyro's `init_to_uniform`, which for this model is a hostile start: log-density
-9557, gradient norm 1.9e4, and 440 of 512 sampled points from the same
`[-2, 2]^7` box have non-finite log-density (86%). The posterior region itself is
clean (0 of 256 non-finite), and the non-finite fraction is identical under both
jax versions. Which basin a chain descends into is decided by chaotic
accumulation over the warmup from that point.

**Seed sweep, `low__dmhmc__window_adaptation_dense_imm`.** Same emitted script,
same box, all draws scored by one procedure under one stack:

| seed | jax 0.11.0 + blackjax 1.6.1 | jax 0.10.0 + blackjax 1.6.dev84 (baseline) |
|---|---|---|
| 682737 (recipe's own) | R-hat 11.10, min-ESS 2.02 | R-hat 11.41, min-ESS 2.02 |
| 682738 | R-hat 1.0131, min-ESS 325.6 | R-hat 1.0131, min-ESS 325.6 |
| 682739 | R-hat 1.3239, min-ESS 5.11 | R-hat 1.0086, min-ESS 333.8 |
| 682740 | R-hat 1.0131, min-ESS 282.1 | R-hat 1.0129, min-ESS 300.0 |
| 682741 | R-hat 123.07, min-ESS 2.00 | R-hat 132.54, min-ESS 2.00 |
| 682742 | R-hat 1.0083, min-ESS 285.7 | R-hat 1.0091, min-ESS 280.3 |

Three of six pass on the current stack, four of six on the baseline stack — a
one-seed difference on n=6, i.e. no detectable version effect. Critically, the
**exact baseline stack fails at the recipe's own recorded seed**, with the same
chain in the same decoy mode (u0 = 2.1803 on both stacks). The committed record
for that cell is R-hat 1.0038. No choice of R-hat estimator turns 11.41 into
1.0038; the difference is whether a chain sits in the decoy basin.

Three distinct pathologies appear across the failing seeds, all warmup-lottery
outcomes and all with zero or near-zero divergences (so the divergence counter
does not catch them):

1. **Decoy-basin capture** (seed 682737): one chain 26.4 reference-SDs away,
   stable there, acceptance 0.99, 0 divergences.
2. **Within-mode under-mixing** (seed 682739): all chains in the true mode but
   one explores half the width of the others.
3. **Step-size collapse** (seed 682741): three of four chains frozen in
   different far-away places — one with position standard deviation exactly 0.0
   over 1000 draws while reporting 0.9999 acceptance. This is the same pathology
   `lessons.md` already records for `mhmc` on this model; `dmhmc` hits it too,
   seed-dependently.

**Caveats on the committed baselines.** The recipes were emitted on x86_64; this
reproduction ran on aarch64, and arch alone is known to flip chaotic-warmup gate
verdicts in this suite. Separately, the recorded `tuning_seed` values derive from
a master seed that the corpus does not record, so an emitted-script run cannot be
guaranteed to reproduce the seeding of the original run. No per-recipe draws are
committed for this model, so the original PASS runs cannot be re-examined. The
jax-0.11-vs-jax-0.10 comparison above is unaffected by either caveat, because
both sides ran on the same box with everything else fixed.

**Comparability note.** Do not compare the `min_bulk_ess` recorded in these
recipes against a freshly computed ESS: the estimator convention has changed
since emission. R-hat is the safe cross-era signal here, and only because the
discrepancy is structural (11.4 vs 1.004) rather than marginal.

### 2026-07-30 — why an explicit init cannot yet be specified for this model

Follow-up to the entry above. The intent was to pin `init_strategy` on the 11
in-scope cells so chains cannot descend into the decoy basin — a deliberate
experimental control on a known bimodal target, not a special case. The schema
already supports it (`uniform_perchain` / `zero_perchain`, validated in
`recipes/_base.py`, allowed for the window-adaptation families). Measurement says
no expressible spec is reliable, so the cells are left unpinned and the finding is
recorded instead.

**Why the geometry is hostile.** In unconstrained space the certified mode sits at
`alpha -0.742, beta -3.037, delta -2.952, gamma -0.641, sigma_obs -0.567,
u0 2.308, v0 1.588` with per-coordinate SDs of 0.016 to 0.083. The coordinates
span -3.04 to +2.31 while the posterior itself is three orders of magnitude
tighter, so **any** single absolute box is 140 to 250 posterior SDs from the mode.
`uniform_perchain` applies ONE scalar `(low, high)` to every coordinate, which is
the crux: the decoy basin surrounds the ORIGIN (its coordinates are `alpha -0.167,
gamma -0.013`, all above the true mode), so a box near zero lands in the decoy,
while a box far enough negative to avoid it lands in a steep tail where step-size
adaptation collapses.

**Candidate scan** (1024 draws per box; "nearer ref" = closer to the certified
mode than to the decoy in unconstrained Euclidean distance):

| box | non-finite | median logp | nearer ref |
|---|---|---|---|
| `[-0.25, 0.25]` | 1.5% | -4692 | 0.0% |
| `[-1.0, 1.0]` | 71.2% | -4302 | 17.6% |
| `[-2.0, 2.0]` (numpyro's effective box) | 83.8% | -6091 | 50.6% |
| `[-3.1, 2.4]` (posterior bounding box) | 81.7% | -7053 | 61.0% |
| `[-1.5, -0.5]` | 2.3% | -33176 | 100.0% |
| `N(0, 0.1^2)` | 0.3% | -4768 | 0.0% |

Validity and basin-side pull in opposite directions: the boxes that are finite are
in the decoy's basin, and the box that is entirely on the reference side sits at
median log-density -33176.

**Gate outcomes**, `dmhmc` × `window_adaptation_dense_imm`, 3 seeds each, run
through the production emit path so the verdict is the production verdict.
"off-mode" counts chains whose per-parameter mean exceeds 5 reference SDs:

| init | PASS | off-mode chains | failure signature |
|---|---|---|---|
| current (single broadcast) | 1/3 | 7/12 | decoy at \|z\| ~26, or all-chain collapse at \|z\| ~131 |
| `uniform_perchain [-1.5, -0.5]` | 2/3 | 2/12 | step-size collapse at \|z\| ~220 |
| `uniform_perchain [-1.0, 0.0]` | 0/3 | 6/12 | 1000+ divergences from non-finite starts |
| `uniform_perchain [-2.0, 0.0]` | 1/3 | 5/12 | 2000 divergences; collapse at \|z\| ~240 |
| `zero_perchain N(0, 0.1^2)` | 0/3 | 6/12 | exactly 2 of 4 chains in the decoy at every seed |

**Widened seed set** (superseding the 3-seed reading above, which understated the
candidate). `uniform_perchain [-1.5, -0.5]` over every seed tried:

| seed | provenance | verdict | R-hat | min-ESS | off-mode |
|---|---|---|---|---|---|
| 11111 | LRD cert seed | PASS | 1.0057 | 2326.93 | 0/4 |
| 22222 | LRD cert seed | PASS | 1.0022 | 2254.67 | 0/4 |
| 33333 | LRD cert seed | PASS | 1.0027 | 2340.34 | 0/4 |
| 682737 | this cell's recorded `tuning_seed` | FAIL | 2.7738 | 4.63 | 2/4 |
| 682738 | exploratory | PASS | 1.0030 | 2339.34 | 0/4 |
| 682739 | exploratory | PASS | 1.0038 | 2320.00 | 0/4 |
| 20260517 | `RECIPE_SEED`, the general path's default | FAIL | 1.5940 | 6.75 | 1/4 |

5 of 7. The current single-broadcast init over the same seeds is 1 of 4
(682738 PASS; 682737, 682739, 20260517 FAIL). Every passing run has **zero**
off-mode chains and zero divergences, so when the init works it works cleanly —
the decoy is absent rather than merely rare.

But it fails at the two seeds that matter institutionally: `RECIPE_SEED`
(20260517), which is what the general emit path uses by default, and 682737,
which is what these cells actually record. The three LRD cert seeds all pass,
so a 3-seed verdict here depends entirely on **which** triple is chosen. Note
also that the historical ≥2-of-3 cert bar governed the low-rank-diagonal MCLMC
path only; `emit_low_recipe_for_cell`, which produces these cells, is single-seed
with no multi-seed bar. The historical direct emitter is retired; current
certification uses the generated recipe lifecycle.
The `zero_perchain` row is the clearest evidence that the origin lies on the basin
boundary: 2 of 4 chains land in the decoy at every seed tested, with R-hat pinned
between 1.7351 and 1.7374.

Note also that `uniform_perchain` performs no validity check of its own — numpyro's
`initialize_model` does (0 of 32 seeds returned a non-finite log-density or
gradient), but the `init_strategy` path bypasses it, which is what produces the
1000-to-2000-divergence rows above.

**Conclusion.** A reliable init for this model needs per-coordinate bounds (or a
validity-guarded draw at posterior scale), neither of which the current
`init_strategy` schema can express. That is a harness change, tracked separately;
these cells stay unpinned until it lands.

**Consequence.** The `dense`/`diag`/`low_rank` distinction is cosmetic for the
low-effort HMC-family cells: they all store the same `step_size`
(0.039880085113090075) and the same length-7 **diagonal** inverse mass matrix
under the same tuning seed, and at baseline they recorded metrics identical to
sixteen digits across preconditioners. Any "invariant across preconditioners"
reasoning about these cells is therefore vacuous — the preconditioner never
varied.

### 2026-07-30 — MEDIUM-tier promotion: seed selection ratified (Belief#1176), 7 of 8 cells promoted

Follow-up to both entries above. JP ratified two policy points that change what
"done" means for the 11 in-scope cells (Belief#1175, Belief#1176): (1) the prior
"no expressible init_strategy fixes this model" conclusion held only at LOW
tier, where `recipes/_base.py` disqualifies ANY specified init by construction;
at MEDIUM tier a specified init is exactly the schema-sanctioned branch-(a)
intervention ("try alternate initialisations"), so the best expressible box
(`uniform_perchain [-1.5, -0.5]`) becomes usable. (2) Seed selection ("seed-
hacking") is a legitimate recipe-authoring practice, not a data-integrity
violation, provided the chosen seed is independently verified through the
normal production gate with no relaxation, and provided the selection is
disclosed with the full x-of-k pass-rate table, not just the winning seed.

**The 5/7 table this promotion is based on** (measured on the reference cell
`dmhmc x window_adaptation_dense_imm`, reproduced from the entry above):

| seed | provenance | verdict | R-hat | min-ESS | off-mode |
|---|---|---|---|---|---|
| 11111 | LRD cert seed | PASS | 1.0057 | 2326.93 | 0/4 |
| 22222 | LRD cert seed | PASS | 1.0022 | 2254.67 | 0/4 |
| 33333 | LRD cert seed | PASS | 1.0027 | 2340.34 | 0/4 |
| 682737 | this cell's recorded `tuning_seed` | FAIL | 2.7738 | 4.63 | 2/4 |
| 682738 | exploratory | PASS | 1.0030 | 2339.34 | 0/4 |
| 682739 | exploratory | PASS | 1.0038 | 2320.00 | 0/4 |
| 20260517 | `RECIPE_SEED`, the general path's default | FAIL | 1.5940 | 6.75 | 1/4 |

The two seeds that fail are exactly the two an unaware rerun would reach for by
default: this cell's own recorded `tuning_seed` (682737) and `RECIPE_SEED`
(20260517), the general emit path's hardcoded default. That is what makes the
disclosure meaningful rather than decorative — a naive re-run of this model
lands on FAIL both ways it could plausibly try.

**Promotion run.** All 8 promotable LOW cells (every in-scope LOW cell except
`mclmc`, which was not part of the 11 recert failures) were re-run with
`init_strategy={"type": "uniform_perchain", "low": -1.5, "high": -0.5}` at
seed=11111 (one member of the passing set above, chosen as the default since
it is already a recognised seed elsewhere in the corpus), through the normal
production emit path (`emit_low_recipe_for_cell`), with every other parameter
(step policy, target_acceptance=0.99, n_warmup=1000, n_samples=1000,
num_chains=4, warmup_inner_kernel where applicable) held identical to the
committed LOW recipe:

| cell | verdict @ seed 11111 | R-hat | min-ESS |
|---|---|---|---|
| `dmhmc` + `window_adaptation_dense_imm` | PASS | 1.0019 | 3844.4 |
| `dmhmc` + `window_adaptation_low_rank_imm` | PASS | 1.0065 | 736.1 |
| `dynamic_hmc` + `window_adaptation_dense_imm` | PASS | 1.0028 | 2029.9 |
| `dynamic_hmc` + `window_adaptation_low_rank_imm` | PASS | 1.0022 | 2385.8 |
| `hmc` + `window_adaptation_diag_imm` (`inner_nuts`) | PASS | 1.0056 | 1405.1 |
| `hmc` + `window_adaptation_low_rank_imm` (`inner_nuts`) | PASS | 1.0056 | 1405.1 |
| `nuts` + `window_adaptation_low_rank_imm` | PASS | 1.0042 | 2221.9 |
| `hmc` + `window_adaptation_dense_imm` (`inner_nuts`) | REVIEW (rhat=1.0134) | — | — |

7 of 8 cleared PASS at seed 11111 on the first attempt and are promoted;
`medium__<method>__<warmup>__gt_informed_init.json` recipes are committed for
each (`notes` field carries the same disclosure as this entry).

**`hmc` + `window_adaptation_dense_imm` + `inner_nuts` does not promote.**
Extended to a 7-seed scan (11111, 22222, 33333, 44444, 55555, 682738, 682739),
all at the same GT-informed init, this cell never clears a clean PASS:

| seed | verdict | R-hat | min-ESS | divergences |
|---|---|---|---|---|
| 11111 | REVIEW | 1.0134 | 14408.2 | 0 |
| 22222 | FAIL | 1.6361 | 6.6 | 0 |
| 33333 | FAIL | 1.8026 | 5.9 | 0 |
| 44444 | FAIL | 1.6007 | 6.6 | 1000 |
| 55555 | REVIEW | 1.0201 | 14408.2 | 0 |
| 682738 | FAIL | 1.6628 | 6.5 | 0 |
| 682739 | REVIEW | 1.0216 | 14408.2 | 0 |

0 of 7 clean PASS (3 REVIEW, 4 FAIL). This is a genuinely harder cell than the
other 7 dense/low_rank/diag combinations at the same init and step policy —
the dense-IMM + `inner_nuts` warmup path has a narrower or differently-shaped
basin of attraction than its diag/low_rank siblings, which is consistent with
the corpus-wide dense-mass-matrix fragility pattern discussed in Belief#1166 /
Belief#1172. Per Belief#1176's "no gate relaxation" clause, none of the REVIEW
results are promoted. This cell **stays at LOW** (its existing, unreliable
`low__hmc__window_adaptation_dense_imm__inner_nuts.json` recipe) with no
MEDIUM counterpart; it is a candidate for either a wider seed/box scan or the
harness-level fix (Issue#255 / Issue#994) before a HIGH-effort escalation is
warranted.

## Citations

**Real-data model**: Lotka-Volterra predator-prey ODE with real population data
