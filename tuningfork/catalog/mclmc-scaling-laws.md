---
status: CURRENT
date: 2026-06-16
tags: [mclmc, warmup, scaling-law, eevpd, routing]
author: statistician
supersedes: []
---

# MCLMC warmup scaling laws and the EEVPD funnel diagnostic

Companion to [`mclmc-routing-taxonomy.md`](mclmc-routing-taxonomy.md). Where the
taxonomy assigns each model to a routing category **by hand**, this note gives the
empirical scaling laws behind MCLMC warmup tuning and a **dynamic, NUTS-free signal
that lets the warmup classify the geometry itself** — dissolving the "you don't know
it's a funnel until you sample" chicken-and-egg.

Derived from a groundtruth-scaffolded study (2026-06-16): MCLMC run at the
groundtruth inverse-mass-matrix (dense, from the certified reference draws), swept
across step size, dimension, and model.

## 1. The √d law (smooth targets)

At a well-conditioned (matched-dense) IMM, MCLMC's tuned operating point scales as a
clean power law in dimension `d`:

| quantity | scaling |
|---|---|
| `mclmc step_size` | **≈ 1.22 · √d** |
| `mclmc L` (decoherence length) | **≈ 0.85 · √d** |
| `nuts ε` | ≈ d^−¼ (classic) |
| `nuts` trajectory length `T = ε·L` | ≈ constant (~3.8, d-independent) |

Consequences:
- **`desired_energy_var = 5e-4` is ≈principled, not arbitrary.** Sweeping step size at
  fixed GT IMM, 2nd-moment bias is U-shaped; the EEVPD=5e-4 step lands at the
  upper edge of the bias-minimum plateau (≈3–5% residual bias) while maximizing
  ESS/grad. The energy-variance tuner therefore produces the √d scaling
  **automatically**, from dimension-independent constants — no NUTS, no d-knowledge.
- **MCLMC out-scales NUTS as `step_mclmc/ε_nuts ≈ 1.1·d^{3/4}`** (a power law, not a
  constant): MCLMC's larger admissible step grows with d, the root of its
  O(d^{1/4}) advantage.

This law is validated on real posteriors: `german_credit` (d=26) and **`irt_1pl`
(d=500)** both follow `step ≈ 1.22√d` (residual 0.97 / 0.95) with EEVPD ≈ 5e-4.

## 2. The EEVPD funnel diagnostic (automatic routing)

The √d law **breaks** on funnel / heavy-tail geometry — and the break is detectable
during warmup, NUTS-free:

| geometry | EEVPD at converged step | step / (1.22√d) |
|---|---|---|
| **Smooth** (german_credit, irt_1pl, ill_cond_50 @ LRD) | ≈ 5e-4 (the target) | ≈ 1.0 |
| **Funnel / heavy-tail** (eight_schools, irt_2pl, stoch_vol, horseshoe) | **≫ 5e-4** (9e-3 … 1e-1) | **≪ 1** (down to 0.02) |

**Why:** a funnel has *position-dependent* curvature (the neck is sharp, the mouth is
flat). A global step that survives the neck is far too small for the mouth, so the
energy-variance tuner cannot reach its target with any global step — it shrinks the
step to survive the neck and the achieved EEVPD stays well above 5e-4. No global IMM
(diagonal, low-rank, or dense) removes this; only reparameterization (e.g. NCP) does.

**Use it as a warmup-time router:**

> Run a cheap diagonal-MCLMC pilot. If EEVPD → ≈5e-4 and the adapted step ≈ 1.22√d →
> **smooth**: proceed with (LRD-)MCLMC. If EEVPD stalls ≫ 5e-4 and the step collapses
> below √d → **position-dependent geometry**: the global-metric MCLMC path won't fit
> it; route to the funnel/heavy-tail handler or to reparameterization.

This is the automatic counterpart to the hand-assigned taxonomy categories — the
geometry is classified from the pilot's own energy error, with no NUTS pilot and no
oracle covariance.

## 3. IMM rank guidance (refines the taxonomy's k-selection)

`κ_eff(k)` — the condition number after applying a rank-`k` LRD IMM — predicts the
rank a model needs. The practical knee is **`κ_eff ≲ 5`**, not 1: ESS/grad reaches
~87% of the dense ceiling once `κ_eff` drops to single digits. On `ill_cond_50`
(κ=1000) that is **k* ≈ 40**, matching the certified LRD recipe.

## 4. Funnel handling — a NUTS-free strategy

Once EEVPD flags a funnel/heavy-tail target (§2), the family has a graceful handler
**without** routing to NUTS: **`adjusted_mclmc_dynamic`** at the same GT IMM. Its
Metropolis correction turns unadjusted MCLMC's *silent* bias into honest, lower-bias
sampling (2nd-moment-bias reductions vs unadjusted, at GT-dense IMM):

| model | unadjusted bias | adjusted_dynamic bias | reduction |
|---|---|---|---|
| horseshoe (d=204, heavy tail) | 1.87 | **0.30** | 6.3× (+ 18× min-ESS) |
| irt_2pl (d=144) | 0.17 | 0.07 | 2.3× |
| stoch_vol (d=503) | 0.22 | 0.13 | 1.7× |
| eight_schools_ncp (d=10) | 0.06 | 0.04 | 1.4× |

On *reparameterized* funnels this is **competitive with NUTS, NUTS-free**:
eight_schools_ncp adjusted_dynamic reaches bias 0.042 (vs NUTS 0.076) at ~66% of
NUTS's ESS/grad.

**The hard limit is universal, not MCLMC-specific.** On a *clean centered* funnel
(`neals_funnel`, exact GT), *every* sampler fails: unadjusted MCLMC bias 0.999
(silent, no flag), adjusted_dynamic 0.970, and **NUTS 0.883 with 2.1% divergences**.
NUTS degrades most gracefully and flags the problem, but does not solve it —
**reparameterization (NCP) is the only fix**, which the EEVPD signal tells you to do.
(`adjusted_mclmc_dynamic` uses a *random*, not U-turn, trajectory length, so it is not
a full NUTS analog; a true U-turn-terminated MCLMC is an open research direction for
closing the residual gap on hard funnels.)

**Net routing (NUTS-free):** smooth → (LRD-)MCLMC; funnel/heavy-tail (EEVPD ≫ 5e-4)
→ `adjusted_mclmc_dynamic`; centered/deep funnel (still EEVPD ≫ 5e-4 *and* adjusted
stuck) → reparameterize. This replaces the taxonomy's static "funnels → NUTS."

## Caveats

- The laws are asymptotic in d (tiny-d models like `logistic_synthetic` at d=3 sit
  below the asymptotic regime and need not match the √d prefactors).
- Unadjusted MCLMC is asymptotically *biased*; the √d operating point carries ~3–5%
  2nd-moment bias on smooth targets. Use adjusted variants where exactness matters.
- Provenance: sandbox study, mostly single-seed; intended as engineering guidance and
  a routing signal, not a certified benchmark result.
