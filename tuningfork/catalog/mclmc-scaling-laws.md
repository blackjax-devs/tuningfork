---
status: CURRENT
date: 2026-06-19
tags: [mclmc, scaling-law, warmup, tuning, adjusted-mclmc, trajectory-length]
author: statistician
supersedes: []
---

# MCLMC Warmup Scaling Laws and the Trajectory-Length Trade-off

Companion to [`mclmc-routing-taxonomy.md`](mclmc-routing-taxonomy.md). This document
gives the empirical scaling laws and tuning principles behind MCLMC's default configuration
for the adjusted variants.

---

## 1. The √d Law (Smooth Targets)

**On smooth, well-conditioned targets**, MCLMC's tuned operating point follows a clean
power law in dimension d:

| quantity | scaling | source |
|---|---|---|
| `step_size` | **≈ 1.22 · √d** | Fitted from S3 iso sweep (d^0.49); validated on `german_credit` d=26 (residual 0.97), `irt_1pl` d=500 (residual 0.95). |
| decoherence length L_dec (unadjusted) | **≈ 0.85 · √d** | Same d^0.50 fit; continuous-time momentum-decoherence scale from `mclmc_find_L_and_step_size`. |
| `desired_energy_var` | **5e-4** | Principled: targets bias-minimum plateau (3–5% residual 2nd-moment bias on smooth) |

**Why two L's matter:** The unadjusted decoherence length L_dec ≈ 0.85√d and the adjusted trajectory length
L_traj = 2·step ≈ 2.44√d are distinct: naive use of L_dec gives avg ≈ 0.70 < 1 (MALA collapse); avg=2 override
sets L_traj = 2·step to escape this. (See research thread line 683.)

**Where it breaks:** On funnels/heavy-tails, the step law fails—EEVPD can't reach 5e-4 with any global step
(step << √d); this is an automatic funnel detector.

---

## 2. Geometric Stiffness → Mixing Degradation (avg=2, Fixed L)

At the default avg=2 calibration, mixing degrades monotonically with geometric stiffness:

| Model | Geometry | d | ESS/grad | Rhat |
|---|---|---|---|---|
| `mvn_10` | Diagonal | 10 | ~2.7k | 1.01 |
| `german_credit` | GLM, correlated | 26 | ~5–8k | 1.01 |
| `eight_schools_ncp` | Hierarchical | 10 | ~5–6k | 1.00 |
| `ill_cond_50` | Rotated κ=1000 | 50 | ~2k | 1.01 |
| `banana` | Curved low-d | 2 | ~34 | 1.23 |
| `horseshoe` | Funnel | 204 | ~5 | 2.8 |

Smooth targets scale predictably; funnels and high curvature break the model even with good preconditioning.

---

## 3. Optimal Trajectory-Length is Geometry-Specific

The `avg` parameter controls mean integration steps per trajectory. Optimal avg is geometry-dependent and
opposite across model classes—there is no universal "bigger is better."

| Model | Geometry | Optimal avg | L-direction | Key finding |
|---|---|---|---|---|
| ill_cond_50, mvn_10 | Rotated / diagonal | avg=2 | SHORT L optimal | Longer L monotonically degrades ESS; avg=2 tuned correctly |
| banana | Curved low-d | avg=18–54 | INTERMEDIATE window | Clean PASS at avg=18–54; avg=108 shows overshoot (acceptance erodes, ESS saturates) |
| horseshoe | Funnel | avg=longest available | LONG L improves, plateaus | ESS improves 5→735 but asymptotes at REVIEW tier (Rhat ~1.2); geometry-hard limit |

**Per-model full sweeps (avg ladder 2/6/18/54/108):** See [`catalog/<model>/lessons.md`](catalog/<model>/lessons.md)
for dynamic-L details (5-row tables, per-seed breakdowns).

---

## 4. Auto-Tuning ESS-Only is Unsafe — 2nd-Moment-Bias Gate Required

The EEVPD energy-variance tuner minimizes `|E[ΔK] - 5e-4|` to select step_size—a *mixing* objective.
Risk: latent bias hides inside high ESS and low Rhat. Example from banana S_long (avg≈8, scoped sweep_stiff_k):

**The ESS-only search strategy (S_search, optimizing trajectory length) selects exactly this avg≈8 in all 3
seeds.** At this operating point:
- All convergence diagnostics: green (Rhat 1.002–1.004, minESS ~900, div 0, acc 0.96)
- Yet 2 of 3 seeds show 2nd-moment bias ≈0.28 (above 0.1 safety gate); aggregated 2mbias 0.207

A mixing-only tuner has walked straight into a silently-biased config that passes every standard diagnostic.

**Solution:** L-selection must gate on independent 2nd-moment bias verification (reference chain with oracle IMM;
reject if bias > 5–10% of reference). See [`catalog/<model>/lessons.md`](catalog/<model>/lessons.md) for per-model
reference biases and safe recipes.

---

## Notes

- √d laws apply to smooth targets only; funnels break them (see boundary, §1).
- Unadjusted MCLMC is asymptotically biased; adjusted variants reduce bias at Metropolis cost.
- Data: n=500/chain × 4 chains × 3 seeds. Max-over-D 2nd-moment-bias is noise-sensitive; read per-cell
  mbias_sd + Rhat as reliable evidence.
- See [mclmc-routing-taxonomy.md §4](mclmc-routing-taxonomy.md#4-the-five-category-routing-architecture)
  for handler decisions (LRD-MCLMC vs adjusted_mclmc_dynamic vs NUTS).
