---
status: CURRENT
date: 2026-06-19
tags: [mclmc, scaling-law, warmup, tuning, adjusted-mclmc, trajectory-length]
author: statistician, tech-writer
supersedes: []
---

# MCLMC Warmup Scaling Laws and the Trajectory-Length Trade-off

Companion to [`mclmc-routing-taxonomy.md`](mclmc-routing-taxonomy.md) — the empirical
warmup scaling laws enabling NUTS-free, constant-free tuning of adjusted MCLMC.
For model-specific findings, see individual `catalog/<model>/lessons.md` files.

---

## 1. The √d Law (Smooth Targets)

On smooth, well-conditioned targets, MCLMC's tuned operating point follows clean power laws.
**These laws apply to smooth fixed-Σ models only.** On funnels/heavy-tails, the step law breaks:
EEVPD cannot reach 5e-4 with any global step → step collapses below √d (automatic funnel detector).

| quantity | law | validation |
|---|---|---|
| `step_size` | **≈ 1.22 · √d** | S3 iso sweep (d^0.49); validated german_credit 0.97, irt_1pl 0.95. |
| **decoherence length L_dec (unadjusted)** | **≈ 0.85 · √d** | Continuous-time momentum-decoherence scale; unadjusted MCLMC target. |
| `desired_energy_var` (EEVPD) | **5e-4** | Bias-minimum plateau on smooth (3–5% residual 2nd-moment bias). |

**Two distinct L's:** Naive use of L_dec gives avg ≈ 0.70 < 1 (MALA collapse, one leapfrog per trajectory).
The **avg=2 override sets L = 2·step ≈ 2.44√d** (distinct, empirically optimal) to escape this. These are
orthogonal findings; do NOT conflate (research thread line 683).

---

## 2. Geometric Stiffness → Mixing Degradation (avg=2, Fixed L)

At default avg=2, mixing degrades monotonically with stiffness:

| Model | Geometry | d | ESS (avg≈2) | Rhat | Notes |
|---|---|---|---|---|---|
| mvn_10 | Diagonal | 10 | 2712 | 1.007 | Baseline. |
| german_credit | GLM, correlated | 26 | ~5–8k | 1.01 | Mild curvature. |
| ill_cond_50 | Rotated κ=1000 | 50 | 1937 | 1.014 | Requires LRD. |
| banana | Curved low-d | 2 | 37 | 1.162 | High curvature. |
| horseshoe | Funnel high-d | 204 | 5 | 2.699 | Position-dependent. |

On fixed-Σ models, degradation is smooth. Funnels break the law (step collapses).

---

## 3. The avg=2 Default is Validated; Larger avg is Geometry-Opposite

Sweeping avg (2, 6, 18, 54, 108) reveals **optimal avg is model-dependent and opposite in direction:**

| Model | Geometry | d | Optimal avg | L-direction | Status | Details |
|---|---|---|---|---|---|---|
| **ill_cond_50** | Stiff (κ=1000) | 50 | **avg=2** | **SHORT L** | avg=2 best; larger avg ↑bias monotone | See `catalog/ill_cond_50/lessons.md` |
| **banana** | Curved low-d | 2 | **avg∈[18,54]** | **INTERMEDIATE-WINDOW** | Clean PASS window avg=18–54; avg=108 overshoot | See `catalog/banana/lessons.md` |
| **mvn_10** | Isotropic | 10 | **avg=2** | **SHORT L** | Bias-indifferent (MC noise); mixing best at avg=2 | See `catalog/mvn_10/lessons.md` |
| **horseshoe** | Funnel high-d | 204 | **longest avail.** | **LONGEST (insufficient)** | ESS improves monotone but Rhat plateaus REVIEW; geometry-hard | See `catalog/horseshoe/lessons.md` |

**There is no universal avg.** Short L works for stiff/isotropic; intermediate for curved; funnels
never reach PASS. See [mclmc-routing-taxonomy.md §4](mclmc-routing-taxonomy.md#4-the-five-category-routing-architecture).

---

## 4. Auto-Tuning ESS-Only is Unsafe; 2nd-Moment-Bias Gate Required

An adaptive tuner optimizing ESS only will converge to silently-biased configs. The canonical example
is **banana avg≈8 (scoped sweep, S_long / S_search), which ESS-only search selects in all 3 seeds:**

**The trap:** Rhat 1.002–1.004, minESS 681–1009, divergence 0, acceptance ~0.96 — all green. Yet 2 of 3
seeds show 2nd-moment bias ≈0.28 (above 0.1 gate); aggregated 0.207. **The crucial reproducible fact:
S_search (mixing-only minimization) selects exactly this avg≈8 in all 3 seeds** — deterministically walks
into the biased region. Per-seed bias swings (0.278, 0.288, 0.056) are high-variance, but that the search
lands here seed-robustly is the failure mode.

**Safe protocol:** L-selection must gate on independent 2nd-moment bias verification:

1. Fit step-size on EEVPD.
2. Estimate L candidates.
3. **For each L, measure 2nd-moment bias vs ground-truth reference.**
4. **Reject any L where bias > 5–10% of reference.** Optimize ESS only among passing L.

Not currently implemented in `adjusted_mclmc_dynamic`; this is the missing safeguard.

---

## 5. Why Funnels Break MCLMC (All Variants)

When EEVPD cannot reach 5e-4 with any global step (step ≪ √d), the model is funnel-like: position-dependent
curvature requires different step sizes in different regions. A single L or preconditioning matrix cannot resolve this.

Unadjusted MCLMC diverges in the neck; adjusted variants fail safely (Rhat > 2). LRD provides no benefit
— the bottleneck is local, not global correlation.

**Remedy:** Reparameterization (NCP), Riemannian MCMC, or NUTS.
See [mclmc-routing-taxonomy.md §4 Category D](mclmc-routing-taxonomy.md#4-the-five-category-routing-architecture).

---

## References

- Per-model calibration: `catalog/<model>/lessons.md`
- Sampler selection: [mclmc-routing-taxonomy.md](mclmc-routing-taxonomy.md)
- EEVPD pilot: [mclmc-routing-taxonomy.md §5](mclmc-routing-taxonomy.md#5-automatic-classification-during-pilot)
- Implementation: `tuningfork/base_method/mclmc.py`, `blackjax/adaptation/mclmc_adaptation.py`
