---
status: CURRENT
date: 2026-06-09
tags: [mclmc, lrd, routing, preconditioning, taxonomy]
author: swe
supersedes: [experimental/mclmc_explore/findings.md, experimental/mclmc_explore/mclmc-lrd-integration.md, experimental/mclmc_explore/README.md, experimental/mclmc_explore/recipes-mclmc-cat-c.md]
---

# MCLMC Routing Taxonomy and LRD Preconditioning

This document is the permanent, production-indexed synthesis of the MCLMC deep-dive
investigation conducted 2026-06-08/09 (tuningfork PR #176, blackjax PR #936).  It
consolidates four experimental files that were deleted from `experimental/` when PR #176
landed on main.

For model-specific findings see the individual `catalog/<model>/lessons.md` files.
For the LRD utility API see `tuningfork/base_method/mclmc.py` and
`tuningfork/base_method/mclmc_lrd_utils.py`.

---

## 1. The LRD Integrator (blackjax upstream)

Standard diagonal MCLMC fails on targets with rotational ill-conditioning.  Dense
preconditioning (O(d³)) destroys MCLMC's O(d^{1/4}) scaling advantage.

The solution: an **O(dk) Low-Rank + Diagonal (LRD) integrator** that injects polymorphic
forward and adjoint projection operators directly into the ESH momentum update closure:

- `adjoint_L(g) = (I + U(√Λ−I)Uᵀ)(σ⊙g)` — gradient → whitened gradient
- `forward_L(y) = σ⊙(y + U(√Λ−I)Uᵀy)` — normalized momentum → position velocity

with L_LR = diag(σ)(I + U(√Λ−I)Uᵀ) the Cholesky-like square root of M⁻¹.

**Mathematical guarantee**: L_LR is position-independent (constant matrix), so the
change-of-variables y = L_LR⁻¹ x is a bijection with constant Jacobian.  The
invariant measure is provably preserved for ALL targets — log-concave or not.
(Numerical check: L_LR @ L_LR^T = M⁻¹ at rel. err 2.8e-7.)

Implementation: `blackjax.mcmc.integrators.isokinetic_mclachlan` dispatches natively
on `blackjax.mcmc.metrics.LowRankInverseMassMatrix` (blackjax PR #936).

The **tuningfork entry point** is `make_lrd_kernel(lrd_imm)` in
`tuningfork/base_method/mclmc.py` — a thin closure that statically binds the LRD mass
matrix so that `mclmc_find_L_and_step_size(diagonal_preconditioning=False)` receives the
correct geometry during warmup.

---

## 2. Validated LRD Results (by model)

| Model | Geometry | Variant | k | Max R-hat | Min ESS | ESS/grad | Verdict |
|---|---|---|---|---|---|---|---|
| `ill_cond_50` | Rotated κ=1000 | Internal LRD mclmc | 40 | **1.0030** | **2079.5** | **0.249** | **PASS** |
| `german_credit` | 26-D GLM | Internal LRD mclmc | 26 | 1.0126 | 520.6 | — | REVIEW |
| `stoch_vol` | 503-D Funnel+AR(1) | Internal LRD mclmc | 50 | 1.0498 | 156.1 | — | REVIEW |
| `stoch_vol` | 503-D Funnel+AR(1) | External LRD mclmc | 50 | 2.0536 | 5.4 | — | FAIL |
| `horseshoe` | 204-D Cauchy heavy tails | Internal LRD adj_mclmc_dyn | 50 | 1.0193 | 270.7 | — | REVIEW |

**Certified PASS**: `ill_cond_50` with internal LRD k=40.  Statistician multi-seed
hardening: seeds 11111/22222/33333 all PASS at ESS 1944–2030 (2026-06-09).
See `catalog/ill_cond_50/recipes/low__mclmc__mclmc_tuning.json`.

---

## 3. The Integrator Ladder (ill_cond_50 development history)

The four stages, each in `tests/mclmc_lrd/`:

| Stage | File | Description |
|---|---|---|
| 1 | `test_dense_mclmc.py` | Dense Cholesky oracle (O(d²)) — baseline upper bound |
| 2 | `test_low_rank_mclmc.py` | External LRD coordinate-whitening (Oracle, rank k=10..40) |
| 3 | `test_adaptive_lrd.py` | Adaptive external LRD (NUTS-pilot → SVD → whitened MCLMC) |
| 4 | `test_internal_lrd.py` | **Internal LRD MCLMC (production path)** — no logdensity wrapping |

The internal integrator matches and slightly exceeds the external Adaptive LRD
(2079.5 ESS vs 1776.9 ESS) because it avoids PyTree flattening and float-precision
boundary effects during logdensity calls.

---

## 4. The Five-Category Routing Architecture

Based on empirical stress-testing across the full catalog:

**Category A — Isotropic / Weakly-Correlated High-D** (e.g., `lgcp` 1600-D, `irt_1pl` 500-D)
- Route to: `mclmc` (diagonal preconditioning)
- Why: Fully realize the O(d^{1/4}) scaling advantage without MH rejection overhead.

**Category B — Correlated Regression / GLMs** (e.g., `german_credit`, `ill_cond_50`)
- Route to: `mclmc` with LRD preconditioning
- Pipeline: Short NUTS Pilot → SVD Extraction → `make_lrd_kernel` → `mclmc_find_L_and_step_size`
- Why: Resolves rotational ill-conditioning in O(dk) without dense-matrix overhead.

**Category C — Heavy Tails / Sparsity** (e.g., `horseshoe`)
- Route to: `adjusted_mclmc_dynamic`
- Why: Cauchy tails do NOT destabilize ESH dynamics. The MH correction provides
  stability. LRD provides no benefit — the bottleneck is local sparsity funnel
  curvature, not global correlation.

**Category D — Hierarchical Funnels** (e.g., `neals_funnel`, `radon`, `irt_2pl`)
- Route to: **NUTS** (MCLMC family is an honest null)
- Why: A global affine IMM transformation cannot resolve position-dependent curvature.
  Unadjusted MCLMC diverges in funnel necks; adjusted variants fail safely but are
  stuck. Any model with funnel-like geometry must use NUTS or Riemannian methods.

**Category E — Multimodal / Stiff ODEs** (e.g., `gmm_25`, `lotka_volterra`)
- Route to: **SMC** or **NUTS** (MCLMC family is an honest null)
- Why: Trajectory integrators cannot cross near-zero density barriers or survive
  extreme Lipschitz constants without microscopic step sizes.

---

## 5. The VI Rank-Collapse Finding (negative result)

Tested whether `multipathfinder` could replace the NUTS pilot run for cheap geometry
discovery.

**Result: catastrophic rank deficiency**
- `german_credit` (26-D): covariance collapses to Rank 1
- `ill_cond_50` (50-D): covariance collapses to Rank 6

**Root cause**: L-BFGS builds inverse Hessian approximations at the MAP mode,
explicitly discarding curvature along dimensions where the objective isn't actively
changing. Multiple paths converge to the same mode (strong likelihood basins funnel
all paths), producing nearly-collinear endpoint vectors with rank-deficient covariance.

**Theoretical conclusion**: Optimization/VI is **structurally incapable** of capturing
the global, elongated covariance of the typical set.  The L-BFGS history is an
approximation to the local Hessian at MAP — it knows nothing about the typical set
spread.  Even with multiple paths (PSIS weighting), strong mode-seeking produces
redundant, degenerate covariance estimates.

**Contrast with NUTS**: detailed balance forces a Hamiltonian chain to physically trace
the full volume of the typical set.  A short NUTS pilot is the minimum viable geometry
discovery step.

See `tests/mclmc_lrd/test_multipathfinder_lrd.py` for the runnable script.

---

## 6. Prior-Centering Preservation Principle

Validated on `stoch_vol` (503-D NCP AR(1)):

- **External coordinate whitening** (FAIL): applies spatial translation x = L_LR(y) + μ.
  Shifting latent coordinates by the posterior mean breaks the prior-centered AR(1)
  structure h_t ~ N(φ h_{t-1}, σ²). Step-size collapses immediately.

- **Internal LRD preconditioning** (REVIEW): rescales and rotates **momentum vectors
  only**, never translating position coordinates. The AR(1) prior structure is
  preserved, enabling partial improvement over diagonal.

**Rule**: For any model with prior-centered NCP (hierarchical models, AR processes,
Gaussian processes), use internal LRD only. Even then, expect REVIEW when non-linear
funnel curvature is the dominant bottleneck.

---

## 7. Key Parameter Notes

### LRD mass matrix construction
From a NUTS pilot (1000 steps) via `run_pilot_nuts` + `extract_lrd_from_samples`:
1. Flatten samples to (N, d) matrix
2. Compute empirical σ (standard deviations per dimension)
3. Standardize and compute SVD: X_std = U S Vᵀ
4. Eigenvalues: λ = S²/N; sort by |λ-1| descending
5. Take top k: `LowRankInverseMassMatrix(sigma, U_k, lam_k)`

### Step-size scaling for adjusted samplers
`mclmc_find_L_and_step_size` (unadjusted warmup) adapts a large step_size.
For `adjusted_mclmc_dynamic`, scale by **0.55** to target ~94% acceptance.
This empirical scaling factor was validated on horseshoe (204-D).

### Rank selection guidance
- k ≥ 20 is sufficient for REVIEW on `ill_cond_50` (κ=1000)
- k = 40 achieves PASS on `ill_cond_50` (captures ~92% Frobenius norm of Σ)
- For d-dimensional GLMs, k=d is full-rank (no scaling advantage; validates pipeline)

---

*This document is the permanent synthesis of the MCLMC LRD integration experiment.*
*Polish pass assigned to @tech-writer (tuningfork PR #176 scope note).*
