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

**Companion:** [`mclmc-scaling-laws.md`](mclmc-scaling-laws.md) — the empirical warmup
scaling laws (`step ≈ 1.22·√d`) and the **EEVPD diagnostic that makes the §4 routing
automatic and NUTS-free** (detects the smooth-vs-funnel boundary from the pilot's own
energy error, dissolving the "you don't know it's a funnel until you sample" problem).

For model-specific findings see the individual `catalog/<model>/lessons.md` files.
For the executable LRD route see the `mclmc_lrd_tuning` emitter in
`tuningfork/recipes/_emit/_warmup.py`.

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

The **tuningfork entry point** is the generated `mclmc_lrd_tuning` route. It emits a
thin closure that statically binds the LRD mass matrix so that
`mclmc_find_L_and_step_size(diagonal_preconditioning=False)` receives the correct
geometry during warmup. The original investigation exposed this closure as
`make_lrd_kernel(lrd_imm)`; codegen now emits the same choreography inline so there is
only one executable route.

---

## 2. Validated LRD Recipe Results

For the empirical √d scaling laws and geometric-stiffness degradation baseline, see
[mclmc-scaling-laws.md §1–2](mclmc-scaling-laws.md#1-the-d-law-smooth-targets).
Recipe-level verdicts (full result numbers in each model's `lessons.md`):

- **PASS** — `ill_cond_50` (rotated κ=1000, 50-D), internal LRD k=40 →
  [`catalog/ill_cond_50/lessons.md`](ill_cond_50/lessons.md) and
  [`low__mclmc_lrd__mclmc_lrd_tuning.json`](ill_cond_50/recipes/low__mclmc_lrd__mclmc_lrd_tuning.json)
- **REVIEW** — `german_credit` (26-D GLM), internal LRD k=26 →
  [`catalog/german_credit/lessons.md`](german_credit/lessons.md)
- **REVIEW** — `stoch_vol` (503-D Funnel+AR(1)), internal LRD k=50 →
  [`catalog/stoch_vol/lessons.md`](stoch_vol/lessons.md)
- **FAIL** — `stoch_vol` (503-D Funnel+AR(1)), external LRD k=50 →
  [`catalog/stoch_vol/lessons.md`](stoch_vol/lessons.md)
- **REVIEW** — `horseshoe` (204-D Cauchy heavy tails), adj_mclmc_dyn LRD k=50 →
  [`catalog/horseshoe/lessons.md`](horseshoe/lessons.md)

**Certified PASS**: `ill_cond_50` with internal LRD k=40. Multi-seed hardening
(seeds 11111/22222/33333) confirmed by statistician — see
[`catalog/ill_cond_50/lessons.md`](ill_cond_50/lessons.md) for full numbers.
Recipe: `catalog/ill_cond_50/recipes/low__mclmc_lrd__mclmc_lrd_tuning.json`.

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

Based on empirical stress-testing across the full catalog and validated against the
[geometric-stiffness degradation law](mclmc-scaling-laws.md#2-geometric-stiffness--mixing-degradation-avg2-fixed-l):

**Category A — Isotropic / Weakly-Correlated High-D** (e.g., `lgcp` 1600-D, `irt_1pl` 500-D)
- Route to: `mclmc` (diagonal preconditioning)
- Why: Fully realize the O(d^{1/4}) scaling advantage without MH rejection overhead.

**Category B — Correlated Regression / GLMs** (e.g., `german_credit`, `ill_cond_50`)
- Route to: `mclmc` with LRD preconditioning
- Pipeline: generated Short NUTS Pilot → SVD Extraction → statically bound LRD kernel
  → `mclmc_find_L_and_step_size`
- Why: Resolves rotational ill-conditioning in O(dk) without dense-matrix overhead.

**Category C — Heavy Tails / Sparsity** (e.g., `horseshoe`)
- Route to: `adjusted_mclmc_dynamic`
- Why: Cauchy tails do NOT destabilize ESH dynamics. The MH correction provides
  stability. LRD provides no benefit — the bottleneck is local sparsity funnel
  curvature, not global correlation.

**Category D — Hierarchical Funnels** (e.g., `neals_funnel`, `radon`, `irt_2pl`)
- Route to: **NUTS** (MCLMC family is an honest null)
- Why: Position-dependent curvature requires different step sizes in different regions.
  A global affine IMM transformation cannot resolve this.
  [See mclmc-scaling-laws.md §5 for empirical failure modes](mclmc-scaling-laws.md#5-why-funnels-break-mclmc-all-variants).
  Unadjusted MCLMC diverges in funnel necks; adjusted variants fail safely but plateau.

**Category E — Multimodal / Stiff ODEs** (e.g., `gmm_25`, `lotka_volterra`)
- Route to: **SMC** or **NUTS** (MCLMC family is an honest null)
- Why: Trajectory integrators cannot cross near-zero density barriers or survive
  extreme Lipschitz constants without microscopic step sizes.

---

## 5. Automatic Classification During Warmup (Exploratory)

For models not in the known catalog, the five-category routing table (§4) is unavailable.
This section describes an exploratory **warmup-time pilot signal** that can automatically
classify a new model into one of these categories, NUTS-free.

**Status**: heuristic, not yet validated on the full catalog. Use as a routing *suggestion*
for unknown models; verify against the per-model `catalog/<model>/lessons.md` once sampling completes.

### The EEVPD Diagnostic

Run a cheap diagonal-MCLMC pilot with the energy-variance tuner
([target `desired_energy_var=5e-4`, validated in mclmc-scaling-laws.md §1](mclmc-scaling-laws.md#1-the-d-law-smooth-targets)).
The achieved EEVPD at convergence **classifies the geometry**:

| EEVPD outcome | step / (1.22√d) | Geometry class | Suggested route |
|---|---|---|---|
| **≈ 5e-4** (target hit) | ≈ 1.0 | Smooth / well-correlated | Category A or B (Diagonal or LRD-MCLMC) |
| **≫ 5e-4** (e.g., 9e-3–0.1) | ≪ 1 (collapses) | Funnel / heavy-tail / position-dependent | Category C or D (adjusted_mclmc_dynamic or NUTS) |

**Why it works**: On smooth targets, the energy-variance tuner can reach its 5e-4 target
with a global step ≈ 1.22√d. On funnels, the step needed to survive position-dependent
curvature (the narrow neck) is too small for the flat regions (the mouth), so the tuner
cannot reach 5e-4 with any global step — it shrinks the step and the achieved EEVPD
stalls well above the target.

### Caveats

- Requires no oracle knowledge or post-hoc reference (runs during cheapest phase)
- But: the threshold (5e-4) is calibrated on smooth/moderately-conditioned targets; very
  high-dimensional or multimodal targets may need tuning
- Use this to *suggest* a route, not to *commit* to one; recheck diagnostics once
  sampling is underway

### Implementation Pseudocode

```python
# Warmup-phase classifier
def classify_geometry(model_logdensity, initial_position, d):
    # Run diagonal MCLMC warmup, target EEVPD=5e-4
    kernel = mclmc(diagonal_preconditioning=True)
    step_size, L, _ = mclmc_find_L_and_step_size(
        kernel, model_logdensity, initial_position,
        desired_energy_var=5e-4, num_steps=1000
    )

    # Compare achieved EEVPD to target
    realized_eevpd = ...  # computed during warmup
    predicted_step = 1.22 * np.sqrt(d)
    step_ratio = step_size / predicted_step

    if realized_eevpd < 2e-3 and step_ratio > 0.7:
        return "smooth", "try_mclmc"
    elif realized_eevpd > 1e-2 and step_ratio < 0.5:
        return "funnel", "try_adjusted_mclmc"
    else:
        return "unclear", "review_diagnostics"
```

### Relationship to the Known-Model Taxonomy (§4)

- **Curated table (§4)**: Ground-truth category assignments for models we've studied.
  Use these when available — they reflect high-confidence routing decisions backed by
  multi-seed validation and per-model reference diagnostics.

- **EEVPD pilot (§5, this section)**: Automatic suggestion for new or unseen models.
  Trade-off: cheaper (no NUTS pilot needed), but exploratory; should be validated
  against per-model diagnostics and `catalog/<model>/lessons.md` references.

### Best Practice Workflow

For a new model:
1. Run the EEVPD pilot first (O(1k) steps).
2. It suggests a route in one of the five categories.
3. Commit to sampling with the suggested handler.
4. Verify diagnostics post-hoc against the per-model `lessons.md` reference (if available).
5. If the [2nd-moment-bias gate](mclmc-scaling-laws.md#4-auto-tuning-ess-only-is-unsafe-2nd-moment-bias-gate-required)
   (see `mclmc-scaling-laws.md` §4) flags issues, escalate to the next-higher handler
   (e.g., diagonal MCLMC → LRD-MCLMC → adjusted_mclmc_dynamic → NUTS).

---

## 6. The VI Rank-Collapse Finding (negative result)

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

## 7. Prior-Centering Preservation Principle

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

## 8. Key Parameter Notes

### LRD mass matrix construction
The generated route emits a 1000-step NUTS pilot followed by SVD extraction:
1. Flatten samples to (N, d) matrix
2. Compute empirical σ (standard deviations per dimension)
3. Standardize and compute SVD: X_std = U S Vᵀ
4. Eigenvalues: λ = S²/N; sort by |λ-1| descending
5. Take top k: `LowRankInverseMassMatrix(sigma, U_k, lam_k)`

The original investigation called these two phases `run_pilot_nuts` and
`extract_lrd_from_samples`; the names are retained here as provenance, but the phases
now exist only inside the emitted recipe program.

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
*Polish pass completed (tuningfork PR #176).*
