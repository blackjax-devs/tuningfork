# MCLMC Advanced Routing and Geometry Discovery: Final Report

This document synthesizes the empirical outcomes and theoretical conclusions from our deep-dive investigation into applying the Microcanonical Langevin Monte Carlo (MCLMC) family to the `tuningfork` registry.

---

## 1. The Low-Rank + Diagonal (LRD) Integrator
We demonstrated that standard diagonal preconditioning fails on targets with rotational ill-conditioning (e.g., highly correlated covariates in Generalized Linear Models like `german_credit`, or synthetic rotated matrices like `ill_cond_50`).

To resolve this without destroying MCLMC's $O(d^{1/4})$ high-dimensional scaling advantage via dense $O(d^3)$ matrices, we implemented an **internal $O(dk)$ Low-Rank + Diagonal (LRD) integrator**.
By injecting polymorphic $O(dk)$ forward and adjoint projection operators directly into the exact ESH momentum update closure, we proved that MCLMC can natively un-rotate complex geometries on the fly.
- *Crucially*, because this internal projection operates purely on momentum/velocity, it preserves the prior-centered state space. This allowed it to successfully sample the highly correlated hierarchical funnels of `stoch_vol`—a geometry where external coordinate whitening fundamentally failed.

**Conclusion:** The internal LRD integrator architecture is mathematically pristine, fully compatible with BlackJAX's existing `LowRankInverseMassMatrix` type, and ready for an upstream PR to `blackjax`.

---

## 2. The Variational Rank Collapse Phenomenon
We tested whether optimization-based Variational Inference (specifically `multipathfinder`) could replace NUTS as an ultra-cheap "Discovery Phase" to extract the LRD covariance matrix.

**Result:** `multipathfinder` exhibited catastrophic rank deficiency. On `german_credit` (26-D), the extracted covariance collapsed to **Rank 1**. On `ill_cond_50` (50-D), it collapsed to **Rank 6**. When fed into the LRD integrator, the sampler completely froze in the zero-variance directions.

**Theoretical Finding:** Optimization engines (L-BFGS) build inverse Hessian approximations *at the MAP mode*, explicitly discarding curvature information along dimensions where the objective isn't actively changing. They are structurally incapable of capturing the global, elongated covariance of the *typical set*. Even with multiple paths (PSIS weighting), strong basins of attraction funnel all trajectories into the same mode, leading to redundant, degenerate covariance estimates.

**Conclusion:** Optimization/VI is fundamentally unsuited for global mass matrix discovery. A short, Hamiltonian MCMC pilot run (e.g., diagonal NUTS) is mathematically indispensable, as detailed balance forces it to physically trace the full volume of the typical set.

---

## 3. The Final `tuningfork` Routing Architecture
Based on our topological stress-testing, we established the definitive, automated routing rules for the MCLMC family:

1. **Category A (Isotropic / Weakly-Correlated High-D):** e.g., `lgcp` (1600-D), `irt_1pl` (500-D).
   - **Route to:** Unadjusted `mclmc` (diagonal preconditioning).
   - **Why:** Absolute champions. They fully realize the $O(d^{1/4})$ scaling advantage without MH rejection bottlenecks.

2. **Category B (Correlated Regression / GLMs):** e.g., `german_credit`, `ill_cond_50`.
   - **Route to:** Unadjusted `mclmc` with LRD Preconditioning.
   - **Pipeline:** Short NUTS Pilot $\rightarrow$ SVD Extraction $\rightarrow$ Internal LRD `mclmc`. Unravels collinearity perfectly.

3. **Category C (Heavy Tails / Sparsity):** e.g., `horseshoe`.
   - **Route to:** `adjusted_mclmc_dynamic`.
   - **Why:** The microcanonical hypersphere naturally bounds the extreme momentum spikes caused by Cauchy tails. The MH correction provides safety. LRD provides no benefit here, as the bottleneck is the local sparsity funnel (transition to zero), not global correlation.

4. **Category D (Hierarchical Funnels):** e.g., `neals_funnel`, `radon`, `irt_2pl`.
   - **Route to:** **NUTS** (MCLMC is an Honest Null).
   - **Why:** Global affine transformations cannot resolve position-dependent curvature. Unadjusted MCLMC explodes in funnel necks; adjusted variants fail safely but get trapped rejecting proposals.

5. **Category E (Multimodal / Stiff ODEs):** e.g., `gmm_25`, `lotka_volterra`.
   - **Route to:** **SMC** or **NUTS** (MCLMC is an Honest Null).
   - **Why:** Trajectory integrators cannot cross near-zero density barriers or survive immense Lipschitz constants without microscopic step sizes.
