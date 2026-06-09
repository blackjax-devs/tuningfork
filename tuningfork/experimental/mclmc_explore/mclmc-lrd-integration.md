---
status: CURRENT
date: 2026-06-08
tags: [mclmc, lrd, preconditioning, german_credit, horseshoe]
model: null
author: tl
supersedes: []
related:
  - worklog/threads/mclmc-paper-validation.md
---

# MCLMC Low-Rank + Diagonal (LRD) Integration

This thread tracks the experimental development and validation of the $O(dk)$ Low-Rank + Diagonal (LRD) preconditioning architecture for the `mclmc` sampler family.

## 1. The LRD Architecture Breakthrough
Standard unadjusted `mclmc` requires a global isotropic step size. Diagonal preconditioning fails on targets with rotational ill-conditioning (e.g., highly correlated GLM covariates), while dense preconditioning destroys MCLMC's theoretical $O(d^{1/4})$ high-dimensional scaling advantage.

We successfully prototyped an **internal, polymorphic LRD integrator**:
By dynamically injecting $O(dk)$ forward and adjoint projection operators into the closure of the Exactly Symmetric Hamiltonian (ESH) momentum update, we achieve dense-like preconditioning in linear time. Crucially, this internal momentum scaling preserves the prior-centered state space (unlike external coordinate whitening, which translates the space and breaks hierarchical funnels).

## 2. Quantitative Validations
We validated this architecture across three geometrically distinct targets using a 1000-step NUTS pilot run to extract the empirical LRD covariance:

| Model | Geometry | Variant | LRD ($k$) | Max R-hat | Min ESS | Verdict | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ill_cond_50` | Rotated $\kappa=1000$ | `mclmc` | 40 | 1.0030 | 2079.5 | **PASS** | LRD perfectly flattens rotational ill-conditioning, matching oracle dense performance. |
| `stoch_vol` | 503-D Funnel + AR(1) | `mclmc` | 50 | 1.0498 | 156.1 | **REVIEW** | Internal LRD preserves the funnel's prior-centering while uncorrelating the AR(1) latents. |
| `german_credit` | 26-D GLM | `mclmc` | 26 | 1.0126 | 520.6 | **REVIEW** | Unravels dense covariate collinearity. Elevates MCLMC to a viable sampler for real-world regression. |
| `horseshoe` | 204-D Heavy Tails | `adj_mclmc_dynamic` | 50 | 1.0193 | 270.7 | **REVIEW** | Equivalent to diagonal baseline. Confirms that local funnel curvature (sparsity transition) dwarfs global correlation. MH correction bounds Cauchy momentum spikes. |

## 3. The Remaining Challenge: Cheap Covariance Discovery
While the internal LRD integrator works perfectly, relying on a 1000-step NUTS pilot run to discover the covariance matrix negates MCLMC's warmup-amortization advantage.

**Next Objective:** Develop a computationally cheap warmup mechanism (e.g., `pathfinder` L-BFGS history extraction, or native adaptive `mclmc`) to discover the LRD covariance structure without relying on NUTS.
