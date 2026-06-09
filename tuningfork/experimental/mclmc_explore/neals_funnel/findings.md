# Position-Dependent Curvature and Funnel Failures in MCLMC

This document records the empirical results and mathematical conclusions from stress-testing the MCLMC family on `neals_funnel`, the quintessential varying-curvature geometry.

---

## 1. The Funnel Topology

Neal's Funnel consists of a variance hyperparameter and a set of latent parameters drawn from a Normal distribution scaled by that hyperparameter.
- **The Mouth**: When the hyperparameter is large, the latents are wide and unconstrained. The optimal Hamiltonian step size is large.
- **The Neck**: When the hyperparameter shrinks, the latents are squeezed into a tiny, highly correlated valley. The local curvature is extreme, requiring a microscopic step size to prevent integration errors.

### Why Global Preconditioning (IMM) Fails
A global Inverse Mass Matrix (IMM)—whether diagonal, dense, or low-rank—is an **affine transformation**. It applies the exact same geometric scaling everywhere in the parameter space. It cannot fix a funnel; it merely tilts or squashes the funnel globally.

Because `mclmc` uses a single global step size to conserve energy, as soon as it wanders from the mouth into the neck, the local curvature tightening causes massive energy spikes. The integrator must either:
1. Diverge and explode (if unadjusted).
2. Reject nearly all proposals (if using `adjusted_mclmc` with a Metropolis-Hastings correction).

---

## 2. Experimental Showdown

We ran an automated showdown on `neals_funnel` using the MH-corrected variants to test if they could serve as a robust safety net.

### `adjusted_mclmc` (Fixed Trajectory Length)
- **10K Warmup Result**: Max R-hat = **1.0821**, Min ESS = **33.1** (FAIL)
- **Diagnosis**: The MH correction successfully prevents outright divergence (producing no NaNs), but the sampler gets trapped rejecting proposals in the neck of the funnel.

### `adjusted_mclmc_dynamic` (Randomized Trajectory Length)
- **50K Warmup Result**: Max R-hat = **1.0552**, Min ESS = **70.0** (FAIL)
- **Diagnosis**: Randomizing the trajectory length significantly improves exploration (preventing the sampler from getting stuck in periodic orbits/U-turns in the funnel), doubling the ESS. However, the varying curvature is too extreme; it still fails to clear the strict (R-hat < 1.01, ESS > 100) auto-gate.

---

## 3. Scientific Conclusion

These results firmly establish the **Curvature-Routing Hypothesis**:
Funnels and hierarchical models with extreme position-dependent curvature are structural "Honest Nulls" for any sampler that uses a global mass matrix and a single global step size.

- **Safety, not Speed**: The adjusted variants act as a robust safety net. They fail gracefully (yielding low ESS without diverging), but they are not viable defaults for funnels.
- **Routing Rule**: Any model with known funnel-like geometries must be explicitly routed to specialized samplers (e.g., NUTS, or algorithms employing Riemannian metrics or non-centered parameterizations).
