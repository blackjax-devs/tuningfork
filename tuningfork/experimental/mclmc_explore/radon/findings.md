# Residual Varying Curvature and Preconditioning Barriers in MCLMC: The `radon` Case Study

This document details the mathematical analysis, empirical outcomes, and scientific conclusions from evaluating the MCLMC family on the 390-dimensional Hierarchical Non-Centered Parameterization (NCP) model `radon`.

---

## 1. The Radon Hierarchical Model and NCP

The `radon` model is a hierarchical regression model of home radon levels with 391 parameters, capturing county-level random effects across 85 counties in Minnesota. Even though Non-Centered Parameterization (NCP) is employed to decouple the county-level random effects from their group-level standard deviation hyperparameter:
- Residual varying curvature still persists due to unequal county-level sample sizes (some counties have many observations, constraining their random effects, while others have very few, leaving them highly unconstrained and close to the group-level prior scale).
- The coordinate axes have highly unequal variance scales (due to the differences in county sample sizes and regression coefficient scales), requiring a diagonal mass matrix (preconditioning) to rotate and scale the space for isotropic exploration.

---

## 2. Quantitative Evaluation Table

We executed comparative runs on CPU with 4 chains and 1,000 sampling steps under master seed `20260517` against certified ground truth draws.

| Sampler / Configuration | Warmup Steps ($n_{\text{warmup}}$) | Max Split-$\hat{R}$ | Min Bulk ESS | Divergences | Auto-Gate Verdict | Status / Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Diagonal NUTS** (Baseline) | 1,000 | **1.0067** | **347.2** | 0 | **REVIEW** (Pass) | **Successful Baseline**. Cleared the auto-gate, achieving excellent mixing. |
| **Unadjusted `mclmc`** | 1,000 | **4.1396** | **4.3** | 0 | **FAIL** | Complete failure. Severe step-size collapse and chain exploration collapse. |
| **`adjusted_mclmc`** (Static) | 10,000 | **1.1456** | **21.6** | 0 | **FAIL** | MH correction prevents divergence but fails safely with low ESS. |
| **`adjusted_mclmc_dynamic`** | 10,000 | **1.1086** | **32.1** | 0 | **FAIL** | Randomized trajectory length improves exploration slightly but still fails. |

---

## 3. Scientific and Structural Discoveries

Evaluating `radon` exposes **two critical structural limitations of standard/adjusted MCLMC**:

### 1. The Necessity of Diagonal Preconditioning (IMM Adaptation)
The most striking contrast is between **Diagonal NUTS** (R-hat 1.0067, ESS 347.2) and **adjusted MCLMC** (R-hat 1.1086, ESS 32.1).
- Standard NUTS adapts a diagonal Inverse Mass Matrix (IMM) during window adaptation, rescaling the 390-dimensional coordinate axes to be isotropic.
- Standard and adjusted MCLMC do NOT adapt a diagonal mass matrix (the mass matrix remains isotropic, i.e., $1.0$). In a 390-dimensional space with highly unequal coordinate scales, an isotropic proposal or momentum update is highly misaligned with the target's geometry. The sampler is forced to adopt an extremely small effective step size or suffer from high rejections, decimating its sampling efficiency.

### 2. The MH Safety Net "Fails Safely"
While unadjusted `mclmc` fails catastrophically with an R-hat of 4.14 and ESS of 4.3 (representing complete chain stagnation), the MH-corrected adjusted variants (`adjusted_mclmc` and `adjusted_mclmc_dynamic`) "fail safely":
- They generate **0 divergences** and zero NaNs.
- They restrict the maximum R-hat to around **1.10 - 1.15**.
While they do not rescue the target (R-hat remains above the 1.05 PASS threshold), the MH safety net successfully prevents pathological numerical explosions, reporting clear and diagnostic non-convergence.

---

## 4. Curvature Routing Rule

NCP successfully reduces the funnel scale-coupling pathology but does not solve coordinate-scale mismatch. Any high-dimensional hierarchical model (like `radon` with $d > 300$) featuring highly unequal coordinate scales must be explicitly routed to samplers supporting **adaptive diagonal/dense mass matrices** (like NUTS) rather than standard MCLMC.
