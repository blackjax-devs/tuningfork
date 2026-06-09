# MCLMC Deep Dive & Routing Roadmap

This experimental directory contains the proof-of-concept code and model-specific tests for advancing MCLMC in `tuningfork`. Code here is intended to prototype geometry-aware routing and advanced preconditioning without destabilizing the main registry.

## Core Directives

1. **Geometry Profiling for Automated Routing**
   Instead of running full tuning sweeps, we aim to use a cheap warm-start phase (e.g., `pathfinder` or a short VI run) to extract geometric heuristics:
   - *Curvature Variation Metric:* Monitor the variance of the L-BFGS diagonal Hessian approximations or the gradient norms. High variance implies funnels/varying curvature -> route away from unadjusted `mclmc`.
   - *Ill-Conditioning Metric:* Estimate the condition number from the Pathfinder history. If the spectrum is heavily skewed, flag for advanced preconditioning.

2. **Preconditioning Strategy (Rotational Ill-Conditioning)**
   Diagonal mass matrices fail on rotational correlations (e.g., `ill_cond_50`). Dense mass matrices (O(d^2)) are intractable at the dimensions where MCLMC shines (e.g., d=1600).
   - *Goal:* Implement and test **low-rank + diagonal mass matrix adaptation** for MCLMC. This reconstructs the dominant correlation directions at O(dk) cost, rescuing MCLMC on rotated targets without destroying the high-dimensional scaling advantage.

3. **Warmup Amortization & the Adjusted Safety Net**
   - Introduce intent flags (e.g., `optimize_for="time_to_first_sample"`) to default to `mclmc`'s ultra-cheap ~1000 grad warmup for rapid prototyping.
   - Use `adjusted_mclmc` as an automatic fallback (safety net) for models flagged as `hierarchical` or those exhibiting extreme energy divergence spikes during tuning.

## Directory Structure
- `README.md` : This scoping document.
- `__init__.py` : Module initialization, alongside any experimental upstream changes needed (e.g., custom MCLMC tuning logic).
- `[model_name]/` : One folder per model for model-specific testing code (e.g., `ill_cond_50/`).
