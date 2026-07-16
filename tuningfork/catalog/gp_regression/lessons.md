# Sampling lessons: gp_regression

## TL;DR

203-D latent-GP regression requires high target_acceptance (0.99) and float64 + jitter ≥1e-4 to avoid silent Cholesky precision failure; posterior has +0.737 correlation between log_lengthscale and log_kernel_scale (identifiability ridge) that diagonal-IMM cannot capture.
[boundary: HIGH-effort NUTS+dense_imm+inner_laplace_hmc PASS at n_warmup=5000, target_acceptance=0.99; elliptical_slice FAIL (hard_direction); meanfield_vi out_of_scope; dense/low_rank IMM not yet tested at MEDIUM; laplace family vmap compile blowup documented elsewhere]

## Canonical recipe

hard-model NUTS (n_warmup=5000, target_acceptance=0.99, max_num_doublings=12; JAX_ENABLE_X64=1, model JITTER=1e-4)

## Sampling quirks

Dense RBF kernel matrix (`200 × 200` points on [0,1]) with float32 + JITTER=1e-6 produces silent NaN in Cholesky gradients (eigenvalue floor below float32 precision); fix is float64 + JITTER≥1e-4. Latent GP requires 4× standard warmup budget (n_warmup ≥5000 for d=203) to adapt diagonal-IMM; 500 steps leaves metric uniform and step_size catastrophically small. Posterior has strong +0.737 correlation between log_lengthscale and log_kernel_scale (classic GP-regression identifiability issue); diagonal-IMM zig-zags along the ridge. Low-rank IMM (rank 2-3) would capture the ridge better but is not yet implemented; workaround is high target_acceptance to keep step_size large. Wall cost O(d² · n_samples) ≈ 50× more expensive per-step than peer models due to dense kernel matrix-vector products (50h budget for full groundtruth certification).

## Known-bad combinations

- `elliptical_slice` + `no_warmup`: **FAIL** (hard_direction — requires Gaussian likelihood matching, which the RBF-GP posterior does not satisfy after conditioning on data). See `recipes/failed__elliptical_slice__no_warmup.json`.
- `meanfield_vi` + `no_warmup`: **FAIL** (out_of_scope — mean-field VI cannot capture the +0.737 lengthscale/kernel-scale correlation). See `recipes/failed__meanfield_vi__no_warmup.json`.

Recorded FAILs not discussed above: all 2 failed recipes are now documented above.

If sampling pathologies emerge as recipe sweeps execute, case studies will be documented here.

## History

The following case studies document the investigation path and distilled lessons:

- 2026-05-12: 500 warmup steps insufficient for d=203 adaptation; covariance windows too short; fixed by n_warmup≥5000
- 2026-05-12: JITTER=1e-6 + float32 produces NaN in Cholesky; silent trap caught by post-hoc warmup inspection; fix is float64 + JITTER≥1e-4
- 2026-05-13: IMM-index diagnostic error pattern (jax.flatten_util.ravel_pytree for correct site labeling); false-alarm high hyperparameter IMM variance was actually latent variance
- 2026-05-13: Posterior ridge is identifiability issue, not prior misspecification; durable Bayesian-modeling lesson (what NOT to do)

## Citations

Rasmussen & Williams (2006, Gaussian Processes for Machine Learning); synthetic RBF regression on [0,1]
