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

### 2026-05-12 — Under-warmupped IMM at d=203

Trial cert at `n_warmup=500, ta=0.80` produced step_size=3.5e-6 (three orders of magnitude below healthy peers), a uniform IMM (2.4e-5 across all 203 dims), and depth-10 tree-saturation on every draw — the chain never moved during warmup.

**Cascade.** BlackJAX's `window_adaptation` covariance windows are calibrated for d ≤ ~50. At d=203 the windows are too short to accumulate meaningful per-dim statistics → IMM stays uniform → step_size dual-averaging targets the uniform-IMM geometry instead of actual anisotropy → step_size is microscopic → NUTS hits the max-tree-depth ceiling on every trajectory. The chain looks statistically valid (0 divergences, R̂≈1 between chunks) but is geometrically stuck near the prior.

**Complete fix.** Raising n_warmup to 2000 unblocked the depth-saturation pathology, but a 16 % steady-state divergence rate persisted at ta=0.80. A side-by-side trial at the same warmup state confirmed the missing lever: `ta=0.80` → 160/1000 divergences; `ta=0.99` → 0 divergences. The production fix requires **both**: `n_warmup ≥ 5000` and `target_acceptance = 0.99`.

**Generalizable rule.** For d > 100, plan `n_warmup ≥ 2000` and `max_num_doublings = 12–15`. The pair matters — better IMM adaptation lowers condition number, and the tree-depth ceiling must give room to traverse whatever condition number the chain actually adapts to.

### 2026-05-12 — Silent float32 Cholesky failure

At the production defaults (float32, JITTER=1e-6), the cert produced R̂≈1.0 and 0 divergences — yet the chain had not moved: ‖Δposition‖=1.5 vs init norm 15.7, IMM condition=1.0×, step_size=3.5e-6, RMSE-to-truth=0.68 (7× data noise).

**Mechanism.** A 200-pt dense RBF kernel matrix on [0,1] routinely has off-diagonal entries in [0.6, 1.0] (points within a few lengthscales of each other). The smallest eigenvalue typically falls below 1e-4; adding JITTER=1e-6 lifts the floor to 1e-6 — above float64 precision (~1e-16) but below float32 precision (~1e-7). `jax.scipy.linalg.cholesky` emits NaN in the triangular factor; NUTS dual-averaging then shrinks step_size until gradients stop returning NaN, which only happens when the chain effectively cannot move.

**Fix.** At `JAX_ENABLE_X64=1 + JITTER=1e-4`, a 3-init probe at n_warmup=200 showed Δpos=13–17 (vs 1.5), IMM cond=1500–3400×, step_size=5–6.5e-3, noise_scale within 14 % of truth across all three inits. All three inits converged to the same posterior, confirming the chain was previously stuck purely due to numerical instability. The model now sets `requires_x64=True` and `JITTER=1e-4` as per-model defaults.

**Durable lesson.** R̂≈1 between chunks of a single long chain does NOT mean convergence to the posterior — it means consistency between chunks all stuck in the same prior-basin neighborhood. Always pair R̂ with either overdispersed multi-chain starts or a posterior sanity check against known truth (e.g., RMSE to synthetic truth). For latent-GP models: default to float64 + jitter ≥ 1e-4.

### 2026-05-13 — IMM-index labeling bug

Three independent scripts assumed flat IMM array indices 0, 1, 2 were the three hyperparameters (log_kernel_scale, log_lengthscale, log_noise_scale). They were actually `f_raw[0..2]`. JAX `tree.flatten` on a plain `dict` flattens leaves alphabetically, so `f_raw` (key `f_`) comes before all three hyperparameter keys.

**Impact.** The claimed `log_lengthscale` IMM value was 0.0015 ("chain hasn't explored the lengthscale ridge"). The correct value is 0.031 — matching the empirical posterior variance. The diagonal IMM was tracking hyperparameter marginals correctly; there was no under-capture.

| Site | Wrong (mislabeled index) | Correct (via `ravel_pytree`) |
|---|---:|---:|
| `log_lengthscale` IMM diag | 0.0015 (was `f_raw[1]`) | 0.031 ✓ matches posterior var |
| `log_kernel_scale` IMM diag | 0.95 (was `f_raw[0]`) | 0.130 ✓ |
| `log_noise_scale` IMM diag | 0.68 (was `f_raw[2]`) | 0.0026 ✓ |

The "600× hyperparam IMM mismatch" narrative and "chain hasn't explored the ridge" conclusions both reversed after correction. The production bottleneck was step_size stability (target_acceptance), not IMM accuracy.

**Fix.** Use `jax.flatten_util.ravel_pytree` to unravel flat arrays by site name. Sanity check after unravel: compare IMM diag against empirical posterior variance from the position trail; a 2× mismatch after convergence means something is wrong with one or the other.

### 2026-05-13 — Encoding posterior correlation as prior

The empirical posterior showed `r(log_lengthscale, log_kernel_scale) = +0.737` (classic GP identifiability ridge — both parameters can jointly shift to fit the data, trading off amplitude against smoothness). The natural reflex: "can we encode this correlation as a prior to cancel the ridge?"

**Why the answer is no.** `posterior ∝ prior × likelihood`. Adding a correlated prior in the same direction as the likelihood's ridge increases off-diagonal *precision* in the posterior, making the ridge sharper, not flatter. Precision matrices add; they do not cancel.

| Option | Verdict |
|---|---|
| MVN prior matching observed posterior correlation | Wrong direction — intensifies the ridge |
| Reparameterize to whitened coordinates (u, v) | Mathematically correct, but the rotation was derived from the posterior (reverse-engineering); ridge direction is dataset-dependent |
| Soft factor penalty on ridge sum `log_ls + log_ks` | Defensible only with pre-data domain knowledge, not applicable to a benchmark |
| Tighten marginal priors | Same reverse-engineering concern |
| **Sampler-side: low-rank or full-rank IMM** | **Correct** — sampler handles correlation without touching the model |

**Generalizable rule.** A diagnostic showing the posterior lives along a correlation ridge is evidence about geometry, not a recipe for prior design. Use it to choose a sampler with a metric that captures the structure (low-rank IMM, Riemannian HMC). The +0.74 ridge here is essentially rank-1 in hyperparameter space; `low_rank_window_adaptation` with max_rank=2–3 should capture it without model changes.

## Citations

Rasmussen & Williams (2006, Gaussian Processes for Machine Learning); synthetic RBF regression on [0,1]
