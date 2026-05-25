"""ExpJ: Phase-1-maxiter=100 confirming run for gp_regression × laplace_mhmc recipe.

Validates that lowering Phase 1 L-BFGS maxiter from 500 → 100 (traversal phase)
preserves recipe quality while cutting Phase 1 wall time ~5×.

KEY CHANGE vs Stage 3b (expI):
  Two separate laplace_marginal_factory instances:
    laplace_phase1: maxiter=100  — rough traversal; lower accuracy acceptable
    laplace_phase2: maxiter=500  — Welford accuracy; required for 3×3 IMM quality

  Phase 1 warmup now uses blackjax.laplace_hmc directly (not blackjax.hmc),
  passing the LaplaceMarginal object as the logdensity_fn argument to
  window_adaptation.  This is valid because:
    - laplace_hmc.init(position, laplace) takes LaplaceMarginal as 2nd arg
    - laplace_hmc kernel(rng_key, state, laplace, step_size, imm, L) — same
    - window_adaptation passes its logdensity_fn arg through to both init & kernel
  BENEFIT vs Stage 3b (blackjax.hmc): warm-start from theta_prev across steps
  (laplace_hmc carries theta_star in state), so L-BFGS starts near solution.
  Combined with maxiter=100, Phase 1 should be ~5× faster.

Stage 3b reference:
  Phase 1 wall: 968.0s  (1.936s/step)  ← this should become ~200s
  Phase 2 wall: 184.7s  (0.924s/step)
  ESS/s (sampling-only): 1.086
  min_ESS: 144.47, max_R-hat: 1.041, n_div: 0

Gates (per §29 spec):
  Phase 1 stationarity: all params within 2σ GT mean
  Phase 1 wall: < 300s  (confirms 5× speedup)
  Phase 2 ridge corr(log_ks, log_ls): ≥ 0.5  (ideally ≥ 0.65)
  Benchmark n_div: = 0
  Benchmark max_R-hat: < 1.05
  Benchmark ESS/s (sampling-only): ≥ 0.977  (≥ 90% of Stage 3b 1.086)
"""

import os
import sys
import time

t0 = time.perf_counter()
print("[t=+0.0s] Script start", flush=True)

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

t_jax = time.perf_counter()
print(
    f"[t=+{t_jax - t0:.1f}s] JAX x64={jax.config.read('jax_enable_x64')} "
    f"backend={jax.default_backend()}",
    flush=True,
)

sys.path.insert(0, "/home/jp/blackjax-devs/tuningfork")
sys.path.insert(0, "/home/jp/blackjax-devs/blackjax")

import blackjax  # noqa: E402
import blackjax.mcmc.hmc as _hmc_module  # noqa: E402
import blackjax.mcmc.laplace_hmc as _lhmc  # noqa: E402
from blackjax.mcmc.laplace_marginal import laplace_marginal_factory  # noqa: E402

from tuningfork.model import MODELS  # noqa: E402
from tuningfork.model._numpyro import build_logdensity_fn  # noqa: E402
from tuningfork.recipes._recipe_runner import _LAPLACE_PHI_THETA_SPLITS  # noqa: E402

t_imports = time.perf_counter()
print(f"[t=+{t_imports - t0:.1f}s] Imports done (stock blackjax 007a9ded)", flush=True)

# ---------------------------------------------------------------------------
# Config — identical to Stage 3b (expI) except MAXITER_PHASE1
# ---------------------------------------------------------------------------
SEED = 20260517
MAXITER_PHASE1 = 100  # ← KEY CHANGE: rough traversal; was 500 in Stage 3b
MAXITER_PHASE2 = 500  # ← unchanged: accurate Hessian for Welford
N_CHAINS = 4
N_SAMPLES = 80  # per chain; 320 total
N_LEAPFROG = 5  # L=5 winner from Stage 2

# Phase 1 warmup config
N_WARMUP_PHASE1 = 500
L_PHASE1 = 10
TARGET_ACC_PHASE1 = 0.9

# Phase 2 warmup config
N_WARMUP_PHASE2 = 200
L_PHASE2 = 5
TARGET_ACC_PHASE2 = 0.93

# Prior-means init (production-realistic; no GT oracle)
PHI_PRIOR_MEANS = {
    "log_kernel_scale": jnp.float64(0.0),
    "log_lengthscale": jnp.float64(0.0),
    "log_noise_scale": jnp.float64(-2.0),
}

# GT posterior mean and stds (for stationarity check only)
GT_MEAN = {
    "log_kernel_scale": 0.40870562293007373,
    "log_lengthscale": -1.0424925985381703,
    "log_noise_scale": -2.34163615643574,
}
GT_STDS = {
    "log_kernel_scale": 0.4278,
    "log_lengthscale": 0.1783,
    "log_noise_scale": 0.0513,
}

# Stage 3b (expI) reference baselines
STAGE3B_PHASE1_WALL = 968.0  # seconds
STAGE2_L5_ESS_PER_S = 197.06 / 138.294  # 1.425 (GT oracle upper bound)
STAGE3A_PRIME_ESS_PER_S_SAMPLING = 157.16 / 134.343  # 1.170
STAGE3B_ESS_PER_S_SAMPLING = 144.47 / 133.089  # 1.086 (Stage 3b sampling-only)

print(
    f"  MAXITER Phase 1 = {MAXITER_PHASE1}  (was 500 in Stage 3b)",
    flush=True,
)
print(
    f"  MAXITER Phase 2 = {MAXITER_PHASE2}  (same as Stage 3b)",
    flush=True,
)
print(
    f"  Stage 3b Phase 1 wall: {STAGE3B_PHASE1_WALL:.1f}s  "
    f"(target: < 300s with maxiter=100)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
rng_key = jax.random.key(SEED)
key_init, key_w1, key_w2, key_run = jax.random.split(rng_key, 4)

entry = MODELS["gp_regression"]
init_position, logdensity_fn, _ = build_logdensity_fn(key_init, entry)

t_model = time.perf_counter()
print(
    f"[t=+{t_model - t0:.1f}s] model built "
    f"d={sum(v.size for v in jax.tree.leaves(init_position))}",
    flush=True,
)

phi_sites, theta_sites = _LAPLACE_PHI_THETA_SPLITS["gp_regression"]
theta_init = {k: init_position[k] for k in theta_sites}


def log_joint_fn(theta, phi):
    return logdensity_fn({**theta, **phi})


# ---------------------------------------------------------------------------
# Two separate laplace factories — KEY CHANGE vs Stage 3b
# ---------------------------------------------------------------------------
laplace_phase1 = laplace_marginal_factory(
    log_joint_fn, theta_init, maxiter=MAXITER_PHASE1
)
laplace_phase2 = laplace_marginal_factory(
    log_joint_fn, theta_init, maxiter=MAXITER_PHASE2
)

t_laplace = time.perf_counter()
print(
    f"[t=+{t_laplace - t0:.1f}s] Two Laplace marginal factories built "
    f"(Phase1 maxiter={MAXITER_PHASE1}, Phase2 maxiter={MAXITER_PHASE2})",
    flush=True,
)

# ---------------------------------------------------------------------------
# Phase 1: diagonal-IMM warmup using blackjax.laplace_hmc + laplace_phase1
# KEY CHANGE vs Stage 3b: uses laplace_hmc (warm-start) instead of plain hmc
# window_adaptation passes logdensity_fn to laplace_hmc.init and kernel,
# so passing LaplaceMarginal object works directly (laplace_hmc expects it).
# ---------------------------------------------------------------------------
print(
    f"\n  --- Phase 1: warmup ({N_WARMUP_PHASE1} steps, laplace_hmc L={L_PHASE1}, "
    f"diag-IMM, maxiter={MAXITER_PHASE1}, target={TARGET_ACC_PHASE1}) ---",
    flush=True,
)

warmup1 = blackjax.window_adaptation(
    blackjax.laplace_hmc,
    laplace_phase1,
    is_mass_matrix_diagonal=True,
    num_integration_steps=L_PHASE1,
    target_acceptance_rate=TARGET_ACC_PHASE1,
)

t_w1_start = time.perf_counter()
(state1, params1), _ = warmup1.run(key_w1, PHI_PRIOR_MEANS, num_steps=N_WARMUP_PHASE1)
jax.block_until_ready(state1)
t_w1_end = time.perf_counter()
wall_w1 = t_w1_end - t_w1_start

step1 = float(params1["step_size"])
imm1_diag = np.array(params1["inverse_mass_matrix"])
phi1_end = state1.position

speedup_phase1 = STAGE3B_PHASE1_WALL / wall_w1
print(
    f"  Phase 1 done: {wall_w1:.3f}s ({wall_w1 / N_WARMUP_PHASE1:.3f}s/step)  "
    f"[Stage 3b was {STAGE3B_PHASE1_WALL:.1f}s → {speedup_phase1:.2f}× speedup]",
    flush=True,
)
print(
    f"  Phase 1 gate: wall < 300s → {'PASS ✓' if wall_w1 < 300 else 'FAIL ✗'}",
    flush=True,
)
print(f"  Adapted step_size (Phase 1) = {step1:.5f}", flush=True)
print(f"  Phase 1 IMM diagonal (stds): {np.sqrt(np.maximum(imm1_diag, 0))}", flush=True)

phi_keys_sorted = sorted(phi1_end.keys())
print("  Final position phi1:", flush=True)
for k in phi_keys_sorted:
    v = float(phi1_end[k])
    gt_m = GT_MEAN[k]
    gt_s = GT_STDS[k]
    dev = abs(v - gt_m) / gt_s
    within = "✓" if dev <= 2.0 else "✗ OUTSIDE 2σ"
    print(
        f"    {k}: {v:.4f}  GT_mean={gt_m:.4f}  dev={dev:.2f}σ  {within}",
        flush=True,
    )

phi1_stationarity_pass = all(
    abs(float(phi1_end[k]) - GT_MEAN[k]) / GT_STDS[k] <= 2.0 for k in phi_keys_sorted
)
print(
    f"\n  Phase 1 stationarity gate (all within 2σ of GT mean): "
    f"{'PASS ✓' if phi1_stationarity_pass else 'FAIL ✗ (too far from posterior)'}",
    flush=True,
)
step1_gate_pass = 0.05 <= step1 <= 2.0
print(
    f"  Phase 1 step gate (0.05-2.0): {step1:.5f} "
    f"{'PASS ✓' if step1_gate_pass else 'FAIL ✗'}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Phase 2: dense 3x3 IMM warmup using blackjax.laplace_hmc + laplace_phase2
# maxiter=500 — full accuracy required for Welford 3×3 IMM
# ---------------------------------------------------------------------------
print(
    f"\n  --- Phase 2: warmup ({N_WARMUP_PHASE2} steps, laplace_hmc L={L_PHASE2}, "
    f"dense-IMM, maxiter={MAXITER_PHASE2}, target={TARGET_ACC_PHASE2}) ---",
    flush=True,
)

warmup2 = blackjax.window_adaptation(
    blackjax.laplace_hmc,
    laplace_phase2,
    is_mass_matrix_diagonal=False,
    num_integration_steps=L_PHASE2,
    initial_step_size=step1,
    target_acceptance_rate=TARGET_ACC_PHASE2,
)

t_w2_start = time.perf_counter()
(state2, params2), _ = warmup2.run(key_w2, phi1_end, num_steps=N_WARMUP_PHASE2)
jax.block_until_ready(state2)
t_w2_end = time.perf_counter()
wall_w2 = t_w2_end - t_w2_start

adapted_step = float(params2["step_size"])
adapted_imm = np.array(params2["inverse_mass_matrix"])
phi2_end = state2.position

print(
    f"  Phase 2 done: {wall_w2:.3f}s ({wall_w2 / N_WARMUP_PHASE2:.3f}s/step)",
    flush=True,
)
print(f"  Adapted step_size (Phase 2) = {adapted_step:.5f}", flush=True)
print("  Adapted IMM 3x3 (dense):", flush=True)
for i, ki in enumerate(phi_keys_sorted):
    row = "  ".join(f"{adapted_imm[i, j]:+.6f}" for j in range(3))
    print(f"    [{ki:20s}] {row}", flush=True)

# Compute ridge correlation from adapted IMM
imm_stds = np.sqrt(np.maximum(np.diag(adapted_imm), 0))
idx_ks = phi_keys_sorted.index("log_kernel_scale")
idx_ls = phi_keys_sorted.index("log_lengthscale")
if imm_stds[idx_ks] > 0 and imm_stds[idx_ls] > 0:
    ridge_corr = adapted_imm[idx_ks, idx_ls] / (imm_stds[idx_ks] * imm_stds[idx_ls])
else:
    ridge_corr = 0.0

print(f"\n  Ridge correlation corr(log_ks, log_ls) = {ridge_corr:.4f}", flush=True)
print(
    "  (gate ≥0.5; ideal ≥0.65; Stage 3b=+0.741; GT=+0.754; expH A2=+0.767)",
    flush=True,
)
print("  Welford IMM stds vs GT stds:", flush=True)
for i, ki in enumerate(phi_keys_sorted):
    gt_s = GT_STDS.get(ki, float("nan"))
    ratio = imm_stds[i] / gt_s if gt_s > 0 else float("nan")
    print(
        f"    {ki:25s}: welford_std={imm_stds[i]:.4f}  "
        f"gt_std={gt_s:.4f}  ratio={ratio:.3f}",
        flush=True,
    )

print("  Final position phi2:", flush=True)
for k in phi_keys_sorted:
    v = float(phi2_end[k])
    gt_m = GT_MEAN[k]
    gt_s = GT_STDS[k]
    dev = abs(v - gt_m) / gt_s
    within = "✓" if dev <= 2.0 else "✗ OUTSIDE 2σ"
    print(
        f"    {k}: {v:.4f}  GT_mean={gt_m:.4f}  dev={dev:.2f}σ  {within}",
        flush=True,
    )

phase2_ridge_pass = ridge_corr >= 0.5
phase2_ridge_ideal = ridge_corr >= 0.65
phase2_step_pass = 0.3 <= adapted_step <= 1.0
print(
    f"\n  Phase 2 ridge gate (corr≥0.5): {ridge_corr:.4f} "
    f"{'PASS ✓' if phase2_ridge_pass else 'FAIL ✗'}"
    f"{'  (ideal ≥0.65 ✓)' if phase2_ridge_ideal else '  (below ideal 0.65)'}",
    flush=True,
)
print(
    f"  Phase 2 step gate (0.3-1.0): {adapted_step:.5f} "
    f"{'PASS ✓' if phase2_step_pass else 'FAIL ✗'}",
    flush=True,
)

wall_warmup_total = wall_w1 + wall_w2
print(
    f"\n  Total warmup wall: {wall_warmup_total:.1f}s "
    f"(Phase 1: {wall_w1:.1f}s + Phase 2: {wall_w2:.1f}s)  "
    f"[Stage 3b total: 1152.7s]",
    flush=True,
)

# ---------------------------------------------------------------------------
# Final benchmark: laplace_mhmc, L=5, adapted step+IMM, 4 chains x 80 samples
# V2 vmap pattern — same as Stage 3b but uses laplace_phase2 (maxiter=500)
# ---------------------------------------------------------------------------
print(
    f"\n  --- Final benchmark ({N_CHAINS} chains x {N_SAMPLES} samples, "
    f"laplace_mhmc L={N_LEAPFROG}, maxiter={MAXITER_PHASE2}) ---",
    flush=True,
)

# Build laplace_mhmc kernel outside vmap (shared across chains)
kernel_mhmc = _lhmc.build_kernel(build_proposal=_hmc_module.multinomial_hmc_proposal)

adapted_imm_jax = jnp.array(adapted_imm, dtype=jnp.float64)
ADAPTED_STEP = jnp.float64(adapted_step)

print("  Init laplace state from Phase 2 end position (maxiter=500) ...", flush=True)
t_init_start = time.perf_counter()
state_mhmc_single = _lhmc.init(phi2_end, laplace_phase2)
jax.block_until_ready(state_mhmc_single)
t_init_end = time.perf_counter()
print(f"  Init done: {t_init_end - t_init_start:.3f}s", flush=True)

# Replicate single state across N_CHAINS
state_batched = jax.tree.map(lambda x: jnp.stack([x] * N_CHAINS), state_mhmc_single)


def _run_chain(rng_key_chain, init_state):
    def scan_fn(state, rng_k):
        new_state, info = kernel_mhmc(
            rng_k, state, laplace_phase2, ADAPTED_STEP, adapted_imm_jax, N_LEAPFROG
        )
        phi_arr = jnp.stack(
            [
                new_state.position["log_kernel_scale"],
                new_state.position["log_lengthscale"],
                new_state.position["log_noise_scale"],
            ]
        )
        return new_state, (phi_arr, info.acceptance_rate, info.is_divergent)

    scan_keys = jax.random.split(rng_key_chain, N_SAMPLES)
    _, (phi_traj, traj_weight_traj, div_traj) = jax.lax.scan(
        scan_fn, init_state, scan_keys
    )
    return phi_traj, traj_weight_traj, div_traj


run_vmap = jax.jit(jax.vmap(_run_chain))
keys_chains = jax.random.split(key_run, N_CHAINS)

t_scan_start = time.perf_counter()
phi, traj_weight, div = run_vmap(keys_chains, state_batched)
jax.block_until_ready(phi)
t_scan_end = time.perf_counter()
wall_scan = t_scan_end - t_scan_start
wall_total = wall_warmup_total + (t_init_end - t_init_start) + wall_scan

total_samples = N_CHAINS * N_SAMPLES
print(
    f"  vmap scan: {wall_scan:.3f}s "
    f"({wall_scan / N_SAMPLES:.3f}s/vmap-step, "
    f"{wall_scan / total_samples:.3f}s/chain-sample, incl JIT)",
    flush=True,
)
print(f"  wall_total (warmup+init+scan): {wall_total:.3f}s", flush=True)
print(
    f"  mean_traj_weight={float(traj_weight.mean()):.4f}  n_div={int(div.sum())}",
    flush=True,
)

param_names = ["log_kernel_scale", "log_lengthscale", "log_noise_scale"]
for c in range(N_CHAINS):
    print(
        f"  chain {c}: "
        f"phi_mean=[{float(phi[c, :, 0].mean()):.4f}, "
        f"{float(phi[c, :, 1].mean()):.4f}, "
        f"{float(phi[c, :, 2].mean()):.4f}]  "
        f"traj_wt={float(traj_weight[c].mean()):.4f}  "
        f"n_div={int(div[c].sum())}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# ESS and R-hat
# ---------------------------------------------------------------------------
ess = blackjax.diagnostics.effective_sample_size(phi)
rhat = blackjax.diagnostics.potential_scale_reduction(phi)

print("\n  --- Diagnostics ---", flush=True)
for i, p in enumerate(param_names):
    print(
        f"  {p}: ESS={float(ess[i]):.2f}  "
        f"ESS/N={float(ess[i]) / total_samples:.4f}  "
        f"R-hat={float(rhat[i]):.4f}",
        flush=True,
    )

min_ess = float(jnp.min(ess))
max_rhat = float(jnp.max(rhat))
ess_per_sec_total = min_ess / wall_total
ess_per_sec_sampling = min_ess / wall_scan
ratio_vs_stage3b_sampling = ess_per_sec_sampling / STAGE3B_ESS_PER_S_SAMPLING
ratio_vs_l5_gt = ess_per_sec_total / STAGE2_L5_ESS_PER_S

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
t_total = time.perf_counter()
sep = "=" * 70
print(f"\n{sep}", flush=True)
print(
    f"ExpJ Phase-1-maxiter=100 Confirming Run (total wall = {t_total - t0:.1f}s)",
    flush=True,
)
print(
    "  blackjax: stock main 007a9ded  |  "
    "gp_regression N=200  |  V2 vmap  |  4 chains x 80 samples",
    flush=True,
)
print(sep, flush=True)
print(
    f"  laplace_mhmc L={N_LEAPFROG}: "
    f"{N_CHAINS} chains x {N_SAMPLES} samples = {total_samples} total",
    flush=True,
)
print(
    "  WARMUP: two-phase laplace_hmc (maxiter=100/500 split) from prior means {0,0,-2}",
    flush=True,
)
print(
    f"  Phase 1: diag-IMM, L={L_PHASE1}, n={N_WARMUP_PHASE1}, maxiter={MAXITER_PHASE1}, "
    f"target={TARGET_ACC_PHASE1}  →  step1={step1:.5f}  wall={wall_w1:.1f}s",
    flush=True,
)
print(
    f"  Phase 2: dense-IMM, L={L_PHASE2}, n={N_WARMUP_PHASE2}, maxiter={MAXITER_PHASE2}, "
    f"target={TARGET_ACC_PHASE2}  →  step={adapted_step:.5f}  wall={wall_w2:.1f}s",
    flush=True,
)
print(
    f"  adapted_step={adapted_step:.5f}  "
    f"(Stage 3b: 0.48109; Stage 3a' ref: 0.57294; Stage 2 oracle: 0.526)",
    flush=True,
)
print(
    f"  ridge_corr(log_ks,log_ls) = {ridge_corr:.4f}  "
    f"(gate ≥0.5; ideal ≥0.65; Stage 3b=+0.741; GT=+0.754)",
    flush=True,
)
print("  Adapted IMM 3x3:", flush=True)
for i, ki in enumerate(phi_keys_sorted):
    row = "  ".join(f"{adapted_imm[i, j]:+.6f}" for j in range(3))
    print(f"    [{ki:20s}] {row}", flush=True)
print(
    f"  wall_w1={wall_w1:.1f}s  wall_w2={wall_w2:.1f}s  "
    f"wall_scan={wall_scan:.1f}s  wall_total={wall_total:.1f}s",
    flush=True,
)
print(
    f"  Stage 3b comparison: w1={STAGE3B_PHASE1_WALL:.1f}s → {wall_w1:.1f}s  "
    f"({speedup_phase1:.2f}× faster)",
    flush=True,
)
print(
    f"  mean_traj_weight={float(traj_weight.mean()):.4f}  n_div={int(div.sum())}  "
    "(traj_weight = exp(sum_log_p_accept)/L, not M-H acc)",
    flush=True,
)
print("  ESS per param:", flush=True)
for i, p in enumerate(param_names):
    print(
        f"    {p}: ESS={float(ess[i]):.2f}  "
        f"ESS/N={float(ess[i]) / total_samples:.4f}  "
        f"R-hat={float(rhat[i]):.4f}",
        flush=True,
    )
print(
    f"  min_ESS={min_ess:.2f}  n_div={int(div.sum())}  max_R-hat={max_rhat:.4f}",
    flush=True,
)
print(
    f"  ESS/s (sampling-only) = {ess_per_sec_sampling:.5f}  "
    f"vs-stage3b={ratio_vs_stage3b_sampling:.3f}x  "
    f"(gate: ≥0.977 = {'PASS ✓' if ratio_vs_stage3b_sampling >= 0.9 else 'FAIL ✗'})",
    flush=True,
)
print(
    f"  ESS/s (total-wall)    = {ess_per_sec_total:.5f}  "
    f"vs-L5-GT={ratio_vs_l5_gt:.3f}x",
    flush=True,
)

# Gate summary
phase1_wall_pass = wall_w1 < 300
rhat_pass = max_rhat < 1.05
ndiv_pass = int(div.sum()) == 0
ess_pass = ratio_vs_stage3b_sampling >= 0.9
print("\n  --- Gate summary ---", flush=True)
print(
    f"  Phase 1 stationarity: {'PASS ✓' if phi1_stationarity_pass else 'FAIL ✗'}",
    flush=True,
)
print(
    f"  Phase 1 wall < 300s: {wall_w1:.1f}s → {'PASS ✓' if phase1_wall_pass else 'FAIL ✗ (>300s → retry at maxiter=200)'}",
    flush=True,
)
print(
    f"  Phase 2 ridge ≥ 0.5: {ridge_corr:.4f} → {'PASS ✓' if phase2_ridge_pass else 'FAIL ✗'}",
    flush=True,
)
print(
    f"  n_div = 0: {int(div.sum())} → {'PASS ✓' if ndiv_pass else 'FAIL ✗'}",
    flush=True,
)
print(
    f"  max_R-hat < 1.05: {max_rhat:.4f} → {'PASS ✓' if rhat_pass else 'FAIL ✗'}",
    flush=True,
)
print(
    f"  ESS/s ≥ 90% Stage 3b: {ratio_vs_stage3b_sampling:.3f}x → {'PASS ✓' if ess_pass else 'FAIL ✗'}",
    flush=True,
)
all_pass = (
    phi1_stationarity_pass
    and phase1_wall_pass
    and phase2_ridge_pass
    and ndiv_pass
    and rhat_pass
    and ess_pass
)
print(
    f"\n  OVERALL: {'ALL GATES PASS ✓ — recipe confirmed' if all_pass else 'SOME GATES FAIL ✗ — investigate'}",
    flush=True,
)
print(sep, flush=True)
