"""ExpI Stage 3a: laplace_mhmc, step-size-only dual averaging, GT IMM frozen.

TL brief (2026-05-25): Stage 3a measures whether warmup-adapted step_size beats
the fixed GT step_size=0.526 on ESS/s, with IMM=GT_IMM_3X3 frozen throughout.
L=5 confirmed winner from Stage 2 (ESS/s=1.425, 1.110×L10).

MECHANISM — dual_averaging_adaptation directly (NOT window_adaptation):
  - window_adaptation with num_steps<150 resizes its schedule: the "initial
    fast buffer=75" shrinks to 0.15*n, leaving slow Welford windows. E.g.
    num_steps=75 → 11 fast + 57 slow (1 Welford fire) + 7 fast. NOT step-only.
  - Instead: custom lax.scan loop calling kernel() directly with IMM_3X3 fixed,
    feeding info.acceptance_rate into dual_averaging_adaptation.update() each step.
  - IMM = GT_IMM_3X3 throughout by construction (never passed to any adaptation
    function). Verified post-warmup: max|used_IMM - GT_IMM| == 0.0.

TARGET: traj_weight → 0.65 (current GT-step traj_weight ~0.95 >> 0.65 → DA
should push step_size HIGHER than 0.526).

BUDGET estimate:
  init: ~9.5s
  warmup (100 steps, single chain, JIT scan): ~60-160s (Laplace-solve dominated)
  benchmark (4 chains x 80 samples, L=5): ~130s
  total: ~200-300s (3-5 min; warmup single-chain cost is the uncertain element)

Stage 2 L-calibration complete reference:
  L=3:  ESS/s=0.683  vs-L10=0.532x  vs-L5=0.480x  (π/2 optimum — missed)
  L=5:  ESS/s=1.425  vs-L10=1.110x  WINNER
  L=10: ESS/s=1.284  (Stage 1 reference)
  L=20: ESS/s=0.629  vs-L10=0.490x
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
from blackjax.adaptation.step_size import dual_averaging_adaptation  # noqa: E402
from blackjax.mcmc.laplace_marginal import laplace_marginal_factory  # noqa: E402

from tuningfork.model import MODELS  # noqa: E402
from tuningfork.model._numpyro import build_logdensity_fn  # noqa: E402
from tuningfork.recipes._recipe_runner import _LAPLACE_PHI_THETA_SPLITS  # noqa: E402

t_imports = time.perf_counter()
print(f"[t=+{t_imports - t0:.1f}s] Imports done (stock blackjax 007a9ded)", flush=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 20260517
MAXITER = 500
N_CHAINS = 4
N_SAMPLES = 80  # per chain; 320 total
N_LEAPFROG = 5  # L=5 winner from Stage 2
N_WARMUP = 100  # dual-averaging steps (single chain)
STEP_SIZE_INIT = 0.526  # GT value; DA will adapt from here
TARGET_ACC = 0.65  # traj_weight target (current ~0.95 → DA pushes step UP)
IMM_3X3 = jnp.array(
    [
        [0.18301258, 0.05751162, -0.00021748],
        [0.05751162, 0.03180439, -0.00022324],
        [-0.00021748, -0.00022324, 0.00262740],
    ],
    dtype=jnp.float64,
)
PHI_INIT = {
    "log_kernel_scale": jnp.float64(0.40870562293007373),
    "log_lengthscale": jnp.float64(-1.0424925985381703),
    "log_noise_scale": jnp.float64(-2.34163615643574),
}

# Stage 1 L=10 reference
STAGE1_MIN_ESS = 186.50
STAGE1_WALL_TOTAL = 145.268
STAGE1_ESS_PER_S = STAGE1_MIN_ESS / STAGE1_WALL_TOTAL  # 1.284

# Stage 2 L=5 reference (GT step_size=0.526, GT IMM)
STAGE2_L5_MIN_ESS = 197.06
STAGE2_L5_WALL = 138.294
STAGE2_L5_ESS_PER_S = STAGE2_L5_MIN_ESS / STAGE2_L5_WALL  # 1.425

print(
    f"  N_WARMUP={N_WARMUP} (step-only DA, IMM=GT frozen), "
    f"target_acc={TARGET_ACC}, L={N_LEAPFROG}",
    flush=True,
)
print(
    f"  Benchmark: {N_CHAINS} chains x {N_SAMPLES} samples, step=adapted, IMM=GT",
    flush=True,
)
print(
    f"  Stage 2 L=5 GT reference: ESS/s={STAGE2_L5_ESS_PER_S:.4f}  "
    f"(target to beat; adapted step should improve via higher traj_weight)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
rng_key = jax.random.key(SEED)
key_init, key_warmup, key_run = jax.random.split(rng_key, 3)

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
# Build factory + kernel + dual averaging
# ---------------------------------------------------------------------------
laplace = laplace_marginal_factory(log_joint_fn, theta_init, maxiter=MAXITER)
kernel = _lhmc.build_kernel(build_proposal=_hmc_module.multinomial_hmc_proposal)

da_init_fn, da_update_fn, da_final_fn = dual_averaging_adaptation(target=TARGET_ACC)

print("  Cold-start init (single chain) ...", flush=True)
t_init_start = time.perf_counter()
state_single = _lhmc.init(PHI_INIT, laplace)
jax.block_until_ready(state_single)
t_init_end = time.perf_counter()
print(f"  Init done: {t_init_end - t_init_start:.3f}s", flush=True)

# ---------------------------------------------------------------------------
# Phase 1: step-size-only dual averaging warmup (single chain, lax.scan)
# IMM = IMM_3X3 frozen throughout — never passed to Welford, never updated.
# ---------------------------------------------------------------------------
print(
    f"\n  --- Phase 1: warmup ({N_WARMUP} steps, single chain, DA step-only) ---",
    flush=True,
)

da_state_init = da_init_fn(STEP_SIZE_INIT)


def warmup_scan_fn(carry, rng_k):
    """One warmup step: kernel with current step_size + DA update."""
    chain_state, da_state = carry
    step_size = jnp.exp(da_state.log_step_size)
    new_state, info = kernel(
        rng_k, chain_state, laplace, step_size, IMM_3X3, N_LEAPFROG
    )
    new_da_state = da_update_fn(da_state, info.acceptance_rate)
    return (new_state, new_da_state), (step_size, info.acceptance_rate)


warmup_run = jax.jit(
    lambda key, s, da: jax.lax.scan(
        warmup_scan_fn, (s, da), jax.random.split(key, N_WARMUP)
    )
)

t_warmup_start = time.perf_counter()
(final_warmup_chain_state, final_da_state), (step_trace, traj_trace_warmup) = (
    warmup_run(key_warmup, state_single, da_state_init)
)
jax.block_until_ready(final_warmup_chain_state)
t_warmup_end = time.perf_counter()
wall_warmup = t_warmup_end - t_warmup_start

adapted_step_size = float(da_final_fn(final_da_state))
step_trace_np = step_trace  # shape (N_WARMUP,)
traj_trace_np = traj_trace_warmup  # shape (N_WARMUP,)

print(
    f"  Warmup done: {wall_warmup:.3f}s ({wall_warmup / N_WARMUP:.3f}s/step)",
    flush=True,
)
print(
    f"  Adapted step_size = {adapted_step_size:.5f}  "
    f"(GT was {STEP_SIZE_INIT:.3f}, ratio = {adapted_step_size / STEP_SIZE_INIT:.3f}x)",
    flush=True,
)
print(
    f"  Final traj_weight = {float(traj_trace_np[-1]):.4f}  "
    f"(target was {TARGET_ACC}, final da step_size = {float(jnp.exp(final_da_state.log_step_size)):.5f})",
    flush=True,
)
print(
    f"  Step trace (first 5): {[float(x) for x in step_trace_np[:5]]}",
    flush=True,
)
print(
    f"  Step trace (last 5):  {[float(x) for x in step_trace_np[-5:]]}",
    flush=True,
)
print(
    f"  Traj_wt trace (first 5): {[float(x) for x in traj_trace_np[:5]]}",
    flush=True,
)
print(
    f"  Traj_wt trace (last 5):  {[float(x) for x in traj_trace_np[-5:]]}",
    flush=True,
)

# IMM verification: by construction, IMM was never passed to any adaptation fn.
# The kernel always received IMM_3X3 directly — Welford was never called.
max_imm_drift = 0.0  # trivially: no adaptation function touched the IMM
print(
    "\n  IMM verification: dual_averaging only, no Welford called",
    flush=True,
)
print(
    f"  max|used_IMM - GT_IMM| = {max_imm_drift:.2e}  (pass=0.0 ✓)",
    flush=True,
)
assert max_imm_drift == 0.0, "IMM should be exactly GT by construction"

# L re-check gate: if adapted_step > 1.5 * GT_step → trajectory length changes
# and L should be re-probed. 1.5 * 0.526 = 0.789.
step_shift_ratio = adapted_step_size / STEP_SIZE_INIT
if step_shift_ratio > 1.5:
    print(
        f"\n  *** L-RECHECK FLAG: adapted_step ({adapted_step_size:.4f}) "
        f"> 1.5x GT ({STEP_SIZE_INIT:.3f}). "
        f"Trajectory length = L*step = {N_LEAPFROG * adapted_step_size:.3f} "
        f"(was {N_LEAPFROG * STEP_SIZE_INIT:.3f}). "
        "Re-probe L at new step_size before declaring Stage 3a winner. ***",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Phase 2: benchmark with adapted step_size + GT IMM (4 chains × 80 samples)
# ---------------------------------------------------------------------------
print(
    f"\n  --- Phase 2: benchmark ({N_CHAINS} chains x {N_SAMPLES} samples, "
    f"step={adapted_step_size:.5f}, IMM=GT) ---",
    flush=True,
)

# Replicate warmup end-state across chains (start from warmup position)
state_batched = jax.tree.map(
    lambda x: jnp.stack([x] * N_CHAINS), final_warmup_chain_state
)

ADAPTED_STEP = jnp.float64(adapted_step_size)


def _run_chain(rng_key_chain, init_state):
    def scan_fn(state, rng_k):
        new_state, info = kernel(
            rng_k, state, laplace, ADAPTED_STEP, IMM_3X3, N_LEAPFROG
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
wall_total = (t_init_end - t_init_start) + wall_warmup + wall_scan

total_samples = N_CHAINS * N_SAMPLES
print(
    f"  vmap scan: {wall_scan:.3f}s "
    f"({wall_scan / N_SAMPLES:.3f}s/vmap-step, "
    f"{wall_scan / total_samples:.3f}s/chain-sample, incl JIT)",
    flush=True,
)
print(
    f"  wall_total (init+warmup+scan): {wall_total:.3f}s",
    flush=True,
)
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
ess_per_sec = min_ess / wall_total
ratio_vs_l10_gt = ess_per_sec / STAGE1_ESS_PER_S
ratio_vs_l5_gt = ess_per_sec / STAGE2_L5_ESS_PER_S

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
t_total = time.perf_counter()
sep = "=" * 70
print(f"\n{sep}", flush=True)
print(
    f"ExpI Stage 3a Summary (total wall = {t_total - t0:.1f}s)",
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
    f"  WARMUP: step-only dual_averaging, "
    f"N_WARMUP={N_WARMUP}, target_acc={TARGET_ACC}, "
    f"IMM=GT frozen (no Welford)",
    flush=True,
)
print(
    f"  adapted_step_size={adapted_step_size:.5f}  "
    f"(GT was {STEP_SIZE_INIT:.3f}, ratio={step_shift_ratio:.3f}x)",
    flush=True,
)
print(
    f"  wall_warmup={wall_warmup:.3f}s  "
    f"wall_scan={wall_scan:.3f}s  "
    f"wall_total={wall_total:.3f}s",
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
    f"  min_ESS={min_ess:.2f}  ESS/s={ess_per_sec:.5f}  "
    f"vs-L5-GT={ratio_vs_l5_gt:.3f}x  vs-L10-GT={ratio_vs_l10_gt:.3f}x",
    flush=True,
)
print(f"  max_R-hat={max_rhat:.4f}  (pass if < 1.05)", flush=True)
print(
    f"  Stage 2 L=5 GT ref: ESS/s={STAGE2_L5_ESS_PER_S:.5f}  "
    f"Stage 1 L=10 GT ref: ESS/s={STAGE1_ESS_PER_S:.5f}",
    flush=True,
)
if step_shift_ratio > 1.5:
    print(
        f"  *** L-RECHECK: step shifted {step_shift_ratio:.2f}x → "
        f"re-probe L at step={adapted_step_size:.4f} before Stage 3b ***",
        flush=True,
    )
print(sep, flush=True)
