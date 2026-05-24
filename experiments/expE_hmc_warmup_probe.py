"""ExpE: HMC window_adaptation on marginal → ridge-capturing IMM probe.

Key question: does window_adaptation(hmc, marginal_logdensity_fn) with dense IMM
capture the +0.754 log_ls/log_ks ridge, providing a viable warmup alternative to
NUTS (which collapses step_size)?

Design
------
Phase 1: window_adaptation(blackjax.hmc, marginal_logdensity_fn,
           is_mass_matrix_diagonal=False, num_integration_steps=10)
         n_warmup=100 → adapted step_size + 3x3 dense IMM (Welford estimator)
         → compute corr(log_ks, log_ls), compare to +0.754

Phase 2: 1-chain laplace_dhmc sampling (10 steps) with adapted params
         (dhmc OK for 1-chain: no vmap, dynamic-loop cost doesn't bite)

Phase 3 (if 1-chain OK): 4-chain vmap with laplace_hmc (fixed-L=10)
         (per TL brief: use hmc not dhmc for 4-chain vmap — expB showed
          dhmc is 4.4x under vmap vs hmc's 1.2x)

Expected outcome
----------------
If HMC window_adaptation captures ridge:
  - IMM[idx_ks, idx_ls] / sqrt(IMM[idx_ks,idx_ks] * IMM[idx_ls,idx_ls]) ~ +0.754
  - Acceptance rate 0.5-0.9 for laplace_dhmc 1-chain with adapted params
  - No divergences for 10 steps

Why this should work vs pathfinder (expD):
  - Welford estimator builds 3x3 covariance from ACTUAL CHAIN SAMPLES
  - HMC (no NUTS) = no tree expansion = no step_size collapse
  - 100 steps from phi_init should reach vicinity of posterior mode
    (HMC with diagonal prior init will random-walk toward the mode)
  - Once near mode, samples build covariance including the +0.754 ridge direction
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
    f"[t=+{t_jax - t0:.1f}s] JAX: x64={jax.config.read('jax_enable_x64')}, "
    f"backend={jax.default_backend()}",
    flush=True,
)

sys.path.insert(0, "/home/jp/blackjax-devs/tuningfork")
sys.path.insert(0, "/home/jp/blackjax-devs/blackjax")

import blackjax  # noqa: E402
from blackjax.util import run_inference_algorithm  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

from tuningfork.model import MODELS  # noqa: E402
from tuningfork.model._numpyro import build_logdensity_fn  # noqa: E402
from tuningfork.recipes._recipe_runner import _build_laplace_components  # noqa: E402

t_imports = time.perf_counter()
print(f"[t=+{t_imports - t0:.1f}s] All imports done", flush=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 20260517
N_WARMUP_STEPS = 100  # HMC window_adaptation steps on 3D marginal
N_LEAPFROG = 10  # fixed-L for HMC and laplace_hmc
N_SAMPLING_STEPS = 10  # laplace sampling steps

# ---------------------------------------------------------------------------
# Setup: gp_regression logdensity + Laplace components
# ---------------------------------------------------------------------------
rng_key = jax.random.key(SEED)
key_init, key_warmup, key_sample_1, key_sample_4 = jax.random.split(rng_key, 4)

entry = MODELS["gp_regression"]
init_position, logdensity_fn, _postprocess_fn = build_logdensity_fn(key_init, entry)

t_model = time.perf_counter()
print(
    f"[t=+{t_model - t0:.1f}s] gp_regression model built, "
    f"d_full={sum(v.size for v in jax.tree.leaves(init_position))}",
    flush=True,
)

laplace_components = _build_laplace_components(
    "gp_regression", init_position, logdensity_fn
)
if laplace_components is None:
    raise RuntimeError("gp_regression not in _LAPLACE_PHI_THETA_SPLITS")

phi_init, log_joint_fn, theta_init, marginal_logdensity_fn = laplace_components

phi_keys_sorted = sorted(phi_init.keys())
phi_flat_init, unravel_fn = ravel_pytree(phi_init)
print(
    f"[t=+{time.perf_counter() - t0:.1f}s] Laplace components built",
    flush=True,
)
print(f"  phi_keys (sorted = flat order): {phi_keys_sorted}", flush=True)
print(f"  phi_init (flat): {np.array(phi_flat_init)}", flush=True)

# ---------------------------------------------------------------------------
# Phase 1: HMC window_adaptation on marginal_logdensity_fn (3D)
# Uses plain HMC (NO NUTS) => no tree expansion => no step_size collapse
# Dense 3x3 IMM: Welford estimator builds full 3x3 covariance from samples
# ---------------------------------------------------------------------------
print(
    f"\n=== Phase 1: HMC window_adaptation on 3D marginal "
    f"(n_warmup={N_WARMUP_STEPS}, L={N_LEAPFROG}, dense IMM) ===",
    flush=True,
)

warmup = blackjax.window_adaptation(
    blackjax.hmc,
    marginal_logdensity_fn,
    is_mass_matrix_diagonal=False,  # dense 3x3 IMM (captures ridge correlation)
    initial_step_size=0.1,
    target_acceptance_rate=0.8,
    num_integration_steps=N_LEAPFROG,
    progress_bar=False,
)

t_warmup_start = time.perf_counter()
(warmup_results, warmup_info) = warmup.run(
    key_warmup, phi_init, num_steps=N_WARMUP_STEPS
)
warmup_state = warmup_results.state
adapted_params = warmup_results.parameters
_ = jax.block_until_ready(warmup_state)
t_warmup_end = time.perf_counter()

warmup_wall = t_warmup_end - t_warmup_start
adapted_step_size = float(adapted_params["step_size"])
adapted_imm = np.array(adapted_params["inverse_mass_matrix"])

print(f"  Warmup wall: {warmup_wall:.2f}s", flush=True)
print(f"  Adapted step_size: {adapted_step_size:.6f}", flush=True)
print("  Adapted IMM (3x3 dense):", flush=True)
for i, key_i in enumerate(phi_keys_sorted):
    row = "  ".join(f"{adapted_imm[i, j]:+.4f}" for j in range(3))
    print(f"    [{key_i:20s}] {row}", flush=True)

# Stds and correlation from adapted IMM
var_diag = np.diag(adapted_imm)
std_diag = np.sqrt(np.maximum(var_diag, 0))
print("\n  Marginal stds:", flush=True)
for i, key_i in enumerate(phi_keys_sorted):
    print(f"    {key_i:25s}: std = {std_diag[i]:.4f}", flush=True)

idx_ks = phi_keys_sorted.index("log_kernel_scale")
idx_ls = phi_keys_sorted.index("log_lengthscale")
cov_ks_ls = adapted_imm[idx_ks, idx_ls]
corr_ks_ls = (
    cov_ks_ls / (std_diag[idx_ks] * std_diag[idx_ls])
    if (std_diag[idx_ks] > 0 and std_diag[idx_ls] > 0)
    else 0.0
)

print("\n  === RIDGE TEST ===", flush=True)
print(f"  corr(log_kernel_scale, log_lengthscale) = {corr_ks_ls:.4f}", flush=True)
print("  Target: +0.754 (from Exp 2 MCLMC posterior ridge)", flush=True)
ridge_captured = abs(corr_ks_ls - 0.754) < 0.15
print(
    f"  Ridge test: {'PASS' if ridge_captured else 'FAIL'} "
    f"(|corr - 0.754| = {abs(corr_ks_ls - 0.754):.4f}, threshold=0.15)",
    flush=True,
)

# Final HMC warmup position (starting point for sampling)
phi_warmup_end = warmup_state.position
phi_warmup_flat = np.array(ravel_pytree(phi_warmup_end)[0])
print(f"\n  Final warmup phi (flat): {phi_warmup_flat}", flush=True)
print(
    f"  Marginal lp at warmup end: {float(marginal_logdensity_fn(phi_warmup_end)):.3f}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Phase 2: 1-chain laplace_dhmc sampling (10 steps)
# dhmc OK for 1-chain: no vmap, dynamic-loop overhead doesn't apply
# ---------------------------------------------------------------------------
print(
    f"\n=== Phase 2: 1-chain laplace_dhmc sampling "
    f"(n_steps={N_SAMPLING_STEPS}, L={N_LEAPFROG}, adapted params) ===",
    flush=True,
)

algorithm_dhmc = blackjax.laplace_dhmc(
    log_joint_fn,
    theta_init,
    step_size=adapted_step_size,
    inverse_mass_matrix=jnp.array(adapted_imm),
)

t_sample1_start = time.perf_counter()
final_state_1, (history_states_1, history_info_1) = run_inference_algorithm(
    key_sample_1,
    algorithm_dhmc,
    num_steps=N_SAMPLING_STEPS,
    initial_position=phi_warmup_end,
    progress_bar=False,
)
_ = jax.block_until_ready(final_state_1)
t_sample1_end = time.perf_counter()

sample1_wall = t_sample1_end - t_sample1_start
accs_1 = np.array(history_info_1.acceptance_rate)
divs_1 = np.array(history_info_1.is_divergent)
n_div_1 = int(divs_1.sum())
mean_acc_1 = float(accs_1.mean())
lps_1 = np.array(history_states_1.logdensity)

print(f"  1-chain wall: {sample1_wall:.2f}s", flush=True)
print(
    f"  Acceptance rate: mean={mean_acc_1:.3f}, min={accs_1.min():.3f}, max={accs_1.max():.3f}",
    flush=True,
)
print(f"  Divergences: {n_div_1}/{N_SAMPLING_STEPS}", flush=True)
print(f"  logdensity range: [{lps_1.min():.2f}, {lps_1.max():.2f}]", flush=True)
phi_positions = np.array(
    jax.vmap(lambda pos: ravel_pytree(pos)[0])(history_states_1.position)
)
print("  phi trace (3 steps):", flush=True)
for step_i in range(min(3, N_SAMPLING_STEPS)):
    print(f"    step {step_i}: {phi_positions[step_i]}", flush=True)

# ---------------------------------------------------------------------------
# Phase 3 (conditional): 4-chain vmap with laplace_hmc (NOT dhmc)
# Per TL brief: use hmc for 4-chain vmap (expB: dhmc=4.4x, hmc=1.2x under vmap)
# ---------------------------------------------------------------------------
vmap_wall = None
vmap_acc_mean = None
vmap_n_div = None

if n_div_1 == 0 and mean_acc_1 > 0.3:
    print(
        f"\n=== Phase 3: 4-chain vmap laplace_hmc "
        f"(n_steps={N_SAMPLING_STEPS}, L={N_LEAPFROG}) ===",
        flush=True,
    )
    print(
        "  NOTE: Using laplace_hmc (not dhmc) per TL brief — avoids 4.4x vmap tax",
        flush=True,
    )

    vmap_keys = jax.random.split(key_sample_4, 4)
    phi_flat_warmup = jnp.array(phi_warmup_flat)
    key_perturb = jax.random.key(SEED + 5)
    phi_starts_4 = phi_flat_warmup[None, :] + 0.1 * jax.random.normal(
        key_perturb, (4, len(phi_warmup_flat))
    )

    def run_chain_hmc(rng_key, phi_flat):
        algo = blackjax.laplace_hmc(
            log_joint_fn,
            theta_init,
            step_size=adapted_step_size,
            inverse_mass_matrix=jnp.array(adapted_imm),
            num_integration_steps=N_LEAPFROG,
        )
        final, hist = run_inference_algorithm(
            rng_key,
            algo,
            num_steps=N_SAMPLING_STEPS,
            initial_position=unravel_fn(phi_flat),
            progress_bar=False,
        )
        return final, hist

    t_vmap_start = time.perf_counter()
    vmap_finals, vmap_hists = jax.vmap(run_chain_hmc)(vmap_keys, phi_starts_4)
    _ = jax.block_until_ready(vmap_finals)
    t_vmap_end = time.perf_counter()

    vmap_hists_states, vmap_hists_info = vmap_hists
    vmap_wall = t_vmap_end - t_vmap_start
    vmap_acc_mean = float(np.array(vmap_hists_info.acceptance_rate).mean())
    vmap_n_div = int(np.array(vmap_hists_info.is_divergent).sum())

    print(f"  4-chain vmap wall: {vmap_wall:.2f}s", flush=True)
    print(f"  Mean acceptance: {vmap_acc_mean:.3f}", flush=True)
    print(f"  Total divergences: {vmap_n_div}/{4 * N_SAMPLING_STEPS}", flush=True)
else:
    print(
        f"\n  Skipping Phase 3 (4-chain vmap): 1-chain not clean "
        f"(n_div={n_div_1}, mean_acc={mean_acc_1:.3f})",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
t_total = time.perf_counter()
sep = "=" * 62
print(f"\n{sep}", flush=True)
print(f"ExpE Summary (total wall = {t_total - t0:.1f}s)", flush=True)
print(sep, flush=True)
print(f"  HMC window_adaptation (n_warmup={N_WARMUP_STEPS}) on marginal:", flush=True)
print(f"    wall = {warmup_wall:.2f}s", flush=True)
print(f"    adapted_step_size = {adapted_step_size:.6f}", flush=True)
print(f"    corr(log_ks, log_ls) = {corr_ks_ls:.4f} (target +0.754)", flush=True)
print(f"    Ridge test: {'PASS' if ridge_captured else 'FAIL'}", flush=True)
print("  laplace_dhmc 1-chain sampling:", flush=True)
print(
    f"    step_size={adapted_step_size:.4f}, mean_acc={mean_acc_1:.3f}, n_div={n_div_1}",
    flush=True,
)
print(f"    wall={sample1_wall:.2f}s", flush=True)
if vmap_wall is not None:
    print("  laplace_hmc 4-chain vmap:", flush=True)
    print(
        f"    wall={vmap_wall:.2f}s, mean_acc={vmap_acc_mean:.3f}, n_div={vmap_n_div}",
        flush=True,
    )
viable = ridge_captured and n_div_1 == 0 and mean_acc_1 > 0.3
print(f"  Overall: {'VIABLE' if viable else 'NEEDS_INVESTIGATION'}", flush=True)
print(sep, flush=True)
