"""ExpF: Dense 3×3 IMM probe from GT-mean init — the A2 hypothesis-isolation probe.

Hypothesis: window_adaptation(laplace_hmc, dense IMM) captures the +0.754
log_ls/log_ks ridge when Welford sees stationary geometry from step 1.

A2 design rationale (statistician 2026-05-24):
  - phi_init = GT mean from groundtruth NUTS run (40K samples, certified).
  - GT mean: log_kernel_scale=+0.41, log_lengthscale=-1.04, log_noise_scale=-2.34
  - Previous expE used phi_init from numpyro prior (log_ls=1.93) → chain stayed
    in prior basin, Welford captured inverted ridge (corr=-0.8085).
  - MCLMC Exp2 "mode" (-1.37, -0.44, -2.32) was from non-converged chain
    (R-hat=1.76); log_ks=-0.44 is below GT q05=-0.265 → that init is in the
    posterior tail. GT mean is strictly better.
  - laplace_hmc (not dhmc): fixed num_integration_steps → no NUTS while_loop →
    no step_size collapse from tree expansion.

Budget (from expE calibration 0.127s/eval at bad init; mode likely faster):
  - Warmup: 200×10×0.127 = 254s. Sampling: 20×10×0.127 = 25s. Compile: ~30s.
  - Total estimate: ~5.2 min ✓ (within 10-min trial budget).

Gates (ratified by statistician):
  1. corr(log_ks, log_ls) from Welford IMM ≥ 0.6   (ridge capture)
  2. adapted step_size ∈ [0.01, 0.5]                (no collapse)
  3. acceptance rate ∈ [0.65, 0.95]                 (valid HMC)
  4. n_div = 0                                       (no numerical failure)

Decision tree:
  - ALL 4 pass → dense-IMM mechanism confirmed; A1 production-init probe next.
  - corr < 0.6, step OK → Welford needs more warmup; try n_warmup=500.
  - step_size < 0.01 → residual collapse even from mode; investigate.
  - acceptance > 0.95 → IMM too tight / step too small (micro-stepping).
  - acceptance < 0.65 or n_div > 0 → step too large; reduce initial_step_size.
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
N_WARMUP_STEPS = 200  # doubled from expE (100) — better Welford estimate at d=3
N_LEAPFROG = 10  # fixed-L for HMC (no tree expansion)
N_SAMPLING_STEPS = 20  # doubled from expE (10) — better acceptance rate estimate

# A2 init: GT posterior mean from certified groundtruth NUTS run (40K samples).
# Source: tuningfork/catalog/gp_regression/reference/summary.json
# CRITICAL: log_kernel_scale=+0.41 (positive — large GP amplitude), NOT -0.44 from
# the non-converged MCLMC Exp2 chain (which was below the GT q05=-0.265).
PHI_A2_INIT = {
    "log_kernel_scale": 0.40870562293007373,  # GT mean — INSIDE [q05=-0.265, q95=+1.137]
    "log_lengthscale": -1.0424925985381703,  # GT mean — INSIDE [q05=-1.361, q95=-0.781]
    "log_noise_scale": -2.34163615643574,  # GT mean — INSIDE [q05=-2.424, q95=-2.255]
}
# Note: the dict keys deliberately match the numpyro site names so unravel_fn works.

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

_phi_init_numpyro, log_joint_fn, theta_init, marginal_logdensity_fn = laplace_components

# A2 override: use GT mean, not the numpyro prior sample.
phi_init = PHI_A2_INIT
phi_keys_sorted = sorted(phi_init.keys())
phi_flat_init, unravel_fn = ravel_pytree(phi_init)

t_laplace = time.perf_counter()
print(
    f"[t=+{t_laplace - t0:.1f}s] Laplace components built",
    flush=True,
)
print(f"  phi_keys (sorted = flat order): {phi_keys_sorted}", flush=True)
print(f"  phi_init A2 (flat): {np.array(phi_flat_init)}", flush=True)

# Sanity check: marginal lp at A2 init should be finite and large
lp_a2 = float(marginal_logdensity_fn(phi_init))
print(
    f"  marginal lp at A2 init: {lp_a2:.3f} (should be finite, ~ -300 to -500)",
    flush=True,
)
if not np.isfinite(lp_a2):
    raise RuntimeError(
        f"marginal logdensity at A2 init is {lp_a2} — init is not in the posterior!"
    )

# For comparison: marginal lp at the numpyro prior sample (bad init from expE)
lp_prior = float(marginal_logdensity_fn(_phi_init_numpyro))
print(
    f"  marginal lp at numpyro prior sample (expE init): {lp_prior:.3f}",
    flush=True,
)
print(
    f"  Δlp(A2 vs prior) = {lp_a2 - lp_prior:+.3f} "
    "(A2 should be much higher — closer to mode)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Phase 1: HMC window_adaptation on marginal_logdensity_fn (3D), dense IMM
# ---------------------------------------------------------------------------
print(
    "\n=== Phase 1: HMC window_adaptation on 3D marginal "
    f"(n_warmup={N_WARMUP_STEPS}, L={N_LEAPFROG}, dense IMM, A2 init) ===",
    flush=True,
)

warmup = blackjax.window_adaptation(
    blackjax.hmc,
    marginal_logdensity_fn,
    is_mass_matrix_diagonal=False,  # dense 3×3 Welford covariance estimator
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
print("  Adapted IMM (3×3 dense):", flush=True)
for i, key_i in enumerate(phi_keys_sorted):
    row = "  ".join(f"{adapted_imm[i, j]:+.4f}" for j in range(3))
    print(f"    [{key_i:20s}] {row}", flush=True)

# Stds and correlation from adapted IMM
var_diag = np.diag(adapted_imm)
std_diag = np.sqrt(np.maximum(var_diag, 0))
print("\n  Marginal stds from Welford IMM:", flush=True)
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

# Welford IMM stds vs GT stds (reference)
gt_stds = {
    "log_kernel_scale": 0.4278,
    "log_lengthscale": 0.1783,
    "log_noise_scale": 0.0513,
}
print("\n  Welford stds vs GT stds:", flush=True)
for i, key_i in enumerate(phi_keys_sorted):
    gt_s = gt_stds.get(key_i, float("nan"))
    ratio = std_diag[i] / gt_s if gt_s > 0 else float("nan")
    print(
        f"    {key_i:25s}: welford={std_diag[i]:.4f}  GT={gt_s:.4f}  ratio={ratio:.2f}",
        flush=True,
    )

print("\n  === GATE 1: Ridge capture ===", flush=True)
print(f"  corr(log_kernel_scale, log_lengthscale) = {corr_ks_ls:.4f}", flush=True)
print(
    "  Target: +0.754 (groundtruth posterior; from Exp2 MCLMC + GT-confirmed)",
    flush=True,
)
gate1_ridge = corr_ks_ls >= 0.6
print(
    f"  Gate 1 (corr ≥ 0.6): {'PASS' if gate1_ridge else 'FAIL'} "
    f"(|corr - 0.754| = {abs(corr_ks_ls - 0.754):.4f})",
    flush=True,
)

print("\n  === GATE 2: Step_size not collapsed ===", flush=True)
gate2_step = 0.01 <= adapted_step_size <= 0.5
print(
    f"  step_size = {adapted_step_size:.6f}  target: [0.01, 0.5]",
    flush=True,
)
print(
    f"  Gate 2 (step_size in [0.01, 0.5]): {'PASS' if gate2_step else 'FAIL'}",
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
# Phase 2: 1-chain laplace_hmc sampling (N_SAMPLING_STEPS steps)
# Using laplace_hmc (not dhmc) — fixed-L, no dynamic while_loop
# Matches the 4-chain vmap kernel so Phase 3 is a direct scale-up
# ---------------------------------------------------------------------------
print(
    "\n=== Phase 2: 1-chain laplace_hmc sampling "
    f"(n_steps={N_SAMPLING_STEPS}, L={N_LEAPFROG}, adapted params) ===",
    flush=True,
)

algorithm_hmc = blackjax.laplace_hmc(
    log_joint_fn,
    theta_init,
    step_size=adapted_step_size,
    inverse_mass_matrix=jnp.array(adapted_imm),
    num_integration_steps=N_LEAPFROG,
)

t_sample1_start = time.perf_counter()
final_state_1, (history_states_1, history_info_1) = run_inference_algorithm(
    key_sample_1,
    algorithm_hmc,
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
phi_positions = np.array(
    jax.vmap(lambda pos: ravel_pytree(pos)[0])(history_states_1.position)
)

print(f"  1-chain wall: {sample1_wall:.2f}s", flush=True)
print(
    f"  Acceptance rate: mean={mean_acc_1:.3f}, min={accs_1.min():.3f}, max={accs_1.max():.3f}",
    flush=True,
)
print(f"  Divergences: {n_div_1}/{N_SAMPLING_STEPS}", flush=True)
print(f"  logdensity range: [{lps_1.min():.2f}, {lps_1.max():.2f}]", flush=True)
print("  phi trace (first 5 steps):", flush=True)
for step_i in range(min(5, N_SAMPLING_STEPS)):
    print(f"    step {step_i}: {phi_positions[step_i]}", flush=True)

print("\n  === GATE 3: Acceptance rate in [0.65, 0.95] ===", flush=True)
gate3_acc = 0.65 <= mean_acc_1 <= 0.95
print(
    f"  mean acceptance = {mean_acc_1:.3f}  target: [0.65, 0.95]",
    flush=True,
)
print(f"  Gate 3: {'PASS' if gate3_acc else 'FAIL'}", flush=True)
if mean_acc_1 > 0.95:
    print(
        "  WARN: acceptance > 0.95 — chain may be micro-stepping (step_size too small "
        "or IMM too tight for posterior scale). Check Welford std vs GT std.",
        flush=True,
    )
elif mean_acc_1 < 0.65:
    print(
        "  WARN: acceptance < 0.65 — step_size too large for adapted IMM. "
        "Reduce initial_step_size to 0.01 and retry.",
        flush=True,
    )

print("\n  === GATE 4: No divergences ===", flush=True)
gate4_divs = n_div_1 == 0
print(f"  n_div = {n_div_1}  target: 0", flush=True)
print(f"  Gate 4: {'PASS' if gate4_divs else 'FAIL'}", flush=True)

# ---------------------------------------------------------------------------
# Phase 3 (conditional): 4-chain vmap with laplace_hmc
# Only runs if 1-chain is clean (gates 3+4 both pass)
# Per expB: laplace_hmc under vmap = 2.5× overhead (vs dhmc 4.4×) — acceptable
# ---------------------------------------------------------------------------
vmap_wall = None
vmap_acc_mean = None
vmap_n_div = None

phase3_eligible = gate3_acc and gate4_divs
if phase3_eligible:
    print(
        "\n=== Phase 3: 4-chain vmap laplace_hmc "
        f"(n_steps={N_SAMPLING_STEPS}, L={N_LEAPFROG}) ===",
        flush=True,
    )
    print(
        "  NOTE: laplace_hmc (not dhmc) per expB — 2.5× vmap overhead vs dhmc's 4.4×",
        flush=True,
    )

    vmap_keys = jax.random.split(key_sample_4, 4)
    key_perturb = jax.random.key(SEED + 5)
    phi_starts_4 = phi_warmup_flat[None, :] + 0.05 * jax.random.normal(
        key_perturb, (4, len(phi_warmup_flat))
    )

    def run_chain_hmc(rng_key_single, phi_flat):
        algo = blackjax.laplace_hmc(
            log_joint_fn,
            theta_init,
            step_size=adapted_step_size,
            inverse_mass_matrix=jnp.array(adapted_imm),
            num_integration_steps=N_LEAPFROG,
        )
        _final, hist = run_inference_algorithm(
            rng_key_single,
            algo,
            num_steps=N_SAMPLING_STEPS,
            initial_position=unravel_fn(phi_flat),
            progress_bar=False,
        )
        return _final, hist

    t_vmap_start = time.perf_counter()
    vmap_finals, vmap_hists = jax.vmap(run_chain_hmc)(vmap_keys, phi_starts_4)
    _ = jax.block_until_ready(vmap_finals)
    t_vmap_end = time.perf_counter()

    _vmap_hist_states, vmap_hists_info = vmap_hists
    vmap_wall = t_vmap_end - t_vmap_start
    vmap_acc_mean = float(np.array(vmap_hists_info.acceptance_rate).mean())
    vmap_n_div = int(np.array(vmap_hists_info.is_divergent).sum())

    print(f"  4-chain vmap wall: {vmap_wall:.2f}s", flush=True)
    print(f"  Mean acceptance: {vmap_acc_mean:.3f}", flush=True)
    print(f"  Total divergences: {vmap_n_div}/{4 * N_SAMPLING_STEPS}", flush=True)
else:
    print(
        f"\n  Skipping Phase 3 (1-chain not clean: acc={mean_acc_1:.3f}, n_div={n_div_1})",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Summary + Gate verdict
# ---------------------------------------------------------------------------
t_total = time.perf_counter()
sep = "=" * 70
print(f"\n{sep}", flush=True)
print("ExpF SUMMARY (A2 probe — GT-mean init, dense 3×3 IMM)", flush=True)
print(f"Total wall = {t_total - t0:.1f}s", flush=True)
print(sep, flush=True)

gates = {
    "Gate 1 (corr ≥ 0.6)": gate1_ridge,
    "Gate 2 (step_size in [0.01, 0.5])": gate2_step,
    "Gate 3 (acceptance in [0.65, 0.95])": gate3_acc,
    "Gate 4 (n_div = 0)": gate4_divs,
}
for name, passed in gates.items():
    print(f"  {name}: {'PASS' if passed else 'FAIL'}", flush=True)

all_pass = all(gates.values())
print(
    f"\n  Overall: {'ALL GATES PASS — dense-IMM mechanism CONFIRMED' if all_pass else 'FAIL — see gate details above'}",
    flush=True,
)

if all_pass:
    print(
        "\n  → Next step: A1 probe (prior-means init) to test production-realistic",
        flush=True,
    )
    print(
        "    starting-position strategy. Or emit LOW recipe for laplace_hmc × W1 ×",
        flush=True,
    )
    print("    gp_regression with phi_init from two-phase approach.", flush=True)
elif not gate1_ridge and gate2_step:
    print(
        "\n  → Welford converging slowly. Try n_warmup=500 (same A2 init).", flush=True
    )
elif not gate2_step and adapted_step_size < 0.01:
    print(
        "\n  → Step_size collapse even from GT-mean init. Investigate marginal",
        flush=True,
    )
    print(
        "    geometry at the posterior mean. Check num_integration_steps.", flush=True
    )
elif not gate3_acc and mean_acc_1 > 0.95:
    print("\n  → Micro-stepping: Welford IMM stds too small vs GT stds.", flush=True)
    print(
        "    Increase n_warmup or check if marginal logdensity is mis-scaled.",
        flush=True,
    )
elif not gate3_acc and mean_acc_1 < 0.65:
    print("\n  → Step_size too large. Retry with initial_step_size=0.01.", flush=True)
elif not gate4_divs:
    print(
        "\n  → Divergences from GT-mean init: unexpected. Check L-BFGS convergence",
        flush=True,
    )
    print(
        "    at the posterior mean. The Laplace approximation may be failing.",
        flush=True,
    )

print(sep, flush=True)
print("\nDone.", flush=True)
