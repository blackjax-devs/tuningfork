"""ExpF: HMC window_adaptation (dense 3x3 IMM, prior-means init) probe.

Key question: does window_adaptation(hmc, marginal_fn, is_mass_matrix_diagonal=False)
with prior-MEANS init (not a random prior sample) capture the +0.754 log_ls/log_ks ridge?

Root cause of expE failure: phi_init was a random prior SAMPLE with log_ls=1.93
(1.9-sigma bad-luck draw), placing the chain 3.3 units from the posterior mode.
This probe uses prior MEANS (log_ls=0.0) — 1.04 units from the GT mode (-1.04),
well within the 1-sigma posterior band. The chain traversal gap is 3.1x smaller.

Spec (TL-ratified 2026-05-24):
  phi_init = {log_kernel_scale: 0.0, log_lengthscale: 0.0, log_noise_scale: -2.0}
  n_warmup=200, initial_step_size=0.1, is_mass_matrix_diagonal=False
  laplace_hmc, num_integration_steps=20, n_samples=20, 1 chain
  Hard kill: 10 min external; surface if compile-phase > 5 min

Stationarity check (gating the ridge verdict):
  "late-warmup log_ls" proxy = final warmup position log_ls + sampling mean log_ls
  Gate: late-warmup mean must be within 0.5 of GT mean (-1.04) to confirm Welford
  window estimated STATIONARY geometry. If still far from -1.04, corr<0.6 is
  inconclusive (transient contamination), not a structural failure.

Go/no-go table (@statistician, TL-ratified):
  corr >= 0.6 AND step_size in [0.01, 0.5] AND acceptance >= 0.65 => UNLOCK
  corr 0.3-0.6 AND step_size reasonable                            => PARTIAL
  corr < 0.3 OR step_size outside [0.001, 0.5]                    => FAIL

If UNLOCK: scale to 4 chains immediately (Phase 3).
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
# Config (TL-ratified spec 2026-05-24)
# ---------------------------------------------------------------------------
SEED = 20260517
N_WARMUP = 200
N_LEAPFROG = 20
N_SAMPLES = 20
INITIAL_STEP_SIZE = 0.1
GT_LOG_LS_MEAN = -1.04  # groundtruth posterior mean for log_lengthscale
STATIONARITY_TOL = 0.5  # late-warmup log_ls must be within this of GT mean

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

phi_init_prior_sample, log_joint_fn, theta_init, marginal_logdensity_fn = (
    laplace_components
)

# Override phi_init to prior MEANS (deterministic, not a random sample)
# phi_init from prior sample had log_ls=1.93 (1.9-sigma bad draw).
# Prior means: log_ls=0.0, log_ks=0.0, log_ns=-2.0 (prior mean of Normal(-2,1))
phi_init = {
    "log_kernel_scale": jnp.float64(0.0),
    "log_lengthscale": jnp.float64(0.0),
    "log_noise_scale": jnp.float64(-2.0),
}

phi_keys_sorted = sorted(phi_init.keys())
phi_flat_init, unravel_fn = ravel_pytree(phi_init)

print(
    f"[t=+{time.perf_counter() - t0:.1f}s] Laplace components built",
    flush=True,
)
print(f"  phi_keys (sorted = flat order): {phi_keys_sorted}", flush=True)
print(
    f"  phi_init PRIOR MEANS (flat): {np.array(phi_flat_init)}  "
    f"[prior sample was {np.array(ravel_pytree(phi_init_prior_sample)[0])}]",
    flush=True,
)
print(f"  GT log_lengthscale mean: {GT_LOG_LS_MEAN}", flush=True)
print(
    f"  Init gap to GT mode: |log_ls_init - GT_mean| = "
    f"|0.0 - {GT_LOG_LS_MEAN}| = {abs(0.0 - GT_LOG_LS_MEAN):.2f} units",
    flush=True,
)

idx_ks = phi_keys_sorted.index("log_kernel_scale")
idx_ls = phi_keys_sorted.index("log_lengthscale")

# ---------------------------------------------------------------------------
# Phase 1: HMC window_adaptation on marginal_logdensity_fn (3D, DENSE IMM)
# n_warmup=200, L=20, initial_step_size=0.1, is_mass_matrix_diagonal=False
# ---------------------------------------------------------------------------
print(
    "\n=== Phase 1: HMC window_adaptation (dense 3x3 IMM, prior-means init) ===",
    flush=True,
)
print(
    f"  n_warmup={N_WARMUP}, L={N_LEAPFROG}, "
    f"initial_step_size={INITIAL_STEP_SIZE}, dense=True",
    flush=True,
)

warmup = blackjax.window_adaptation(
    blackjax.hmc,
    marginal_logdensity_fn,
    is_mass_matrix_diagonal=False,
    initial_step_size=INITIAL_STEP_SIZE,
    target_acceptance_rate=0.8,
    num_integration_steps=N_LEAPFROG,
    progress_bar=False,
)

t_warmup_start = time.perf_counter()
(warmup_result, warmup_info) = warmup.run(key_warmup, phi_init, num_steps=N_WARMUP)
warmup_state = warmup_result.state
adapted_params = warmup_result.parameters
_ = jax.block_until_ready(warmup_state)
t_warmup_end = time.perf_counter()

warmup_wall = t_warmup_end - t_warmup_start
adapted_step_size = float(adapted_params["step_size"])
adapted_imm = np.array(adapted_params["inverse_mass_matrix"])

print(f"  Warmup wall: {warmup_wall:.2f}s", flush=True)

# Check for 5-min compile threshold
if warmup_wall > 300:
    print(
        f"  WARNING: warmup wall={warmup_wall:.0f}s > 5-min threshold — "
        "dense-vmap architectural issue suspected",
        flush=True,
    )

print(f"  Adapted step_size: {adapted_step_size:.6f}", flush=True)
print("  Adapted IMM (3x3 dense):", flush=True)
for i, key_i in enumerate(phi_keys_sorted):
    row = "  ".join(f"{adapted_imm[i, j]:+.4f}" for j in range(3))
    print(f"    [{key_i:20s}] {row}", flush=True)

# Stds and correlation from adapted IMM
var_diag = np.diag(adapted_imm)
std_diag = np.sqrt(np.maximum(var_diag, 0))
print("\n  Marginal stds from IMM:", flush=True)
for i, key_i in enumerate(phi_keys_sorted):
    print(f"    {key_i:25s}: std = {std_diag[i]:.4f}", flush=True)

cov_ks_ls = adapted_imm[idx_ks, idx_ls]
corr_ks_ls = (
    cov_ks_ls / (std_diag[idx_ks] * std_diag[idx_ls])
    if (std_diag[idx_ks] > 0 and std_diag[idx_ls] > 0)
    else 0.0
)

print("\n  === RIDGE TEST ===", flush=True)
print(
    f"  corr(log_kernel_scale, log_lengthscale) = {corr_ks_ls:.4f}",
    flush=True,
)
print("  Target: +0.754 (from Exp 2 MCLMC posterior ridge)", flush=True)

# Apply @statistician's gates
if corr_ks_ls >= 0.6 and 0.01 <= adapted_step_size <= 0.5:
    ridge_verdict = "UNLOCK"
elif 0.3 <= corr_ks_ls < 0.6:
    ridge_verdict = "PARTIAL"
else:
    ridge_verdict = "FAIL"

print(
    f"  Ridge gate (corr>=0.6 + step_size 0.01-0.5): {ridge_verdict}",
    flush=True,
)
step_size_gate_str = "PASS" if 0.01 <= adapted_step_size <= 0.5 else "FAIL"
print(
    f"  step_size gate (0.01-0.5): {step_size_gate_str}",
    flush=True,
)

# Stationarity proxy: final warmup position log_ls
phi_warmup_end = warmup_state.position
log_ls_warmup_final = float(phi_warmup_end["log_lengthscale"])
phi_flat_warmup = np.array(ravel_pytree(phi_warmup_end)[0])

print("\n  === STATIONARITY CHECK (proxy: final warmup position) ===", flush=True)
print(
    f"  log_ls at init:         {0.0:.4f}  (prior mean)",
    flush=True,
)
print(
    f"  log_ls at warmup end:   {log_ls_warmup_final:.4f}  "
    f"(GT mean = {GT_LOG_LS_MEAN})",
    flush=True,
)
dist_from_gt = abs(log_ls_warmup_final - GT_LOG_LS_MEAN)
stationary_warmup = dist_from_gt <= STATIONARITY_TOL
stationary_warmup_str = (
    "STATIONARY" if stationary_warmup else "TRANSIENT (corr may be false negative)"
)
print(
    f"  |final - GT_mean| = {dist_from_gt:.4f}  "
    f"(tol={STATIONARITY_TOL}) => {stationary_warmup_str}",
    flush=True,
)
print(f"  Final warmup phi (flat): {phi_flat_warmup}", flush=True)
marginal_lp_end = float(marginal_logdensity_fn(phi_warmup_end))
print(
    f"  Marginal lp at warmup end: {marginal_lp_end:.3f}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Phase 2: 1-chain laplace_hmc sampling (20 steps, adapted params)
# Using hmc (not dhmc) per TL brief — consistent with 4-chain vmap in Phase 3
# ---------------------------------------------------------------------------
print(
    f"\n=== Phase 2: 1-chain laplace_hmc sampling "
    f"(n_steps={N_SAMPLES}, L={N_LEAPFROG}, adapted params) ===",
    flush=True,
)

algorithm = blackjax.laplace_hmc(
    log_joint_fn,
    theta_init,
    step_size=adapted_step_size,
    inverse_mass_matrix=jnp.array(adapted_imm),
    num_integration_steps=N_LEAPFROG,
)

t_sample1_start = time.perf_counter()
final_state_1, (history_states_1, history_info_1) = run_inference_algorithm(
    key_sample_1,
    algorithm,
    num_steps=N_SAMPLES,
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

# Sampling log_ls trajectory (stationarity proxy)
phi_positions = np.array(
    jax.vmap(lambda pos: ravel_pytree(pos)[0])(history_states_1.position)
)
log_ls_samples = phi_positions[:, idx_ls]
log_ls_sample_mean = float(log_ls_samples.mean())
log_ls_sample_std = float(log_ls_samples.std())
dist_sampling_from_gt = abs(log_ls_sample_mean - GT_LOG_LS_MEAN)
stationary_sampling = dist_sampling_from_gt <= STATIONARITY_TOL

print(f"  1-chain wall: {sample1_wall:.2f}s", flush=True)
print(
    f"  Acceptance rate: mean={mean_acc_1:.3f}, "
    f"min={accs_1.min():.3f}, max={accs_1.max():.3f}",
    flush=True,
)
print(f"  Divergences: {n_div_1}/{N_SAMPLES}", flush=True)
print(
    f"  logdensity range: [{lps_1.min():.2f}, {lps_1.max():.2f}]",
    flush=True,
)

print("\n  === STATIONARITY CHECK (sampling trajectory) ===", flush=True)
print(
    f"  log_ls sampling: mean={log_ls_sample_mean:.4f}, "
    f"std={log_ls_sample_std:.4f}  (GT mean={GT_LOG_LS_MEAN})",
    flush=True,
)
print(
    f"  |sampling_mean - GT_mean| = {dist_sampling_from_gt:.4f}  "
    f"(tol={STATIONARITY_TOL}) => "
    + ("STATIONARY" if stationary_sampling else "TRANSIENT"),
    flush=True,
)

# Acceptance gate
acc_gate = mean_acc_1 >= 0.65
acc_gate_str = "PASS" if acc_gate else "FAIL"
print(f"\n  acceptance gate (>=0.65): {acc_gate_str}", flush=True)

# Composite verdict
unlock = corr_ks_ls >= 0.6 and 0.01 <= adapted_step_size <= 0.5 and acc_gate

print("\n  phi trace (5 steps):", flush=True)
for step_i in range(min(5, N_SAMPLES)):
    print(f"    step {step_i}: {phi_positions[step_i]}", flush=True)

# ---------------------------------------------------------------------------
# Phase 3 (conditional on UNLOCK): 4-chain vmap laplace_hmc
# Using laplace_hmc (not dhmc) per TL brief — avoids 4.4x vmap tax
# ---------------------------------------------------------------------------
vmap_wall = None
vmap_acc_mean = None
vmap_n_div = None

if unlock:
    print(
        f"\n=== Phase 3: 4-chain vmap laplace_hmc (UNLOCK triggered) "
        f"(n_steps={N_SAMPLES}, L={N_LEAPFROG}) ===",
        flush=True,
    )
    vmap_keys = jax.random.split(key_sample_4, 4)
    phi_flat_warmup_j = jnp.array(phi_flat_warmup)
    key_perturb = jax.random.key(SEED + 7)
    # Small perturbations around the warmup endpoint
    phi_starts_4 = phi_flat_warmup_j[None, :] + 0.05 * jax.random.normal(
        key_perturb, (4, len(phi_flat_warmup))
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
            num_steps=N_SAMPLES,
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

    # 4-chain log_ls means
    vmap_phi_flat = np.array(
        jax.vmap(
            lambda states: jax.vmap(lambda pos: ravel_pytree(pos)[0])(states.position)
        )(vmap_hists_states)
    )
    vmap_log_ls_mean = float(vmap_phi_flat[:, :, idx_ls].mean())
    vmap_log_ls_std = float(vmap_phi_flat[:, :, idx_ls].std())

    print(f"  4-chain vmap wall: {vmap_wall:.2f}s", flush=True)
    print(f"  Mean acceptance: {vmap_acc_mean:.3f}", flush=True)
    print(
        f"  Total divergences: {vmap_n_div}/{4 * N_SAMPLES}",
        flush=True,
    )
    print(
        f"  4-chain log_ls: mean={vmap_log_ls_mean:.4f}, "
        f"std={vmap_log_ls_std:.4f}  (GT mean={GT_LOG_LS_MEAN})",
        flush=True,
    )
else:
    print(
        f"\n  Skipping Phase 3 (UNLOCK not triggered): "
        f"corr={corr_ks_ls:.4f}, step_size={adapted_step_size:.6f}, "
        f"acc={mean_acc_1:.3f}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Summary — three-number verdict tuple
# ---------------------------------------------------------------------------
t_total = time.perf_counter()
sep = "=" * 66
print(f"\n{sep}", flush=True)
print(f"ExpF Summary (total wall = {t_total - t0:.1f}s)", flush=True)
print(sep, flush=True)

print("  === THREE-NUMBER VERDICT TUPLE ===", flush=True)
print(
    f"  (1) corr(log_ks, log_ls) = {corr_ks_ls:.4f}  " f"(target +0.754, gate >=0.6)",
    flush=True,
)
print(
    f"  (2) adapted_step_size    = {adapted_step_size:.6f}  " f"(gate 0.01-0.5)",
    flush=True,
)
print(
    f"  (3) mean acceptance      = {mean_acc_1:.3f}          " f"(gate >=0.65)",
    flush=True,
)
print(f"  Ridge verdict: {ridge_verdict}", flush=True)
print(
    f"  Acceptance gate: {'PASS' if acc_gate else 'FAIL'}",
    flush=True,
)
print(
    f"  Stationarity (warmup final): "
    f"{'STATIONARY' if stationary_warmup else 'TRANSIENT'}  "
    f"(log_ls={log_ls_warmup_final:.3f}, GT={GT_LOG_LS_MEAN})",
    flush=True,
)
print(
    f"  Stationarity (sampling mean): "
    f"{'STATIONARY' if stationary_sampling else 'TRANSIENT'}  "
    f"(log_ls={log_ls_sample_mean:.3f}, GT={GT_LOG_LS_MEAN})",
    flush=True,
)
print(
    f"  OVERALL: {'UNLOCK — laplace track UNLOCKED' if unlock else 'NO UNLOCK'}",
    flush=True,
)

if vmap_wall is not None:
    print("  4-chain vmap:", flush=True)
    print(
        f"    wall={vmap_wall:.2f}s, acc={vmap_acc_mean:.3f}, " f"n_div={vmap_n_div}",
        flush=True,
    )

print(sep, flush=True)
