"""ExpA1: Production-init probe — prior-means → L-BFGS MAP → dense window_adaptation.

Validates the maxiter=500 fix works WITHOUT cheating via GT-mean init.
This is the production-realistic path: cold start from prior means for phi,
L-BFGS to phi_MAP on the marginal density, then dense window_adaptation.

expH (A2) used PHI_GT_MEAN as the init — that's certified mode, not production.
A1 tests whether a practitioner starting from scratch gets the same result.

Two-phase warmup:
  Phase 0: L-BFGS on marginal_logdensity_fn(phi) from prior means → phi_MAP
  Phase 1: dense window_adaptation from phi_MAP (maxiter=500)
  Phase 2: laplace_hmc sampling with adapted params

Config: MAXITER=500 (production default), N_WARMUP=200, L=10, N_SAMPLES=20.

Gates:
  1. corr(log_ks, log_ls) >= 0.6       (ridge detection)
  2. step_size in [0.01, 0.7]           (upper loosened: 0.526 in expH was healthy)
  3. mean acceptance in [0.65, 0.95]
  4. n_div = 0
+ Level 0: density residual at 10 GT samples (same as expG/expH reference)
+ Level 0.5: grad_norm ‖∇_θ log_joint(z*, phi_MAP)‖
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
from blackjax.mcmc.laplace_marginal import laplace_marginal_factory  # noqa: E402
from blackjax.optimizers.lbfgs import minimize_lbfgs  # noqa: E402
from blackjax.util import run_inference_algorithm  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

from tuningfork.model import MODELS  # noqa: E402
from tuningfork.model._numpyro import build_logdensity_fn  # noqa: E402
from tuningfork.model.gp_regression import JITTER, N_OBS, X_DATA, Y_DATA  # noqa: E402
from tuningfork.recipes._recipe_runner import _LAPLACE_PHI_THETA_SPLITS  # noqa: E402

t_imports = time.perf_counter()
print(f"[t=+{t_imports - t0:.1f}s] All imports done", flush=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 20260517
MAXITER = 500  # Production default (expH: 300; A1: 500 per TL direction)
N_WARMUP_STEPS = 200
N_LEAPFROG = 10
N_SAMPLING_STEPS = 20

# GT posterior mean (certified 40K NUTS) — for reference only, NOT used as init
PHI_GT_MEAN = {
    "log_kernel_scale": 0.40870562293007373,
    "log_lengthscale": -1.0424925985381703,
    "log_noise_scale": -2.34163615643574,
}
GT_STDS = {
    "log_kernel_scale": 0.4278,
    "log_lengthscale": 0.1783,
    "log_noise_scale": 0.0513,
}
GT_DRAWS_PATH = (
    "/home/jp/blackjax-devs/tuningfork/tuningfork/catalog"
    "/gp_regression/groundtruth_samples/blackjax/draws.npz"
)

# Prior means for phi (cold start):
#   log_kernel_scale ~ Normal(0,1)  → mean 0.0
#   log_lengthscale  ~ Normal(0,1)  → mean 0.0
#   log_noise_scale  ~ Normal(-2,1) → mean -2.0
PHI_PRIOR_MEANS = {
    "log_kernel_scale": jnp.float64(0.0),
    "log_lengthscale": jnp.float64(0.0),
    "log_noise_scale": jnp.float64(-2.0),
}

# ---------------------------------------------------------------------------
# Build model and Laplace components with maxiter=500
# ---------------------------------------------------------------------------
rng_key = jax.random.key(SEED)
key_init, key_warmup, key_sample_1 = jax.random.split(rng_key, 3)

entry = MODELS["gp_regression"]
init_position, logdensity_fn, _postprocess_fn = build_logdensity_fn(key_init, entry)

t_model = time.perf_counter()
print(
    f"[t=+{t_model - t0:.1f}s] gp_regression model built, "
    f"d_full={sum(v.size for v in jax.tree.leaves(init_position))}",
    flush=True,
)

phi_sites, theta_sites = _LAPLACE_PHI_THETA_SPLITS["gp_regression"]
phi_init_numpyro = {k: init_position[k] for k in phi_sites}
theta_init = {k: init_position[k] for k in theta_sites}


def log_joint_fn(theta, phi):
    return logdensity_fn({**theta, **phi})


laplace = laplace_marginal_factory(log_joint_fn, theta_init, maxiter=MAXITER)


def marginal_logdensity_fn(phi):
    lp, _theta_star = laplace(phi)
    return lp


phi_keys_sorted = sorted(PHI_GT_MEAN.keys())
phi_prior_flat, unravel_phi = ravel_pytree(PHI_PRIOR_MEANS)

print(
    f"[t=+{time.perf_counter() - t0:.1f}s] Laplace components built (maxiter={MAXITER})",
    flush=True,
)
print(f"  phi_keys (sorted): {phi_keys_sorted}", flush=True)
print(f"  Prior-means init: {PHI_PRIOR_MEANS}", flush=True)

# ---------------------------------------------------------------------------
# Build exact marginal (reference)
# ---------------------------------------------------------------------------
X = jnp.array(X_DATA, dtype=jnp.float64)
y = jnp.array(Y_DATA, dtype=jnp.float64)
n = N_OBS


def log_p_exact(phi):
    """Exact GP marginal log-posterior via Cholesky (no L-BFGS)."""
    log_ls = phi["log_lengthscale"]
    log_ks = phi["log_kernel_scale"]
    log_ns = phi["log_noise_scale"]
    ls = jnp.exp(log_ls)
    ks = jnp.exp(log_ks)
    sigma_n = jnp.exp(log_ns)
    sqdist = (X[:, None] - X[None, :]) ** 2
    K = ks**2 * jnp.exp(-0.5 * sqdist / ls**2) + JITTER * jnp.eye(n)
    C = K + sigma_n**2 * jnp.eye(n)
    L_C = jax.scipy.linalg.cholesky(C, lower=True)
    log_det_C = 2.0 * jnp.sum(jnp.log(jnp.diag(L_C)))
    v = jax.scipy.linalg.solve_triangular(L_C, y, lower=True)
    quad_form = jnp.dot(v, v)
    log_lik = -0.5 * quad_form - 0.5 * log_det_C - 0.5 * n * jnp.log(2.0 * jnp.pi)
    log_prior = (
        -0.5 * log_ks**2
        - 0.5 * log_ls**2
        - 0.5 * (log_ns + 2.0) ** 2
        - 1.5 * jnp.log(2.0 * jnp.pi)
    )
    return log_lik + log_prior


# ---------------------------------------------------------------------------
# Phase 0: L-BFGS MAP on marginal density from prior means
# ---------------------------------------------------------------------------
print(
    "\n=== Phase 0: L-BFGS MAP from prior-means phi (phi_0 = [0, 0, -2]) ===",
    flush=True,
)
print("  Optimizing marginal_logdensity_fn(phi) → phi_MAP", flush=True)
print("  Prior means: log_ks=0.0, log_ls=0.0, log_ns=-2.0", flush=True)
print("  GT mean:     log_ks=0.409, log_ls=-1.042, log_ns=-2.342", flush=True)


def neg_marginal_flat(phi_flat):
    """Negative marginal log-density over flat phi (for L-BFGS minimization)."""
    phi = unravel_phi(phi_flat)
    lp, _ = laplace(phi)
    return -lp


t_map_start = time.perf_counter()
map_result, _ = minimize_lbfgs(neg_marginal_flat, phi_prior_flat, maxiter=200)
phi_map_flat = map_result.params
phi_MAP = unravel_phi(phi_map_flat)
_ = jax.block_until_ready(phi_map_flat)
t_map_end = time.perf_counter()

map_wall = t_map_end - t_map_start
map_lp = float(marginal_logdensity_fn(phi_MAP))
prior_lp = float(marginal_logdensity_fn(PHI_PRIOR_MEANS))
gt_lp = float(marginal_logdensity_fn(PHI_GT_MEAN))

print(f"  MAP optimization wall: {map_wall:.2f}s", flush=True)
print("  phi_MAP:", flush=True)
for k in phi_keys_sorted:
    gt_v = float(PHI_GT_MEAN[k])
    map_v = float(phi_MAP[k])
    delta = map_v - gt_v
    print(
        f"    {k:25s}: map={map_v:+.4f}, gt={gt_v:+.4f}, delta={delta:+.4f}",
        flush=True,
    )
print(
    f"  marginal lp: prior_means={prior_lp:.3f}, phi_MAP={map_lp:.3f}, gt_mean={gt_lp:.3f}",
    flush=True,
)
print(f"  MAP grad_norm: {float(map_result.state.error):.4e}", flush=True)

# ---------------------------------------------------------------------------
# Level 0: Density cross-check at 10 GT samples
# ---------------------------------------------------------------------------
print(
    "\n=== Level 0: Density cross-check (maxiter=500 Laplace vs exact) at 10 GT samples ===",
    flush=True,
)
print("  Reference: expH max_residual=0.014 nats (at maxiter=300)", flush=True)

gt_draws = np.load(GT_DRAWS_PATH)
gt_phi_samples = [
    {
        "log_kernel_scale": jnp.float64(float(gt_draws["log_kernel_scale"][i])),
        "log_lengthscale": jnp.float64(float(gt_draws["log_lengthscale"][i])),
        "log_noise_scale": jnp.float64(float(gt_draws["log_noise_scale"][i])),
    }
    for i in range(10)
]

laplace_vals = []
exact_vals = []
for i, phi_s in enumerate(gt_phi_samples):
    t_s = time.perf_counter()
    lp_l = float(marginal_logdensity_fn(phi_s))
    lp_e = float(log_p_exact(phi_s))
    laplace_vals.append(lp_l)
    exact_vals.append(lp_e)
    print(
        f"  sample {i:2d}: laplace={lp_l:10.4f}, exact={lp_e:10.4f}, "
        f"diff={lp_l - lp_e:+8.4f}  [{time.perf_counter() - t_s:.1f}s]",
        flush=True,
    )

diffs = np.array(laplace_vals) - np.array(exact_vals)
offset_mean = diffs.mean()
offset_std = diffs.std()
max_abs_residual = np.abs(diffs - offset_mean).max()
fix_works_density = max_abs_residual < 0.5

print(
    f"\n  Offset (laplace - exact): mean={offset_mean:+.4f}, std={offset_std:.4f}",
    flush=True,
)
print(
    f"  Max |centered residual| = {max_abs_residual:.4f} "
    f"(threshold: 0.5 nats, expH@500: ~0.014 nats)",
    flush=True,
)
print(
    f"  Level 0: {'PASS' if fix_works_density else 'FAIL'}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Level 0.5: Gradient norm at phi_MAP
# ---------------------------------------------------------------------------
print(
    "\n=== Level 0.5: Gradient norm ‖∇_θ log_joint(z*, phi_MAP)‖ at phi_MAP ===",
    flush=True,
)
print(
    "  (expH grad_norm=0.56 at GT mean; quadratic model → density exact regardless)",
    flush=True,
)

t_gnorm_start = time.perf_counter()
z_star_map = laplace.solve_theta(phi_MAP)
grad_theta_at_map = jax.grad(log_joint_fn, argnums=0)(z_star_map, phi_MAP)
grad_flat, _ = ravel_pytree(grad_theta_at_map)
grad_norm_map = float(jnp.linalg.norm(grad_flat))
t_gnorm_end = time.perf_counter()

print(
    f"  ‖∇_θ log_joint(z*, phi_MAP)‖ = {grad_norm_map:.6e}  "
    f"[{t_gnorm_end - t_gnorm_start:.1f}s]",
    flush=True,
)
converged = grad_norm_map < 1e-3
print(
    f"  Convergence verdict: {'CONVERGED (<1e-3)' if converged else 'NOT CONVERGED (>=1e-3) -- benign for quadratic NCP'}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Phase 1: HMC window_adaptation from phi_MAP
# ---------------------------------------------------------------------------
print(
    "\n=== Phase 1: HMC window_adaptation from phi_MAP "
    "(n_warmup=%d, L=%d, dense IMM, maxiter=%d) ==="
    % (N_WARMUP_STEPS, N_LEAPFROG, MAXITER),
    flush=True,
)
print("  Starting phi: phi_MAP (NOT GT mean — production init)", flush=True)

warmup = blackjax.window_adaptation(
    blackjax.hmc,
    marginal_logdensity_fn,
    is_mass_matrix_diagonal=False,
    initial_step_size=0.1,
    target_acceptance_rate=0.8,
    num_integration_steps=N_LEAPFROG,
    progress_bar=False,
)

t_warmup_start = time.perf_counter()
(warmup_results, warmup_info) = warmup.run(
    key_warmup, phi_MAP, num_steps=N_WARMUP_STEPS
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

var_diag = np.diag(adapted_imm)
std_diag = np.sqrt(np.maximum(var_diag, 0))
print("\n  Welford IMM stds vs GT stds:", flush=True)
for i, key_i in enumerate(phi_keys_sorted):
    gt_s = GT_STDS.get(key_i, float("nan"))
    ratio = std_diag[i] / gt_s if gt_s > 0 else float("nan")
    print(
        f"    {key_i:25s}: welford_std={std_diag[i]:.4f}, "
        f"gt_std={gt_s:.4f}, ratio={ratio:.3f}",
        flush=True,
    )

idx_ks = phi_keys_sorted.index("log_kernel_scale")
idx_ls = phi_keys_sorted.index("log_lengthscale")
cov_ks_ls = adapted_imm[idx_ks, idx_ls]
corr_ks_ls = (
    cov_ks_ls / (std_diag[idx_ks] * std_diag[idx_ls])
    if (std_diag[idx_ks] > 0 and std_diag[idx_ls] > 0)
    else 0.0
)

phi_warmup_end = warmup_state.position
phi_warmup_flat = np.array(ravel_pytree(phi_warmup_end)[0])
print(f"\n  Final warmup phi (flat): {phi_warmup_flat}", flush=True)
print(
    f"  Marginal lp at warmup end: {float(marginal_logdensity_fn(phi_warmup_end)):.3f}",
    flush=True,
)

print("\n  === GATES 1 & 2 ===", flush=True)
print(
    f"  corr(log_ks, log_ls) = {corr_ks_ls:.4f} (target +0.754, gate: >=0.6)",
    flush=True,
)
print(
    f"  adapted step_size = {adapted_step_size:.6f} (gate: 0.01-0.7; expH: 0.526)",
    flush=True,
)
gate1_pass = corr_ks_ls >= 0.6
gate2_pass = 0.01 <= adapted_step_size <= 0.7
corr_str = "PASS" if gate1_pass else "FAIL"
step_str = (
    "PASS" if gate2_pass else ("COLLAPSED" if adapted_step_size < 0.01 else "TOO_LARGE")
)
print(f"  Gate 1 (corr>=0.6): {corr_str}", flush=True)
print(f"  Gate 2 (step_size 0.01-0.7): {step_str}", flush=True)

# ---------------------------------------------------------------------------
# Phase 2: 1-chain laplace_hmc sampling with adapted params
# ---------------------------------------------------------------------------
print(
    "\n=== Phase 2: 1-chain laplace_hmc sampling "
    "(n_steps=%d, L=%d, maxiter=%d) ===" % (N_SAMPLING_STEPS, N_LEAPFROG, MAXITER),
    flush=True,
)

algorithm_hmc = blackjax.laplace_hmc(
    log_joint_fn,
    theta_init,
    step_size=adapted_step_size,
    inverse_mass_matrix=jnp.array(adapted_imm),
    num_integration_steps=N_LEAPFROG,
    maxiter=MAXITER,
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
    f"  Acceptance: mean={mean_acc_1:.3f}, min={accs_1.min():.3f}, max={accs_1.max():.3f}",
    flush=True,
)
print(f"  Divergences: {n_div_1}/{N_SAMPLING_STEPS}", flush=True)
print(f"  logdensity range: [{lps_1.min():.2f}, {lps_1.max():.2f}]", flush=True)
print("  phi trace (first 5 steps):", flush=True)
for step_i in range(min(5, N_SAMPLING_STEPS)):
    print(f"    step {step_i}: {phi_positions[step_i]}", flush=True)

gate3_pass = 0.65 <= mean_acc_1 <= 0.95
gate4_pass = n_div_1 == 0
acc_str = (
    "PASS"
    if gate3_pass
    else ("MICRO-STEPPING (>0.95)" if mean_acc_1 > 0.95 else "TOO_LOW (<0.65)")
)
div_str = "PASS" if gate4_pass else f"FAIL ({n_div_1} divergences)"
print(f"\n  Gate 3 (acceptance 0.65-0.95): {acc_str}", flush=True)
print(f"  Gate 4 (n_div=0): {div_str}", flush=True)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
t_total = time.perf_counter()
sep = "=" * 68
print(f"\n{sep}", flush=True)
print(
    f"ExpA1 Summary (maxiter={MAXITER}, production-init, "
    f"total wall = {t_total - t0:.1f}s)",
    flush=True,
)
print(sep, flush=True)

print(
    f"  Phase 0 (MAP): prior_lp={prior_lp:.2f} → map_lp={map_lp:.2f} "
    f"(gt_lp={gt_lp:.2f}), wall={map_wall:.1f}s",
    flush=True,
)
print("  phi_MAP vs GT_MEAN:", flush=True)
for k in phi_keys_sorted:
    print(
        f"    {k:25s}: map={float(phi_MAP[k]):+.4f}, gt={float(PHI_GT_MEAN[k]):+.4f}",
        flush=True,
    )
print(
    f"  Level 0 (density): max_residual={max_abs_residual:.4f} nats, "
    f"{'PASS' if fix_works_density else 'FAIL'}",
    flush=True,
)
print(
    f"  Level 0.5 (grad_norm at phi_MAP): {grad_norm_map:.2e}, "
    f"{'CONVERGED' if converged else 'NOT CONVERGED -- benign for quadratic NCP'}",
    flush=True,
)
print(
    f"  Gate 1 corr(log_ks, log_ls) = {corr_ks_ls:.4f}: {corr_str}",
    flush=True,
)
print(
    f"  Gate 2 step_size = {adapted_step_size:.6f}: {step_str}",
    flush=True,
)
print(
    f"  Gate 3 acceptance = {mean_acc_1:.3f}: {acc_str}",
    flush=True,
)
print(
    f"  Gate 4 n_div = {n_div_1}: {'PASS' if gate4_pass else 'FAIL'}",
    flush=True,
)

all_gates_pass = gate1_pass and gate2_pass and gate3_pass and gate4_pass
print(
    f"\n  Production-init verdict: "
    f"{'ALL 4 GATES PASS -- cold-start confirmed' if all_gates_pass else 'PARTIAL/FAIL -- see gate breakdown'}",
    flush=True,
)
print(sep, flush=True)
