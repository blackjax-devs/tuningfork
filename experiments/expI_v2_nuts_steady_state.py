"""ExpI v2: laplace_hmc vs NUTS efficiency benchmark — NUTS Arm B corrected.

v1 finding: Arm B (NUTS) was initialized at f_raw=zeros (NCP prior mode, far
from posterior), giving depths 2–4 (7–31 leapfrogs) vs GT steady-state depth
~11 (2047 leapfrogs). The "5× faster" claim in Phase A v1 was an artifact.

v2 fix (TL + statistician design):
  (a) Compute f_raw posterior mean (closed-form for linear-Gaussian NCP model)
      at GT phi → use as NUTS init instead of zeros.
  (b) Burn NUTS in for up to 30 steps; gate: median leapfrog ≥ 500 after 10.
  (c) Time 3 steps AFTER gate passes; report per-step leapfrog counts to
      confirm GT-representativeness.

Validity note (@tl): if timed NUTS leapfrogs are materially below 2047
(@statistician will check), a scaling adjustment is needed before speedup
verdict. Raw counts are always reported.

f_raw posterior mean formula (linear-Gaussian GP NCP):
  Prior:  f_raw ~ N(0, I)
  Likelihood: y ~ N(L @ f_raw, σ²I)
  Posterior precision: Λ = I + Lᵀ L / σ²
  Posterior mean:      μ = Λ⁻¹ Lᵀ y / σ²
  i.e., solve(I + L.T @ L / σ², L.T @ y) / σ²   (TL-verified formula)
"""

import json
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
    f"[t=+{t_jax - t0:.1f}s] JAX: x64={jax.config.read('jax_enable_x64')}, "
    f"backend={jax.default_backend()}",
    flush=True,
)

sys.path.insert(0, "/home/jp/blackjax-devs/tuningfork")
sys.path.insert(0, "/home/jp/blackjax-devs/blackjax")

# ---------------------------------------------------------------------------
# Patch laplace_hmc to use a callback-free laplace_marginal_factory.
# (Same as v1 — avoids jax.debug.callback overhead from parked PR branch.)
# ---------------------------------------------------------------------------
from blackjax.mcmc.laplace_marginal import LaplaceMarginal  # noqa: E402
from blackjax.optimizers.lbfgs import minimize_lbfgs  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402


def laplace_marginal_factory_nocb(log_joint_fn, theta_init, **optimizer_kwargs):
    """Stock laplace_marginal_factory — NO jax.debug.callback overhead."""
    optimizer_kwargs.setdefault("maxiter", 30)

    theta_flat_init, unravel_theta = ravel_pytree(theta_init)
    d = theta_flat_init.shape[0]

    def solve_theta(phi, theta_prev=None):
        initial = theta_prev if theta_prev is not None else theta_init

        def objective(theta):
            return -log_joint_fn(theta, phi)

        result, _ = minimize_lbfgs(objective, initial, **optimizer_kwargs)
        return result.params

    def get_theta_star(phi, theta_prev=None):
        def f_residual(theta_flat):
            theta = unravel_theta(theta_flat)
            grad_theta = jax.grad(log_joint_fn, argnums=0)(theta, phi)
            grad_flat, _ = ravel_pytree(grad_theta)
            return grad_flat

        def solve_root(f, x0):
            del f
            theta_star = solve_theta(phi, theta_prev)
            theta_star_flat, _ = ravel_pytree(theta_star)
            return theta_star_flat

        def tangent_solve(g, y):
            J = jax.jacobian(g)(jnp.zeros_like(theta_flat_init))
            return jnp.linalg.solve(J, y)

        theta_flat_star = jax.lax.custom_root(
            f_residual, theta_flat_init, solve_root, tangent_solve
        )
        return unravel_theta(theta_flat_star)

    def log_marginal(phi, theta_prev=None):
        theta_star = get_theta_star(phi, theta_prev)
        theta_flat_star, _ = ravel_pytree(theta_star)

        def log_joint_flat(t_flat):
            return log_joint_fn(unravel_theta(t_flat), phi)

        log_p_star = log_joint_flat(theta_flat_star)
        neg_hess = jax.hessian(lambda t: -log_joint_flat(t))(theta_flat_star)
        _, log_abs_det = jnp.linalg.slogdet(neg_hess)
        lp = log_p_star - 0.5 * log_abs_det + 0.5 * d * jnp.log(2.0 * jnp.pi)
        return lp, theta_star

    def sample_theta(rng_key, phi, theta_star):
        theta_flat_star, _ = ravel_pytree(theta_star)

        def log_joint_flat(t_flat):
            return log_joint_fn(unravel_theta(t_flat), phi)

        neg_hess = jax.hessian(lambda t: -log_joint_flat(t))(theta_flat_star)
        L = jnp.linalg.cholesky(neg_hess)
        z = jax.random.normal(rng_key, (d,))
        x_flat = jax.lax.linalg.triangular_solve(
            L, z, left_side=True, lower=True, transpose_a=True
        )
        return unravel_theta(theta_flat_star + x_flat)

    return LaplaceMarginal(
        solve_theta=solve_theta,
        get_theta_star=get_theta_star,
        log_marginal=log_marginal,
        sample_theta=sample_theta,
    )


import blackjax.mcmc.laplace_hmc as _lhmc  # noqa: E402

_lhmc.laplace_marginal_factory = laplace_marginal_factory_nocb

import blackjax  # noqa: E402

from tuningfork.model import MODELS  # noqa: E402
from tuningfork.model._numpyro import build_logdensity_fn  # noqa: E402
from tuningfork.recipes._recipe_runner import _LAPLACE_PHI_THETA_SPLITS  # noqa: E402

t_imports = time.perf_counter()
print(
    f"[t=+{t_imports - t0:.1f}s] All imports done (laplace_hmc patched)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 20260517
MAXITER = 500
N_PILOT = 3  # timed steps after burn-in gate
N_LEAPFROG_LAPLACE = 10

BURN_IN_MIN = 10  # run at least this many burn-in steps before gate check
BURN_IN_MAX = 30  # give up and time anyway after this many burn-in steps
LEAPFROG_GATE = 500  # median leapfrog threshold to confirm steady-state

# Arm A GT params
PHI_INIT = {
    "log_kernel_scale": jnp.float64(0.40870562293007373),
    "log_lengthscale": jnp.float64(-1.0424925985381703),
    "log_noise_scale": jnp.float64(-2.34163615643574),
}
STEP_SIZE_LAPLACE = 0.526

IMM_3X3 = jnp.array(
    [
        [0.18301258, 0.05751162, -0.00021748],
        [0.05751162, 0.03180439, -0.00022324],
        [-0.00021748, -0.00022324, 0.00262740],
    ],
    dtype=jnp.float64,
)

# Arm B GT params
ADAPTATION_JSON_PATH = (
    "/home/jp/blackjax-devs/tuningfork/tuningfork/catalog"
    "/gp_regression/reference/adaptation.json"
)
with open(ADAPTATION_JSON_PATH) as f:
    adaptation_data = json.load(f)

STEP_SIZE_NUTS = adaptation_data["step_size"]
IMM_NUTS = jnp.array(adaptation_data["inverse_mass_matrix"], dtype=jnp.float64)
MAX_NUM_DOUBLINGS = 12

print(
    f"  NUTS step_size={STEP_SIZE_NUTS:.8f}, IMM shape={IMM_NUTS.shape}, "
    f"max_doublings={MAX_NUM_DOUBLINGS}",
    flush=True,
)
print(f"  Laplace step_size={STEP_SIZE_LAPLACE}, L={N_LEAPFROG_LAPLACE}", flush=True)

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
rng_key = jax.random.key(SEED)
key_init, key_a, key_b = jax.random.split(rng_key, 3)

entry = MODELS["gp_regression"]
init_position, logdensity_fn, _postprocess_fn = build_logdensity_fn(key_init, entry)

t_model = time.perf_counter()
print(
    f"[t=+{t_model - t0:.1f}s] gp_regression model built, "
    f"d_full={sum(v.size for v in jax.tree.leaves(init_position))}",
    flush=True,
)

phi_sites, theta_sites = _LAPLACE_PHI_THETA_SPLITS["gp_regression"]
theta_init = {k: init_position[k] for k in theta_sites}


def log_joint_fn(theta, phi):
    return logdensity_fn({**theta, **phi})


# ---------------------------------------------------------------------------
# Compute f_raw posterior mean (closed-form, linear-Gaussian NCP)
#
# Model: f_raw ~ N(0,I), y ~ N(L_K @ f_raw, σ²I)
# Posterior mean: μ = (I + Lᵀ L / σ²)⁻¹ @ (Lᵀ y / σ²)
#
# We compute L_K at GT phi values.
# ---------------------------------------------------------------------------
from tuningfork.model.gp_regression import JITTER, N_OBS, X_DATA, Y_DATA  # noqa: E402

print("\n=== Computing f_raw posterior mean at GT phi ===", flush=True)

kernel_scale_gt = float(jnp.exp(PHI_INIT["log_kernel_scale"]))
lengthscale_gt = float(jnp.exp(PHI_INIT["log_lengthscale"]))
noise_scale_gt = float(jnp.exp(PHI_INIT["log_noise_scale"]))

print(
    f"  GT phi: kernel_scale={kernel_scale_gt:.4f}, "
    f"lengthscale={lengthscale_gt:.4f}, noise_scale={noise_scale_gt:.4f}",
    flush=True,
)

sqdist = (X_DATA[:, None] - X_DATA[None, :]) ** 2
K_gt = kernel_scale_gt**2 * jnp.exp(
    -0.5 * sqdist / lengthscale_gt**2
) + JITTER * jnp.eye(N_OBS)
L_K_gt = jax.scipy.linalg.cholesky(K_gt, lower=True)

sigma2 = noise_scale_gt**2
# Posterior precision: Λ = I + L.T @ L / σ²
precision = jnp.eye(N_OBS) + L_K_gt.T @ L_K_gt / sigma2
# Posterior mean: μ = Λ⁻¹ (L.T @ y / σ²)
f_raw_posterior_mean = jnp.linalg.solve(precision, L_K_gt.T @ Y_DATA / sigma2)

f_raw_init_norm = float(jnp.linalg.norm(f_raw_posterior_mean))
print(f"  f_raw_posterior_mean norm = {f_raw_init_norm:.4f}", flush=True)
print(
    "  (v1 used f_raw=zeros with norm=0.0 — this is the posterior mean, "
    "far from the prior mode)",
    flush=True,
)

# NUTS full-joint initial position: phi at GT mean + f_raw at posterior mean
full_init_position = {
    **{k: PHI_INIT[k] for k in phi_sites},
    "f_raw": f_raw_posterior_mean,
}

# ---------------------------------------------------------------------------
# Build algorithms
# ---------------------------------------------------------------------------
alg_a = blackjax.laplace_hmc(
    log_joint_fn,
    theta_init,
    step_size=STEP_SIZE_LAPLACE,
    inverse_mass_matrix=IMM_3X3,
    num_integration_steps=N_LEAPFROG_LAPLACE,
    maxiter=MAXITER,
)

alg_b = blackjax.nuts(
    logdensity_fn,
    step_size=STEP_SIZE_NUTS,
    inverse_mass_matrix=IMM_NUTS,
    max_num_doublings=MAX_NUM_DOUBLINGS,
)

t_build = time.perf_counter()
print(
    f"[t=+{t_build - t0:.1f}s] Algorithms built (no warmup)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Phase A v2 — Arm A (laplace_hmc): 3 timed steps, no burn-in needed
# (laplace_hmc init is already at phi_GT so it's in the stationary regime)
# ---------------------------------------------------------------------------
print(
    f"\n=== Phase A v2: Arm A — laplace_hmc (d=3, maxiter={MAXITER}, L={N_LEAPFROG_LAPLACE}) ===",
    flush=True,
)
print("  No burn-in (phi_GT init is already representative)", flush=True)

t_a_init_start = time.perf_counter()
state_a = alg_a.init(PHI_INIT)
jax.block_until_ready(state_a)
t_a_init_end = time.perf_counter()
print(f"  Arm A init: {t_a_init_end - t_a_init_start:.3f}s", flush=True)

step_a_jit = jax.jit(alg_a.step)

t_a_pilot_start = time.perf_counter()
step_times_a = []
for i in range(N_PILOT):
    t_step_start = time.perf_counter()
    key_a, subkey = jax.random.split(key_a)
    state_a, info_a = step_a_jit(subkey, state_a)
    jax.block_until_ready(state_a)
    t_step_end = time.perf_counter()
    dt = t_step_end - t_step_start
    step_times_a.append(dt)
    print(
        f"  step {i + 1}: {dt:.3f}s  "
        f"acc={float(info_a.acceptance_rate):.4f}  "
        f"L_fixed={N_LEAPFROG_LAPLACE}  "
        f"phi=[{float(state_a.position['log_kernel_scale']):.4f}, "
        f"{float(state_a.position['log_lengthscale']):.4f}, "
        f"{float(state_a.position['log_noise_scale']):.4f}]",
        flush=True,
    )
t_a_pilot_end = time.perf_counter()

wall_a_total = t_a_pilot_end - t_a_pilot_start
t_per_step_a = wall_a_total / N_PILOT
# Post-JIT is steps 2+ (step 1 includes JIT compile)
t_per_step_a_postjit = sum(step_times_a[1:]) / max(len(step_times_a) - 1, 1)
print(
    f"  Arm A pilot: {wall_a_total:.3f}s total, "
    f"{t_per_step_a:.3f}s/step (incl JIT), "
    f"{t_per_step_a_postjit:.3f}s/step (post-JIT)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Phase A v2 — Arm B (NUTS): burn-in until median leapfrog ≥ 500, then time
# ---------------------------------------------------------------------------
print(
    f"\n=== Phase A v2: Arm B — NUTS (d=203, max_doublings={MAX_NUM_DOUBLINGS}) ===",
    flush=True,
)
print(
    f"  f_raw init: posterior mean (norm={f_raw_init_norm:.4f}), phi at GT mean",
    flush=True,
)
print(
    f"  Burn-in protocol: run {BURN_IN_MIN} steps, gate median_leapfrog >= {LEAPFROG_GATE}",
    flush=True,
)
print(f"  Max burn-in: {BURN_IN_MAX} steps. Then time {N_PILOT} steps.", flush=True)

t_b_init_start = time.perf_counter()
state_b = alg_b.init(full_init_position)
jax.block_until_ready(state_b)
t_b_init_end = time.perf_counter()
print(f"  Arm B init: {t_b_init_end - t_b_init_start:.3f}s", flush=True)

step_b_jit = jax.jit(alg_b.step)

# --- Burn-in phase ---
print(
    f"\n  [Burn-in] up to {BURN_IN_MAX} steps, gate: median_leapfrog >= {LEAPFROG_GATE}",
    flush=True,
)
burnin_leapfrogs = []
gate_passed = False
t_burnin_start = time.perf_counter()

for i in range(BURN_IN_MAX):
    t_step_start = time.perf_counter()
    key_b, subkey = jax.random.split(key_b)
    state_b, info_b = step_b_jit(subkey, state_b)
    jax.block_until_ready(state_b)
    t_step_end = time.perf_counter()

    n_lf = int(getattr(info_b, "num_integration_steps", 0))
    burnin_leapfrogs.append(n_lf)
    print(
        f"  burn {i + 1:2d}: {t_step_end - t_step_start:.3f}s  "
        f"n_leapfrog={n_lf:5d}  acc={float(info_b.acceptance_rate):.4f}",
        flush=True,
    )

    # Check gate after min burn-in
    if i + 1 >= BURN_IN_MIN:
        recent = burnin_leapfrogs[max(0, len(burnin_leapfrogs) - 5) :]
        median_recent = sorted(recent)[len(recent) // 2]
        if median_recent >= LEAPFROG_GATE:
            print(
                f"\n  [Burn-in] GATE PASSED at step {i + 1}: "
                f"median(last {len(recent)} steps) = {median_recent} >= {LEAPFROG_GATE}",
                flush=True,
            )
            gate_passed = True
            break

t_burnin_end = time.perf_counter()
print(
    f"  Burn-in total: {t_burnin_end - t_burnin_start:.3f}s, "
    f"{len(burnin_leapfrogs)} steps",
    flush=True,
)
if not gate_passed:
    print(
        f"  [WARN] Gate NOT passed after {BURN_IN_MAX} burn-in steps. "
        f"Last 5 leapfrog counts: {burnin_leapfrogs[-5:]}. "
        f"Proceeding to timing anyway — report these counts to @statistician.",
        flush=True,
    )

# --- Timed phase ---
print(f"\n  [Timed] {N_PILOT} steps after burn-in:", flush=True)
t_b_pilot_start = time.perf_counter()
step_times_b = []
timed_leapfrogs = []

for i in range(N_PILOT):
    t_step_start = time.perf_counter()
    key_b, subkey = jax.random.split(key_b)
    state_b, info_b = step_b_jit(subkey, state_b)
    jax.block_until_ready(state_b)
    t_step_end = time.perf_counter()

    dt = t_step_end - t_step_start
    n_lf = int(getattr(info_b, "num_integration_steps", 0))
    step_times_b.append(dt)
    timed_leapfrogs.append(n_lf)
    print(
        f"  step {i + 1}: {dt:.3f}s  "
        f"n_leapfrog={n_lf:5d}  acc={float(info_b.acceptance_rate):.4f}",
        flush=True,
    )

t_b_pilot_end = time.perf_counter()

wall_b_total = t_b_pilot_end - t_b_pilot_start
t_per_step_b = wall_b_total / N_PILOT
median_timed_lf = sorted(timed_leapfrogs)[len(timed_leapfrogs) // 2]

print(
    f"\n  Arm B timed: {wall_b_total:.3f}s total, " f"{t_per_step_b:.3f}s/step",
    flush=True,
)
print(
    f"  Timed leapfrog counts: {timed_leapfrogs}  "
    f"(median={median_timed_lf}, GT=2047)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Phase A v2 Summary
# ---------------------------------------------------------------------------
budget_sec = 120.0
n_samples_laplace = max(50, min(200, int(budget_sec / t_per_step_a_postjit)))
n_samples_nuts = max(10, min(200, int(budget_sec / t_per_step_b)))

# Leapfrog-count-aware scaling (@statistician verdict check)
scale_factor = 2047.0 / max(median_timed_lf, 1)
t_per_step_b_scaled = t_per_step_b * scale_factor
ratio_raw = t_per_step_b / t_per_step_a_postjit
ratio_scaled = t_per_step_b_scaled / t_per_step_a_postjit

t_total = time.perf_counter()
sep = "=" * 68
print(f"\n{sep}", flush=True)
print(f"ExpI Phase A v2 Summary (total wall = {t_total - t0:.1f}s)", flush=True)
print(sep, flush=True)
print(
    f"  Arm A (laplace_hmc): {t_per_step_a_postjit:.3f} s/step (post-JIT, L=10 fixed)",
    flush=True,
)
print(
    f"  Arm B (NUTS d=203):  {t_per_step_b:.3f} s/step (timed, "
    f"median_leapfrog={median_timed_lf})",
    flush=True,
)
print(
    f"  GT NUTS median leapfrog = 2047 → scaling factor = {scale_factor:.2f}",
    flush=True,
)
print(
    f"  Arm B scaled to GT-representative: {t_per_step_b_scaled:.3f} s/step",
    flush=True,
)
print(
    f"\n  Ratio B/A (raw, timed leapfrogs):      {ratio_raw:.2f}×",
    flush=True,
)
print(
    f"  Ratio B/A (scaled to GT 2047 lf):     {ratio_scaled:.2f}×",
    flush=True,
)
print("  (> 1 means NUTS is slower per step)", flush=True)
print(
    f"\n  Verdict validity (@statistician check): "
    f"timed median={median_timed_lf} vs GT=2047. "
    f"{'NEAR GT — no adjustment needed' if median_timed_lf >= 1500 else 'BELOW GT — scaled ratio is the valid comparand'}",
    flush=True,
)
print(f"\n  Burn-in leapfrog sequence: {burnin_leapfrogs}", flush=True)
print(f"  Timed leapfrog sequence:   {timed_leapfrogs}", flush=True)
print(
    f"\n  Phase B projection (budget={budget_sec:.0f}s per arm):",
    flush=True,
)
print(
    f"    n_samples_laplace (4 chains vmap)    = {n_samples_laplace}",
    flush=True,
)
print(
    f"    n_samples_nuts (2 chains sequential) = {n_samples_nuts}",
    flush=True,
)
print("\n  Phase B awaits TL spot-review.", flush=True)
print(sep, flush=True)
