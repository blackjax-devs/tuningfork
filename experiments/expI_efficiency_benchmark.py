"""ExpI: laplace_hmc vs NUTS efficiency benchmark on gp_regression.

Question: Does laplace_hmc (d=3 marginal) beat NUTS (d=203 joint) on
ESS(phi)/wall_seconds when both get GT-calibrated step_size + IMM, no warmup?

Both arms use ground-truth parameters (no adaptation noise).  This is the
upper-bound on laplace_hmc efficiency — if it loses here, warmup engineering
is pointless.

Phase A (this run): 3-step pilot × 1 chain per arm.
  - Measures per-step wall time including JIT compile.
  - Reports projected n_samples for Phase B at a 120s budget.
  - Output goes to @tl + @statistician for spot-review before Phase B.

Phase B (pending TL review): full benchmark with n_chains=4 (laplace_hmc)
  and n_chains=2 sequential (NUTS, never vmap — expB proved vmap-NUTS hangs).

Hard guards (from TL brief):
  - Stock blackjax (NO jax.debug.callback in Arm A timing path)
  - NO debug_callback / io_callback / host-sync inside timed loops
  - block_until_ready ONLY at timed-block boundaries
  - JAX_ENABLE_X64=1
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
#
# The local blackjax is on branch laplace-convergence-check-925, which adds
# jax.debug.callback(ordered=True) to every solve_theta call.  That callback
# forces a host-sync per L-BFGS solve, artificially inflating Arm A timing.
# Fix: monkey-patch blackjax.mcmc.laplace_hmc.laplace_marginal_factory with
# a callback-free version before creating the algorithm.
# ---------------------------------------------------------------------------
from blackjax.mcmc.laplace_marginal import LaplaceMarginal  # noqa: E402
from blackjax.optimizers.lbfgs import minimize_lbfgs  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402


def laplace_marginal_factory_nocb(log_joint_fn, theta_init, **optimizer_kwargs):
    """Stock laplace_marginal_factory — NO jax.debug.callback overhead.

    Identical to blackjax.mcmc.laplace_marginal.laplace_marginal_factory
    from commit 007a9ded (main), which has no convergence-warning callback.
    Used here so Arm A timing is not contaminated by host-sync overhead.
    """
    optimizer_kwargs.setdefault("maxiter", 30)  # stock default; caller overrides

    theta_flat_init, unravel_theta = ravel_pytree(theta_init)
    d = theta_flat_init.shape[0]

    def solve_theta(phi, theta_prev=None):
        initial = theta_prev if theta_prev is not None else theta_init

        def objective(theta):
            return -log_joint_fn(theta, phi)

        result, _ = minimize_lbfgs(objective, initial, **optimizer_kwargs)
        # NO jax.debug.callback here — avoids host-sync overhead
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


# Patch BEFORE importing blackjax (which triggers laplace_hmc import)
import blackjax.mcmc.laplace_hmc as _lhmc  # noqa: E402

_lhmc.laplace_marginal_factory = laplace_marginal_factory_nocb

import blackjax  # noqa: E402

from tuningfork.model import MODELS  # noqa: E402
from tuningfork.model._numpyro import build_logdensity_fn  # noqa: E402
from tuningfork.recipes._recipe_runner import _LAPLACE_PHI_THETA_SPLITS  # noqa: E402

t_imports = time.perf_counter()
print(f"[t=+{t_imports - t0:.1f}s] All imports done (laplace_hmc patched)", flush=True)

# ---------------------------------------------------------------------------
# Config (from §16)
# ---------------------------------------------------------------------------
SEED = 20260517
MAXITER = 500  # Production default; zero perf cost (just optimizer_kwargs)
N_PILOT = 3  # Phase A: 3 steps per arm
N_LEAPFROG_LAPLACE = 10  # Arm A: L=10

# Arm A — laplace_hmc GT params
PHI_INIT = {
    "log_kernel_scale": jnp.float64(0.40870562293007373),
    "log_lengthscale": jnp.float64(-1.0424925985381703),
    "log_noise_scale": jnp.float64(-2.34163615643574),
}
STEP_SIZE_LAPLACE = 0.526  # expH calibrated

# GT phi posterior covariance (computed from 40K certified NUTS draws)
# key order: log_kernel_scale, log_lengthscale, log_noise_scale (alphabetical)
IMM_3X3 = jnp.array(
    [
        [0.18301258, 0.05751162, -0.00021748],
        [0.05751162, 0.03180439, -0.00022324],
        [-0.00021748, -0.00022324, 0.00262740],
    ],
    dtype=jnp.float64,
)

# Arm B — NUTS GT params (from adaptation.json)
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


# NUTS full-joint initial position: phi at GT mean + f_raw=zeros (NCP prior mean)
full_init_position = {
    **{k: PHI_INIT[k] for k in phi_sites},
    "f_raw": jnp.zeros(200, dtype=jnp.float64),
}

print(
    f"  phi_sites={phi_sites}, theta_sites={theta_sites[:1]}...(200D)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Build algorithms (NO warmup)
# ---------------------------------------------------------------------------

# Arm A: laplace_hmc with GT 3x3 IMM (uses patched callback-free factory)
alg_a = blackjax.laplace_hmc(
    log_joint_fn,
    theta_init,
    step_size=STEP_SIZE_LAPLACE,
    inverse_mass_matrix=IMM_3X3,
    num_integration_steps=N_LEAPFROG_LAPLACE,
    maxiter=MAXITER,
)

# Arm B: NUTS on full joint (d=203), GT diagonal IMM
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
# Phase A — 3-step pilot, Arm A (laplace_hmc)
# ---------------------------------------------------------------------------
print(
    f"\n=== Phase A Pilot: Arm A — laplace_hmc (d=3, maxiter={MAXITER}, L={N_LEAPFROG_LAPLACE}) ===",
    flush=True,
)
print("  3 steps × 1 chain; includes JIT compile in step 1", flush=True)

t_a_init_start = time.perf_counter()
state_a = alg_a.init(PHI_INIT)
jax.block_until_ready(state_a)
t_a_init_end = time.perf_counter()
print(f"  Arm A init: {t_a_init_end - t_a_init_start:.3f}s", flush=True)

# JIT the step function once (shared across pilot calls)
step_a_jit = jax.jit(alg_a.step)

t_a_pilot_start = time.perf_counter()
for i in range(N_PILOT):
    t_step_start = time.perf_counter()
    key_a, subkey = jax.random.split(key_a)
    state_a, info_a = step_a_jit(subkey, state_a)
    jax.block_until_ready(state_a)
    t_step_end = time.perf_counter()
    print(
        f"  step {i + 1}: {t_step_end - t_step_start:.3f}s  "
        f"acc={float(info_a.acceptance_rate):.4f}  "
        f"phi_flat={[float(state_a.position[k]) for k in sorted(state_a.position.keys())]}",
        flush=True,
    )
t_a_pilot_end = time.perf_counter()

wall_a_total = t_a_pilot_end - t_a_pilot_start
t_per_step_a = wall_a_total / N_PILOT
print(f"  Arm A pilot: {wall_a_total:.3f}s total, {t_per_step_a:.3f}s/step", flush=True)

# ---------------------------------------------------------------------------
# Phase A — 3-step pilot, Arm B (NUTS, d=203 joint)
# ---------------------------------------------------------------------------
print(
    f"\n=== Phase A Pilot: Arm B — NUTS (d=203, max_doublings={MAX_NUM_DOUBLINGS}) ===",
    flush=True,
)
print(
    "  3 steps × 1 chain sequential (NEVER vmap NUTS — compile-hang risk)", flush=True
)

# Warn if NUTS is expected to be slow
expected_leapfrog_per_step = 2047  # from adaptation.json median
print(
    f"  GT median leapfrog/step = {expected_leapfrog_per_step} "
    f"(is_turning=0.9997 → essentially deterministic)",
    flush=True,
)

t_b_init_start = time.perf_counter()
state_b = alg_b.init(full_init_position)
jax.block_until_ready(state_b)
t_b_init_end = time.perf_counter()
print(f"  Arm B init: {t_b_init_end - t_b_init_start:.3f}s", flush=True)

step_b_jit = jax.jit(alg_b.step)

t_b_pilot_start = time.perf_counter()
for i in range(N_PILOT):
    t_step_start = time.perf_counter()
    key_b, subkey = jax.random.split(key_b)
    state_b, info_b = step_b_jit(subkey, state_b)
    jax.block_until_ready(state_b)
    t_step_end = time.perf_counter()
    n_leapfrog_actual = int(
        getattr(info_b, "num_integration_steps", expected_leapfrog_per_step)
    )
    print(
        f"  step {i + 1}: {t_step_end - t_step_start:.3f}s  "
        f"acc={float(info_b.acceptance_rate):.4f}  "
        f"n_leapfrog={n_leapfrog_actual}",
        flush=True,
    )
    elapsed = t_step_end - t_b_pilot_start
    if elapsed > 60 and i == 0:
        print(
            "[WARN] NUTS arm step 1 took >60s. "
            "Steps 2-3 should be faster (JIT warm). "
            "If >300s total, consider reporting timing only.",
            flush=True,
        )
t_b_pilot_end = time.perf_counter()

wall_b_total = t_b_pilot_end - t_b_pilot_start
t_per_step_b = wall_b_total / N_PILOT
print(f"  Arm B pilot: {wall_b_total:.3f}s total, {t_per_step_b:.3f}s/step", flush=True)

# ---------------------------------------------------------------------------
# Phase A summary and Phase B projection
# ---------------------------------------------------------------------------
budget_sec = 120.0
n_samples_laplace = max(50, min(200, int(budget_sec / t_per_step_a)))
n_samples_nuts = max(10, min(200, int(budget_sec / t_per_step_b)))

if budget_sec / t_per_step_b < 10:
    print(
        f"[WARN] NUTS arm may time out at 10 samples "
        f"(projected {budget_sec / t_per_step_b:.1f} samples at {budget_sec}s budget). "
        "Consider reducing to 5 or reporting timing only.",
        flush=True,
    )

t_total = time.perf_counter()
sep = "=" * 68
print(f"\n{sep}", flush=True)
print(
    f"ExpI Phase A Summary (total wall = {t_total - t0:.1f}s)",
    flush=True,
)
print(sep, flush=True)
print(
    f"  Arm A (laplace_hmc): {t_per_step_a:.3f} s/step  "
    f"(includes JIT compile in step 1)",
    flush=True,
)
print(
    f"  Arm B (NUTS d=203):  {t_per_step_b:.3f} s/step  "
    f"(includes JIT compile in step 1)",
    flush=True,
)
print(
    f"  Ratio Arm B / Arm A: {t_per_step_b / t_per_step_a:.1f}×  "
    f"(> 1 means NUTS is slower per step)",
    flush=True,
)
print(f"\n  Phase B projection (budget={budget_sec:.0f}s per arm):", flush=True)
print(
    f"    n_samples_laplace (4 chains vmap) = {n_samples_laplace}",
    flush=True,
)
print(
    f"    n_samples_nuts (2 chains sequential) = {n_samples_nuts}",
    flush=True,
)
print(
    f"    Expected laplace wall = {n_samples_laplace * t_per_step_a:.1f}s "
    f"(JIT already warm from pilot)",
    flush=True,
)
print(
    f"    Expected NUTS wall    = {n_samples_nuts * t_per_step_b:.1f}s "
    f"(JIT already warm from pilot)",
    flush=True,
)
print("\n  Phase B awaits TL spot-review.", flush=True)
print(sep, flush=True)
