"""ExpI v3: Verify Concern 1 (maxiter pass-through) and diagnose actual iter_num.

TL Concern: maxiter=500 may not flow through the setdefault — Arm A may be at maxiter=30.
Diagnosis: instrument the factory to log optimizer_kwargs + actual iter_num per solve.

Also: run 1 step using stock blackjax (007a9ded) by switching off monkey-patch,
to confirm Concern 2 (reimplementation drift) is a non-issue.
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

sys.path.insert(0, "/home/jp/blackjax-devs/tuningfork")
sys.path.insert(0, "/home/jp/blackjax-devs/blackjax")

from blackjax.mcmc.laplace_marginal import LaplaceMarginal  # noqa: E402
from blackjax.optimizers.lbfgs import minimize_lbfgs  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

# Shared state for instrumentation
_solve_count = [0]
_iter_log = []

MAXITER_PASSED = 500  # what we pass to blackjax.laplace_hmc


def factory_instrumented(log_joint_fn, theta_init, **optimizer_kwargs):
    """Instrumented version: logs optimizer_kwargs and iter_num from each solve."""
    print(
        f"\n  [factory_instrumented] optimizer_kwargs RECEIVED: {optimizer_kwargs}",
        flush=True,
    )
    optimizer_kwargs.setdefault("maxiter", 30)
    print(
        f"  [factory_instrumented] optimizer_kwargs AFTER setdefault: {optimizer_kwargs}",
        flush=True,
    )
    print(
        f"  [factory_instrumented] => maxiter in use = {optimizer_kwargs['maxiter']}",
        flush=True,
    )

    theta_flat_init, unravel_theta = ravel_pytree(theta_init)
    d = theta_flat_init.shape[0]

    def solve_theta(phi, theta_prev=None):
        initial = theta_prev if theta_prev is not None else theta_init

        def objective(theta):
            return -log_joint_fn(theta, phi)

        result, _ = minimize_lbfgs(objective, initial, **optimizer_kwargs)

        def _log_call(iter_num, error):
            _solve_count[0] += 1
            _iter_log.append(int(iter_num))
            if _solve_count[0] <= 5:
                print(
                    f"  [solve #{_solve_count[0]}] iter_num={int(iter_num):4d}  "
                    f"error={float(error):.3e}  "
                    f"maxiter_in_kwargs={optimizer_kwargs['maxiter']}",
                    flush=True,
                )

        jax.debug.callback(_log_call, result.state.iter_num, result.state.error)
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

_lhmc.laplace_marginal_factory = factory_instrumented

import blackjax  # noqa: E402

from tuningfork.model import MODELS  # noqa: E402
from tuningfork.model._numpyro import build_logdensity_fn  # noqa: E402
from tuningfork.recipes._recipe_runner import _LAPLACE_PHI_THETA_SPLITS  # noqa: E402

print(f"[t=+{time.perf_counter() - t0:.1f}s] Imports done", flush=True)

# Build model
rng_key = jax.random.key(20260517)
key_init, key_a = jax.random.split(rng_key, 2)

entry = MODELS["gp_regression"]
init_position, logdensity_fn, _ = build_logdensity_fn(key_init, entry)

phi_sites, theta_sites = _LAPLACE_PHI_THETA_SPLITS["gp_regression"]
theta_init = {k: init_position[k] for k in theta_sites}

PHI_INIT = {
    "log_kernel_scale": jnp.float64(0.40870562293007373),
    "log_lengthscale": jnp.float64(-1.0424925985381703),
    "log_noise_scale": jnp.float64(-2.34163615643574),
}

IMM_3X3 = jnp.array(
    [
        [0.18301258, 0.05751162, -0.00021748],
        [0.05751162, 0.03180439, -0.00022324],
        [-0.00021748, -0.00022324, 0.00262740],
    ],
    dtype=jnp.float64,
)


def log_joint_fn(theta, phi):
    return logdensity_fn({**theta, **phi})


print(f"\n=== Concern 1: maxiter={MAXITER_PASSED} pass-through check ===", flush=True)
print(f"Calling blackjax.laplace_hmc(..., maxiter={MAXITER_PASSED})", flush=True)

_solve_count[0] = 0
_iter_log.clear()

alg = blackjax.laplace_hmc(
    log_joint_fn,
    theta_init,
    step_size=0.526,
    inverse_mass_matrix=IMM_3X3,
    num_integration_steps=10,
    maxiter=MAXITER_PASSED,
)

print("\n--- alg.init (cold start from phi_GT) ---", flush=True)
t_init = time.perf_counter()
state = alg.init(PHI_INIT)
jax.block_until_ready(state)
print(
    f"  Init wall: {time.perf_counter() - t_init:.3f}s, "
    f"solve calls={_solve_count[0]}, iter_nums={_iter_log[:10]}",
    flush=True,
)

print("\n--- 3 warm steps (warm-start from theta_star) ---", flush=True)
step_jit = jax.jit(alg.step)
for i in range(3):
    _solve_count[0] = 0
    _iter_log.clear()
    t_step = time.perf_counter()
    state, info = step_jit(key_a, state)
    jax.block_until_ready(state)
    key_a, _ = jax.random.split(key_a)
    print(
        f"  step {i + 1}: wall={time.perf_counter() - t_step:.3f}s  "
        f"solve_calls={_solve_count[0]}  iter_nums={_iter_log}",
        flush=True,
    )

print("\n=== VERDICT ===", flush=True)
print(
    "If all iter_nums << 500: warm-starting converges early; v2 timing IS representative of production.",
    flush=True,
)
print(
    "If all iter_nums == 500: hits ceiling every solve; need to re-time at maxiter=500 vs 30.",
    flush=True,
)
print(
    "The setdefault('maxiter', 30) is irrelevant when maxiter=500 is explicitly passed.",
    flush=True,
)
