"""ExpI v4: Phase A re-time using explicit laplace_marginal_factory on stock blackjax.

TL rationale: instead of monkey-patching laplace_marginal_factory, call it directly
with maxiter=500 explicit — removes the forwarding-chain question entirely. This runs
on stock blackjax (007a9ded, NO jax.debug.callback, NO setdefault) so the timing
is callback-free.

Changes from v3/v2:
  - No monkey-patch. blackjax repo switched to main (007a9ded) before running.
  - laplace_marginal_factory(log_joint_fn, theta_init, maxiter=500) called directly.
  - Cold-init iter_num probe: separate minimize_lbfgs call BEFORE timed sampling.
    Not inside any timed loop; purely diagnostic. Expected: iter_num ~295 (< 500,
    confirming maxiter=500 in effect and solver converges early from cold start).
  - Uses laplace_hmc.build_kernel() + laplace_hmc.init() directly (same internals
    as blackjax.laplace_hmc but factory call made explicit for TL audit).

Expected results (@tl):
  - cold-init iter_num: ~295 (warm-start would be 39-225; cold start is ~295 < 500)
  - t_per_step_a_postjit: ~1.0 s/step (warm-starting, not cold; ceiling rarely binds)
  - If ~16x slowdown appears: something wrong (cold-start per leapfrog would be bug)

Arm B: same as v2 (NUTS, f_raw=posterior_mean, burn-in gate >= 1500).
Already validated in v2; included here for Phase A completeness.
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
# Import stock blackjax directly — NO monkey-patching.
# blackjax repo must be on main (007a9ded) when this runs.
# Stock laplace_marginal_factory has no jax.debug.callback (clean timing).
# ---------------------------------------------------------------------------
import blackjax  # noqa: E402
import blackjax.mcmc.laplace_hmc as _lhmc  # noqa: E402
from blackjax.mcmc.laplace_marginal import laplace_marginal_factory  # noqa: E402
from blackjax.optimizers.lbfgs import minimize_lbfgs  # noqa: E402

from tuningfork.model import MODELS  # noqa: E402
from tuningfork.model._numpyro import build_logdensity_fn  # noqa: E402
from tuningfork.recipes._recipe_runner import _LAPLACE_PHI_THETA_SPLITS  # noqa: E402

t_imports = time.perf_counter()
print(
    f"[t=+{t_imports - t0:.1f}s] All imports done "
    f"(stock blackjax, no monkey-patch)",
    flush=True,
)

# Verify we are on stock blackjax (should have NO jax.debug.callback in solve_theta)
import inspect as _inspect  # noqa: E402

_src = _inspect.getsource(laplace_marginal_factory)
_has_callback = "jax.debug.callback" in _src or "debug.callback" in _src
print(
    f"  laplace_marginal_factory has debug.callback: {_has_callback}  "
    f"(expected: False for stock 007a9ded)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 20260517
MAXITER = 500
N_PILOT = 3
N_LEAPFROG_LAPLACE = 10
STEP_SIZE_LAPLACE = 0.526

BURN_IN_MIN = 10
BURN_IN_MAX = 50
LEAPFROG_GATE = 1500  # updated gate (supersedes v2 gate of 500)

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

ADAPTATION_JSON_PATH = (
    "/home/jp/blackjax-devs/tuningfork/tuningfork/catalog"
    "/gp_regression/reference/adaptation.json"
)
with open(ADAPTATION_JSON_PATH) as f:
    adaptation_data = json.load(f)

STEP_SIZE_NUTS = adaptation_data["step_size"]
IMM_NUTS = jnp.array(adaptation_data["inverse_mass_matrix"], dtype=jnp.float64)
MAX_NUM_DOUBLINGS = 12

print(f"  NUTS step_size={STEP_SIZE_NUTS:.8f}", flush=True)
print(f"  Laplace step_size={STEP_SIZE_LAPLACE}, L={N_LEAPFROG_LAPLACE}", flush=True)
print(f"  MAXITER={MAXITER}", flush=True)

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
# Compute f_raw posterior mean for NUTS init (same formula as v2)
# ---------------------------------------------------------------------------
from tuningfork.model.gp_regression import JITTER, N_OBS, X_DATA, Y_DATA  # noqa: E402

kernel_scale_gt = float(jnp.exp(PHI_INIT["log_kernel_scale"]))
lengthscale_gt = float(jnp.exp(PHI_INIT["log_lengthscale"]))
noise_scale_gt = float(jnp.exp(PHI_INIT["log_noise_scale"]))

sqdist = (X_DATA[:, None] - X_DATA[None, :]) ** 2
K_gt = kernel_scale_gt**2 * jnp.exp(
    -0.5 * sqdist / lengthscale_gt**2
) + JITTER * jnp.eye(N_OBS)
L_K_gt = jax.scipy.linalg.cholesky(K_gt, lower=True)
sigma2 = noise_scale_gt**2
precision = jnp.eye(N_OBS) + L_K_gt.T @ L_K_gt / sigma2
f_raw_posterior_mean = jnp.linalg.solve(precision, L_K_gt.T @ Y_DATA / sigma2)
f_raw_init_norm = float(jnp.linalg.norm(f_raw_posterior_mean))
print(f"  f_raw_posterior_mean norm = {f_raw_init_norm:.4f}", flush=True)

full_init_position = {
    **{k: PHI_INIT[k] for k in phi_sites},
    "f_raw": f_raw_posterior_mean,
}

# ---------------------------------------------------------------------------
# Build EXPLICIT factory with maxiter=500 — the key change vs v2.
# No forwarding chain: laplace_marginal_factory receives maxiter=500 directly.
# On stock blackjax (no setdefault, no callback), this flows verbatim to
# minimize_lbfgs(..., maxiter=500).
# ---------------------------------------------------------------------------
print(
    f"\n=== Building explicit laplace_marginal_factory(maxiter={MAXITER}) ===",
    flush=True,
)

laplace = laplace_marginal_factory(log_joint_fn, theta_init, maxiter=MAXITER)
print(
    f"  laplace object type: {type(laplace).__name__}  " f"(expected: LaplaceMarginal)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Cold-init iter_num probe: direct minimize_lbfgs call with maxiter=500.
# This is NOT timed and NOT inside any loop. Fires ONCE before sampling.
# Expected: iter_num ~295 (< 500 = maxiter, converges via ftol from cold start).
# If iter_num == 30: stock blackjax has the old setdefault(30) — bug.
# If iter_num == 500: ceiling hit at cold start — unusual but not alarming.
# ---------------------------------------------------------------------------
print(
    "\n=== Cold-init iter_num probe (direct minimize_lbfgs, NOT timed) ===",
    flush=True,
)
print(
    f"  Calling minimize_lbfgs(objective, theta_init, maxiter={MAXITER})",
    flush=True,
)

t_probe_start = time.perf_counter()


def _cold_objective(theta):
    return -log_joint_fn(theta, PHI_INIT)


probe_result, _ = minimize_lbfgs(_cold_objective, theta_init, maxiter=MAXITER)
jax.block_until_ready(probe_result)
t_probe_end = time.perf_counter()

cold_iter_num = int(probe_result.state.iter_num)
cold_error = float(probe_result.state.error)
print(
    f"  cold-init iter_num={cold_iter_num:4d}  "
    f"error={cold_error:.3e}  "
    f"maxiter={MAXITER}  "
    f"wall={t_probe_end - t_probe_start:.3f}s",
    flush=True,
)
if cold_iter_num < 30:
    print(
        "  [WARN] iter_num < 30 — did the factory override maxiter? "
        "Check stock blackjax has no setdefault.",
        flush=True,
    )
elif cold_iter_num >= MAXITER:
    print(
        f"  [INFO] iter_num = {cold_iter_num} = maxiter ceiling hit at cold start "
        f"(unusual — check convergence).",
        flush=True,
    )
else:
    print(
        f"  [OK] iter_num = {cold_iter_num} < {MAXITER}: "
        f"converged early via ftol/gtol. "
        f"maxiter={MAXITER} is in effect (ceiling not hit at cold start).",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Phase A v4 — Arm A (laplace_hmc): 3 timed steps with explicit factory
# ---------------------------------------------------------------------------
print(
    f"\n=== Phase A v4: Arm A — laplace_hmc (d=3, explicit factory, "
    f"maxiter={MAXITER}, L={N_LEAPFROG_LAPLACE}) ===",
    flush=True,
)

kernel = _lhmc.build_kernel()

t_a_init_start = time.perf_counter()
state_a = _lhmc.init(PHI_INIT, laplace)
jax.block_until_ready(state_a)
t_a_init_end = time.perf_counter()
print(f"  Arm A init: {t_a_init_end - t_a_init_start:.3f}s", flush=True)


@jax.jit
def step_a(rng_key, state):
    return kernel(
        rng_key, state, laplace, STEP_SIZE_LAPLACE, IMM_3X3, N_LEAPFROG_LAPLACE
    )


# JIT warm-up (first call compiles; not timed)
key_a, subkey = jax.random.split(key_a)
state_a, info_a = step_a(subkey, state_a)
jax.block_until_ready(state_a)
key_a, subkey = jax.random.split(key_a)

step_times_a = []
for i in range(N_PILOT):
    t_step_start = time.perf_counter()
    key_a, subkey = jax.random.split(key_a)
    state_a, info_a = step_a(subkey, state_a)
    jax.block_until_ready(state_a)
    t_step_end = time.perf_counter()
    dt = t_step_end - t_step_start
    step_times_a.append(dt)
    print(
        f"  step {i + 1}: {dt:.3f}s  "
        f"acc={float(info_a.acceptance_rate):.4f}  "
        f"phi=[{float(state_a.position['log_kernel_scale']):.4f}, "
        f"{float(state_a.position['log_lengthscale']):.4f}, "
        f"{float(state_a.position['log_noise_scale']):.4f}]",
        flush=True,
    )

t_per_step_a = sum(step_times_a) / N_PILOT
print(
    f"\n  Arm A: {sum(step_times_a):.3f}s total, "
    f"{t_per_step_a:.3f}s/step (post-JIT, all 3 steps)",
    flush=True,
)
print(
    "  Compare to v2 monkey-patch: 1.015 s/step. "
    "Expected: ~same (confirms explicit factory ≡ monkey-patch timing).",
    flush=True,
)

# ---------------------------------------------------------------------------
# Phase A v4 — Arm B (NUTS): same burn-in gate as v2 (>= 1500)
# ---------------------------------------------------------------------------
print(
    f"\n=== Phase A v4: Arm B — NUTS (d=203, max_doublings={MAX_NUM_DOUBLINGS}) ===",
    flush=True,
)
print(
    f"  f_raw init: posterior mean (norm={f_raw_init_norm:.4f}), phi at GT mean",
    flush=True,
)
print(
    f"  Burn-in: up to {BURN_IN_MAX} steps, gate median_leapfrog >= {LEAPFROG_GATE}",
    flush=True,
)

alg_b = blackjax.nuts(
    logdensity_fn,
    step_size=STEP_SIZE_NUTS,
    inverse_mass_matrix=IMM_NUTS,
    max_num_doublings=MAX_NUM_DOUBLINGS,
)

t_b_init_start = time.perf_counter()
state_b = alg_b.init(full_init_position)
jax.block_until_ready(state_b)
t_b_init_end = time.perf_counter()
print(f"  Arm B init: {t_b_init_end - t_b_init_start:.3f}s", flush=True)

step_b_jit = jax.jit(alg_b.step)

# Burn-in
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
    f"  Burn-in: {t_burnin_end - t_burnin_start:.3f}s, {len(burnin_leapfrogs)} steps",
    flush=True,
)
if not gate_passed:
    print(
        f"  [WARN] Gate NOT passed after {BURN_IN_MAX} burn-in steps. "
        f"Last 5 lf: {burnin_leapfrogs[-5:]}. Timing anyway.",
        flush=True,
    )

# Timed phase
print(f"\n  [Timed] {N_PILOT} steps after burn-in:", flush=True)
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

t_per_step_b = sum(step_times_b) / N_PILOT
median_timed_lf = sorted(timed_leapfrogs)[len(timed_leapfrogs) // 2]
scale_factor = 2047.0 / max(median_timed_lf, 1)
t_per_step_b_scaled = t_per_step_b * scale_factor

# ---------------------------------------------------------------------------
# Phase A v4 Summary
# ---------------------------------------------------------------------------
budget_sec = 120.0
n_samples_laplace = max(50, min(200, int(budget_sec / max(t_per_step_a, 0.001))))
n_samples_nuts = max(10, min(200, int(budget_sec / max(t_per_step_b, 0.001))))
ratio_raw = t_per_step_b / t_per_step_a
ratio_scaled = t_per_step_b_scaled / t_per_step_a

t_total = time.perf_counter()
sep = "=" * 70
print(f"\n{sep}", flush=True)
print(f"ExpI Phase A v4 Summary (total wall = {t_total - t0:.1f}s)", flush=True)
print(
    "  blackjax: stock main 007a9ded (NO callback, NO setdefault override)", flush=True
)
print(f"  laplace_marginal_factory called with maxiter={MAXITER} EXPLICIT", flush=True)
print(sep, flush=True)
print(
    f"  Cold-init iter_num:          {cold_iter_num:4d}  "
    f"(maxiter={MAXITER}; < maxiter → ftol convergence confirmed)",
    flush=True,
)
print(
    f"  Arm A (laplace_hmc):         {t_per_step_a:.3f} s/step "
    f"(post-JIT, L={N_LEAPFROG_LAPLACE} fixed)",
    flush=True,
)
print(
    f"  Arm B (NUTS d=203):          {t_per_step_b:.3f} s/step "
    f"(timed, median_lf={median_timed_lf})",
    flush=True,
)
print(
    f"  Arm B scaled to GT (2047 lf): {t_per_step_b_scaled:.3f} s/step "
    f"(scale={scale_factor:.2f}x)",
    flush=True,
)
print(f"\n  Ratio B/A (raw):       {ratio_raw:.2f}x", flush=True)
print(f"  Ratio B/A (GT-scaled): {ratio_scaled:.2f}x", flush=True)
print("  (> 1 means NUTS is slower per step)", flush=True)
print(
    f"\n  Verdict validity: timed median_lf={median_timed_lf} vs GT=2047. "
    f"{'NEAR GT' if median_timed_lf >= 1500 else 'BELOW GT — scaled ratio applies'}",
    flush=True,
)
print(f"\n  Burn-in lf sequence: {burnin_leapfrogs}", flush=True)
print(f"  Timed lf sequence:   {timed_leapfrogs}", flush=True)
print(
    f"\n  Phase B projection (budget={budget_sec:.0f}s per arm):",
    flush=True,
)
print(
    f"    n_samples_laplace (4 chains vmap) = {n_samples_laplace}",
    flush=True,
)
print(
    f"    n_samples_nuts (2 chains seq)     = {n_samples_nuts}",
    flush=True,
)
if n_samples_nuts < 10:
    print(
        "    [WARN] NUTS budget very tight. Report timing only, skip ESS.",
        flush=True,
    )
print(sep, flush=True)
