"""ExpI Stage 1 ESS: laplace_dmhmc (dynamic steps + multinomial proposal).

TL brief (2026-05-25): Tight ESS run post Phase A pilot. 50 samples/chain x 4
chains.  Budget: ~3.86 s/vmap-step x 50 + 17 s overhead ~ 210 s (~3.5 min).

Note on acceptance_rate for multinomial HMC:
  For multinomial proposals, info.acceptance_rate is a **trajectory-weight
  diagnostic** = exp(sum_log_p_accept) / num_integration_steps, NOT a standard
  M-H acceptance probability. info.is_accepted is always True (no rejection).
  The field is labelled traj_weight in this script to avoid confusion with the
  M-H acc_rate of laplace_hmc.
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
import blackjax.mcmc.laplace_dynamic_hmc as _ldynhmc  # noqa: E402
from blackjax.mcmc.laplace_dynamic_hmc import LaplaceDynamicHMCState  # noqa: E402
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
N_SAMPLES = 50  # per chain; 200 total
STEP_SIZE = 0.526
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
# integration_steps_fn: Uniform[5, 14) -> mean ~9.5, matches fixed L=10
INTEGRATION_STEPS_FN = lambda k: jax.random.randint(k, (), 5, 14)  # noqa: E731

print(
    f"  {N_CHAINS} chains x {N_SAMPLES} samples/chain = {N_CHAINS * N_SAMPLES} total, "
    f"step_size={STEP_SIZE}, integration_steps=Uniform[5,14] (mean~9.5), "
    f"maxiter={MAXITER}",
    flush=True,
)
print(
    "  acceptance_rate field = trajectory-weight diagnostic (not M-H acc)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
rng_key = jax.random.key(SEED)
key_init, key_run, key_rand = jax.random.split(rng_key, 3)

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
# Build factory + kernel (outside vmap — V2 pattern)
# ---------------------------------------------------------------------------
import blackjax.mcmc.laplace_hmc as _lhmc  # noqa: E402

laplace = laplace_marginal_factory(log_joint_fn, theta_init, maxiter=MAXITER)
kernel = _ldynhmc.build_kernel(
    integration_steps_fn=INTEGRATION_STEPS_FN,
    build_proposal=_hmc_module.multinomial_hmc_proposal,
)

# Single cold-start init via laplace_hmc.init (shared between mhmc + dmhmc)
print("  Cold-start init (via laplace_hmc.init, shared) ...", flush=True)
t_init_start = time.perf_counter()
state_hmc_single = _lhmc.init(PHI_INIT, laplace)
jax.block_until_ready(state_hmc_single)
t_init_end = time.perf_counter()
print(f"  Init done: {t_init_end - t_init_start:.3f}s", flush=True)

# Build batched LaplaceDynamicHMCState from the lhmc init result
# Each chain gets its own random_generator_arg (independent step counts)
chain_rand_keys = jax.random.split(key_rand, N_CHAINS)
state_batched = LaplaceDynamicHMCState(
    position=jax.tree.map(
        lambda x: jnp.stack([x] * N_CHAINS), state_hmc_single.position
    ),
    logdensity=jnp.stack([state_hmc_single.logdensity] * N_CHAINS),
    logdensity_grad=jax.tree.map(
        lambda x: jnp.stack([x] * N_CHAINS), state_hmc_single.logdensity_grad
    ),
    theta_star=jax.tree.map(
        lambda x: jnp.stack([x] * N_CHAINS), state_hmc_single.theta_star
    ),
    random_generator_arg=chain_rand_keys,
)

# ---------------------------------------------------------------------------
# vmap scan: N_CHAINS chains x N_SAMPLES steps
# ---------------------------------------------------------------------------


def _run_chain(rng_key_chain, init_state):
    def scan_fn(state, rng_k):
        new_state, info = kernel(rng_k, state, laplace, STEP_SIZE, IMM_3X3)
        phi_arr = jnp.stack(
            [
                new_state.position["log_kernel_scale"],
                new_state.position["log_lengthscale"],
                new_state.position["log_noise_scale"],
            ]
        )
        # traj_weight: trajectory-weight diagnostic (exp(sum_log_p_accept)/L)
        # n_lf: actual number of leapfrog steps drawn (Uniform[5,14])
        return new_state, (
            phi_arr,
            info.acceptance_rate,
            info.is_divergent,
            info.num_integration_steps,
        )

    scan_keys = jax.random.split(rng_key_chain, N_SAMPLES)
    _, (phi_traj, traj_weight_traj, div_traj, lf_traj) = jax.lax.scan(
        scan_fn, init_state, scan_keys
    )
    return phi_traj, traj_weight_traj, div_traj, lf_traj


run_vmap = jax.jit(jax.vmap(_run_chain))
keys_chains = jax.random.split(key_run, N_CHAINS)

print(
    f"  Running vmap scan ({N_CHAINS} chains x {N_SAMPLES} steps, incl JIT) ...",
    flush=True,
)
t_scan_start = time.perf_counter()
phi, traj_weight, div, lf_counts = run_vmap(keys_chains, state_batched)
jax.block_until_ready(phi)
t_scan_end = time.perf_counter()
wall_scan = t_scan_end - t_scan_start
wall_total = (t_init_end - t_init_start) + wall_scan

total_samples = N_CHAINS * N_SAMPLES
print(
    f"  vmap scan: {wall_scan:.3f}s "
    f"({wall_scan / N_SAMPLES:.3f}s/vmap-step, "
    f"{wall_scan / total_samples:.3f}s/chain-sample, incl JIT)",
    flush=True,
)
print(f"  wall_dmhmc_total (init+scan): {wall_total:.3f}s", flush=True)
print(
    f"  mean_traj_weight={float(traj_weight.mean()):.4f}  n_div={int(div.sum())}  "
    f"lf_mean={float(lf_counts.mean()):.1f}",
    flush=True,
)

# Per-chain traj_weight + phi means
param_names = ["log_kernel_scale", "log_lengthscale", "log_noise_scale"]
for c in range(N_CHAINS):
    print(
        f"  chain {c}: "
        f"phi_mean=[{float(phi[c, :, 0].mean()):.4f}, "
        f"{float(phi[c, :, 1].mean()):.4f}, "
        f"{float(phi[c, :, 2].mean()):.4f}]  "
        f"traj_wt={float(traj_weight[c].mean()):.4f}  "
        f"n_div={int(div[c].sum())}  "
        f"lf_mean={float(lf_counts[c].mean()):.1f}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# ESS and R-hat
# ---------------------------------------------------------------------------
ess = blackjax.diagnostics.effective_sample_size(phi)  # shape (3,)
rhat = blackjax.diagnostics.potential_scale_reduction(phi)  # shape (3,)

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

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
t_total = time.perf_counter()
sep = "=" * 70
print(f"\n{sep}", flush=True)
print(
    f"ExpI Stage 1 dmhmc ESS Summary (total wall = {t_total - t0:.1f}s)",
    flush=True,
)
print(
    "  blackjax: stock main 007a9ded  |  "
    "gp_regression N=200  |  V2 vmap  |  4 chains x 50 samples",
    flush=True,
)
print(sep, flush=True)
print(
    f"  laplace_dmhmc: {N_CHAINS} chains x {N_SAMPLES} samples = {total_samples} total",
    flush=True,
)
print(
    "  proposal: multinomial (dynamic L=Uniform[5,14], "
    "step_size=0.526, dense 3x3 IMM)",
    flush=True,
)
print(
    f"  wall_total={wall_total:.3f}s  "
    f"(init={t_init_end - t_init_start:.3f}s + scan={wall_scan:.3f}s)",
    flush=True,
)
print(
    f"  mean_traj_weight={float(traj_weight.mean()):.4f}  n_div={int(div.sum())}  "
    f"lf_mean={float(lf_counts.mean()):.1f}  "
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
    f"  min_ESS={min_ess:.2f}  ESS/s (min_ESS/wall_total)={ess_per_sec:.5f}",
    flush=True,
)
print(f"  max_R-hat={max_rhat:.4f}  (pass if < 1.05)", flush=True)
print(sep, flush=True)
