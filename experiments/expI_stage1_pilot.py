"""ExpI Stage 1 pilot: per-step timing for laplace_mhmc + laplace_dmhmc.

TL brief (2026-05-25): Phase A pilot to measure per-step wall time before the
full ESS run. Two multinomial variants on gp_regression, V2 vmap pattern,
GT-calibrated params, 4 chains x N_PILOT_STEPS.

Variants:
  laplace_mhmc:  fixed-L + multinomial proposal (LaplaceHMCState, same as lhmc)
  laplace_dmhmc: dynamic (integration_steps_fn=Uniform[5,14]) + multinomial

API notes (stock blackjax 007a9ded):
  - laplace_mhmc: _lhmc.build_kernel(build_proposal=multinomial_hmc_proposal)
  - laplace_dmhmc: _ldynhmc.build_kernel(integration_steps_fn=...,
                                          build_proposal=multinomial_hmc_proposal)
    - init signature: _ldynhmc.init(position, laplace, random_generator_arg)
    - random_generator_arg = PRNG key (advanced by next_random_arg_fn each step)
    - kernel signature: kernel(rng_key, state, laplace, step_size, imm)
  - Shared single cold-start init (avoids 2x L-BFGS from separate init calls)

Budget: ~5 pilot steps x 4 chains x ~2.5 s/vmap-step x 2 variants = ~25s each
        + 9s init = ~60s total (<< 3 min).
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

import blackjax.mcmc.hmc as _hmc_module  # noqa: E402
import blackjax.mcmc.laplace_dynamic_hmc as _ldynhmc  # noqa: E402
import blackjax.mcmc.laplace_hmc as _lhmc  # noqa: E402
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
N_PILOT = 5  # per chain — just timing, not ESS
N_LEAPFROG = 10  # fixed-L for mhmc; dmhmc uses Uniform[5,14] (mean ~9.5)
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
# integration_steps_fn for dmhmc: Uniform[5, 14) → mean 9.5, matches L=10
INTEGRATION_STEPS_FN = lambda k: jax.random.randint(k, (), 5, 14)  # noqa: E731

print(
    f"  N_CHAINS={N_CHAINS}, N_PILOT={N_PILOT}, "
    f"step_size={STEP_SIZE}, L={N_LEAPFROG} (mhmc fixed)",
    flush=True,
)
print(
    "  dmhmc: integration_steps_fn=Uniform[5,14] (mean~9.5)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
rng_key = jax.random.key(SEED)
key_init, key_mhmc, key_dmhmc, key_dmhmc_rand = jax.random.split(rng_key, 4)

entry = MODELS["gp_regression"]
init_position, logdensity_fn, _ = build_logdensity_fn(key_init, entry)

t_model = time.perf_counter()
d_total = sum(v.size for v in jax.tree.leaves(init_position))
print(
    f"[t=+{t_model - t0:.1f}s] model built d={d_total}",
    flush=True,
)

phi_sites, theta_sites = _LAPLACE_PHI_THETA_SPLITS["gp_regression"]
theta_init = {k: init_position[k] for k in theta_sites}


def log_joint_fn(theta, phi):
    return logdensity_fn({**theta, **phi})


# ---------------------------------------------------------------------------
# Build factory (shared) and kernels
# ---------------------------------------------------------------------------
laplace = laplace_marginal_factory(log_joint_fn, theta_init, maxiter=MAXITER)

kernel_mhmc = _lhmc.build_kernel(build_proposal=_hmc_module.multinomial_hmc_proposal)
kernel_dmhmc = _ldynhmc.build_kernel(
    integration_steps_fn=INTEGRATION_STEPS_FN,
    build_proposal=_hmc_module.multinomial_hmc_proposal,
)

# ---------------------------------------------------------------------------
# Single shared cold-start init (shared between both variants)
# ---------------------------------------------------------------------------
print("\n  Cold-start init (shared between mhmc + dmhmc) ...", flush=True)
t_init_start = time.perf_counter()
state_hmc_single = _lhmc.init(PHI_INIT, laplace)
jax.block_until_ready(state_hmc_single)
t_init_end = time.perf_counter()
print(
    f"  Init done: {t_init_end - t_init_start:.3f}s",
    flush=True,
)

# Replicate for N_CHAINS (mhmc uses LaplaceHMCState)
state_mhmc_batched = jax.tree.map(lambda x: jnp.stack([x] * N_CHAINS), state_hmc_single)

# dmhmc uses LaplaceDynamicHMCState: same fields + random_generator_arg per chain
chain_rand_keys = jax.random.split(key_dmhmc_rand, N_CHAINS)
state_dmhmc_single = LaplaceDynamicHMCState(
    position=state_hmc_single.position,
    logdensity=state_hmc_single.logdensity,
    logdensity_grad=state_hmc_single.logdensity_grad,
    theta_star=state_hmc_single.theta_star,
    random_generator_arg=chain_rand_keys[0],
)
state_dmhmc_batched = LaplaceDynamicHMCState(
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
# Pilot run: laplace_mhmc
# ---------------------------------------------------------------------------
param_names = ["log_kernel_scale", "log_lengthscale", "log_noise_scale"]
print(
    f"\n=== laplace_mhmc pilot ({N_CHAINS} chains x {N_PILOT} steps, incl JIT) ===",
    flush=True,
)


def _run_mhmc_chain(rng_key_chain, init_state):
    def scan_fn(state, rng_k):
        new_state, info = kernel_mhmc(
            rng_k, state, laplace, STEP_SIZE, IMM_3X3, N_LEAPFROG
        )
        phi_arr = jnp.stack(
            [
                new_state.position["log_kernel_scale"],
                new_state.position["log_lengthscale"],
                new_state.position["log_noise_scale"],
            ]
        )
        return new_state, (phi_arr, info.acceptance_rate, info.is_divergent)

    scan_keys = jax.random.split(rng_key_chain, N_PILOT)
    _, (phi_traj, acc_traj, div_traj) = jax.lax.scan(scan_fn, init_state, scan_keys)
    return phi_traj, acc_traj, div_traj


run_vmap_mhmc = jax.jit(jax.vmap(_run_mhmc_chain))
keys_mhmc = jax.random.split(key_mhmc, N_CHAINS)

t_mhmc_start = time.perf_counter()
phi_mhmc, acc_mhmc, div_mhmc = run_vmap_mhmc(keys_mhmc, state_mhmc_batched)
jax.block_until_ready(phi_mhmc)
t_mhmc_end = time.perf_counter()
wall_mhmc = t_mhmc_end - t_mhmc_start

print(
    f"  wall={wall_mhmc:.3f}s  "
    f"({wall_mhmc / N_PILOT:.3f}s/vmap-step, "
    f"{wall_mhmc / (N_CHAINS * N_PILOT):.3f}s/chain-sample)",
    flush=True,
)
print(
    f"  mean_acc={float(acc_mhmc.mean()):.4f}  n_div={int(div_mhmc.sum())}",
    flush=True,
)
for c in range(N_CHAINS):
    print(
        f"  chain {c}: phi_mean="
        f"[{float(phi_mhmc[c, :, 0].mean()):.4f}, "
        f"{float(phi_mhmc[c, :, 1].mean()):.4f}, "
        f"{float(phi_mhmc[c, :, 2].mean()):.4f}]  "
        f"acc={float(acc_mhmc[c].mean()):.4f}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Pilot run: laplace_dmhmc
# ---------------------------------------------------------------------------
print(
    f"\n=== laplace_dmhmc pilot ({N_CHAINS} chains x {N_PILOT} steps, incl JIT) ===",
    flush=True,
)
print(
    "  integration_steps_fn=Uniform[5,14]",
    flush=True,
)


def _run_dmhmc_chain(rng_key_chain, init_state):
    def scan_fn(state, rng_k):
        new_state, info = kernel_dmhmc(rng_k, state, laplace, STEP_SIZE, IMM_3X3)
        phi_arr = jnp.stack(
            [
                new_state.position["log_kernel_scale"],
                new_state.position["log_lengthscale"],
                new_state.position["log_noise_scale"],
            ]
        )
        n_lf = info.num_integration_steps
        return new_state, (phi_arr, info.acceptance_rate, info.is_divergent, n_lf)

    scan_keys = jax.random.split(rng_key_chain, N_PILOT)
    _, (phi_traj, acc_traj, div_traj, lf_traj) = jax.lax.scan(
        scan_fn, init_state, scan_keys
    )
    return phi_traj, acc_traj, div_traj, lf_traj


run_vmap_dmhmc = jax.jit(jax.vmap(_run_dmhmc_chain))
keys_dmhmc = jax.random.split(key_dmhmc, N_CHAINS)

t_dmhmc_start = time.perf_counter()
phi_dmhmc, acc_dmhmc, div_dmhmc, lf_dmhmc = run_vmap_dmhmc(
    keys_dmhmc, state_dmhmc_batched
)
jax.block_until_ready(phi_dmhmc)
t_dmhmc_end = time.perf_counter()
wall_dmhmc = t_dmhmc_end - t_dmhmc_start

print(
    f"  wall={wall_dmhmc:.3f}s  "
    f"({wall_dmhmc / N_PILOT:.3f}s/vmap-step, "
    f"{wall_dmhmc / (N_CHAINS * N_PILOT):.3f}s/chain-sample)",
    flush=True,
)
print(
    f"  mean_acc={float(acc_dmhmc.mean()):.4f}  n_div={int(div_dmhmc.sum())}",
    flush=True,
)
print(
    f"  lf_counts (all chains x steps): {lf_dmhmc.tolist()}",
    flush=True,
)
for c in range(N_CHAINS):
    print(
        f"  chain {c}: phi_mean="
        f"[{float(phi_dmhmc[c, :, 0].mean()):.4f}, "
        f"{float(phi_dmhmc[c, :, 1].mean()):.4f}, "
        f"{float(phi_dmhmc[c, :, 2].mean()):.4f}]  "
        f"acc={float(acc_dmhmc[c].mean()):.4f}  "
        f"lf_mean={float(lf_dmhmc[c].mean()):.1f}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
t_total = time.perf_counter()
sep = "=" * 70
print(f"\n{sep}", flush=True)
print(
    f"ExpI Stage 1 Pilot Summary (total wall = {t_total - t0:.1f}s)",
    flush=True,
)
print(
    "  blackjax: stock main 007a9ded  |  "
    "gp_regression N=200  |  V2 vmap  |  4 chains x 5 pilot steps",
    flush=True,
)
print(sep, flush=True)
print(
    "  laplace_hmc (expI baseline): ~2.55 s/vmap-step  (from expI_laplace_ess.log)",
    flush=True,
)
print(
    f"  laplace_mhmc (fixed-L multinomial): {wall_mhmc / N_PILOT:.3f}s/vmap-step  "
    f"acc={float(acc_mhmc.mean()):.4f}  n_div={int(div_mhmc.sum())}",
    flush=True,
)
print(
    f"  laplace_dmhmc (dynamic multinomial): {wall_dmhmc / N_PILOT:.3f}s/vmap-step  "
    f"acc={float(acc_dmhmc.mean()):.4f}  n_div={int(div_dmhmc.sum())}  "
    f"lf_mean={float(lf_dmhmc.mean()):.1f}",
    flush=True,
)
print(
    f"  lhmc/mhmc ratio: {2.55 / (wall_mhmc / N_PILOT):.2f}x  (mhmc slower if >1)",
    flush=True,
)
print(
    f"  lhmc/dmhmc ratio: {2.55 / (wall_dmhmc / N_PILOT):.2f}x  (dmhmc slower if >1)",
    flush=True,
)
print(sep, flush=True)
