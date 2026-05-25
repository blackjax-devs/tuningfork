"""ExpI Phase B: ESS/wall benchmark — laplace_hmc (4-chain vmap) vs NUTS (2-chain seq).

TL brief (2026-05-25): Phase A validated per-step timing (laplace 1.104 s/step vs
NUTS 5.413 s/step = 4.90x). Phase B adds the ESS dimension for the full efficiency
verdict. Run on stock blackjax main (007a9ded), NO monkey-patch.

Design:
  Arm A: laplace_hmc, 4 chains vmap (V2 pattern: kernel outside vmap, single shared
         init to avoid 4x cold-start overhead), explicit maxiter=500 factory,
         GT step_size=0.526, GT 3x3 IMM, L=10, N_SAMPLES_A=15/chain.
  Arm B: NUTS, 2 chains sequential (never vmap), f_raw=posterior-mean init,
         BURN_IN_MIN=5 (reduced from 10 — burn step 1 already at 1535 lf >= 1500,
         gate passes at step 5 reliably), N_SAMPLES_B=5/chain, GT params.

Budget: target < 3 min total.
  Arm A: single-chain init ~9s + vmap JIT+scan 15 steps ~40s = ~49s
  Arm B: 2 chains x (5 burn-in + 5 timed) x ~5.4s = ~108s
  Misc:  ~20s
  Total: ~177s ~ 3 min (tight; BURN_IN_MIN=5 is the budget-critical choice)

NOTE for @statistician: N_SAMPLES_B=5 per chain gives ESS input shape (2, 5, 3).
ESS estimates at n=5 are noisy. Flag this when computing ESS/s speedup.
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
    f"[t=+{t_jax - t0:.1f}s] JAX x64={jax.config.read('jax_enable_x64')} "
    f"backend={jax.default_backend()}",
    flush=True,
)

sys.path.insert(0, "/home/jp/blackjax-devs/tuningfork")
sys.path.insert(0, "/home/jp/blackjax-devs/blackjax")

import blackjax  # noqa: E402
import blackjax.mcmc.laplace_hmc as _lhmc  # noqa: E402
from blackjax.mcmc.laplace_marginal import laplace_marginal_factory  # noqa: E402

from tuningfork.model import MODELS  # noqa: E402
from tuningfork.model._numpyro import build_logdensity_fn  # noqa: E402
from tuningfork.recipes._recipe_runner import _LAPLACE_PHI_THETA_SPLITS  # noqa: E402

t_imports = time.perf_counter()
print(f"[t=+{t_imports - t0:.1f}s] Imports done (stock blackjax)", flush=True)

import inspect as _inspect  # noqa: E402

_src = _inspect.getsource(laplace_marginal_factory)
_has_cb = "debug.callback" in _src
print(
    f"  laplace_marginal_factory debug.callback={_has_cb} (expected False)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 20260517
MAXITER = 500
N_CHAINS_A = 4
N_SAMPLES_A = 15  # per chain; 60 total
N_LEAPFROG_A = 10
STEP_SIZE_A = 0.526
IMM_3X3 = jnp.array(
    [
        [0.18301258, 0.05751162, -0.00021748],
        [0.05751162, 0.03180439, -0.00022324],
        [-0.00021748, -0.00022324, 0.00262740],
    ],
    dtype=jnp.float64,
)

N_CHAINS_B = 2
N_SAMPLES_B = 5  # per chain; 10 total
BURN_IN_MIN_B = 5  # reduced from 10 — gate passes at step 5 from posterior-mean init
BURN_IN_MAX_B = 15
LEAPFROG_GATE = 1500
MAX_NUM_DOUBLINGS = 12

PHI_INIT = {
    "log_kernel_scale": jnp.float64(0.40870562293007373),
    "log_lengthscale": jnp.float64(-1.0424925985381703),
    "log_noise_scale": jnp.float64(-2.34163615643574),
}

ADAPTATION_JSON_PATH = (
    "/home/jp/blackjax-devs/tuningfork/tuningfork/catalog"
    "/gp_regression/reference/adaptation.json"
)
with open(ADAPTATION_JSON_PATH) as f:
    adaptation_data = json.load(f)
STEP_SIZE_B = adaptation_data["step_size"]
IMM_NUTS = jnp.array(adaptation_data["inverse_mass_matrix"], dtype=jnp.float64)

print(
    f"  Arm A: {N_CHAINS_A} chains x {N_SAMPLES_A} samples, "
    f"step_size={STEP_SIZE_A}, L={N_LEAPFROG_A}, maxiter={MAXITER}",
    flush=True,
)
print(
    f"  Arm B: {N_CHAINS_B} chains (seq) x {N_SAMPLES_B} samples, "
    f"step_size={STEP_SIZE_B:.6f}, max_doublings={MAX_NUM_DOUBLINGS}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
rng_key = jax.random.key(SEED)
key_init, key_a, key_b = jax.random.split(rng_key, 3)

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


# f_raw posterior mean for NUTS init (linear-Gaussian NCP closed form)
from tuningfork.model.gp_regression import JITTER, N_OBS, X_DATA, Y_DATA  # noqa: E402

ks_gt = float(jnp.exp(PHI_INIT["log_kernel_scale"]))
ls_gt = float(jnp.exp(PHI_INIT["log_lengthscale"]))
ns_gt = float(jnp.exp(PHI_INIT["log_noise_scale"]))
sqdist = (X_DATA[:, None] - X_DATA[None, :]) ** 2
K_gt = ks_gt**2 * jnp.exp(-0.5 * sqdist / ls_gt**2) + JITTER * jnp.eye(N_OBS)
L_K_gt = jax.scipy.linalg.cholesky(K_gt, lower=True)
sigma2 = ns_gt**2
precision = jnp.eye(N_OBS) + L_K_gt.T @ L_K_gt / sigma2
f_raw_pm = jnp.linalg.solve(precision, L_K_gt.T @ Y_DATA / sigma2)
print(
    f"  f_raw posterior-mean norm = {float(jnp.linalg.norm(f_raw_pm)):.4f}", flush=True
)

full_init_pos = {**{k: PHI_INIT[k] for k in phi_sites}, "f_raw": f_raw_pm}

# ---------------------------------------------------------------------------
# ARM A: laplace_hmc, 4 chains vmap
# V2 pattern: kernel + factory outside vmap; single init replicated to 4 chains.
# ---------------------------------------------------------------------------
print(
    f"\n=== Arm A: laplace_hmc {N_CHAINS_A} chains vmap, "
    f"{N_SAMPLES_A} samples/chain ===",
    flush=True,
)

laplace = laplace_marginal_factory(log_joint_fn, theta_init, maxiter=MAXITER)
kernel_a = _lhmc.build_kernel()

# Single init (outside vmap): avoids 4x cold-start overhead
print("  Building factory + single chain init ...", flush=True)
t_a_init_start = time.perf_counter()
state_a_single = _lhmc.init(PHI_INIT, laplace)
jax.block_until_ready(state_a_single)
t_a_init_end = time.perf_counter()
print(f"  Single-chain init: {t_a_init_end - t_a_init_start:.3f}s", flush=True)

# Replicate init state for N_CHAINS_A chains
state_a_batched = jax.tree.map(lambda x: jnp.stack([x] * N_CHAINS_A), state_a_single)


# Define per-chain sampling function (vmap axes: rng_key and init_state)
def _run_laplace_chain(rng_key_chain, init_state):
    def scan_fn(state, rng_k):
        new_state, info = kernel_a(
            rng_k, state, laplace, STEP_SIZE_A, IMM_3X3, N_LEAPFROG_A
        )
        phi_arr = jnp.stack(
            [
                new_state.position["log_kernel_scale"],
                new_state.position["log_lengthscale"],
                new_state.position["log_noise_scale"],
            ]
        )  # shape (3,)
        return new_state, (phi_arr, info.acceptance_rate, info.is_divergent)

    scan_keys = jax.random.split(rng_key_chain, N_SAMPLES_A)
    _, (phi_traj, acc_traj, div_traj) = jax.lax.scan(scan_fn, init_state, scan_keys)
    return phi_traj, acc_traj, div_traj
    # phi_traj: (N_SAMPLES_A, 3)


run_vmap_a = jax.jit(jax.vmap(_run_laplace_chain))
keys_a_chains = jax.random.split(key_a, N_CHAINS_A)

print(
    f"  Running vmap scan ({N_CHAINS_A} chains x {N_SAMPLES_A} steps, incl JIT) ...",
    flush=True,
)
t_a_scan_start = time.perf_counter()
phi_a, acc_a, div_a = run_vmap_a(keys_a_chains, state_a_batched)
jax.block_until_ready(phi_a)
t_a_scan_end = time.perf_counter()
wall_a_scan = t_a_scan_end - t_a_scan_start
# phi_a shape: (N_CHAINS_A, N_SAMPLES_A, 3)
total_samples_a = N_CHAINS_A * N_SAMPLES_A
wall_a_total = (t_a_init_end - t_a_init_start) + wall_a_scan

print(
    f"  Arm A vmap scan: {wall_a_scan:.3f}s "
    f"({wall_a_scan / total_samples_a:.3f}s/sample, incl JIT)",
    flush=True,
)
print(
    f"  Arm A total (init + scan): {wall_a_total:.3f}s",
    flush=True,
)
print(
    f"  mean acceptance: {float(acc_a.mean()):.4f}  "
    f"total divergences: {int(div_a.sum())}",
    flush=True,
)
for c in range(N_CHAINS_A):
    print(
        f"  chain {c}: "
        f"phi_mean=[{float(phi_a[c, :, 0].mean()):.4f}, "
        f"{float(phi_a[c, :, 1].mean()):.4f}, "
        f"{float(phi_a[c, :, 2].mean()):.4f}]  "
        f"acc={float(acc_a[c].mean()):.4f}",
        flush=True,
    )

# ESS Arm A
ess_a = blackjax.diagnostics.effective_sample_size(phi_a)  # shape (3,)
param_names = ["log_kernel_scale", "log_lengthscale", "log_noise_scale"]
print(
    "  ESS Arm A: "
    + " ".join(f"{p}={float(v):.2f}" for p, v in zip(param_names, ess_a)),
    flush=True,
)
min_ess_a = float(jnp.min(ess_a))
ess_per_sec_a = min_ess_a / wall_a_total
print(
    f"  min_ESS_a={min_ess_a:.2f}  ESS/s_a={ess_per_sec_a:.4f} "
    "(min_ESS / total_wall incl init)",
    flush=True,
)

# ---------------------------------------------------------------------------
# ARM B: NUTS, 2 chains sequential
# ---------------------------------------------------------------------------
print(
    f"\n=== Arm B: NUTS {N_CHAINS_B} chains sequential, "
    f"{N_SAMPLES_B} samples/chain ===",
    flush=True,
)
print(
    f"  BURN_IN_MIN={BURN_IN_MIN_B}, gate=median(last 5 steps)>={LEAPFROG_GATE}",
    flush=True,
)
print(
    "  NOTE: BURN_IN_MIN reduced from 10 to 5 for budget; step 1 hits ~1535 lf "
    "from posterior-mean init, gate passes at step 5.",
    flush=True,
)

alg_b = blackjax.nuts(
    logdensity_fn,
    step_size=STEP_SIZE_B,
    inverse_mass_matrix=IMM_NUTS,
    max_num_doublings=MAX_NUM_DOUBLINGS,
)
step_b_jit = jax.jit(alg_b.step)

phi_b_chains = []  # will be (N_CHAINS_B, N_SAMPLES_B, 3)
acc_b_chains = []
div_b_chains = []
lf_b_chains = []
wall_b_sampling = 0.0  # timed sampling only (excludes burn-in)

for chain_i in range(N_CHAINS_B):
    print(f"\n  --- Chain {chain_i} ---", flush=True)
    key_b, subkey_init = jax.random.split(key_b)

    # Init
    state_b = alg_b.init(full_init_pos)
    jax.block_until_ready(state_b)

    # Burn-in
    burnin_lf = []
    gate_passed = False
    for i in range(BURN_IN_MAX_B):
        key_b, subkey = jax.random.split(key_b)
        t_bi = time.perf_counter()
        state_b, info_b = step_b_jit(subkey, state_b)
        jax.block_until_ready(state_b)
        n_lf = int(getattr(info_b, "num_integration_steps", 0))
        burnin_lf.append(n_lf)
        print(
            f"  burn {i + 1:2d}: {time.perf_counter() - t_bi:.3f}s  "
            f"lf={n_lf:5d}  acc={float(info_b.acceptance_rate):.4f}",
            flush=True,
        )
        if i + 1 >= BURN_IN_MIN_B:
            recent = burnin_lf[max(0, len(burnin_lf) - 5) :]
            if sorted(recent)[len(recent) // 2] >= LEAPFROG_GATE:
                print(
                    f"  [GATE PASSED] step {i + 1}: "
                    f"median(last 5)={sorted(recent)[len(recent) // 2]} >= {LEAPFROG_GATE}",
                    flush=True,
                )
                gate_passed = True
                break

    if not gate_passed:
        print(
            f"  [WARN] Gate not passed after {BURN_IN_MAX_B} steps — "
            f"timing anyway. Last 5 lf: {burnin_lf[-5:]}",
            flush=True,
        )

    # Timed sampling
    phi_chain = []
    acc_chain = []
    div_chain = []
    lf_chain = []
    for j in range(N_SAMPLES_B):
        key_b, subkey = jax.random.split(key_b)
        t_step = time.perf_counter()
        state_b, info_b = step_b_jit(subkey, state_b)
        jax.block_until_ready(state_b)
        dt = time.perf_counter() - t_step
        wall_b_sampling += dt
        n_lf = int(getattr(info_b, "num_integration_steps", 0))
        phi_chain.append(
            [
                float(state_b.position["log_kernel_scale"]),
                float(state_b.position["log_lengthscale"]),
                float(state_b.position["log_noise_scale"]),
            ]
        )
        acc_chain.append(float(info_b.acceptance_rate))
        div_chain.append(int(info_b.is_divergent))
        lf_chain.append(n_lf)
        print(
            f"  step {j + 1}: {dt:.3f}s  lf={n_lf:5d}  "
            f"acc={float(info_b.acceptance_rate):.4f}",
            flush=True,
        )

    phi_b_chains.append(phi_chain)
    acc_b_chains.append(acc_chain)
    div_b_chains.append(div_chain)
    lf_b_chains.append(lf_chain)
    print(
        f"  chain {chain_i} done: " f"burn_lf={burnin_lf}  timed_lf={lf_chain}",
        flush=True,
    )

# Stack Arm B samples: (N_CHAINS_B, N_SAMPLES_B, 3)
phi_b = jnp.array(phi_b_chains)
acc_b = jnp.array(acc_b_chains)
div_b = jnp.array(div_b_chains)

print(
    f"\n  Arm B sampling wall (timed steps only, excl burn-in): {wall_b_sampling:.3f}s",
    flush=True,
)
print(
    f"  mean acceptance: {float(acc_b.mean()):.4f}  "
    f"total divergences: {int(div_b.sum())}",
    flush=True,
)
print(
    f"  all timed lf counts: {lf_b_chains}",
    flush=True,
)

# ESS Arm B
ess_b = blackjax.diagnostics.effective_sample_size(phi_b)  # shape (3,)
print(
    "  ESS Arm B: "
    + " ".join(f"{p}={float(v):.2f}" for p, v in zip(param_names, ess_b)),
    flush=True,
)
min_ess_b = float(jnp.min(ess_b))
# Wall for ESS/s: use timed sampling wall only (burn-in excluded per spec §16)
ess_per_sec_b = min_ess_b / wall_b_sampling
print(
    f"  min_ESS_b={min_ess_b:.2f}  ESS/s_b={ess_per_sec_b:.4f} "
    "(min_ESS / timed sampling wall, excl burn-in)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
t_total = time.perf_counter()
sep = "=" * 70
print(f"\n{sep}", flush=True)
print(f"ExpI Phase B Summary (total wall = {t_total - t0:.1f}s)", flush=True)
print(
    "  blackjax: stock main 007a9ded, laplace_marginal_factory(maxiter=500) explicit",
    flush=True,
)
print(sep, flush=True)
print(
    f"  Arm A (laplace_hmc, {N_CHAINS_A} chains x {N_SAMPLES_A} samples):", flush=True
)
print(
    f"    wall_a_total={wall_a_total:.3f}s  "
    f"(init={t_a_init_end - t_a_init_start:.3f}s + scan={wall_a_scan:.3f}s)",
    flush=True,
)
print(
    "    ESS per param: "
    + " ".join(f"{p}={float(v):.2f}" for p, v in zip(param_names, ess_a)),
    flush=True,
)
print(
    f"    min_ESS={min_ess_a:.2f}  ESS/s={ess_per_sec_a:.4f}",
    flush=True,
)
print(
    f"    mean_acc={float(acc_a.mean()):.4f}  n_div={int(div_a.sum())}",
    flush=True,
)
print(
    f"\n  Arm B (NUTS, {N_CHAINS_B} chains x {N_SAMPLES_B} samples, sequential):",
    flush=True,
)
print(
    f"    wall_b_sampling={wall_b_sampling:.3f}s  (timed steps only, excl burn-in)",
    flush=True,
)
print(
    "    ESS per param: "
    + " ".join(f"{p}={float(v):.2f}" for p, v in zip(param_names, ess_b)),
    flush=True,
)
print(
    f"    min_ESS={min_ess_b:.2f}  ESS/s={ess_per_sec_b:.4f}",
    flush=True,
)
print(
    f"    mean_acc={float(acc_b.mean()):.4f}  n_div={int(div_b.sum())}",
    flush=True,
)
print(
    f"    lf_counts (all chains): {lf_b_chains}",
    flush=True,
)
print("\n  Raw ESS/s ratio (Arm A / Arm B): ", end="", flush=True)
if ess_per_sec_b > 0:
    ratio = ess_per_sec_a / ess_per_sec_b
    print(f"{ratio:.2f}x (laplace faster if > 1)", flush=True)
else:
    print("undefined (ESS_b=0)", flush=True)
print(
    f"\n  NOTE: N_SAMPLES_B={N_SAMPLES_B}/chain is intentionally small "
    "(budget constraint). ESS_b estimates are noisy. "
    "@statistician: flag in verdict.",
    flush=True,
)
print(sep, flush=True)
