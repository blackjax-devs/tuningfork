"""ExpD: Pathfinder-style warmup probe for gp_regression × laplace_hmc.

Key question: does L-BFGS MAP + inverse Hessian IMM capture the +0.754 ridge?

Why this matters
----------------
Exp 4/5 (window_adaptation × laplace_dhmc/dmhmc) blocked by NUTS step_size
collapse (step_size → 2.4e-4 after n_warmup=50; → 2e-7 after n_warmup=1000).
Root cause: window_adaptation uses NUTS which hits depth-1024 trees for this
model+seed. Pathfinder warmup avoids NUTS entirely.

Design (revised after v1 timeout)
----------------------------------
v1 used blackjax.vi.pathfinder.approximate() which internally does:
  1. L-BFGS on marginal → history with alpha/beta/gamma at each path step
  2. ELBO estimation: jax.vmap(51 iters, jax.vmap(50 samples, marginal_eval))
     → 51 × 50 = 2550 marginal_logdensity_fn evaluations
     → TIMEOUT: each eval is ~0.03s, 2550 × 0.03s ≈ 76s + compile overhead

v2 (this script): skip ELBO estimation, use MAP + final inverse Hessian
  1. Run _minimize_lbfgs on marginal_logdensity_fn (3D) → MAP + L-BFGS history
  2. Reconstruct 3×3 IMM from final-step L-BFGS history (alpha, S, Z)
  3. Use MAP as starting position for laplace_hmc sampling

The IMM from the final L-BFGS step is the inverse Hessian of -log_p_marginal
evaluated at the MAP. For a unimodal posterior, this is a good approximation
to the posterior covariance. The ELBO-selection step in pathfinder just picks
the path iteration with the best Gaussian fit — for a simple 3D problem, the
final step should be optimal anyway.

Phases
------
1. L-BFGS optimization on marginal_logdensity_fn → MAP phi + 3×3 IMM
   → compute corr(log_ks, log_ls), compare to +0.754 (ridge test)
2. laplace_hmc sampling: 10 steps with L-BFGS IMM + step-size probe
   → report wall, divergences, acceptance rate
3. (conditional) 4-chain vmap if 1-chain succeeds
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
from blackjax.optimizers.lbfgs import (  # noqa: E402
    _minimize_lbfgs,
    lbfgs_inverse_hessian_factors,
    lbfgs_inverse_hessian_formula_1,
)
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
LBFGS_MAXITER = 30  # L-BFGS outer iterations on 3D marginal
LBFGS_MAXCOR = 10  # L-BFGS history size
LBFGS_MAXLS = 50  # L-BFGS max linesearch steps
N_SAMPLING_STEPS = 10  # laplace_hmc steps
L_FIXED = 10  # num_integration_steps (fixed-L, no dynamic while_loop)
STEP_SIZES = [0.5, 0.2, 0.1, 0.05]  # step_size probe (3 steps each)

# ---------------------------------------------------------------------------
# Setup: gp_regression logdensity + Laplace components
# ---------------------------------------------------------------------------
rng_key = jax.random.key(SEED)
key_init, key_sampling = jax.random.split(rng_key)

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

phi_init, log_joint_fn, theta_init, marginal_logdensity_fn = laplace_components

phi_keys_sorted = sorted(phi_init.keys())
phi_flat_init, unravel_fn = ravel_pytree(phi_init)
t_laplace = time.perf_counter()
print(
    f"[t=+{t_laplace - t0:.1f}s] Laplace components built",
    flush=True,
)
print(f"  phi_keys (sorted = flat order): {phi_keys_sorted}", flush=True)
print(f"  phi_init (flat): {np.array(phi_flat_init)}", flush=True)

# ---------------------------------------------------------------------------
# Phase 1: L-BFGS MAP on marginal_logdensity_fn (3D)
# ---------------------------------------------------------------------------
print(
    f"\n=== Phase 1: L-BFGS MAP on 3D marginal "
    f"(maxiter={LBFGS_MAXITER}, maxcor={LBFGS_MAXCOR}, maxls={LBFGS_MAXLS}) ===",
    flush=True,
)


# Flat objective: minimize -log_marginal(phi)
def flat_objective(x_flat):
    return -marginal_logdensity_fn(unravel_fn(x_flat))


t_lbfgs_start = time.perf_counter()

last_step, history = _minimize_lbfgs(
    flat_objective,
    phi_flat_init,
    LBFGS_MAXITER,
    LBFGS_MAXCOR,
    1e-8,  # gtol
    1e-5,  # ftol
    LBFGS_MAXLS,
)

# Force materialization
phi_map_flat = np.array(jax.block_until_ready(last_step.params))
t_lbfgs_end = time.perf_counter()

lbfgs_wall = t_lbfgs_end - t_lbfgs_start
phi_map = unravel_fn(jnp.array(phi_map_flat))
print(f"  L-BFGS wall (incl. JIT compile): {lbfgs_wall:.2f}s", flush=True)
print(f"  MAP phi (flat): {phi_map_flat}", flush=True)
final_lp = float(marginal_logdensity_fn(phi_map))
print(f"  MAP marginal log p: {final_lp:.3f}", flush=True)
print(f"  L-BFGS grad norm: {float(last_step.state.error):.6f}", flush=True)

# ---------------------------------------------------------------------------
# Phase 2: Reconstruct 3×3 IMM from L-BFGS history
# ---------------------------------------------------------------------------
print("\n=== Phase 2: Reconstruct 3x3 IMM from L-BFGS history ===", flush=True)

# The history has shape (LBFGS_MAXITER+1, d):
#   history.x[i]      = phi position at step i
#   history.g[i]      = gradient of -log_marginal at step i
#   history.alpha[i]  = diagonal inv-Hessian approximation at step i
#   history.update_mask[i] = whether step i update was included

# Replicate pathfinder's sliding-window construction for final step
alpha_final = history.alpha[-1]  # (d,) = (3,)
s_diff = jnp.diff(history.x, axis=0)  # (maxiter, d)
z_diff = jnp.diff(history.g, axis=0)  # (maxiter, d)
update_mask = history.update_mask[1:]  # (maxiter, d) — valid update flags

s_masked = jnp.where(update_mask, s_diff, jnp.zeros_like(s_diff))
z_masked = jnp.where(update_mask, z_diff, jnp.zeros_like(z_diff))

# Pad with LBFGS_MAXCOR zeros at front (same as pathfinder)
s_padded = jnp.pad(s_masked, ((LBFGS_MAXCOR, 0), (0, 0)), mode="constant")
z_padded = jnp.pad(z_masked, ((LBFGS_MAXCOR, 0), (0, 0)), mode="constant")

# Final step window: indices LBFGS_MAXITER to LBFGS_MAXITER+LBFGS_MAXCOR
s_final_window = s_padded[LBFGS_MAXITER : LBFGS_MAXITER + LBFGS_MAXCOR]  # (maxcor, d)
z_final_window = z_padded[LBFGS_MAXITER : LBFGS_MAXITER + LBFGS_MAXCOR]  # (maxcor, d)

beta_final, gamma_final = lbfgs_inverse_hessian_factors(
    s_final_window.T, z_final_window.T, alpha_final
)
imm_3x3 = lbfgs_inverse_hessian_formula_1(alpha_final, beta_final, gamma_final)

# Force eval
imm_np = np.array(jax.block_until_ready(imm_3x3))
alpha_np = np.array(alpha_final)

print(f"  alpha (diag inv-Hessian): {alpha_np}", flush=True)
print("\n  3x3 IMM (inverse Hessian of -log_marginal at MAP):", flush=True)
for i, key_i in enumerate(phi_keys_sorted):
    row = "  ".join(f"{imm_np[i, j]:+.4f}" for j in range(3))
    print(f"    [{key_i:20s}] {row}", flush=True)

# Variances and stds from IMM diagonal
var_diag = np.diag(imm_np)
std_diag = np.sqrt(np.maximum(var_diag, 0))
print("\n  Marginal stds (from IMM diagonal):", flush=True)
for i, key_i in enumerate(phi_keys_sorted):
    print(f"    {key_i:25s}: std = {std_diag[i]:.4f}", flush=True)

# RIDGE TEST: corr(log_kernel_scale, log_lengthscale)
# Flat order (alphabetical): 0=log_kernel_scale, 1=log_lengthscale, 2=log_noise_scale
idx_ks = phi_keys_sorted.index("log_kernel_scale")
idx_ls = phi_keys_sorted.index("log_lengthscale")
cov_ks_ls = imm_np[idx_ks, idx_ls]
corr_ks_ls = cov_ks_ls / (std_diag[idx_ks] * std_diag[idx_ls])

print("\n  === RIDGE TEST ===", flush=True)
print(f"  corr(log_kernel_scale, log_lengthscale) = {corr_ks_ls:.4f}", flush=True)
print("  Target: +0.754 (from Exp 2 MCLMC posterior ridge)", flush=True)
ridge_captured = abs(corr_ks_ls - 0.754) < 0.15
print(
    f"  Ridge test: {'PASS ✓' if ridge_captured else 'FAIL ✗'} "
    f"(threshold |corr - 0.754| < 0.15)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Phase 3: Step-size probe (3 steps each at multiple step_sizes)
# ---------------------------------------------------------------------------
print("\n=== Phase 3: Step-size probe (3 steps each) ===", flush=True)

best_step_size = None
probe_results = {}

for step_size_c in STEP_SIZES:
    try:
        algo_probe = blackjax.laplace_hmc(
            log_joint_fn,
            theta_init,
            step_size=step_size_c,
            inverse_mass_matrix=imm_3x3,
            num_integration_steps=L_FIXED,
        )
        probe_state = jax.jit(algo_probe.init)(phi_map)
        _ = jax.block_until_ready(probe_state)

        # 3-step probe
        k0, k1, k2 = jax.random.split(jax.random.key(SEED + int(step_size_c * 1000)), 3)
        accs = []
        divs = []
        s = probe_state
        for ki in [k0, k1, k2]:
            s, info = jax.jit(algo_probe.step)(ki, s)
            _ = jax.block_until_ready(s)
            accs.append(float(info.acceptance_rate))
            divs.append(int(info.is_divergent))

        mean_acc = float(np.mean(accs))
        n_div = int(sum(divs))
        probe_results[step_size_c] = (mean_acc, n_div)
        print(
            f"  step_size={step_size_c:.3f}: mean_acc={mean_acc:.3f}, n_div={n_div}/3",
            flush=True,
        )
        if 0.4 <= mean_acc <= 0.95 and n_div == 0:
            if best_step_size is None:
                best_step_size = step_size_c
    except Exception as e:
        print(f"  step_size={step_size_c:.3f}: ERROR — {e}", flush=True)

if best_step_size is None:
    best_step_size = 0.05
    print(
        f"  WARNING: no clean step_size found, using fallback={best_step_size}",
        flush=True,
    )
else:
    print(f"  Selected step_size: {best_step_size:.3f}", flush=True)

# ---------------------------------------------------------------------------
# Phase 4: Full 10-step laplace_hmc run with L-BFGS IMM
# ---------------------------------------------------------------------------
print(
    f"\n=== Phase 4: laplace_hmc sampling "
    f"(n_steps={N_SAMPLING_STEPS}, L={L_FIXED}, step_size={best_step_size}) ===",
    flush=True,
)

algorithm = blackjax.laplace_hmc(
    log_joint_fn,
    theta_init,
    step_size=best_step_size,
    inverse_mass_matrix=imm_3x3,
    num_integration_steps=L_FIXED,
)

t_sample_start = time.perf_counter()
final_state, (history_states, history_info) = run_inference_algorithm(
    key_sampling,
    algorithm,
    num_steps=N_SAMPLING_STEPS,
    initial_position=phi_map,
    progress_bar=False,
)
_ = jax.block_until_ready(final_state)
t_sample_end = time.perf_counter()

sampling_wall = t_sample_end - t_sample_start
acceptances_arr = np.array(history_info.acceptance_rate)
divergences_arr = np.array(history_info.is_divergent)
n_div = int(divergences_arr.sum())
mean_acc = float(acceptances_arr.mean())
lps = np.array(history_states.logdensity)

print(f"  Sampling wall (10 steps): {sampling_wall:.2f}s", flush=True)
print(
    f"  Acceptance rate: mean={mean_acc:.3f}, min={acceptances_arr.min():.3f}, max={acceptances_arr.max():.3f}",
    flush=True,
)
print(f"  Divergences: {n_div}/{N_SAMPLING_STEPS}", flush=True)
print(f"  logdensity range: [{lps.min():.2f}, {lps.max():.2f}]", flush=True)

final_phi_flat = np.array(ravel_pytree(final_state.position)[0])
print(f"  Final phi (flat): {final_phi_flat}", flush=True)

# Trace of phi positions across 10 steps
positions_flat = np.array(
    jax.vmap(lambda pos: ravel_pytree(pos)[0])(history_states.position)
)
print("  phi trace (first 3 steps, flat):", flush=True)
for step_i in range(min(3, N_SAMPLING_STEPS)):
    print(f"    step {step_i}: {positions_flat[step_i]}", flush=True)

# ---------------------------------------------------------------------------
# Phase 5 (conditional): 4-chain vmap
# ---------------------------------------------------------------------------
vmap_wall = None
vmap_accs_mean = None
vmap_n_div = None

if n_div == 0 and mean_acc > 0.3:
    print(f"\n=== Phase 5: 4-chain vmap (n_steps={N_SAMPLING_STEPS}) ===", flush=True)

    vmap_keys = jax.random.split(jax.random.key(SEED + 2), 4)

    def run_one_chain(rng_key, phi_start):
        algo = blackjax.laplace_hmc(
            log_joint_fn,
            theta_init,
            step_size=best_step_size,
            inverse_mass_matrix=imm_3x3,
            num_integration_steps=L_FIXED,
        )
        final, hist = run_inference_algorithm(
            rng_key,
            algo,
            num_steps=N_SAMPLING_STEPS,
            initial_position=phi_start,
            progress_bar=False,
        )
        return final, hist

    # 4 independent starting positions (perturbed MAP)
    phi_map_flat_j = jnp.array(phi_map_flat)
    key_perturb = jax.random.key(SEED + 3)
    phi_starts_4 = phi_map_flat_j[None, :] + 0.1 * jax.random.normal(
        key_perturb, (4, len(phi_map_flat))
    )

    def run_chain_flat(rng_key, phi_flat):
        return run_one_chain(rng_key, unravel_fn(phi_flat))

    t_vmap_start = time.perf_counter()
    vmap_finals, vmap_hists_tup = jax.vmap(run_chain_flat)(vmap_keys, phi_starts_4)
    _ = jax.block_until_ready(vmap_finals)
    t_vmap_end = time.perf_counter()

    vmap_hists_states, vmap_hists_info = vmap_hists_tup
    vmap_wall = t_vmap_end - t_vmap_start
    vmap_accs_mean = float(np.array(vmap_hists_info.acceptance_rate).mean())
    vmap_n_div = int(np.array(vmap_hists_info.is_divergent).sum())

    print(f"  4-chain vmap wall: {vmap_wall:.2f}s", flush=True)
    print(f"  Mean acceptance: {vmap_accs_mean:.3f}", flush=True)
    print(f"  Total divergences: {vmap_n_div}/{4 * N_SAMPLING_STEPS}", flush=True)
else:
    print(
        f"\n  Skipping Phase 5 (4-chain vmap): 1-chain not clean "
        f"(n_div={n_div}, mean_acc={mean_acc:.3f})",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
t_total = time.perf_counter()
sep = "=" * 62
print(f"\n{sep}", flush=True)
print(f"ExpD Summary (total wall = {t_total - t0:.1f}s)", flush=True)
print(sep, flush=True)
print(f"  L-BFGS MAP wall: {lbfgs_wall:.2f}s", flush=True)
print(f"  3×3 IMM diagonal (alpha): {alpha_np}", flush=True)
print(
    f"  Ridge test: corr(log_ks, log_ls) = {corr_ks_ls:.4f} (target +0.754)", flush=True
)
print(f"  Ridge captured: {'YES ✓' if ridge_captured else 'NO ✗'}", flush=True)
print("  laplace_hmc 10 steps:", flush=True)
print(
    f"    step_size={best_step_size}, mean_acc={mean_acc:.3f}, n_div={n_div}",
    flush=True,
)
print(f"    wall={sampling_wall:.2f}s", flush=True)
if vmap_wall is not None:
    print(
        f"  4-chain vmap: wall={vmap_wall:.2f}s, mean_acc={vmap_accs_mean:.3f}, n_div={vmap_n_div}",
        flush=True,
    )
viable = ridge_captured and n_div == 0 and mean_acc > 0.3
print(f"  Overall: {'VIABLE' if viable else 'NEEDS_INVESTIGATION'}", flush=True)
print(sep, flush=True)

# ---------------------------------------------------------------------------
# Phase 6: Test full pathfinder.approximate with minimal params
# (to determine if vmap(vmap(custom_root)) compile time is feasible)
# ---------------------------------------------------------------------------
print(
    "\n=== Phase 6: Full pathfinder.approximate compile/run test "
    "(num_samples=3, maxiter=10) ===",
    flush=True,
)
print("  NOTE: This tests whether pathfinder ELBO compilation is feasible.", flush=True)
print(
    "  The key question: can vmap(10 iters, vmap(3 samples, custom_root)) compile?",
    flush=True,
)

import blackjax.vi.pathfinder as pathfinder_vi  # noqa: E402

key_pf6 = jax.random.key(SEED + 10)
t_pf6_start = time.perf_counter()
try:
    pf_state6, pf_info6 = pathfinder_vi.approximate(
        key_pf6,
        marginal_logdensity_fn,
        phi_init,
        num_samples=3,  # minimal ELBO samples
        maxiter=10,  # minimal L-BFGS iterations
        maxls=30,  # conservative linesearch cap
    )
    pf_elbo6 = float(jax.block_until_ready(pf_state6.elbo))
    t_pf6_end = time.perf_counter()

    pf_wall = t_pf6_end - t_pf6_start
    print(f"  Pathfinder compile+run wall: {pf_wall:.2f}s", flush=True)
    print(f"  Best ELBO: {pf_elbo6:.3f}", flush=True)
    print(
        f"  Best position: {np.array(ravel_pytree(pf_state6.position)[0])}", flush=True
    )

    # Reconstruct IMM from pathfinder factors
    imm_pf = lbfgs_inverse_hessian_formula_1(
        pf_state6.alpha, pf_state6.beta, pf_state6.gamma
    )
    imm_pf_np = np.array(imm_pf)
    var_pf = np.diag(imm_pf_np)
    std_pf = np.sqrt(np.maximum(var_pf, 0))

    print("\n  Pathfinder 3×3 IMM:", flush=True)
    for i, key_i in enumerate(phi_keys_sorted):
        row = "  ".join(f"{imm_pf_np[i, j]:+.4f}" for j in range(3))
        print(f"    [{key_i:20s}] {row}", flush=True)

    cov_pf_ks_ls = imm_pf_np[idx_ks, idx_ls]
    corr_pf_ks_ls = (
        cov_pf_ks_ls / (std_pf[idx_ks] * std_pf[idx_ls])
        if (std_pf[idx_ks] > 0 and std_pf[idx_ls] > 0)
        else 0.0
    )

    print("\n  Pathfinder RIDGE TEST:", flush=True)
    print(f"  corr(log_ks, log_ls) = {corr_pf_ks_ls:.4f} (target +0.754)", flush=True)
    pf_ridge = abs(corr_pf_ks_ls - 0.754) < 0.15
    print(f"  Ridge test: {'PASS ✓' if pf_ridge else 'FAIL ✗'}", flush=True)
    print(
        f"  Phase 6 conclusion: pathfinder IS {'feasible' if pf_wall < 120 else 'too slow'} "
        f"(wall={pf_wall:.1f}s, budget=120s)",
        flush=True,
    )

except Exception as e:
    t_pf6_end = time.perf_counter()
    print(f"  ERROR after {t_pf6_end - t_pf6_start:.1f}s: {e}", flush=True)

print(f"\n[t=+{time.perf_counter() - t0:.1f}s] Script complete", flush=True)
