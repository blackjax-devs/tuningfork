"""ExpG: Validate marginal_logdensity_fn vs exact GP marginal likelihood.

Deciding probe: does the Laplace marginal reproduce the exact analytical GP marginal?

If the Laplace approximation is exact under NCP (Gaussian conditional), then
marginal_logdensity_fn(phi) should equal log_p_exact(phi) up to a constant offset.
A phi-dependent residual >0.5 nats => BUG. A Hessian ratio >5x => BUG.

Design (per TL spec 2026-05-24T22:16:47):
  Build log_p_exact(phi) = log p(phi) + log N(y; 0, K(phi) + sigma_n^2*I)
    directly from RBF kernel + Cholesky (NO L-BFGS, NO Laplace machinery).
  X/y from MODELS["gp_regression"].

  Step 1 (value cross-check): marginal_logdensity_fn vs log_p_exact at 10 GT phi
    samples. Constant offset => correct; phi-dependent residual >0.5 nats => BUG.

  Step 2 (curvature): jax.hessian of both at the GT posterior mean.
    H_laplace > 5x H_exact => BUG.

  Step 3 (2D ridge profile): 7x7 grid over (log_ks, log_ls) +/-1.5 GT-std;
    does the +0.754-angle ridge appear in both?

Verdict:
  (a) value-offset-constant + H_laplace~H_exact + stds match GT => correct marginal,
      genuine multiscale pathology => close laplace track as structural dead-end.
  (b) phi-dependent value residual OR H_laplace >> H_exact => BUG in marginal_logdensity_fn.
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

from jax.flatten_util import ravel_pytree  # noqa: E402

from tuningfork.model import MODELS  # noqa: E402
from tuningfork.model._numpyro import build_logdensity_fn  # noqa: E402
from tuningfork.model.gp_regression import JITTER, N_OBS, X_DATA, Y_DATA  # noqa: E402
from tuningfork.recipes._recipe_runner import _build_laplace_components  # noqa: E402

t_imports = time.perf_counter()
print(f"[t=+{t_imports - t0:.1f}s] All imports done", flush=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 20260517
GT_DRAWS_PATH = (
    "/home/jp/blackjax-devs/tuningfork/tuningfork/catalog"
    "/gp_regression/groundtruth_samples/blackjax/draws.npz"
)
# GT posterior means (certified 40K NUTS)
GT_MEANS = {
    "log_kernel_scale": 0.40870562293007373,
    "log_lengthscale": -1.0424925985381703,
    "log_noise_scale": -2.34163615643574,
}
# GT posterior stds
GT_STDS = {
    "log_kernel_scale": 0.427794,
    "log_lengthscale": 0.178336,
    "log_noise_scale": 0.051258,
}

# ---------------------------------------------------------------------------
# Build Laplace marginal_logdensity_fn
# ---------------------------------------------------------------------------
rng_key = jax.random.key(SEED)
key_init = rng_key

entry = MODELS["gp_regression"]
init_position, logdensity_fn, _postprocess_fn = build_logdensity_fn(key_init, entry)

laplace_components = _build_laplace_components(
    "gp_regression", init_position, logdensity_fn
)
if laplace_components is None:
    raise RuntimeError("gp_regression not in _LAPLACE_PHI_THETA_SPLITS")

phi_init, log_joint_fn, theta_init, marginal_logdensity_fn = laplace_components
phi_keys_sorted = sorted(phi_init.keys())
print(f"[t=+{time.perf_counter() - t0:.1f}s] Laplace components built", flush=True)
print(f"  phi_keys (sorted): {phi_keys_sorted}", flush=True)

# ---------------------------------------------------------------------------
# Build exact GP marginal log-posterior
# ---------------------------------------------------------------------------
# The GP marginal (integrating out f analytically):
#   log p(y | phi) = -0.5 * y^T @ C^{-1} @ y
#                   - 0.5 * log det(C)
#                   - n/2 * log(2*pi)
# where C = K(phi) + sigma_n^2 * I
#       K(phi) = kernel_scale^2 * exp(-0.5 * sqdist / ls^2) + JITTER * I
#
# The NCP Laplace approximation is EXACT for this model (Gaussian conditional):
#   marginal_logdensity_fn(phi) should = log p(y|phi) + log p(phi) + const
#
# This is the reference implementation using no L-BFGS machinery.

X = jnp.array(X_DATA, dtype=jnp.float64)
y = jnp.array(Y_DATA, dtype=jnp.float64)
n = N_OBS


def log_p_exact(phi: dict) -> float:
    """Exact GP marginal log-posterior (no Laplace approximation, no L-BFGS).

    Integrates out f analytically: p(y|phi) = N(y; 0, K + sigma_n^2 * I).
    Includes phi prior: Normal(log_ks; 0, 1) + Normal(log_ls; 0, 1) + Normal(log_ns; -2, 1).
    """
    log_ls = phi["log_lengthscale"]
    log_ks = phi["log_kernel_scale"]
    log_ns = phi["log_noise_scale"]

    ls = jnp.exp(log_ls)
    ks = jnp.exp(log_ks)
    sigma_n = jnp.exp(log_ns)

    # RBF kernel (same formula as gp_regression.py)
    sqdist = (X[:, None] - X[None, :]) ** 2
    K = ks**2 * jnp.exp(-0.5 * sqdist / ls**2) + JITTER * jnp.eye(n)

    # Observation covariance C = K + sigma_n^2 * I
    C = K + sigma_n**2 * jnp.eye(n)

    # Cholesky of C for numerically stable log det + quadratic form
    L_C = jax.scipy.linalg.cholesky(C, lower=True)
    # log det(C) = 2 * sum(log diag(L_C))
    log_det_C = 2.0 * jnp.sum(jnp.log(jnp.diag(L_C)))
    # y^T C^{-1} y = ||L_C^{-1} y||^2
    v = jax.scipy.linalg.solve_triangular(L_C, y, lower=True)
    quad_form = jnp.dot(v, v)

    log_lik = -0.5 * quad_form - 0.5 * log_det_C - 0.5 * n * jnp.log(2.0 * jnp.pi)

    # Phi prior: Normal(log_ks; 0, 1), Normal(log_ls; 0, 1), Normal(log_ns; -2, 1)
    log_prior = (
        -0.5 * log_ks**2
        - 0.5 * log_ls**2
        - 0.5 * (log_ns + 2.0) ** 2
        - 1.5 * jnp.log(2.0 * jnp.pi)
    )

    return log_lik + log_prior


print(
    f"[t=+{time.perf_counter() - t0:.1f}s] Exact marginal function defined", flush=True
)

# ---------------------------------------------------------------------------
# Load 10 GT phi samples
# ---------------------------------------------------------------------------
gt_draws = np.load(GT_DRAWS_PATH)
n_samples_gt = 10
gt_phi_samples = [
    {
        "log_kernel_scale": jnp.float64(float(gt_draws["log_kernel_scale"][i])),
        "log_lengthscale": jnp.float64(float(gt_draws["log_lengthscale"][i])),
        "log_noise_scale": jnp.float64(float(gt_draws["log_noise_scale"][i])),
    }
    for i in range(n_samples_gt)
]
print(
    f"[t=+{time.perf_counter() - t0:.1f}s] GT phi samples loaded (n={n_samples_gt})",
    flush=True,
)

# ---------------------------------------------------------------------------
# Step 1: Value cross-check at 10 GT phi samples
# ---------------------------------------------------------------------------
print(
    "\n=== Step 1: Value cross-check (marginal_fn vs exact) at 10 GT phi samples ===",
    flush=True,
)
print(
    "  Expected: constant offset (approx exact => same up to const); "
    "phi-dependent residual >0.5 => BUG",
    flush=True,
)

laplace_vals = []
exact_vals = []

for i, phi_sample in enumerate(gt_phi_samples):
    t_step = time.perf_counter()
    lp_laplace = float(marginal_logdensity_fn(phi_sample))
    lp_exact = float(log_p_exact(phi_sample))
    laplace_vals.append(lp_laplace)
    exact_vals.append(lp_exact)
    print(
        f"  sample {i:2d}: laplace={lp_laplace:10.4f}, exact={lp_exact:10.4f}, "
        f"diff={lp_laplace - lp_exact:+8.4f}  [{time.perf_counter() - t_step:.1f}s]",
        flush=True,
    )

laplace_arr = np.array(laplace_vals)
exact_arr = np.array(exact_vals)
diffs = laplace_arr - exact_arr
offset_mean = diffs.mean()
offset_std = diffs.std()

print(
    f"\n  Offset (laplace - exact): mean={offset_mean:+.6f}, std={offset_std:.6f}",
    flush=True,
)
print(
    "  NOTE: A constant offset is expected (different normalisation constants).",
    flush=True,
)
residuals_centered = diffs - offset_mean
max_abs_residual = np.abs(residuals_centered).max()
print(f"  Max |residual - mean_offset| = {max_abs_residual:.6f}", flush=True)

step1_bug = max_abs_residual > 0.5
step1_verdict = "BUG" if step1_bug else "OK (constant offset)"
print(f"  Step 1 verdict: {step1_verdict}", flush=True)

# ---------------------------------------------------------------------------
# Step 2: Hessian comparison at GT posterior mean
# ---------------------------------------------------------------------------
print(
    "\n=== Step 2: Hessian comparison at GT posterior mean ===",
    flush=True,
)
phi_gt_mean = {k: jnp.float64(GT_MEANS[k]) for k in phi_keys_sorted}

print("  Computing Hessian of Laplace marginal...", flush=True)
t_h_laplace = time.perf_counter()

phi_flat_gt, unravel_phi = ravel_pytree(phi_gt_mean)


def laplace_flat(phi_flat):
    return marginal_logdensity_fn(unravel_phi(phi_flat))


def exact_flat(phi_flat):
    return log_p_exact(unravel_phi(phi_flat))


H_laplace = np.array(jax.hessian(laplace_flat)(phi_flat_gt))
print(
    f"  [t=+{time.perf_counter() - t0:.1f}s] Laplace Hessian done ({time.perf_counter() - t_h_laplace:.1f}s)",
    flush=True,
)

print("  Computing Hessian of exact marginal...", flush=True)
t_h_exact = time.perf_counter()
H_exact = np.array(jax.hessian(exact_flat)(phi_flat_gt))
print(
    f"  [t=+{time.perf_counter() - t0:.1f}s] Exact Hessian done ({time.perf_counter() - t_h_exact:.1f}s)",
    flush=True,
)

# Eigenvalues of -H (negative Hessian = precision matrix at mode)
neg_H_laplace = -H_laplace
neg_H_exact = -H_exact

eigvals_laplace = np.linalg.eigvalsh(neg_H_laplace)
eigvals_exact = np.linalg.eigvalsh(neg_H_exact)

# Std = 1/sqrt(eigenvalue) of the precision matrix
stds_laplace = 1.0 / np.sqrt(np.maximum(eigvals_laplace, 1e-12))
stds_exact = 1.0 / np.sqrt(np.maximum(eigvals_exact, 1e-12))

print(f"\n  phi order (sorted): {phi_keys_sorted}", flush=True)
print(
    "\n  Per-dimension diagnostics (using diagonal of -H for phi-aligned stds):",
    flush=True,
)
for i, k in enumerate(phi_keys_sorted):
    std_laplace_diag = 1.0 / np.sqrt(max(-H_laplace[i, i], 1e-12))
    std_exact_diag = 1.0 / np.sqrt(max(-H_exact[i, i], 1e-12))
    ratio = -H_laplace[i, i] / -H_exact[i, i] if -H_exact[i, i] != 0 else float("nan")
    print(
        f"    {k:25s}: std_laplace={std_laplace_diag:.4f}, std_exact={std_exact_diag:.4f}, "
        f"H_ratio(laplace/exact)={ratio:.3f}  [GT_std={GT_STDS[k]:.4f}]",
        flush=True,
    )

print("\n  Eigenvalue stds (principal axes):", flush=True)
print(f"    Laplace stds (ascending): {sorted(stds_laplace)}", flush=True)
print(f"    Exact stds (ascending):   {sorted(stds_exact)}", flush=True)

max_h_ratio = np.max(eigvals_laplace / np.maximum(eigvals_exact, 1e-12))
print(f"\n  Max eigenvalue ratio (laplace/exact): {max_h_ratio:.3f}", flush=True)

step2_bug = max_h_ratio > 5.0
step2_verdict = (
    "BUG (H_laplace >> H_exact)" if step2_bug else "OK (H_laplace ~ H_exact)"
)
print(f"  Step 2 verdict: {step2_verdict}", flush=True)

# Correlation matrix from -H_exact (shows ridge structure)
diag_inv_exact = 1.0 / np.sqrt(np.diag(neg_H_exact))
corr_exact = neg_H_exact * diag_inv_exact[:, None] * diag_inv_exact[None, :]
print(
    "\n  Correlation matrix of phi from EXACT marginal Hessian at GT mean "
    "(should show +0.754 ks-ls ridge if marginal is correct):",
    flush=True,
)
for i, ki in enumerate(phi_keys_sorted):
    row = "  ".join(f"{corr_exact[i, j]:+.4f}" for j in range(3))
    print(f"    [{ki:20s}] {row}", flush=True)
idx_ks = phi_keys_sorted.index("log_kernel_scale")
idx_ls = phi_keys_sorted.index("log_lengthscale")
corr_ks_ls_exact = corr_exact[idx_ks, idx_ls]
print(
    f"\n  corr(log_ks, log_ls) from EXACT Hessian = {corr_ks_ls_exact:.4f} "
    f"(target from GT draws: ~+0.754)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Step 3: 2D ridge profile (7x7 grid over log_ks x log_ls)
# ---------------------------------------------------------------------------
print(
    "\n=== Step 3: 2D ridge profile (7x7 grid, log_ks x log_ls ±1.5 GT-std) ===",
    flush=True,
)
n_grid = 7
ks_range = 1.5 * GT_STDS["log_kernel_scale"]
ls_range = 1.5 * GT_STDS["log_lengthscale"]
ks_grid = np.linspace(
    GT_MEANS["log_kernel_scale"] - ks_range,
    GT_MEANS["log_kernel_scale"] + ks_range,
    n_grid,
)
ls_grid = np.linspace(
    GT_MEANS["log_lengthscale"] - ls_range,
    GT_MEANS["log_lengthscale"] + ls_range,
    n_grid,
)

print(
    f"  log_ks grid: [{ks_grid[0]:.3f}, ..., {ks_grid[-1]:.3f}] (GT mean={GT_MEANS['log_kernel_scale']:.3f})",
    flush=True,
)
print(
    f"  log_ls grid: [{ls_grid[0]:.3f}, ..., {ls_grid[-1]:.3f}] (GT mean={GT_MEANS['log_lengthscale']:.3f})",
    flush=True,
)

grid_laplace = np.zeros((n_grid, n_grid))
grid_exact = np.zeros((n_grid, n_grid))

t_grid_start = time.perf_counter()
for i, ks_val in enumerate(ks_grid):
    for j, ls_val in enumerate(ls_grid):
        phi_grid = {
            "log_kernel_scale": jnp.float64(float(ks_val)),
            "log_lengthscale": jnp.float64(float(ls_val)),
            "log_noise_scale": jnp.float64(GT_MEANS["log_noise_scale"]),
        }
        grid_laplace[i, j] = float(marginal_logdensity_fn(phi_grid))
        grid_exact[i, j] = float(log_p_exact(phi_grid))
    print(
        f"  Grid row {i + 1}/{n_grid} done [{time.perf_counter() - t_grid_start:.1f}s]",
        flush=True,
    )

# Normalize grids (subtract max for visual comparison)
grid_laplace_norm = grid_laplace - grid_laplace.max()
grid_exact_norm = grid_exact - grid_exact.max()

print("\n  Laplace marginal (normalized, log_ks rows × log_ls cols):", flush=True)
print(f"  log_ls: {ls_grid.round(2)}", flush=True)
for i, ks_val in enumerate(ks_grid):
    row = " ".join(f"{grid_laplace_norm[i, j]:+6.2f}" for j in range(n_grid))
    print(f"  log_ks={ks_val:+.3f}: {row}", flush=True)

print("\n  Exact marginal (normalized, log_ks rows × log_ls cols):", flush=True)
for i, ks_val in enumerate(ks_grid):
    row = " ".join(f"{grid_exact_norm[i, j]:+6.2f}" for j in range(n_grid))
    print(f"  log_ks={ks_val:+.3f}: {row}", flush=True)

# Check ridge alignment: the ridge in the exact marginal should have
# positive slope in (log_ks, log_ls) space — higher ks paired with higher ls
# Measure: correlation of (i, j) positions weighted by normalized log-density

# Use exact grid to estimate ridge direction
flat_vals = grid_exact_norm.ravel()
exp_vals = np.exp(np.clip(flat_vals, -20, 0))  # softmax-like weighting
exp_vals /= exp_vals.sum()
ks_idxs, ls_idxs = np.meshgrid(np.arange(n_grid), np.arange(n_grid), indexing="ij")
ks_flat = ks_grid[ks_idxs.ravel()]
ls_flat = ls_grid[ls_idxs.ravel()]
mean_ks = (ks_flat * exp_vals).sum()
mean_ls = (ls_flat * exp_vals).sum()
cov_ks_ls = ((ks_flat - mean_ks) * (ls_flat - mean_ls) * exp_vals).sum()
var_ks = ((ks_flat - mean_ks) ** 2 * exp_vals).sum()
var_ls = ((ls_flat - mean_ls) ** 2 * exp_vals).sum()
corr_grid_exact = cov_ks_ls / np.sqrt(max(var_ks * var_ls, 1e-12))

# Same for laplace grid
flat_vals_lap = grid_laplace_norm.ravel()
exp_vals_lap = np.exp(np.clip(flat_vals_lap, -20, 0))
exp_vals_lap /= exp_vals_lap.sum()
cov_ks_ls_lap = ((ks_flat - mean_ks) * (ls_flat - mean_ls) * exp_vals_lap).sum()
corr_grid_laplace = cov_ks_ls_lap / np.sqrt(max(var_ks * var_ls, 1e-12))

print(
    f"\n  Weighted corr(log_ks, log_ls) from 2D exact grid:   {corr_grid_exact:+.4f}",
    flush=True,
)
print(
    f"  Weighted corr(log_ks, log_ls) from 2D laplace grid: {corr_grid_laplace:+.4f}",
    flush=True,
)
print("  (Positive = ridge in same direction as GT +0.754)", flush=True)

step3_ridge_exact = corr_grid_exact > 0.3
step3_ridge_laplace = corr_grid_laplace > 0.3
step3_match = abs(corr_grid_exact - corr_grid_laplace) < 0.3
print(
    f"\n  Step 3 verdict: exact has ridge={step3_ridge_exact}, "
    f"laplace has ridge={step3_ridge_laplace}, grids match={step3_match}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Summary and (a)/(b) verdict
# ---------------------------------------------------------------------------
t_total = time.perf_counter()
sep = "=" * 68
print(f"\n{sep}", flush=True)
print(f"ExpG Summary (total wall = {t_total - t0:.1f}s)", flush=True)
print(sep, flush=True)

print("\nStep 1 (value cross-check at 10 GT phi samples):", flush=True)
print(f"  offset mean = {offset_mean:.6f}, std = {offset_std:.6f}", flush=True)
print(
    f"  max |centered residual| = {max_abs_residual:.6f} (threshold: 0.5 nats)",
    flush=True,
)
print(f"  Result: {step1_verdict}", flush=True)

print("\nStep 2 (Hessian ratio at GT posterior mean):", flush=True)
print(
    f"  max eigenvalue ratio (laplace/exact) = {max_h_ratio:.3f} (threshold: 5.0)",
    flush=True,
)
print(
    f"  corr(log_ks, log_ls) from exact Hessian = {corr_ks_ls_exact:.4f} (target: ~+0.754)",
    flush=True,
)
print(f"  Result: {step2_verdict}", flush=True)

print("\nStep 3 (2D ridge profile):", flush=True)
print(
    f"  corr from exact grid={corr_grid_exact:+.4f}, laplace grid={corr_grid_laplace:+.4f}",
    flush=True,
)
print(f"  Grids match: {step3_match}", flush=True)

# Final verdict
is_bug = step1_bug or step2_bug
if is_bug:
    verdict = "(b) BUG"
    explanation = (
        "marginal_logdensity_fn disagrees with the exact GP marginal. "
        "phi-dependent residual or H_laplace >> H_exact detected. "
        "Likely cause: L-BFGS non-convergence or log|H_theta| correction error. "
        "Escalate to blackjax — fixing may unlock all laplace cells."
    )
else:
    verdict = "(a) CORRECT MARGINAL — genuine multiscale geometry"
    explanation = (
        "marginal_logdensity_fn agrees with the exact GP marginal (constant offset only, "
        "H_laplace ~ H_exact). The 20-60x curvature mismatch vs GT posterior covariance "
        "is an intrinsic property of the GP marginal landscape: the marginal has narrow "
        "local curvature at the mode but wide global spread (non-Gaussian shape). "
        "window_adaptation step_size collapse is structural, not a bug. "
        "=> Close laplace track for gp_regression as structural dead-end."
    )

sep2 = "=" * 68
print(f"\n{sep2}", flush=True)
print(f"FINAL VERDICT: {verdict}", flush=True)
print(f"Explanation: {explanation}", flush=True)
print("=" * 68, flush=True)
