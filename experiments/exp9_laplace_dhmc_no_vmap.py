"""Exp 9: run_inference_algorithm without vmap — laplace_dhmc × gp_regression.

Statistician's decision table probe:
  Config: window_adaptation_diag_imm, n_warmup=50, n_samples=10, num_chains=1
  Budget: 10 min hard kill

Purpose: isolate whether the 83× slowdown (exp7 vs exp6) is caused by the
``jax.vmap`` wrapper in the recipe runner, or by ``run_inference_algorithm``
itself (which uses ``lax.scan`` internally).

Decision table:
  exp9 ≈ exp6 (~6-20s): ``run_inference_algorithm`` itself is fast; the 83×
    cost sits entirely in ``jax.vmap(_run_one_chain)``. Fix: drop the vmap
    (sequential chains, or vmap-of-jitted-step).
  exp9 ≈ exp7 (~500s): ``run_inference_algorithm`` is slow independent of
    vmap.  vmap adds an outer wrapper but the inner scan compilation is the
    bottleneck.  Fix: reduce L-BFGS maxiter or avoid the L-BFGS scan.

Context:
  exp6 (bare jit+scan, 10 steps): 6.0s → FAST
  exp7 (emit_low_recipe_for_cell, 1 chain, 10 samples): ~500s → SLOW (83×)
  exp8 (emit_low_recipe_for_cell, 4 chains, 10 samples): ~2696s → 5.4× exp7

Setup: same as exp7 but calling run_inference_algorithm directly (no vmap).
  1. Build laplace components for gp_regression
  2. Run window_adaptation_diag_imm warmup (1 chain, n_warmup=50)
  3. Build laplace_dhmc kernel with adapted step_size / IMM
  4. Re-init laplace state from warmup position (same as recipe runner)
  5. Call run_inference_algorithm(rng_key, kernel, num_steps=10, ...) directly
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
    f"[t=+{t_jax - t0:.1f}s] JAX imported: "
    f"x64={jax.config.read('jax_enable_x64')}, "
    f"backend={jax.default_backend()}",
    flush=True,
)

sys.path.insert(0, "/home/jp/blackjax-devs/tuningfork")

from blackjax.util import run_inference_algorithm  # noqa: E402

from tuningfork.base_method import BASE_METHODS  # noqa: E402
from tuningfork.calibration.tune import default_params_for  # noqa: E402
from tuningfork.model import MODELS  # noqa: E402
from tuningfork.model._numpyro import build_logdensity_fn  # noqa: E402
from tuningfork.recipes._recipe_runner import _build_laplace_components  # noqa: E402
from tuningfork.warmup import WARMUPS  # noqa: E402

t_imports = time.perf_counter()
print(f"[t=+{t_imports - t0:.1f}s] All imports done", flush=True)

# ---------------------------------------------------------------------------
# Build laplace components for gp_regression (same as exp6)
# ---------------------------------------------------------------------------
model = MODELS["gp_regression"]
master_key = jax.random.key(20260517)
init_key, warmup_key, sample_key = jax.random.split(master_key, 3)

print(f"[t=+{time.perf_counter() - t0:.1f}s] Building logdensity_fn...", flush=True)
full_position, joint_logdensity_fn, _data = build_logdensity_fn(init_key, model)

laplace_result = _build_laplace_components(
    "gp_regression", full_position, joint_logdensity_fn
)
assert laplace_result is not None, "gp_regression not in _LAPLACE_PHI_THETA_SPLITS"
phi_init, log_joint_fn, theta_init, marginal_logdensity_fn = laplace_result

t_built = time.perf_counter()
print(
    f"[t=+{t_built - t0:.1f}s] Laplace components built: "
    f"phi_dim={len(phi_init)}, theta_dim={theta_init['f_raw'].shape[0]}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Run warmup: window_adaptation_diag_imm, 1 chain, n_warmup=50
# (same warmup as exp7; warmup is fast ~20s)
# ---------------------------------------------------------------------------
base_method = BASE_METHODS["laplace_dhmc"]
warmup = WARMUPS["window_adaptation_diag_imm"]

print(
    f"[t=+{time.perf_counter() - t0:.1f}s] "
    "Running warmup (window_adaptation_diag_imm, n_warmup=50, num_chains=1)...",
    flush=True,
)
t_warmup0 = time.perf_counter()

batched_state, batched_params = warmup.runner(
    warmup_key,
    phi_init,
    50,  # n_warmup
    base_method,
    logdensity_fn=marginal_logdensity_fn,
    num_chains=1,
)

t_warmup = time.perf_counter() - t_warmup0
step_size = float(jnp.asarray(batched_params["step_size"]).ravel()[0])
imm = jnp.asarray(batched_params["inverse_mass_matrix"])[0]

print(
    f"[t=+{time.perf_counter() - t0:.1f}s] "
    f"Warmup done in {t_warmup:.1f}s. "
    f"step_size={step_size:.4g}, imm_shape={imm.shape}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Build laplace_dhmc kernel with adapted params; re-init from warmup position
# ---------------------------------------------------------------------------
# The warmup ran HMC (laplace substitute); warmup state is an HMCState.
# We need to re-init laplace_dhmc from the warmup position (same as recipe runner).
defaults = default_params_for(base_method)
shared_kwargs = {
    k: v for k, v in defaults.items() if k not in ("step_size", "inverse_mass_matrix")
}

kernel = base_method.factory(
    marginal_logdensity_fn,  # positional slot: logdensity_fn (marginal over phi)
    log_joint_fn=log_joint_fn,
    theta_init=theta_init,
    step_size=step_size,
    inverse_mass_matrix=imm,
    **shared_kwargs,
)

# warmup state has leading dim=1 (vmap over 1 chain); take chain 0
warmup_position_0 = jax.tree.map(lambda x: x[0], batched_state.position)

# laplace_dhmc.init(phi_position, rng_key) — rng_key seeds random_generator_arg
reinit_key, run_key = jax.random.split(sample_key)
state_0 = kernel.init(warmup_position_0, reinit_key)

t_kernel = time.perf_counter()
print(
    f"[t=+{t_kernel - t0:.1f}s] Kernel + state init done",
    flush=True,
)

# ---------------------------------------------------------------------------
# Call run_inference_algorithm directly — NO jax.vmap
# ---------------------------------------------------------------------------
print(
    f"[t=+{time.perf_counter() - t0:.1f}s] "
    "Invoking run_inference_algorithm(laplace_dhmc, n_steps=10, no vmap)...",
    flush=True,
)
t_ria_start = time.perf_counter()

final_state, (all_states, all_infos) = run_inference_algorithm(
    rng_key=run_key,
    inference_algorithm=kernel,
    num_steps=10,
    initial_state=state_0,
)
# Force all XLA evaluation to complete before taking the timestamp
_ = jax.block_until_ready(final_state.position)

t_done = time.perf_counter()
ria_wall = t_done - t_ria_start
total_wall = t_done - t0

print(f"[t=+{total_wall:.1f}s] block_until_ready returned", flush=True)

print("\n=== Exp 9 Result ===", flush=True)
print(f"  run_inference_algorithm wall: {ria_wall:.1f}s", flush=True)
print(f"  warmup wall: {t_warmup:.1f}s", flush=True)
print(f"  total wall: {total_wall:.1f}s", flush=True)

# Reference points from prior probes
exp6_wall = 6.0
exp7_ria_wall = 500.0  # ~all of exp7's 500s was sampling compilation

ratio_vs_exp6 = ria_wall / exp6_wall
ratio_vs_exp7_ria = ria_wall / exp7_ria_wall

print(f"  exp6_wall (bare jit+scan, 10 steps): {exp6_wall:.1f}s", flush=True)
print(f"  exp7_sampling_wall (~all compile): {exp7_ria_wall:.1f}s", flush=True)
print(f"  ratio (exp9/exp6): {ratio_vs_exp6:.1f}×", flush=True)
print(f"  ratio (exp9/exp7_sampling): {ratio_vs_exp7_ria:.3f}×", flush=True)

if ria_wall < 60.0:
    print(
        f"\nExp 9 FAST ({ria_wall:.1f}s, {ratio_vs_exp6:.1f}× exp6): "
        "run_inference_algorithm is cheap without vmap.",
        flush=True,
    )
    print(
        "  -> Root cause confirmed: jax.vmap(_run_one_chain) is the 83× "
        "compile blowup.",
        flush=True,
    )
    print(
        "  -> Fix: drop vmap-of-run_inference_algorithm; use "
        "vmap-of-jitted-step OR sequential chains.",
        flush=True,
    )
elif ria_wall < 300.0:
    print(
        f"\nExp 9 MODERATE ({ria_wall:.1f}s, {ratio_vs_exp6:.1f}× exp6): "
        "run_inference_algorithm adds overhead but not the full 83×.",
        flush=True,
    )
    print(
        "  -> Partial vmap tax. Also check if scan compilation in "
        "run_inference_algorithm differs from bare jit+scan.",
        flush=True,
    )
else:
    print(
        f"\nExp 9 SLOW ({ria_wall:.1f}s, {ratio_vs_exp7_ria:.2f}× exp7_sampling): "
        "run_inference_algorithm itself is the bottleneck.",
        flush=True,
    )
    print(
        "  -> vmap is NOT the root cause; the lax.scan inside "
        "run_inference_algorithm triggers XLA specialisation.",
        flush=True,
    )
    print(
        "  -> Fix: reduce L-BFGS maxiter or replace static scan with "
        "while_loop in the L-BFGS body.",
        flush=True,
    )

print("\nDone.", flush=True)
