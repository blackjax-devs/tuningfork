"""Exp 6: bare kernel sanity check — laplace_dhmc × gp_regression, no warmup.

Statistician's decision table probe:
  Config: no_warmup, n_samples=10, num_chains=1
  Budget: 10 min hard kill

Purpose: isolate the XLA compilation cost of the L-BFGS kernel body itself
(without the scan-of-1000 warmup overhead). If even 10 bare kernel steps
take >10 min, the bottleneck is in the kernel body (L-BFGS specialisation),
not the warmup scan unrolling.

Direct bare-kernel path (bypasses emit_low_recipe_for_cell):
  no_warmup + laplace_dhmc cannot go through the recipe runner because
  no_warmup.runner raises NotImplementedError for base methods with
  extra_required_kwargs=("log_joint_fn", "theta_init"). The recipe runner's
  `batched_params["step_size"]` also KeyErrors when no_warmup returns {}.
  This script drives the laplace_dhmc kernel directly.

Timestamps:
  t0 = script start (before any import)
  t_jax = after `import jax`
  t_imports = after all tuningfork imports
  t_jit_start = before jit-compiled 10-step scan is invoked
  t_done = after block_until_ready on final position

If t_done - t_jit_start > 600s (10 min): compilation has failed the budget.
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

from tuningfork.base_method import BASE_METHODS  # noqa: E402
from tuningfork.calibration.tune import default_params_for  # noqa: E402
from tuningfork.model import MODELS  # noqa: E402
from tuningfork.model._numpyro import build_logdensity_fn  # noqa: E402
from tuningfork.recipes._recipe_runner import _build_laplace_components  # noqa: E402

t_imports = time.perf_counter()
print(f"[t=+{t_imports - t0:.1f}s] All imports done", flush=True)

# ---------------------------------------------------------------------------
# Build laplace components for gp_regression
# ---------------------------------------------------------------------------
model = MODELS["gp_regression"]
master_key = jax.random.key(20260517)
init_key, sample_key = jax.random.split(master_key)

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
# Build bare kernel (single chain, default step_size = ~0.126, identity IMM)
# ---------------------------------------------------------------------------
base_method = BASE_METHODS["laplace_dhmc"]
defaults = default_params_for(base_method)

step_size = float(defaults["step_size"])  # ~0.126
phi_dim = sum(jnp.asarray(v).size for v in phi_init.values())  # = 3
inverse_mass_matrix = jnp.ones(phi_dim, dtype=jnp.float64)

print(
    f"[t=+{time.perf_counter() - t0:.1f}s] "
    f"Building kernel: step_size={step_size:.4g}, phi_dim={phi_dim}",
    flush=True,
)

# blackjax.laplace_dhmc(log_joint_fn, ...) — first positional IS log_joint_fn
# (NOT logdensity_fn). Use base_method.factory for the correct interface:
#   _factory(logdensity_fn_unused, log_joint_fn=..., theta_init=..., ...)
kernel = base_method.factory(
    marginal_logdensity_fn,  # positional slot 1: logdensity_fn (unused by laplace family)
    log_joint_fn=log_joint_fn,
    theta_init=theta_init,
    step_size=step_size,
    inverse_mass_matrix=inverse_mass_matrix,
)
# laplace_dhmc.init(position, rng_key) — rng_key seeds random_generator_arg
# for the quasi-random integration-step counter (pass_rng_key_to_init=True).
init_state_key, sample_key = jax.random.split(sample_key)
state = kernel.init(phi_init, init_state_key)

t_kernel = time.perf_counter()
print(f"[t=+{t_kernel - t0:.1f}s] Kernel + state init done", flush=True)

# ---------------------------------------------------------------------------
# JIT-compile + run 10 steps via lax.scan (single chain)
# ---------------------------------------------------------------------------


@jax.jit
def run_10_steps(rng_key: jax.Array, state: object) -> tuple:
    """Run 10 laplace_dhmc steps via lax.scan (JIT-compiled)."""

    def one_step(carry, key):
        s, info = kernel.step(key, carry)
        return s, info

    keys = jax.random.split(rng_key, 10)
    final_state, infos = jax.lax.scan(one_step, state, keys)
    return final_state, infos


print(
    f"[t=+{time.perf_counter() - t0:.1f}s] "
    "Invoking JIT: vmap(scan(laplace_dhmc_kernel, n=10))...",
    flush=True,
)
t_jit_start = time.perf_counter()

final_state, infos = run_10_steps(sample_key, state)
# Force all XLA evaluation to complete before taking the timestamp
_ = jax.block_until_ready(final_state.position)

t_done = time.perf_counter()
compile_and_run = t_done - t_jit_start
total = t_done - t0

print(f"[t=+{total:.1f}s] block_until_ready returned", flush=True)
print("", flush=True)
print("=== Exp 6 Result ===", flush=True)
print(f"  compile+run (10 steps): {compile_and_run:.1f}s", flush=True)
print(f"  total wall: {total:.1f}s", flush=True)
print(f"  final phi position: {final_state.position}", flush=True)
print(f"  acceptance_rate (last step): {infos.acceptance_rate[-1]:.4f}", flush=True)
print(
    f"  n_finite: {int(jnp.sum(jnp.isfinite(jnp.stack(list(final_state.position.values())))))}",
    flush=True,
)

if compile_and_run < 60:
    verdict = "FAST (<1 min)"
elif compile_and_run < 300:
    verdict = "MODERATE (1-5 min)"
elif compile_and_run < 600:
    verdict = "SLOW (5-10 min)"
else:
    verdict = "EXCEED_BUDGET (>10 min)"

print(f"  verdict: {verdict}", flush=True)
print("", flush=True)
if compile_and_run < 60:
    print("Exp 6 FAST: L-BFGS kernel body compiles quickly.", flush=True)
    print(
        "  -> exp7 (scan probe) will reveal if scan unrolling is the bottleneck.",
        flush=True,
    )
else:
    print(
        f"Exp 6 SLOW ({compile_and_run:.0f}s): kernel body is the compile bottleneck.",
        flush=True,
    )
    print(
        "  -> Fix direction: reduce L-BFGS max_iter or switch to dynamic while_loop.",
        flush=True,
    )
