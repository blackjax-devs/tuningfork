"""ExpA: minimal blackjax reproducer for laplace_dhmc vmap compile blowup.

Pure blackjax — no tuningfork runner, no numpyro model.  Synthetic Gaussian
hierarchical model: d_phi=3, d_theta=200.

Purpose: isolate which layer of the laplace_dhmc × vmap × run_inference_algorithm
stack causes the 83× compile-time blowup observed in exp7/8.

Decision table:
  V2 ≈ V1 AND V3 >> V1 → bug is in run_inference_algorithm + vmap interaction;
    shared kernel step vmaps fine; the per-chain kernel construction + RIA is slow.
  V2 >> V1 → bug is in vmap(laplace_dhmc_step) itself (e.g., vmap(while_loop) in
    L-BFGS linesearch); shared or per-chain kernel doesn't matter.

Three variants (n_samples=10, n_chains=4, same d_phi=3, d_theta=200):

  V1 (baseline):
    - Single chain, no vmap
    - jit(lax.scan(kernel.step, n=10))
    - Expected: ~6s (matches exp6)

  V2 (shared kernel, vmap over scan):
    - 4 chains, kernel built OUTSIDE vmap
    - jax.vmap(jit(lax.scan(kernel.step, n=10))) with per-chain rng+state
    - Tests: is vmap(laplace_step) itself expensive?
    - If V2 ~ V1: vmap(step) is cheap; issue is in RIA or per-chain kernel
    - If V2 >> V1: vmap(step) is expensive (likely vmap(while_loop) in linesearch)

  V3 (recipe runner pattern):
    - 4 chains, kernel built INSIDE vmap (exact recipe runner pattern)
    - jax.vmap(_run_one_chain) where _run_one_chain builds kernel + calls
      run_inference_algorithm, same as _recipe_runner.py
    - Tests: total recipe runner overhead vs V2

Hypothesis from code reading:
  The L-BFGS optimizer uses optax.scale_by_zoom_linesearch(max_linesearch_steps=1000)
  which internally uses lax.while_loop. Under vmap, jax transforms this to
  vmap(while_loop(...)) which is known to have complex XLA compilation behavior.
  The static outer scan (maxiter=30 LBFGS iterations) under vmap is likely fine,
  but the inner while_loop (linesearch, maxls=1000) under vmap may cause the blowup.

  Secondary hypothesis: jax.lax.custom_root in get_theta_star (for IFT gradients)
  has complex batching rules under vmap, adding to compile time.

Model:
  log_joint(theta, phi) = log p(theta | phi) + log p(phi) + log p(y | theta)
  where:
    phi ~ N(0, I_3)
    theta ~ N(0, exp(2*phi[0]) * I_200)   (scale driven by phi[0])
    y_obs = 0.0 (synthetic single observation)
    log p(y | theta) = -0.5 * theta[0]^2  (theta[0] observed at 0)

  The Laplace approximation is exact since p(theta | phi, y) is Gaussian.

Hard kill: 30 min per variant (1800s timeout).
"""

import os
import signal
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

sys.path.insert(0, "/home/jp/blackjax-devs/blackjax")

import blackjax  # noqa: E402
from blackjax.util import run_inference_algorithm  # noqa: E402

t_imports = time.perf_counter()
print(f"[t=+{t_imports - t0:.1f}s] All imports done", flush=True)

# ---------------------------------------------------------------------------
# Synthetic hierarchical model: d_phi=3, d_theta=200
# ---------------------------------------------------------------------------
D_PHI = 3
D_THETA = 200
N_SAMPLES = 10
N_CHAINS = 4
STEP_SIZE = 0.1
VARIANT_BUDGET = 1800.0  # 30 min hard kill per variant

_theta_init = jnp.zeros(D_THETA, dtype=jnp.float64)
_phi_init = jnp.zeros(D_PHI, dtype=jnp.float64)


def log_joint(theta, phi):
    """Hierarchical Gaussian model: theta ~ N(0, exp(2*phi[0]) * I), phi ~ N(0, I)."""
    log_sigma = phi[0]
    sigma2 = jnp.exp(2.0 * log_sigma)
    log_prior_phi = -0.5 * jnp.sum(phi**2)
    log_prior_theta = -0.5 * jnp.sum(theta**2) / sigma2 - D_THETA * log_sigma
    # Synthetic observation: y=0 observed through theta[0]
    log_lik = -0.5 * theta[0] ** 2
    return log_prior_phi + log_prior_theta + log_lik


print(
    f"[t=+{time.perf_counter() - t0:.1f}s] "
    f"Model: d_phi={D_PHI}, d_theta={D_THETA}, "
    f"n_samples={N_SAMPLES}, n_chains={N_CHAINS}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Build initial state for 1 chain (used by V1 and as template for V2/V3)
# ---------------------------------------------------------------------------
master_key = jax.random.key(42)
init_key, v1_key, v2_key, v3_key = jax.random.split(master_key, 4)

# Build a laplace_dhmc kernel (used by V1 and V2)
imm_1d = jnp.ones(D_PHI, dtype=jnp.float64)

t_kernel_build = time.perf_counter()
print(
    f"[t=+{time.perf_counter() - t0:.1f}s] "
    "Building laplace_dhmc kernel (step_size=0.1, identity IMM)...",
    flush=True,
)

kernel_shared = blackjax.laplace_dhmc(
    log_joint,
    theta_init=_theta_init,
    step_size=STEP_SIZE,
    inverse_mass_matrix=imm_1d,
)

print(
    f"[t=+{time.perf_counter() - t0:.1f}s] Kernel built in "
    f"{time.perf_counter() - t_kernel_build:.2f}s",
    flush=True,
)

# Initial state for single chain
state_1chain = kernel_shared.init(_phi_init, init_key)

print(
    f"[t=+{time.perf_counter() - t0:.1f}s] Initial state built. "
    f"theta_star shape: {state_1chain.theta_star.shape}",
    flush=True,
)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
class VariantTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise VariantTimeout


def time_variant(name, fn, budget=VARIANT_BUDGET):
    """Run fn(); return wall time (s) or None if timed out / errored."""
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(budget))
    t_start = time.perf_counter()
    try:
        fn()
        wall = time.perf_counter() - t_start
        signal.alarm(0)
        return wall
    except VariantTimeout:
        wall = time.perf_counter() - t_start
        print(
            f"  [TIMEOUT] {name} exceeded {budget:.0f}s budget " f"after {wall:.0f}s",
            flush=True,
        )
        return None
    except Exception as exc:
        wall = time.perf_counter() - t_start
        print(f"  [ERROR] {name} failed after {wall:.1f}s: {exc}", flush=True)
        signal.alarm(0)
        return None


# ---------------------------------------------------------------------------
# V1: bare jit+scan, 1 chain, no vmap
# ---------------------------------------------------------------------------
print(
    f"\n[t=+{time.perf_counter() - t0:.1f}s] "
    "=== V1: bare jit(lax.scan(kernel.step, n=10)), 1 chain, no vmap ===",
    flush=True,
)


@jax.jit
def _v1_run(rng_key, state):
    def one_step(s, k):
        new_s, info = kernel_shared.step(k, s)
        return new_s, info

    return jax.lax.scan(one_step, state, jax.random.split(rng_key, N_SAMPLES))


def _run_v1():
    final_state, infos = _v1_run(v1_key, state_1chain)
    _ = jax.block_until_ready(final_state.position)
    print(
        f"  V1 acceptance_rate[-1]: {float(infos.acceptance_rate[-1]):.4f}",
        flush=True,
    )


t_v1 = time_variant("V1", _run_v1)
if t_v1 is not None:
    print(
        f"[t=+{time.perf_counter() - t0:.1f}s] V1 done: {t_v1:.2f}s "
        f"({t_v1 / 6.0:.1f}× exp6 baseline)",
        flush=True,
    )
else:
    print(f"[t=+{time.perf_counter() - t0:.1f}s] V1 TIMEOUT/ERROR", flush=True)

# ---------------------------------------------------------------------------
# V2: vmap of jit+scan, shared kernel (built OUTSIDE vmap)
# ---------------------------------------------------------------------------
print(
    f"\n[t=+{time.perf_counter() - t0:.1f}s] "
    "=== V2: vmap(jit(lax.scan(kernel.step, n=10))), "
    f"{N_CHAINS} chains, SHARED kernel ===",
    flush=True,
)

# Build initial states for N_CHAINS chains (all starting from _phi_init)
v2_init_keys = jax.random.split(v2_key, N_CHAINS)
v2_states = jax.vmap(lambda k: kernel_shared.init(_phi_init, k))(v2_init_keys)


@jax.jit
def _v2_run_one_chain(rng_key, state):
    """Single-chain scan using the shared kernel (built outside vmap)."""

    def one_step(s, k):
        new_s, info = kernel_shared.step(k, s)
        return new_s, info

    return jax.lax.scan(one_step, state, jax.random.split(rng_key, N_SAMPLES))


# vmap over chain dimension
_v2_run_all_chains = jax.vmap(_v2_run_one_chain)


def _run_v2():
    chain_keys_v2 = jax.random.split(jax.random.key(100), N_CHAINS)
    final_states, infos = _v2_run_all_chains(chain_keys_v2, v2_states)
    _ = jax.block_until_ready(final_states.position)
    print(
        f"  V2 acceptance_rate[0,-1]: {float(infos.acceptance_rate[0, -1]):.4f}",
        flush=True,
    )


print(
    f"[t=+{time.perf_counter() - t0:.1f}s] "
    "Launching V2 (shared kernel, 4 chains via vmap). "
    f"Budget: {VARIANT_BUDGET:.0f}s.",
    flush=True,
)

t_v2 = time_variant("V2", _run_v2)
if t_v2 is not None:
    ratio_vs_v1 = (t_v2 / t_v1) if t_v1 else float("nan")
    print(
        f"[t=+{time.perf_counter() - t0:.1f}s] V2 done: {t_v2:.2f}s "
        f"({ratio_vs_v1:.1f}× V1)",
        flush=True,
    )
else:
    print(
        f"[t=+{time.perf_counter() - t0:.1f}s] V2 TIMEOUT: "
        "vmap(lax.scan(laplace_dhmc_step)) is expensive.",
        flush=True,
    )
    print(
        "  -> ROOT CAUSE CANDIDATE: vmap(laplace_dhmc_step) itself is slow. "
        "Likely vmap(while_loop) in L-BFGS linesearch.",
        flush=True,
    )

# ---------------------------------------------------------------------------
# V3: recipe runner exact pattern (kernel built INSIDE vmap, uses RIA)
# ---------------------------------------------------------------------------
print(
    f"\n[t=+{time.perf_counter() - t0:.1f}s] "
    "=== V3: recipe runner pattern — vmap(_run_one_chain), "
    f"{N_CHAINS} chains, kernel INSIDE vmap ===",
    flush=True,
)

# Per-chain step sizes and IMMs (slightly varied to mimic real adaptation output)
v3_step_sizes = jnp.full((N_CHAINS,), STEP_SIZE, dtype=jnp.float64)
v3_imms = jnp.tile(imm_1d, (N_CHAINS, 1))  # (N_CHAINS, D_PHI)
v3_init_positions = jnp.tile(_phi_init, (N_CHAINS, 1))  # (N_CHAINS, D_PHI)


def _run_one_chain_v3(rng_key, init_position, step_size, imm):
    """Exact recipe runner pattern: build kernel INSIDE vmap, use run_inference_algorithm."""
    kernel = blackjax.laplace_dhmc(
        log_joint,
        theta_init=_theta_init,
        step_size=step_size,
        inverse_mass_matrix=imm,
    )
    reinit_key, run_key = jax.random.split(rng_key)
    init_state = kernel.init(init_position, reinit_key)
    _, (states, infos) = run_inference_algorithm(
        rng_key=run_key,
        inference_algorithm=kernel,
        num_steps=N_SAMPLES,
        initial_state=init_state,
    )
    return states, infos


_v3_run_all_chains = jax.vmap(_run_one_chain_v3)


def _run_v3():
    chain_keys_v3 = jax.random.split(jax.random.key(200), N_CHAINS)
    final_states, infos = _v3_run_all_chains(
        chain_keys_v3, v3_init_positions, v3_step_sizes, v3_imms
    )
    _ = jax.block_until_ready(final_states[0].position)
    print(
        f"  V3 acceptance_rate[0,-1]: {float(infos.acceptance_rate[0, -1]):.4f}",
        flush=True,
    )


print(
    f"[t=+{time.perf_counter() - t0:.1f}s] "
    "Launching V3 (recipe runner pattern). "
    f"Budget: {VARIANT_BUDGET:.0f}s.",
    flush=True,
)

t_v3 = time_variant("V3", _run_v3)
if t_v3 is not None:
    ratio_vs_v1 = (t_v3 / t_v1) if t_v1 else float("nan")
    ratio_vs_v2 = (t_v3 / t_v2) if t_v2 else float("nan")
    print(
        f"[t=+{time.perf_counter() - t0:.1f}s] V3 done: {t_v3:.2f}s "
        f"({ratio_vs_v1:.1f}× V1, {ratio_vs_v2:.1f}× V2)",
        flush=True,
    )
else:
    print(
        f"[t=+{time.perf_counter() - t0:.1f}s] V3 TIMEOUT: "
        "recipe runner pattern is expensive.",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Summary + diagnosis
# ---------------------------------------------------------------------------
total_wall = time.perf_counter() - t0
print(f"\n[t=+{total_wall:.1f}s] === ExpA Summary ===", flush=True)
print(
    f"  V1 (bare scan, 1 chain):           {f'{t_v1:.1f}s' if t_v1 else 'TIMEOUT'}",
    flush=True,
)
print(
    f"  V2 (vmap+scan, shared kernel, 4ch):{f' {t_v2:.1f}s' if t_v2 else ' TIMEOUT'}",
    flush=True,
)
print(
    f"  V3 (vmap+RIA, per-chain kernel):   {f'{t_v3:.1f}s' if t_v3 else 'TIMEOUT'}",
    flush=True,
)
print("", flush=True)

if t_v1 and t_v2 and t_v3:
    if t_v2 < 5 * t_v1 and t_v3 > 50 * t_v1:
        print(
            "DIAGNOSIS: V2≈V1, V3>>V1 → "
            "vmap(laplace_step) is fine; bug is in "
            "run_inference_algorithm + kernel-inside-vmap interaction.",
            flush=True,
        )
        print(
            "SUSPECT: run_inference_algorithm's scan body under vmap, "
            "or the kernel factory call inside vmap.",
            flush=True,
        )
    elif t_v2 > 50 * t_v1:
        print(
            "DIAGNOSIS: V2>>V1 → vmap(laplace_step) itself is the bottleneck.",
            flush=True,
        )
        print(
            "SUSPECT: vmap(while_loop) in L-BFGS linesearch "
            "(optax.scale_by_zoom_linesearch maxls=1000). "
            "Or vmap(custom_root) in IFT gradient.",
            flush=True,
        )
    else:
        print(
            "DIAGNOSIS: unclear. Check ratios manually.",
            flush=True,
        )
elif t_v2 is None:
    print(
        "DIAGNOSIS: V2 TIMED OUT → vmap(laplace_dhmc_step) itself is expensive "
        "(independent of run_inference_algorithm or per-chain kernel construction).",
        flush=True,
    )
    print(
        "PRIME SUSPECT: vmap(while_loop(linesearch, maxls=1000)) inside L-BFGS. "
        "Confirm with py-spy: look for XLA while_loop / xla::WhileThunk frames.",
        flush=True,
    )
elif t_v3 is None and t_v2 is not None:
    print(
        f"DIAGNOSIS: V2 completed ({t_v2:.1f}s), V3 TIMED OUT → "
        "the overhead is in run_inference_algorithm + kernel-inside-vmap.",
        flush=True,
    )

print("\nDone.", flush=True)
