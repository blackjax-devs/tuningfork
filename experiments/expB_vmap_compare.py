"""ExpB: vmap compile blowup comparison on real gp_regression.

Three kernels × three variants on the real GP 3D phi-marginal.

Purpose: isolate whether vmap(laplace_dhmc_step) causes compile blowup on the
real GP model (d_phi=3, d_theta=200), compared to a fixed-L HMC variant and a
NUTS control on the phi-only marginal.

Three kernels:
  laplace_dhmc   — dynamic HMC on Laplace-approx marginal (dynamic step count)
  laplace_hmc    — fixed-L HMC on Laplace-approx marginal (L=10 steps)
  nuts_control   — NUTS on the phi-only marginal logdensity (same inner laplace
                   call, but NUTS leapfrog structure)

Three variants (same as expA):
  V1 — 1 chain, no vmap; jit(lax.scan(kernel.step, n=N_SAMPLES))
  V2 — N_CHAINS chains, shared kernel built OUTSIDE vmap; vmap(jit(scan))
  V3 — N_CHAINS chains, per-chain kernel built INSIDE vmap (recipe runner pattern)

py-spy snapshot at T+15s during V3 compile for each kernel — written to
experiments/expB_pyspy_v3_<kernel>.txt.

Decision matrix (same as expA):
  V2 ≈ V1, V3 >> V1  → bug is in run_inference_algorithm + kernel-inside-vmap
  V2 >> V1            → vmap(laplace_dhmc_step) itself is expensive
    Sub-question: does nuts_control show same pattern? If nuts blowup >> nuts V1
    but ~proportional to laplace_dhmc V2 blowup → shared root cause is L-BFGS
    while_loop under vmap. If nuts_control V2 ≈ V1 → problem is unique to the
    laplace kernel's own while_loop structure.
"""

import os
import signal
import subprocess
import sys
import threading
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

import blackjax  # noqa: E402
from blackjax.util import run_inference_algorithm  # noqa: E402

from tuningfork.model import MODELS  # noqa: E402
from tuningfork.model._numpyro import build_logdensity_fn  # noqa: E402
from tuningfork.recipes._recipe_runner import _build_laplace_components  # noqa: E402

t_imports = time.perf_counter()
print(f"[t=+{t_imports - t0:.1f}s] All imports done", flush=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SEED = 20260517
N_SAMPLES = 5
N_CHAINS = 4
STEP_SIZE = 0.1
L_HMC = 10  # fixed integration steps for laplace_hmc
PYSPY_DELAY = 15.0  # seconds into V3 to capture py-spy snapshot
V1_BUDGET = 120.0
V2_BUDGET = 600.0
V3_BUDGET = 900.0
OUTPUT_DIR = "experiments"
PYSPY_DIR = "experiments/expB_pyspy"

os.makedirs(PYSPY_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Build gp_regression model components
# ---------------------------------------------------------------------------
model = MODELS["gp_regression"]
init_key, warmup_key = jax.random.split(jax.random.key(SEED), 2)

full_position, raw_joint_fn, _postprocess = build_logdensity_fn(init_key, model)

laplace_result = _build_laplace_components("gp_regression", full_position, raw_joint_fn)
assert laplace_result is not None, "gp_regression not in _LAPLACE_PHI_THETA_SPLITS"
phi_init, log_joint_fn, theta_init, marginal_logdensity_fn = laplace_result

d_phi = sum(jnp.asarray(v).size for v in phi_init.values())
d_theta = sum(jnp.asarray(v).size for v in theta_init.values())
imm_phi = jnp.ones(d_phi, dtype=jnp.float64)

print(
    f"[t=+{time.perf_counter() - t0:.1f}s] "
    f"d_phi={d_phi}, d_theta={d_theta}, N_CHAINS={N_CHAINS}, N_SAMPLES={N_SAMPLES}",
    flush=True,
)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
class VariantTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise VariantTimeout


_VARIANT_TIMEOUT = "TIMEOUT"
_VARIANT_ERROR = "ERROR"


def time_variant(name, fn, budget):
    """Run fn(); return wall time (s), _VARIANT_TIMEOUT, or _VARIANT_ERROR."""
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
            f"  [TIMEOUT] {name} exceeded {budget:.0f}s budget after {wall:.0f}s",
            flush=True,
        )
        return _VARIANT_TIMEOUT
    except Exception as exc:
        wall = time.perf_counter() - t_start
        print(f"  [ERROR] {name} failed after {wall:.1f}s: {exc}", flush=True)
        signal.alarm(0)
        return _VARIANT_ERROR


# ---------------------------------------------------------------------------
# py-spy snapshot helper
# ---------------------------------------------------------------------------
def launch_pyspy_snapshot(out_path, delay_s=PYSPY_DELAY):
    """Start a daemon thread that captures py-spy dump at delay_s seconds."""
    pid = os.getpid()

    def _capture():
        time.sleep(delay_s)
        try:
            r = subprocess.run(
                ["py-spy", "dump", "--pid", str(pid), "--nonblocking"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            with open(out_path, "w") as f:
                f.write(r.stdout)
                if r.stderr:
                    f.write("\n--- stderr ---\n" + r.stderr)
            print(
                f"  [py-spy] snapshot written to {out_path} "
                f"({len(r.stdout)} chars)",
                flush=True,
            )
        except FileNotFoundError:
            print(
                f"  [py-spy] not found — no snapshot at {out_path}",
                flush=True,
            )
        except Exception as e:
            print(f"  [py-spy] error: {e}", flush=True)

    t = threading.Thread(target=_capture, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Per-kernel V1 / V2 / V3 runs
# ---------------------------------------------------------------------------

_fmt = lambda t: f"{t:.2f}s" if isinstance(t, float) else str(t)  # noqa: E731

kernel_names = ["laplace_dhmc", "laplace_hmc", "nuts_control"]
results = {}

for kernel_name in kernel_names:
    print(
        f"\n[t=+{time.perf_counter() - t0:.1f}s] "
        f"{'=' * 60}\n  Kernel: {kernel_name}\n{'=' * 60}",
        flush=True,
    )

    # -------------------------------------------------------------------
    # Build shared kernel (used by V1 and V2)
    # -------------------------------------------------------------------
    if kernel_name == "laplace_dhmc":
        kernel_shared = blackjax.laplace_dhmc(
            log_joint_fn,
            theta_init=theta_init,
            step_size=STEP_SIZE,
            inverse_mass_matrix=imm_phi,
        )
    elif kernel_name == "laplace_hmc":
        kernel_shared = blackjax.laplace_hmc(
            log_joint_fn,
            theta_init=theta_init,
            step_size=STEP_SIZE,
            inverse_mass_matrix=imm_phi,
            num_integration_steps=L_HMC,
        )
    else:  # nuts_control
        kernel_shared = blackjax.nuts(
            marginal_logdensity_fn,
            step_size=STEP_SIZE,
            inverse_mass_matrix=imm_phi,
        )

    print(f"[t=+{time.perf_counter() - t0:.1f}s] Kernel built", flush=True)

    # -------------------------------------------------------------------
    # Build initial states for all chains
    # -------------------------------------------------------------------
    rng_keys_v1, rng_keys_v2, rng_keys_v3 = (
        jax.random.split(jax.random.key(SEED + i), N_CHAINS) for i in range(3)
    )
    rng_key_v1 = rng_keys_v1[0]

    t_init = time.perf_counter()
    if kernel_name == "laplace_dhmc":
        # init takes (position, rng_key)
        state_1chain = kernel_shared.init(phi_init, rng_key_v1)
        v2_states = jax.vmap(lambda k: kernel_shared.init(phi_init, k))(rng_keys_v2)
    else:
        # laplace_hmc and nuts init take (position) only
        state_1chain = kernel_shared.init(phi_init)
        v2_states = jax.vmap(lambda _: kernel_shared.init(phi_init))(
            jnp.arange(N_CHAINS)
        )
    _ = jax.block_until_ready(jax.tree.map(jnp.asarray, state_1chain))
    print(
        f"[t=+{time.perf_counter() - t0:.1f}s] "
        f"Initial states built in {time.perf_counter() - t_init:.1f}s",
        flush=True,
    )

    # -------------------------------------------------------------------
    # V1: bare jit+scan, 1 chain, no vmap
    # -------------------------------------------------------------------
    print(
        f"\n[t=+{time.perf_counter() - t0:.1f}s] "
        f"  [{kernel_name}] V1: jit(lax.scan(step, n={N_SAMPLES})), 1 chain",
        flush=True,
    )

    @jax.jit
    def _v1_scan(rng_key, state):
        def one_step(s, k):
            return kernel_shared.step(k, s)

        return jax.lax.scan(one_step, state, jax.random.split(rng_key, N_SAMPLES))

    def _run_v1():
        final_state, infos = _v1_scan(rng_key_v1, state_1chain)
        _ = jax.block_until_ready(final_state.position)
        if hasattr(infos, "acceptance_rate"):
            print(
                f"    V1 acceptance_rate[-1]: "
                f"{float(infos.acceptance_rate[-1]):.4f}",
                flush=True,
            )

    t_v1 = time_variant(f"{kernel_name}/V1", _run_v1, V1_BUDGET)
    print(
        f"[t=+{time.perf_counter() - t0:.1f}s] " f"  V1: {_fmt(t_v1)}",
        flush=True,
    )

    # -------------------------------------------------------------------
    # V2: vmap(jit(scan)), shared kernel, N_CHAINS chains
    # -------------------------------------------------------------------
    print(
        f"\n[t=+{time.perf_counter() - t0:.1f}s] "
        f"  [{kernel_name}] V2: vmap(jit(scan)), shared kernel, {N_CHAINS} chains",
        flush=True,
    )

    @jax.jit
    def _v2_one_chain(rng_key, state):
        def one_step(s, k):
            return kernel_shared.step(k, s)

        return jax.lax.scan(one_step, state, jax.random.split(rng_key, N_SAMPLES))

    _v2_all_chains = jax.vmap(_v2_one_chain)

    def _run_v2():
        chain_keys = jax.random.split(jax.random.key(SEED + 100), N_CHAINS)
        final_states, infos = _v2_all_chains(chain_keys, v2_states)
        _ = jax.block_until_ready(final_states.position)
        if hasattr(infos, "acceptance_rate"):
            print(
                f"    V2 acceptance_rate[0,-1]: "
                f"{float(infos.acceptance_rate[0, -1]):.4f}",
                flush=True,
            )

    t_v2 = time_variant(f"{kernel_name}/V2", _run_v2, V2_BUDGET)
    ratio_v2 = (
        (t_v2 / t_v1) if isinstance(t_v2, float) and isinstance(t_v1, float) else "N/A"
    )
    ratio_str = f"{ratio_v2:.1f}× V1" if isinstance(ratio_v2, float) else ratio_v2
    print(
        f"[t=+{time.perf_counter() - t0:.1f}s] " f"  V2: {_fmt(t_v2)} ({ratio_str})",
        flush=True,
    )

    # -------------------------------------------------------------------
    # V3: per-chain kernel inside vmap (recipe runner pattern)
    # -------------------------------------------------------------------
    print(
        f"\n[t=+{time.perf_counter() - t0:.1f}s] "
        f"  [{kernel_name}] V3: vmap(per-chain kernel + RIA), {N_CHAINS} chains",
        flush=True,
    )

    if kernel_name == "laplace_dhmc":

        def _run_one_chain_v3(rng_key, init_position, step_size, imm):
            k = blackjax.laplace_dhmc(
                log_joint_fn,
                theta_init=theta_init,
                step_size=step_size,
                inverse_mass_matrix=imm,
            )
            reinit_key, run_key = jax.random.split(rng_key)
            init_state = k.init(init_position, reinit_key)
            _, (states, infos) = run_inference_algorithm(
                rng_key=run_key,
                inference_algorithm=k,
                num_steps=N_SAMPLES,
                initial_state=init_state,
            )
            return states, infos

    elif kernel_name == "laplace_hmc":

        def _run_one_chain_v3(rng_key, init_position, step_size, imm):
            k = blackjax.laplace_hmc(
                log_joint_fn,
                theta_init=theta_init,
                step_size=step_size,
                inverse_mass_matrix=imm,
                num_integration_steps=L_HMC,
            )
            _, run_key = jax.random.split(rng_key)
            init_state = k.init(init_position)
            _, (states, infos) = run_inference_algorithm(
                rng_key=run_key,
                inference_algorithm=k,
                num_steps=N_SAMPLES,
                initial_state=init_state,
            )
            return states, infos

    else:  # nuts_control

        def _run_one_chain_v3(rng_key, init_position, step_size, imm):
            k = blackjax.nuts(
                marginal_logdensity_fn,
                step_size=step_size,
                inverse_mass_matrix=imm,
            )
            _, run_key = jax.random.split(rng_key)
            init_state = k.init(init_position)
            _, (states, infos) = run_inference_algorithm(
                rng_key=run_key,
                inference_algorithm=k,
                num_steps=N_SAMPLES,
                initial_state=init_state,
            )
            return states, infos

    _v3_all_chains = jax.vmap(_run_one_chain_v3)
    v3_step_sizes = jnp.full((N_CHAINS,), STEP_SIZE, dtype=jnp.float64)
    v3_imms = jnp.tile(imm_phi, (N_CHAINS, 1))
    phi_init_flat, unravel = jax.flatten_util.ravel_pytree(phi_init)
    v3_init_positions = jnp.tile(phi_init_flat, (N_CHAINS, 1))

    # Adapt _run_one_chain_v3 to receive flat phi
    if kernel_name == "laplace_dhmc":

        def _run_one_chain_v3_flat(rng_key, init_position_flat, step_size, imm):
            return _run_one_chain_v3(
                rng_key, unravel(init_position_flat), step_size, imm
            )

    else:

        def _run_one_chain_v3_flat(rng_key, init_position_flat, step_size, imm):
            return _run_one_chain_v3(
                rng_key, unravel(init_position_flat), step_size, imm
            )

    _v3_all_chains = jax.vmap(_run_one_chain_v3_flat)

    # Launch py-spy snapshot BEFORE starting V3 (will fire at T+15s into V3)
    pyspy_out = f"{PYSPY_DIR}/v3_{kernel_name}.txt"
    _pyspy_thread = launch_pyspy_snapshot(pyspy_out, delay_s=PYSPY_DELAY)

    def _run_v3():
        chain_keys = jax.random.split(jax.random.key(SEED + 200), N_CHAINS)
        final_states, infos = _v3_all_chains(
            chain_keys, v3_init_positions, v3_step_sizes, v3_imms
        )
        _ = jax.block_until_ready(final_states.position)
        if hasattr(infos, "acceptance_rate"):
            print(
                f"    V3 acceptance_rate[0,-1]: "
                f"{float(infos.acceptance_rate[0, -1]):.4f}",
                flush=True,
            )

    t_v3 = time_variant(f"{kernel_name}/V3", _run_v3, V3_BUDGET)
    ratio_v3_v1 = (
        (t_v3 / t_v1) if isinstance(t_v3, float) and isinstance(t_v1, float) else "N/A"
    )
    ratio_v3_v2 = (
        (t_v3 / t_v2) if isinstance(t_v3, float) and isinstance(t_v2, float) else "N/A"
    )
    ratio_v3_v1_str = (
        f"{ratio_v3_v1:.1f}× V1" if isinstance(ratio_v3_v1, float) else ratio_v3_v1
    )
    ratio_v3_v2_str = (
        f"{ratio_v3_v2:.1f}× V2" if isinstance(ratio_v3_v2, float) else ratio_v3_v2
    )
    print(
        f"[t=+{time.perf_counter() - t0:.1f}s] "
        f"  V3: {_fmt(t_v3)} ({ratio_v3_v1_str}, {ratio_v3_v2_str})",
        flush=True,
    )

    results[kernel_name] = {"v1": t_v1, "v2": t_v2, "v3": t_v3}

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
total_wall = time.perf_counter() - t0
print(f"\n[t=+{total_wall:.1f}s] === ExpB Summary ===", flush=True)
print(
    f"  {'Kernel':>15}  {'V1':>10}  {'V2':>10}  {'V3':>10}  "
    f"{'V2/V1':>8}  {'V3/V1':>8}",
    flush=True,
)
for kname in kernel_names:
    r = results.get(kname, {})
    v1, v2, v3 = r.get("v1", "N/A"), r.get("v2", "N/A"), r.get("v3", "N/A")
    r21 = (
        f"{v2 / v1:.1f}×" if isinstance(v2, float) and isinstance(v1, float) else "N/A"
    )
    r31 = (
        f"{v3 / v1:.1f}×" if isinstance(v3, float) and isinstance(v1, float) else "N/A"
    )
    v1s = f"{v1:.1f}s" if isinstance(v1, float) else str(v1)
    v2s = f"{v2:.1f}s" if isinstance(v2, float) else str(v2)
    v3s = f"{v3:.1f}s" if isinstance(v3, float) else str(v3)
    print(
        f"  {kname:>15}  {v1s:>10}  {v2s:>10}  {v3s:>10}  {r21:>8}  {r31:>8}",
        flush=True,
    )

# Diagnosis
print("\n=== Diagnosis ===", flush=True)
dhmc = results.get("laplace_dhmc", {})
hmc = results.get("laplace_hmc", {})
nuts = results.get("nuts_control", {})

v1_d, v2_d, v3_d = dhmc.get("v1"), dhmc.get("v2"), dhmc.get("v3")
v1_h, v2_h = hmc.get("v1"), hmc.get("v2")
v1_n, v2_n = nuts.get("v1"), nuts.get("v2")

if v2_d == _VARIANT_TIMEOUT:
    print(
        "laplace_dhmc V2 TIMED OUT → vmap(laplace_dhmc_step) itself is expensive.",
        flush=True,
    )
    if v2_n == _VARIANT_TIMEOUT:
        print(
            "  nuts_control V2 also timed out → root cause is vmap(while_loop) "
            "inside L-BFGS (shared by all kernels via marginal_logdensity_fn).",
            flush=True,
        )
    elif isinstance(v2_n, float) and isinstance(v1_n, float) and v2_n < 5 * v1_n:
        print(
            "  nuts_control V2 ≈ V1 → vmap(NUTS_trajectory) is fine; "
            "blowup is specific to vmap(laplace_dhmc) — likely vmap(while_loop) "
            "inside L-BFGS run as part of the dynamic HMC trajectory builder.",
            flush=True,
        )
elif isinstance(v2_d, float) and isinstance(v1_d, float) and v2_d > 20 * v1_d:
    print(
        f"laplace_dhmc V2 >> V1 ({v2_d / v1_d:.0f}×) → "
        "vmap(laplace_dhmc_step) is expensive even with shared kernel.",
        flush=True,
    )
elif isinstance(v2_d, float) and isinstance(v1_d, float):
    ratio = v2_d / v1_d
    print(
        f"laplace_dhmc V2/V1 = {ratio:.1f}× — {'linear (expected)' if ratio < 6 else 'super-linear (unexpected)'}.",
        flush=True,
    )

if v3_d == _VARIANT_TIMEOUT and isinstance(v2_d, float):
    print(
        "laplace_dhmc V3 TIMED OUT but V2 completed → "
        "extra overhead is in run_inference_algorithm + kernel-inside-vmap.",
        flush=True,
    )

print(f"\npy-spy snapshots written to: {PYSPY_DIR}/", flush=True)
print("Done.", flush=True)
