"""ExpC: per-L-BFGS-call linesearch step distribution for gp_regression.

Purpose: answer the TL's questions:
  (a) How often does the zoom linesearch hit the maxls cap?
  (b) What is the convergence-step distribution under different maxls values?
  (c) Are warmup and sampling phases different in their effective linesearch demand?

Method:
  - Monkey-patch optax.scale_by_zoom_linesearch with an instrumented wrapper
    that uses jax.debug.callback to record per-outer-iteration linesearch counts.
  - Run two phases:
      Phase 1 (standalone): 20 direct laplace(phi) calls at random phi values
        → gives per-call linesearch distribution without NUTS dynamics noise
      Phase 2 (warmup sample): NUTS window_adaptation, n_warmup=5, 1 chain, no vmap
        → gives distribution during actual warmup phase
  - Repeat for maxls in {1000, 200, 50, 20, 10} with FIXED SEED across all values.
  - Save structured data to experiments/expC_logs/ as npz.

Instrumentation approach: option (ii) from TL — jax.debug.callback side-channel.
  Performance cost: ~1 callback per outer L-BFGS iteration. For 20 phi calls × 30
  max outer iterations = ≤600 callbacks per maxls value. Acceptable for diagnostic.
  For n_warmup=5 warmup phase: ~5 steps × 2^max_doublings leapfrog × 30 outer
  iterations = O(50000) callbacks max at max tree depth. We use n_warmup=5 to limit.

Does not modify blackjax source.
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

# ---------------------------------------------------------------------------
# Monkey-patch optax.scale_by_zoom_linesearch BEFORE any blackjax import
# ---------------------------------------------------------------------------
import optax  # noqa: E402

_orig_scale_by_zoom_linesearch = optax.scale_by_zoom_linesearch

# Python-side accumulators for jax.debug.callback data
_ls_counts_buf: list[int] = []
_ls_dec_err_buf: list[float] = []
_ls_curv_err_buf: list[float] = []


def _record_ls_step(n_steps: int, dec_err: float, curv_err: float) -> None:
    _ls_counts_buf.append(int(n_steps))
    _ls_dec_err_buf.append(float(dec_err))
    _ls_curv_err_buf.append(float(curv_err))


def _patched_scale_by_zoom_linesearch(
    max_linesearch_steps, **kwargs
) -> optax.GradientTransformationExtraArgs:
    """Wrapper that records per-outer-iteration linesearch step counts via callback."""
    base_ls = _orig_scale_by_zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps, **kwargs
    )

    def patched_update(updates, state, params, **extra):
        new_updates, new_state = base_ls.update(updates, state, params, **extra)
        jax.debug.callback(
            _record_ls_step,
            new_state.info.num_linesearch_steps,
            new_state.info.decrease_error,
            new_state.info.curvature_error,
        )
        return new_updates, new_state

    return optax.GradientTransformationExtraArgs(
        init=base_ls.init,
        update=patched_update,
    )


# Patch at optax module level (blackjax accesses via optax.scale_by_zoom_linesearch)
optax.scale_by_zoom_linesearch = _patched_scale_by_zoom_linesearch

# ---------------------------------------------------------------------------
# Now import blackjax (will pick up the patched linesearch)
# ---------------------------------------------------------------------------
import blackjax  # noqa: E402
from blackjax.util import run_inference_algorithm  # noqa: E402

from tuningfork.model import MODELS  # noqa: E402
from tuningfork.model._numpyro import build_logdensity_fn  # noqa: E402
from tuningfork.recipes._recipe_runner import _build_laplace_components  # noqa: E402

t_imports = time.perf_counter()
print(f"[t=+{t_imports - t0:.1f}s] All imports done (optax patched)", flush=True)

# ---------------------------------------------------------------------------
# Build gp_regression model components (fixed seed, same across all maxls)
# ---------------------------------------------------------------------------
SEED = 20260517  # FIXED for all maxls — no confound
N_PHI_PROBES = 20  # standalone phase: 20 direct laplace(phi) calls
N_WARMUP_SAMPLE = 5  # warmup phase: 5 warmup steps (limits callback volume)
N_SAMPLING_STEPS = 5  # sampling phase: 5 steps
MAXLS_VALUES = [1000, 200, 50, 20, 10]
PROBE_NOISE_SCALE = 0.1  # std of Gaussian noise for phi probe perturbations
OUTPUT_DIR = "experiments/expC_logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = MODELS["gp_regression"]
init_key, probe_key, warmup_key, sample_key = jax.random.split(jax.random.key(SEED), 4)

full_position, raw_joint_fn, _postprocess = build_logdensity_fn(init_key, model)

laplace_result = _build_laplace_components("gp_regression", full_position, raw_joint_fn)
assert laplace_result is not None, "gp_regression not in _LAPLACE_PHI_THETA_SPLITS"
phi_init, log_joint_fn, theta_init, _ = laplace_result

d_phi = sum(jnp.asarray(v).size for v in phi_init.values())
d_theta = sum(jnp.asarray(v).size for v in theta_init.values())
print(
    f"[t=+{time.perf_counter() - t0:.1f}s] d_phi={d_phi}, d_theta={d_theta}",
    flush=True,
)

# Build probe phi values: phi_init + small Gaussian noise for diversity
phi_flat, unravel_phi = jax.flatten_util.ravel_pytree(phi_init)
probe_keys = jax.random.split(probe_key, N_PHI_PROBES)
probe_phi_flat = jax.vmap(
    lambda k: phi_flat + PROBE_NOISE_SCALE * jax.random.normal(k, phi_flat.shape)
)(probe_keys)
probe_phis = [unravel_phi(probe_phi_flat[i]) for i in range(N_PHI_PROBES)]

print(
    f"[t=+{time.perf_counter() - t0:.1f}s] "
    f"Built {N_PHI_PROBES} probe phi values (±{PROBE_NOISE_SCALE} noise)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Per-maxls sweep
# ---------------------------------------------------------------------------

all_results = {}

for maxls in MAXLS_VALUES:
    print(
        f"\n[t=+{time.perf_counter() - t0:.1f}s] "
        f"=== maxls={maxls} — Phase 1: standalone probe ({N_PHI_PROBES} calls) ===",
        flush=True,
    )

    # Clear callback buffers
    _ls_counts_buf.clear()
    _ls_dec_err_buf.clear()
    _ls_curv_err_buf.clear()

    # Build laplace with this maxls (patched linesearch picks up automatically)
    from blackjax.mcmc.laplace_marginal import laplace_marginal_factory  # noqa: E402

    laplace = laplace_marginal_factory(log_joint_fn, theta_init, maxls=maxls)

    # Phase 1: standalone probe — direct laplace(phi) calls (JIT-compiled)
    @jax.jit
    def single_laplace_call(phi_flat):
        phi = unravel_phi(phi_flat)
        lp, _ = laplace(phi)
        return lp

    t_phase1_start = time.perf_counter()
    for i, phi in enumerate(probe_phis):
        phi_flat_i, _ = jax.flatten_util.ravel_pytree(phi)
        lp_val = float(single_laplace_call(phi_flat_i))
        if i == 0:
            # First call triggers JIT compile — time separately
            t_first = time.perf_counter() - t_phase1_start
            print(f"  First call (JIT): {t_first:.1f}s, lp={lp_val:.3f}", flush=True)

    # Force all callbacks to flush (JAX async)
    jax.effects_barrier()
    t_phase1_wall = time.perf_counter() - t_phase1_start

    phase1_counts = np.array(_ls_counts_buf[:])
    phase1_dec_errors = np.array(_ls_dec_err_buf[:])
    phase1_curv_errors = np.array(_ls_curv_err_buf[:])

    # Stats
    n_calls = len(phase1_counts)
    cap_hits = int(np.sum(phase1_counts >= maxls)) if len(phase1_counts) > 0 else 0

    # Outer L-BFGS iterations per phi call (callback fires once per outer iter)
    # n_calls = N_PHI_PROBES × mean_outer_iters
    outer_iters_per_call = n_calls / N_PHI_PROBES if N_PHI_PROBES > 0 else 0

    print(
        f"  Phase1: {n_calls} outer-L-BFGS callbacks, "
        f"mean outer iters/call={outer_iters_per_call:.1f}, "
        f"wall={t_phase1_wall:.1f}s",
        flush=True,
    )
    if len(phase1_counts) > 0:
        print(
            f"  LS steps: mean={np.mean(phase1_counts):.1f}, "
            f"median={np.median(phase1_counts):.0f}, "
            f"max={np.max(phase1_counts)}, "
            f"p95={np.percentile(phase1_counts, 95):.0f}, "
            f"cap_hits={cap_hits}/{n_calls} "
            f"({100 * cap_hits / n_calls:.0f}%)",
            flush=True,
        )
        print(
            f"  Convergence: dec_err>0 (fail): {int(np.sum(phase1_dec_errors > 0))}/{n_calls} "
            f"({100 * np.mean(phase1_dec_errors > 0):.0f}%)",
            flush=True,
        )

    # Phase 2: warmup — NUTS window_adaptation, n_warmup=5
    print(
        f"\n[t=+{time.perf_counter() - t0:.1f}s] "
        f"  maxls={maxls} — Phase 2: warmup sample (n_warmup={N_WARMUP_SAMPLE}) ===",
        flush=True,
    )

    _ls_counts_buf.clear()
    _ls_dec_err_buf.clear()
    _ls_curv_err_buf.clear()

    def marginal_logdensity_fn(phi):
        lp, _ = laplace(phi)
        return lp

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        marginal_logdensity_fn,
        progress_bar=False,
    )

    wkey = jax.random.key(SEED + maxls * 7919)  # deterministic but different per maxls

    t_warmup_start = time.perf_counter()
    (warmup_state, adapted_params), _warmup_info = warmup.run(
        wkey, phi_init, N_WARMUP_SAMPLE
    )
    step_size = float(jnp.asarray(adapted_params["step_size"]))
    jax.effects_barrier()  # flush all debug callbacks
    t_warmup_wall = time.perf_counter() - t_warmup_start

    phase2_counts = np.array(_ls_counts_buf[:])
    phase2_dec_errors = np.array(_ls_dec_err_buf[:])
    cap_hits2 = int(np.sum(phase2_counts >= maxls)) if len(phase2_counts) > 0 else 0

    print(
        f"  Phase2 warmup: {len(phase2_counts)} outer-L-BFGS callbacks, "
        f"wall={t_warmup_wall:.1f}s, step_size={step_size:.4g}",
        flush=True,
    )
    if len(phase2_counts) > 0:
        print(
            f"  LS steps: mean={np.mean(phase2_counts):.1f}, "
            f"median={np.median(phase2_counts):.0f}, "
            f"max={np.max(phase2_counts)}, "
            f"p95={np.percentile(phase2_counts, 95):.0f}, "
            f"cap_hits={cap_hits2}/{len(phase2_counts)} "
            f"({100 * cap_hits2 / len(phase2_counts):.0f}%)",
            flush=True,
        )

    # Phase 3: sampling — laplace_dhmc, n_sampling=5
    print(
        f"  maxls={maxls} — Phase 3: sampling ({N_SAMPLING_STEPS} steps) ===",
        flush=True,
    )

    _ls_counts_buf.clear()
    _ls_dec_err_buf.clear()
    _ls_curv_err_buf.clear()

    imm = jnp.asarray(adapted_params["inverse_mass_matrix"])
    kernel = blackjax.laplace_dhmc(
        log_joint_fn,
        theta_init=theta_init,
        step_size=step_size,
        inverse_mass_matrix=imm,
        maxls=maxls,
    )
    init_state = kernel.init(phi_init, jax.random.key(SEED + 1))
    t_sample_start = time.perf_counter()
    final_state, _ = run_inference_algorithm(
        rng_key=jax.random.key(SEED + 2),
        inference_algorithm=kernel,
        num_steps=N_SAMPLING_STEPS,
        initial_state=init_state,
    )
    _ = jax.block_until_ready(final_state.position)
    jax.effects_barrier()
    t_sample_wall = time.perf_counter() - t_sample_start

    phase3_counts = np.array(_ls_counts_buf[:])
    cap_hits3 = int(np.sum(phase3_counts >= maxls)) if len(phase3_counts) > 0 else 0

    print(
        f"  Phase3 sampling: {len(phase3_counts)} callbacks, "
        f"wall={t_sample_wall:.1f}s",
        flush=True,
    )
    if len(phase3_counts) > 0:
        print(
            f"  LS steps: mean={np.mean(phase3_counts):.1f}, "
            f"median={np.median(phase3_counts):.0f}, "
            f"max={np.max(phase3_counts)}, "
            f"p95={np.percentile(phase3_counts, 95):.0f}, "
            f"cap_hits={cap_hits3}/{len(phase3_counts)} "
            f"({100 * cap_hits3 / len(phase3_counts):.0f}%)",
            flush=True,
        )

    # Save this maxls result
    out_path = f"{OUTPUT_DIR}/maxls_{maxls}.npz"
    np.savez(
        out_path,
        maxls=np.array(maxls),
        seed=np.array(SEED),
        # Phase 1 (standalone probe)
        phase1_ls_counts=phase1_counts,
        phase1_dec_errors=phase1_dec_errors,
        phase1_curv_errors=phase1_curv_errors,
        phase1_wall=np.array(t_phase1_wall),
        phase1_n_phi_probes=np.array(N_PHI_PROBES),
        # Phase 2 (warmup sample)
        phase2_ls_counts=phase2_counts,
        phase2_dec_errors=phase2_dec_errors,
        phase2_wall=np.array(t_warmup_wall),
        phase2_step_size=np.array(step_size),
        phase2_n_warmup=np.array(N_WARMUP_SAMPLE),
        # Phase 3 (sampling)
        phase3_ls_counts=phase3_counts,
        phase3_wall=np.array(t_sample_wall),
        phase3_n_steps=np.array(N_SAMPLING_STEPS),
    )
    print(f"  Saved → {out_path}", flush=True)

    all_results[maxls] = {
        "phase1_counts": phase1_counts,
        "phase2_counts": phase2_counts,
        "phase3_counts": phase3_counts,
        "phase1_wall": t_phase1_wall,
        "phase2_wall": t_warmup_wall,
        "phase3_wall": t_sample_wall,
        "step_size": step_size,
    }

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
total_wall = time.perf_counter() - t0
print(f"\n[t=+{total_wall:.1f}s] === ExpC Summary ===", flush=True)
print(
    f"  {'maxls':>6}  {'p1_mean':>8}  {'p1_max':>7}  {'p1_cap%':>7}  "
    f"{'p2_mean':>8}  {'p2_max':>7}  {'p2_cap%':>7}  "
    f"{'p2_wall':>9}  {'step_size':>10}",
    flush=True,
)
for maxls in MAXLS_VALUES:
    r = all_results.get(maxls)
    if r is None:
        print(f"  {maxls:>6}  (no data)", flush=True)
        continue
    p1 = r["phase1_counts"]
    p2 = r["phase2_counts"]
    p1_mean = f"{np.mean(p1):.1f}" if len(p1) > 0 else "N/A"
    p1_max = f"{np.max(p1)}" if len(p1) > 0 else "N/A"
    p1_cap = f"{100 * np.mean(p1 >= maxls):.0f}%" if len(p1) > 0 else "N/A"
    p2_mean = f"{np.mean(p2):.1f}" if len(p2) > 0 else "N/A"
    p2_max = f"{np.max(p2)}" if len(p2) > 0 else "N/A"
    p2_cap = f"{100 * np.mean(p2 >= maxls):.0f}%" if len(p2) > 0 else "N/A"
    p2_wall = f"{r['phase2_wall']:.1f}s"
    ss = f"{r['step_size']:.4g}"
    print(
        f"  {maxls:>6}  {p1_mean:>8}  {p1_max:>7}  {p1_cap:>7}  "
        f"{p2_mean:>8}  {p2_max:>7}  {p2_cap:>7}  "
        f"{p2_wall:>9}  {ss:>10}",
        flush=True,
    )

print(
    "\nColumns: maxls | "
    "phase1 mean/max LS steps | phase1 cap hit % | "
    "phase2 mean/max LS steps | phase2 cap hit % | "
    "phase2 warmup wall | step_size",
    flush=True,
)
print(
    "Phase 1 = standalone direct laplace(phi) calls; "
    "Phase 2 = NUTS warmup; "
    "Phase 3 = laplace_dhmc sampling.",
    flush=True,
)

# Diagnosis
print("\n=== Diagnosis ===", flush=True)
r1000 = all_results.get(1000)
r50 = all_results.get(50)
if r1000 is not None and r50 is not None:
    p1_mean_1000 = (
        np.mean(r1000["phase1_counts"])
        if len(r1000["phase1_counts"]) > 0
        else float("nan")
    )
    p1_mean_50 = (
        np.mean(r50["phase1_counts"]) if len(r50["phase1_counts"]) > 0 else float("nan")
    )
    cap_rate_1000 = (
        np.mean(r1000["phase1_counts"] >= 1000)
        if len(r1000["phase1_counts"]) > 0
        else float("nan")
    )
    cap_rate_50 = (
        np.mean(r50["phase1_counts"] >= 50)
        if len(r50["phase1_counts"]) > 0
        else float("nan")
    )

    if cap_rate_1000 > 0.1:
        print(
            f"DIAGNOSIS: linesearch hits cap at maxls=1000 in {100 * cap_rate_1000:.0f}% of outer iters. "
            "Linesearch budget IS a binding constraint. Reducing maxls should help.",
            flush=True,
        )
    elif p1_mean_1000 < 50:
        print(
            f"DIAGNOSIS: mean LS steps at maxls=1000 = {p1_mean_1000:.1f} (<50). "
            "Cap is NOT binding — linesearch converges early. "
            "Reducing maxls will NOT reduce wall time. "
            "Bottleneck is per-evaluation cost (Cholesky + gradient), not linesearch budget.",
            flush=True,
        )
    else:
        print(
            f"DIAGNOSIS: mixed. mean LS steps at maxls=1000 = {p1_mean_1000:.1f}. "
            f"Cap hit rate = {100 * cap_rate_1000:.0f}%. Manual inspection recommended.",
            flush=True,
        )

print("\nDone.", flush=True)
