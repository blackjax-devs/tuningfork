"""ExpB: laplace_dhmc × gp_regression — maxls sweep to confirm linesearch bottleneck.

Purpose: confirm that the ~427s warmup wall in exp9 is driven by
max_linesearch_steps=1000 (L-BFGS default).  If warmup_wall scales with maxls,
the fix is a smaller default or exposing the kwarg in the recipe runner.

Wiring check (confirmed by code read 2026-05-24):
  blackjax.laplace_dhmc(**optimizer_kwargs)
    → as_top_level_api(**optimizer_kwargs)
    → laplace_marginal_factory(**optimizer_kwargs)
    → minimize_lbfgs(maxls=optimizer_kwargs.get("maxls", 1000))

  tuningfork _recipe_runner.py line 327:
    laplace = laplace_marginal_factory(log_joint_fn, theta_init)  # NO maxls → 1000
  tuningfork _recipe_runner.py line 715-721:
    kernel = base_method.factory(..., **shared_kwargs)            # NO maxls → 1000

  -> Wiring verdict: (b) configurable in blackjax (optimizer_kwargs plumbed through),
     UNSET by tuningfork — always runs at maxls=1000.  Fix lives in tuningfork runner
     (or a saner blackjax default).

  Note: laplace_dynamic_hmc.py as_top_level_api docstring omits maxls from "Useful
  keys" list (documents maxiter/gtol/ftol only) — documentation inconsistency.

Config: gp_regression, n_warmup=50, n_samples=10, 1 chain, NO vmap.
  Using blackjax.window_adaptation directly (bypasses tuningfork runner vmap).
  float(step_size) after warmup forces async-dispatch materialization.

maxls sweep: {1000 (default), 50, 20, 10}.
Expected: warmup_wall ∝ maxls if linesearch dominates.
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
    f"[t=+{t_jax - t0:.1f}s] JAX: x64={jax.config.read('jax_enable_x64')}, "
    f"backend={jax.default_backend()}",
    flush=True,
)

sys.path.insert(0, "/home/jp/blackjax-devs/tuningfork")
sys.path.insert(0, "/home/jp/blackjax-devs/blackjax")

import blackjax  # noqa: E402
from blackjax.mcmc.laplace_marginal import laplace_marginal_factory  # noqa: E402
from blackjax.util import run_inference_algorithm  # noqa: E402

from tuningfork.model import MODELS  # noqa: E402
from tuningfork.model._numpyro import build_logdensity_fn  # noqa: E402
from tuningfork.recipes._recipe_runner import _build_laplace_components  # noqa: E402

t_imports = time.perf_counter()
print(f"[t=+{t_imports - t0:.1f}s] All imports done", flush=True)

# ---------------------------------------------------------------------------
# Build gp_regression model components
# ---------------------------------------------------------------------------
SEED = 20260517
N_WARMUP = 50
N_SAMPLES = 10
MAXLS_VALUES = [1000, 50, 20, 10]

model = MODELS["gp_regression"]
init_key, warmup_key, run_key = jax.random.split(jax.random.key(SEED), 3)

full_position, raw_joint_fn, _postprocess = build_logdensity_fn(init_key, model)

laplace_result = _build_laplace_components("gp_regression", full_position, raw_joint_fn)
assert laplace_result is not None, "gp_regression not in _LAPLACE_PHI_THETA_SPLITS"
phi_init, log_joint_fn, theta_init, _ = laplace_result
# Note: log_joint_fn is the 2-arg (theta, phi) → float function from
# _build_laplace_components; we rebuild the marginal per-maxls below.

d_phi = sum(jnp.asarray(v).size for v in phi_init.values())
d_theta = sum(jnp.asarray(v).size for v in theta_init.values())
print(
    f"[t=+{time.perf_counter() - t0:.1f}s] " f"d_phi={d_phi}, d_theta={d_theta}",
    flush=True,
)


# ---------------------------------------------------------------------------
# Per-maxls sweep
# ---------------------------------------------------------------------------

results = []

for maxls in MAXLS_VALUES:
    print(
        f"\n[t=+{time.perf_counter() - t0:.1f}s] === maxls={maxls} ===",
        flush=True,
    )
    wkey, ikey, skey = jax.random.split(jax.random.key(SEED + maxls), 3)

    # Build marginal logdensity with THIS maxls value
    laplace = laplace_marginal_factory(log_joint_fn, theta_init, maxls=maxls)

    def marginal_logdensity_fn(phi):
        lp, _ = laplace(phi)
        return lp

    # --- WARMUP (no vmap, single chain, blackjax.window_adaptation direct) ---
    print(
        f"  Building warmup (window_adaptation+nuts, n_warmup={N_WARMUP})...",
        flush=True,
    )
    t_warmup_start = time.perf_counter()

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        marginal_logdensity_fn,
        progress_bar=False,
    )
    (warmup_state, adapted_params), _warmup_info = warmup.run(wkey, phi_init, N_WARMUP)

    # Force async-dispatch materialization (adapted_params is a dict)
    step_size = float(jnp.asarray(adapted_params["step_size"]))
    imm = jnp.asarray(adapted_params["inverse_mass_matrix"])
    t_warmup_wall = time.perf_counter() - t_warmup_start

    print(
        f"  maxls={maxls}: warmup wall={t_warmup_wall:.1f}s, "
        f"step_size={step_size:.4g}, imm_shape={imm.shape}",
        flush=True,
    )

    # --- SAMPLING (laplace_dhmc, n_samples=10, 1 chain, no vmap) ---
    print(
        f"  Running sampling (laplace_dhmc, n_samples={N_SAMPLES})...",
        flush=True,
    )
    t_sampling_start = time.perf_counter()

    kernel = blackjax.laplace_dhmc(
        log_joint_fn,
        theta_init=theta_init,
        step_size=step_size,
        inverse_mass_matrix=imm,
        maxls=maxls,
    )
    init_state = kernel.init(phi_init, ikey)
    final_state, _ = run_inference_algorithm(
        rng_key=skey,
        inference_algorithm=kernel,
        num_steps=N_SAMPLES,
        initial_state=init_state,
    )
    _ = jax.block_until_ready(final_state.position)
    t_sampling_wall = time.perf_counter() - t_sampling_start

    print(
        f"  maxls={maxls}: sampling wall={t_sampling_wall:.1f}s",
        flush=True,
    )

    results.append(
        {
            "maxls": maxls,
            "warmup_wall": t_warmup_wall,
            "sampling_wall": t_sampling_wall,
            "step_size": step_size,
        }
    )

# ---------------------------------------------------------------------------
# Summary + diagnosis
# ---------------------------------------------------------------------------
total_wall = time.perf_counter() - t0
print(f"\n[t=+{total_wall:.1f}s] === ExpB Summary ===", flush=True)
print(
    f"  {'maxls':>6}  {'warmup_wall':>12}  {'sampling_wall':>14}  "
    f"{'warmup_ratio':>12}  {'step_size':>10}",
    flush=True,
)

baseline_w = results[0]["warmup_wall"] if results else None
for r in results:
    ww = f"{r['warmup_wall']:.1f}s"
    sw = f"{r['sampling_wall']:.1f}s"
    ratio = f"{r['warmup_wall'] / baseline_w:.3f}×" if baseline_w else "N/A"
    ss = f"{r['step_size']:.4g}"
    print(
        f"  {r['maxls']:>6}  {ww:>12}  {sw:>14}  {ratio:>12}  {ss:>10}",
        flush=True,
    )

# Theoretical linear-scaling prediction: warmup_wall ∝ maxls
if baseline_w and len(results) >= 2:
    print("", flush=True)
    print("  Theoretical prediction if warmup_wall ∝ maxls:", flush=True)
    for r in results:
        pred = baseline_w * r["maxls"] / 1000
        print(f"    maxls={r['maxls']:4d}: predicted={pred:.1f}s", flush=True)

    # Verdict
    r10 = next((r for r in results if r["maxls"] == 10), None)
    r50 = next((r for r in results if r["maxls"] == 50), None)
    if r10:
        ratio_10 = r10["warmup_wall"] / baseline_w
        if ratio_10 < 0.05:
            print(
                "\nDIAGNOSIS: warmup cost IS proportional to maxls. "
                "Fix: set maxls ≤ 50 in laplace_marginal_factory (or expose as recipe kwarg).",
                flush=True,
            )
        elif ratio_10 > 0.5:
            print(
                "\nDIAGNOSIS: warmup cost does NOT scale with maxls. "
                "Linesearch budget exhausted quickly; bottleneck elsewhere.",
                flush=True,
            )
        else:
            print(
                f"\nDIAGNOSIS: partial sensitivity (ratio10={ratio_10:.3f}). "
                "Linesearch is a factor but not the only bottleneck.",
                flush=True,
            )

print("\nDone.", flush=True)
