"""Exp 7: scan probe — laplace_dhmc × window_adaptation_diag_imm × gp_regression.

Statistician's decision table probe:
  Config: window_adaptation_diag_imm, n_warmup=50, n_samples=10, num_chains=1
  Budget: 20 min hard kill

Purpose: isolate whether the warmup scan (n_warmup=50 steps unrolled) is the
bottleneck. If exp7 takes >5× exp6's 6.0s compile+run, scan unrolling of the
L-BFGS body is the root cause — n_warmup=1000 would be ~100× slower still.

Decision context (exp6 result):
  exp6 compile+run (10 steps, no warmup): 6.0s → FAST
  Kernel body compiles quickly; bottleneck must be elsewhere.

If exp7 is SLOW (>5× exp6 = >30s):
  -> Diagnosis: scan unrolling of L-BFGS body at n_warmup=50 steps
  -> Fix direction: n_warmup ≤ 100 non-viable at 1000; need while_loop or
     reduce L-BFGS maxiter (default 30)

Uses emit_low_recipe_for_cell (window_adaptation_diag_imm works fine with
laplace_dhmc — no extra_required_kwargs blocking the warmup path).
"""

import os
import sys
import time

t0 = time.perf_counter()
print("[t=+0.0s] Script start", flush=True)

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax  # noqa: E402

t_jax = time.perf_counter()
print(
    f"[t=+{t_jax - t0:.1f}s] JAX imported: "
    f"x64={jax.config.read('jax_enable_x64')}, "
    f"backend={jax.default_backend()}",
    flush=True,
)

sys.path.insert(0, "/home/jp/blackjax-devs/tuningfork")

from tuningfork.recipes._recipe_runner import (  # noqa: E402
    CellResult,
    emit_low_recipe_for_cell,
)

t_imports = time.perf_counter()
print(f"[t=+{t_imports - t0:.1f}s] All imports done", flush=True)

print(
    "Running emit_low_recipe_for_cell("
    "gp_regression × window_adaptation_diag_imm × laplace_dhmc)...",
    flush=True,
)
print("n_warmup=50, n_samples=10, num_chains=1, seed=20260517", flush=True)

t_run_start = time.perf_counter()

result: CellResult = emit_low_recipe_for_cell(
    model_name="gp_regression",
    warmup_name="window_adaptation_diag_imm",
    sampler_name="laplace_dhmc",
    n_warmup=50,
    n_samples=10,
    num_chains=1,
    seed=20260517,
    verbose=True,
)

t_done = time.perf_counter()
run_wall = t_done - t_run_start
total_wall = t_done - t0

print("\n=== Exp 7 Result ===", flush=True)
print(f"  warmup+sample wall: {run_wall:.1f}s", flush=True)
print(f"  total wall: {total_wall:.1f}s", flush=True)
print(f"  verdict: {result.verdict}", flush=True)
print(f"  min_ess: {result.gate_min_ess}", flush=True)
print(f"  max_rhat: {result.gate_rhat_max}", flush=True)
print(f"  n_div: {result.gate_n_div}", flush=True)

# Decision table: exp6 compile+run = 6.0s; ratio threshold = 5×
exp6_compile_run = 6.0
ratio = run_wall / exp6_compile_run
print(f"  exp6_compile_run: {exp6_compile_run:.1f}s", flush=True)
print(f"  ratio (exp7/exp6): {ratio:.1f}×", flush=True)

if run_wall < 30.0:
    print("\nExp 7 FAST (<5× exp6): scan overhead is not the bottleneck.", flush=True)
    print("  -> exp8 (vmap probe) will check if 4-chain vmap is the issue.", flush=True)
elif run_wall < 1200.0:
    print(
        f"\nExp 7 SLOW ({run_wall:.0f}s, {ratio:.1f}× exp6): "
        "scan unrolling is the bottleneck.",
        flush=True,
    )
    print(
        "  -> Fix: n_warmup ≤ 100 required, or reduce L-BFGS maxiter from 30.",
        flush=True,
    )
else:
    print(
        f"\nExp 7 EXCEEDED BUDGET ({run_wall:.0f}s > 1200s): "
        "scan unrolling catastrophic.",
        flush=True,
    )
    print(
        "  -> Fix: L-BFGS body must be converted to dynamic while_loop.",
        flush=True,
    )

print("\nDone.", flush=True)
