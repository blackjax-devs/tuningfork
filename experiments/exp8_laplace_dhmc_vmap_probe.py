"""Exp 8: vmap probe — laplace_dhmc × window_adaptation_diag_imm × gp_regression.

Statistician's decision table probe:
  Config: window_adaptation_diag_imm, n_warmup=50, n_samples=10, num_chains=4
  Budget: 40 min hard kill

Purpose: check if vmap(4) over laplace_dhmc_kernel adds significant overhead
vs 1-chain case (exp7). Distinguishes two hypotheses:

A) If exp8 ≈ exp7 (~500s): compile time doesn't scale with num_chains.
   Root cause is vmap(any) over the laplace_dhmc L-BFGS body — even 1 chain
   triggers the slow compilation. Fix: sequential chains (no vmap).

B) If exp8 ≈ 4× exp7 (~2000s): compile time scales linearly with chains.
   Root cause is the per-chain specialisation in XLA. Fix: same (sequential chains).

Context (exp6 + exp7 results):
  exp6 (jit(scan(laplace_dhmc, n=10)), no vmap): 6.0s → FAST
  exp7 (vmap(run_inference_algorithm(laplace_dhmc, n=10)), 1 chain): 500s → SLOW
  Key finding: sampling compilation is bottleneck, NOT warmup (warmup=19.3s).
  The recipe runner's vmap(run_inference_algorithm) is 83× slower than a bare jit.
  Hypothesis: vmap over L-BFGS line-search creates complex XLA graph.

This uses emit_low_recipe_for_cell (same as exp7, just num_chains=4).
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
print("n_warmup=50, n_samples=10, num_chains=4, seed=20260517", flush=True)

t_run_start = time.perf_counter()

result: CellResult = emit_low_recipe_for_cell(
    model_name="gp_regression",
    warmup_name="window_adaptation_diag_imm",
    sampler_name="laplace_dhmc",
    n_warmup=50,
    n_samples=10,
    num_chains=4,
    seed=20260517,
    verbose=True,
)

t_done = time.perf_counter()
run_wall = t_done - t_run_start
total_wall = t_done - t0

print("\n=== Exp 8 Result ===", flush=True)
print(f"  warmup+sample wall: {run_wall:.1f}s", flush=True)
print(f"  total wall: {total_wall:.1f}s", flush=True)
print(f"  verdict: {result.verdict}", flush=True)
print(f"  min_ess: {result.gate_min_ess}", flush=True)
print(f"  max_rhat: {result.gate_rhat_max}", flush=True)
print(f"  n_div: {result.gate_n_div}", flush=True)

# Compare to exp7 (1 chain, 50 warmup, 10 samples → 500s)
exp7_wall = 500.0
ratio_vs_exp7 = run_wall / exp7_wall
exp6_compile_run = 6.0
print(f"  exp7_wall (1-chain): {exp7_wall:.0f}s", flush=True)
print(f"  ratio (exp8/exp7): {ratio_vs_exp7:.1f}×", flush=True)
print(f"  exp6_compile_run: {exp6_compile_run:.1f}s", flush=True)

if run_wall < exp7_wall * 1.5:
    print(
        f"\nExp 8 ~ exp7 (ratio={ratio_vs_exp7:.1f}×): "
        "compile cost doesn't scale with num_chains.",
        flush=True,
    )
    print(
        "  -> Root cause: vmap(any) over L-BFGS triggers slow XLA path.",
        flush=True,
    )
    print(
        "  -> Fix direction: sequential chains (skip vmap) OR avoid L-BFGS in JIT.",
        flush=True,
    )
elif run_wall < exp7_wall * 6.0:
    print(
        f"\nExp 8 ~ {ratio_vs_exp7:.1f}× exp7: "
        "compile cost scales roughly linearly with num_chains.",
        flush=True,
    )
    print("  -> Fix direction: sequential chains OR reduce L-BFGS maxiter.", flush=True)
else:
    print(
        f"\nExp 8 >> {ratio_vs_exp7:.1f}× exp7: "
        "superlinear scaling with num_chains — interaction effect.",
        flush=True,
    )
    print(
        "  -> Fix direction: sequential chains is mandatory; also reduce maxiter.",
        flush=True,
    )

print("\nDone.", flush=True)
