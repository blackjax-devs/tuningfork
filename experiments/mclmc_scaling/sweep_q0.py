"""Q0 sweep: isolate E1 (√d warm-start) from IMM quality by holding IMM fixed at GT-dense.

Design:
  - IMM is FIXED at GT-dense (k=d) for all runs — removes the pilot-IMM bottleneck.
  - Only the TUNING budget (n_warmup) varies: {50, 100, 200, 500, 1000, 2000}.
  - For each (model, n_warmup), we run run_mclmc_fixed_imm TWICE:
      (a) default init  -- tune_init_step=None (DA starts at 0.25*sqrt(d))
      (b) sqrt(d) warm-start -- tune_init_step=1.22*sqrt(d), tune_init_L=0.85*sqrt(d)
  - Records: tuned step, tuned L, ess/grad, min_bulk_ess, max_bias.
  - Signal we want: smallest n_warmup at which warm-start reaches plateau ess/grad vs default.

Full sweep: {mvn_10, ill_cond_50, german_credit, irt_1pl}
  n_warmup in {50, 100, 200, 500, 1000, 2000}, n_samples=3000, num_chains=4.

Smoke mode (--smoke): {mvn_10, ill_cond_50}, n_warmup in {50, 200},
  n_samples=300, num_chains=2.

Usage:
  python sweep_q0.py --smoke
  python sweep_q0.py --outfile results_q0.jsonl
"""

import argparse
import json
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Add experiment dir to sys.path so local imports work
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gt_imm import gt_cov, gt_from_draws, gt_lrd_imm
from run_fixed_imm import run_mclmc_fixed_imm

# ---------------------------------------------------------------------------
# Panel configuration
# ---------------------------------------------------------------------------

FULL_MODELS = ["mvn_10", "ill_cond_50", "german_credit", "irt_1pl"]
SMOKE_MODELS = ["mvn_10", "ill_cond_50"]

FULL_N_WARMUP = [50, 100, 200, 500, 1000, 2000]
SMOKE_N_WARMUP = [50, 200]

FULL_N_SAMPLES = 3000
FULL_NUM_CHAINS = 4

SMOKE_N_SAMPLES = 300
SMOKE_NUM_CHAINS = 2

# Base seed for all runs
BASE_SEED = 12345


# ---------------------------------------------------------------------------
# Model setup helper
# ---------------------------------------------------------------------------


def _get_model_gt(model_name: str):
    """Return (imm_dense, gt_var, gt_mean, d) for a model.

    Synthetic models: analytic Sigma, gt_lrd_imm(Sigma, k=d).
    Real models: gt_from_draws(model_name, k=d).
    """
    _SYNTHETIC = {"ill_cond_50", "mvn_10"}
    if model_name in _SYNTHETIC:
        Sigma, _ = gt_cov(model_name)
        d = Sigma.shape[0]
        imm_dense = gt_lrd_imm(Sigma, k=d)
        gt_var = np.diag(Sigma)
        gt_mean = np.zeros(d, dtype=np.float64)
        return imm_dense, gt_var, gt_mean, d
    else:
        imm_dense, gt_var, gt_mean, d = gt_from_draws(
            model_name, k=d if False else None
        )
        # gt_from_draws with k=None uses k=d (dense) by default
        return imm_dense, gt_var, gt_mean, d


# ---------------------------------------------------------------------------
# Main sweep function
# ---------------------------------------------------------------------------


def run_sweep(
    models: list[str],
    n_warmup_list: list[int],
    n_samples: int,
    num_chains: int,
    outfile: str | None,
):
    """Run Q0 sweep: default vs √d warm-start across (model, n_warmup) grid.

    For each model, pre-loads the GT-dense IMM once, then iterates over
    n_warmup values. For each (model, n_warmup) point runs two calls to
    run_mclmc_fixed_imm — one with default init, one with √d warm-start.
    """
    results = []

    for model_name in models:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")

        # Load GT-dense IMM once per model
        t0 = time.time()
        imm_dense, gt_var, gt_mean, d = _get_model_gt(model_name)
        sqrt_d = math.sqrt(d)
        step_warm = 1.22 * sqrt_d
        L_warm = 0.85 * sqrt_d
        print(
            f"  d={d}, sqrt(d)={sqrt_d:.2f}, warm_start: step={step_warm:.3f}, L={L_warm:.3f}"
        )
        print(f"  GT IMM loaded in {time.time()-t0:.1f}s")
        sys.stdout.flush()

        _is_real = model_name not in {"ill_cond_50", "mvn_10"}
        _gt_mean_arg = gt_mean if _is_real else None
        _gt_var_arg = gt_var if _is_real else None

        for n_warmup in n_warmup_list:
            seed = BASE_SEED + hash((model_name, n_warmup)) % 10000

            # (a) Default init
            t0 = time.time()
            res_def = run_mclmc_fixed_imm(
                model_name=model_name,
                imm=imm_dense,
                n_warmup=n_warmup,
                n_samples=n_samples,
                num_chains=num_chains,
                seed=seed,
                tune_init_step=None,
                tune_init_L=None,
                gt_mean=_gt_mean_arg,
                gt_var=_gt_var_arg,
            )
            t_def = time.time() - t0

            # (b) √d warm-start
            t0 = time.time()
            res_warm = run_mclmc_fixed_imm(
                model_name=model_name,
                imm=imm_dense,
                n_warmup=n_warmup,
                n_samples=n_samples,
                num_chains=num_chains,
                seed=seed,
                tune_init_step=step_warm,
                tune_init_L=L_warm,
                gt_mean=_gt_mean_arg,
                gt_var=_gt_var_arg,
            )
            t_warm = time.time() - t0

            # Print comparison row
            print(
                f"  n_w={n_warmup:4d} | "
                f"default: step={res_def['step_size']:.3f} L={res_def['L']:.3f} "
                f"epg={res_def['ess_per_grad']:.5f} ess={res_def['min_bulk_ess']:.1f} bias={res_def['max_bias']:.4f} "
                f"({t_def:.1f}s) | "
                f"warm:    step={res_warm['step_size']:.3f} L={res_warm['L']:.3f} "
                f"epg={res_warm['ess_per_grad']:.5f} ess={res_warm['min_bulk_ess']:.1f} bias={res_warm['max_bias']:.4f} "
                f"({t_warm:.1f}s)"
            )
            sys.stdout.flush()

            # Record row
            row = {
                "model": model_name,
                "d": d,
                "n_warmup": n_warmup,
                "n_samples": n_samples,
                "num_chains": num_chains,
                "seed": seed,
                "sqrt_d": sqrt_d,
                "warm_start_step": step_warm,
                "warm_start_L": L_warm,
                # default init results
                "def_step": res_def["step_size"],
                "def_L": res_def["L"],
                "def_ess_per_grad": res_def["ess_per_grad"],
                "def_min_bulk_ess": res_def["min_bulk_ess"],
                "def_max_bias": res_def["max_bias"],
                "def_total_grads": res_def["total_grads"],
                # warm-start results
                "warm_step": res_warm["step_size"],
                "warm_L": res_warm["L"],
                "warm_ess_per_grad": res_warm["ess_per_grad"],
                "warm_min_bulk_ess": res_warm["min_bulk_ess"],
                "warm_max_bias": res_warm["max_bias"],
                "warm_total_grads": res_warm["total_grads"],
                # relative speedup
                "epg_ratio_warm_over_def": (
                    res_warm["ess_per_grad"] / max(res_def["ess_per_grad"], 1e-30)
                ),
            }
            results.append(row)

            if outfile:
                with open(outfile, "a") as f:
                    f.write(json.dumps(row) + "\n")

    print(f"\n{'='*70}")
    print(f"Sweep complete. {len(results)} rows.")
    if outfile:
        print(f"Results written to: {outfile}")
    sys.stdout.flush()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Q0 sweep: default vs sqrt(d) warm-start"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke mode: {mvn_10, ill_cond_50} x {50,200} warmup, n_samples=300, 2 chains",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Path to JSONL output file (append mode). Default: results_q0_smoke.jsonl or results_q0.jsonl",
    )
    args = parser.parse_args()

    if args.smoke:
        models = SMOKE_MODELS
        n_warmup_list = SMOKE_N_WARMUP
        n_samples = SMOKE_N_SAMPLES
        num_chains = SMOKE_NUM_CHAINS
        outfile = args.outfile or os.path.join(_HERE, "results_q0_smoke.jsonl")
        print(
            f"SMOKE MODE: models={models}, n_warmup={n_warmup_list}, "
            f"n_samples={n_samples}, num_chains={num_chains}"
        )
    else:
        models = FULL_MODELS
        n_warmup_list = FULL_N_WARMUP
        n_samples = FULL_N_SAMPLES
        num_chains = FULL_NUM_CHAINS
        outfile = args.outfile or os.path.join(_HERE, "results_q0.jsonl")
        print(
            f"FULL SWEEP: models={models}, n_warmup={n_warmup_list}, "
            f"n_samples={n_samples}, num_chains={num_chains}"
        )

    sys.stdout.flush()

    # Clear output file if it exists (fresh run)
    if outfile and os.path.exists(outfile):
        os.remove(outfile)
        print(f"Cleared existing output file: {outfile}")

    results = run_sweep(
        models=models,
        n_warmup_list=n_warmup_list,
        n_samples=n_samples,
        num_chains=num_chains,
        outfile=outfile,
    )

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE (ess/grad ratio warm/default)")
    print(f"{'Model':<16} {'n_w':>6} {'def_epg':>10} {'warm_epg':>10} {'ratio':>7}")
    print(f"{'-'*16} {'-'*6} {'-'*10} {'-'*10} {'-'*7}")
    for r in results:
        print(
            f"{r['model']:<16} {r['n_warmup']:>6} "
            f"{r['def_ess_per_grad']:>10.5f} {r['warm_ess_per_grad']:>10.5f} "
            f"{r['epg_ratio_warm_over_def']:>7.3f}"
        )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
