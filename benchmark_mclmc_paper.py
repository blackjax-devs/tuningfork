# Copyright 2026- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Directional paper replication script comparing unadjusted MCLMC vs. NUTS.

Utilizes tuningfork.recipes._recipe_runner's auto-gate to execute runs
and outputs a standardized, structured results.json.
"""

import argparse
import json
import time
from pathlib import Path

from tuningfork.recipes._base import Recipe
from tuningfork.recipes._recipe_runner import emit_low_recipe_for_cell


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCLMC paper validation benchmark using _recipe_runner"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in downscaled smoke-test mode to verify E2E flow quickly",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Path to save the structured JSON results (default: results.json)",
    )
    args = parser.parse_args()

    # Determine execution scale
    if args.smoke:
        print("Executing in SMOKE mode...")
        n_warmup = 10
        n_samples = 20
        num_chains = 2
    else:
        print("Executing in PRODUCTION mode...")
        n_warmup = 1000
        n_samples = 1000
        num_chains = 4

    seed = 42
    models = ["irt_1pl", "irt_2pl", "lgcp"]
    samplers = ["nuts", "mclmc"]

    results = []

    for m in models:
        for s in samplers:
            warmup = "window_adaptation_diag_imm" if s == "nuts" else "mclmc_tuning"
            print("\n==========================================")
            print(f"Running {m} with {s} ({warmup})...")
            print("==========================================")

            t_start = time.perf_counter()
            cell_res = emit_low_recipe_for_cell(
                model_name=m,
                warmup_name=warmup,
                sampler_name=s,
                n_warmup=n_warmup,
                n_samples=n_samples,
                num_chains=num_chains,
                seed=seed,
                verbose=True,
            )
            wall_total = time.perf_counter() - t_start

            # Extract metrics from cell_res or saved recipe JSON
            verdict = cell_res.verdict
            min_ess = cell_res.gate_min_ess
            warmup_grads = cell_res.warmup_grad_evals
            sampling_grads = getattr(cell_res, "sampling_grad_evals", None)
            sampling_time = None
            warmup_time = None

            # On PASS or REVIEW, emit_low_recipe_for_cell writes a recipe.
            # We can load it to get the precise metrics.
            if (
                verdict in ("PASS", "REVIEW")
                and cell_res.recipe_path
                and cell_res.recipe_path.exists()
            ):
                try:
                    recipe = Recipe.load(cell_res.recipe_path)
                    if recipe.headline_basis:
                        sampling_grads = recipe.headline_basis.get("total_grad_evals")
                        min_ess = recipe.headline_basis.get("min_bulk_ess")
                    if recipe.calibration_budget:
                        warmup_grads = recipe.calibration_budget.get(
                            "warmup_grad_evals"
                        )
                        sampling_time = recipe.calibration_budget.get(
                            "sampling_wall_seconds"
                        )
                        warmup_time = recipe.calibration_budget.get(
                            "warmup_wall_seconds"
                        )
                except Exception as e:
                    print(f"Warning: failed to load emitted recipe JSON: {e}")

            # Fallbacks / Manual calculation for FAIL or load failure
            if min_ess is None:
                min_ess = (
                    cell_res.gate_min_ess if cell_res.gate_min_ess is not None else 0.0
                )
            if warmup_grads is None:
                # Fallback to CellResult value or None
                warmup_grads = cell_res.warmup_grad_evals

            if sampling_grads is None:
                if s == "mclmc":
                    # Unadjusted McLachlan uses 2 gradient evaluations per step.
                    sampling_grads = 2 * n_samples * num_chains
                else:
                    # NUTS - should not fail, but if it does, we estimate based on default/previous
                    sampling_grads = 0

            # Calculate total and efficiency
            total_grads = (warmup_grads if warmup_grads is not None else 0) + (
                sampling_grads if sampling_grads is not None else 0
            )

            ess_per_grad_sampling = (
                float(min_ess / sampling_grads) if sampling_grads and min_ess else 0.0
            )
            ess_per_grad_total = (
                float(min_ess / total_grads) if total_grads and min_ess else 0.0
            )

            row = {
                "model": m,
                "sampler": s,
                "min_ess": float(min_ess) if min_ess is not None else None,
                "warmup_grads": warmup_grads,
                "sampling_grads": sampling_grads,
                "total_grads": total_grads,
                "ess_per_grad_sampling": ess_per_grad_sampling,
                "ess_per_grad_total": ess_per_grad_total,
                "sampling_time": (
                    sampling_time
                    if sampling_time is not None
                    else (wall_total - (warmup_time or 0.0))
                ),
                "warmup_time": warmup_time,
                "verdict": verdict,
                "note": cell_res.note,
            }
            results.append(row)

            print(f"Result for {m} ({s}):")
            print(f"  verdict:               {verdict}")
            print(f"  min_ess:               {min_ess}")
            print(f"  warmup_grads:          {warmup_grads}")
            print(f"  sampling_grads:        {sampling_grads}")
            print(f"  total_grads:           {total_grads}")
            print(f"  ess_per_grad_sampling: {ess_per_grad_sampling:.6e}")
            print(f"  ess_per_grad_total:    {ess_per_grad_total:.6e}")

    # Output path
    base_dir = Path(__file__).parent

    # Save structured JSON
    json_path = base_dir / "results.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Save formatted text table to results.txt
    txt_path = base_dir / "results.txt"
    with txt_path.open("w") as f:
        f.write("FINAL COMPARISON TABLE (ESS per Gradient Evaluation)\n")
        f.write("==========================================\n")
        f.write(
            f"{'Model':<12} | {'Sampler':<7} | {'Min ESS':<8} | "
            f"{'Grads (Samp)':<12} | {'ESS/Grad (Samp)':<16} | {'ESS/Grad (Total)':<16}\n"
        )
        f.write("-" * 80 + "\n")
        for r in results:
            # We format the floats beautifully
            min_ess_val = r["min_ess"]
            min_ess_str = f"{min_ess_val:.2f}" if min_ess_val is not None else "N/A"
            f.write(
                f"{r['model']:<12} | {r['sampler']:<7} | {min_ess_str:<8} | "
                f"{r['sampling_grads']:<12} | "
                f"{r['ess_per_grad_sampling']:<16.6e} | {r['ess_per_grad_total']:<16.6e}\n"
            )

    print(
        f"\nBenchmark run complete! Results written to:\n  JSON: {json_path}\n  TXT:  {txt_path}"
    )


if __name__ == "__main__":
    main()
