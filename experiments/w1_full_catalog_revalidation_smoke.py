"""Smoke test for w1_full_catalog_revalidation.py.

Tests all code paths at tiny N to verify the E2E flow before committing the
full ~133-cell sweep.  Also measures per-cell wall time to project full sweep
duration.

Three tests:
  smoke-A  Path A (cached draws) + W1 gate
           Cell: eight_schools_ncp × dynamic_hmc × window_adaptation_diag_imm
  smoke-B  Path B (skip_warmup resample) + W1 gate
           Cell: mvn_10 × nuts × window_adaptation_diag_imm (first PASS B-cell found)
  smoke-4  Step 4 irt_2pl×chees at tiny nc=4, n_warmup=50, n_samples=30

Expected output (all three):
  smoke-A  PASS (well-behaved cached draws against certified GT)
  smoke-B  PASS (fresh resampled draws from tight model)
  smoke-4  Any verdict — just verifies the pipeline runs without error

Usage:
  JAX_PLATFORM_NAME=cpu uv run python experiments/w1_full_catalog_revalidation_smoke.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

# Set JAX backend before any JAX import
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# ---------------------------------------------------------------------------
# Import production helpers (sys.path extended so 'experiments' is importable)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from experiments.w1_full_catalog_revalidation import (  # noqa: E402  # type: ignore
    CATALOG,
    W1_ALPHA,
    W1_SEED,
    apply_w1_gate,
    collect_eligible_cells,
    load_cached_draws,
    load_gt_data,
    resample_recipe_draws,
    run_irt2pl_chees,
)

SMOKE_W1_B = 200  # fast bootstrap for smoke (not production-accurate)
SMOKE_RESAMPLE_N = 50
SMOKE_CHEES_N_CHAINS = 4
SMOKE_CHEES_N_WARMUP = 50
SMOKE_CHEES_N_SAMPLES = 30

PRINT_SEP = "=" * 60


def _verdict_str(w1_result) -> str:
    return (
        f"verdict={w1_result.verdict} "
        f"max_w1σ={w1_result.max_w1_sigma:.4f} "
        f"floor={w1_result.floor_of_max:.4f} "
        f"n_dims={w1_result.n_dims}"
    )


def smoke_a() -> dict:
    """Path A: cached draws for eight_schools_ncp × dynamic_hmc × window_adaptation_diag_imm."""
    print(f"\n{PRINT_SEP}")
    print("smoke-A: Path A (cached draws) — eight_schools_ncp × dynamic_hmc × diag_imm")
    print(PRINT_SEP)

    recipe_path = (
        CATALOG
        / "eight_schools_ncp"
        / "recipes"
        / "low__dynamic_hmc__window_adaptation_diag_imm.json"
    )
    if not recipe_path.exists():
        print(f"  SKIP: recipe not found at {recipe_path}")
        return {"status": "SKIP", "reason": "recipe not found"}

    t0 = time.perf_counter()
    draws = load_cached_draws(recipe_path)
    if draws is None:
        print("  SKIP: no cache file found")
        return {"status": "SKIP", "reason": "no cache"}

    print(
        f"  Loaded cached draws: {{{', '.join(f'{k}: {v.shape}' for k, v in draws.items())}}}"
    )

    gt_data = load_gt_data("eight_schools_ncp")
    if gt_data is None:
        print("  SKIP: no GT data found")
        return {"status": "SKIP", "reason": "no GT"}
    gt_draws_per_site, gt_summary_per_site = gt_data

    print(f"  GT sites: {list(gt_draws_per_site.keys())}")

    t_w1 = time.perf_counter()
    w1_result = apply_w1_gate(
        draws,
        gt_summary_per_site,
        gt_draws_per_site,
        B=SMOKE_W1_B,
        alpha=W1_ALPHA,
        seed=W1_SEED,
    )
    w1_elapsed = time.perf_counter() - t_w1
    total_elapsed = time.perf_counter() - t0

    print(f"  W1 gate ({w1_elapsed:.1f}s): {_verdict_str(w1_result)}")
    print(f"  Total elapsed: {total_elapsed:.1f}s")
    print(
        f"  smoke-A: {'PASS' if w1_result.verdict in ('PASS', 'SKIP') else 'FAIL (unexpected)'}"
    )
    return {
        "status": "OK",
        "w1_verdict": w1_result.verdict,
        "max_w1_sigma": float(w1_result.max_w1_sigma),
        "floor_of_max": float(w1_result.floor_of_max),
        "n_dims": w1_result.n_dims,
        "elapsed_s": round(total_elapsed, 1),
        "w1_elapsed_s": round(w1_elapsed, 1),
    }


def smoke_b() -> dict:
    """Path B: skip_warmup resample for a simple model."""
    print(f"\n{PRINT_SEP}")
    print("smoke-B: Path B (skip_warmup resample) — first eligible B cell")
    print(PRINT_SEP)

    # Find first path-B cell (skip_warmup eligible, no cache)
    cells = collect_eligible_cells()
    b_cells = [(m, p, c) for m, p, c in cells if c == "B"]
    if not b_cells:
        print("  SKIP: no path-B cells found")
        return {"status": "SKIP", "reason": "no B cells"}

    # Prefer a simple model (mvn_10 first)
    b_cell = None
    for model, recipe_path, code in b_cells:
        if model == "mvn_10":
            b_cell = (model, recipe_path, code)
            break
    if b_cell is None:
        b_cell = b_cells[0]

    model, recipe_path, code = b_cell
    print(f"  Cell: {model}/{recipe_path.stem}")

    t0 = time.perf_counter()
    draws = resample_recipe_draws(recipe_path, SMOKE_RESAMPLE_N, skip_warmup=True)
    resample_elapsed = time.perf_counter() - t0
    print(
        f"  Resampled draws ({resample_elapsed:.1f}s): "
        f"{{{', '.join(f'{k}: {v.shape}' for k, v in draws.items())}}}"
    )

    gt_data = load_gt_data(model)
    if gt_data is None:
        print("  SKIP: no GT data found")
        return {"status": "SKIP", "reason": "no GT"}
    gt_draws_per_site, gt_summary_per_site = gt_data

    t_w1 = time.perf_counter()
    w1_result = apply_w1_gate(
        draws,
        gt_summary_per_site,
        gt_draws_per_site,
        B=SMOKE_W1_B,
        alpha=W1_ALPHA,
        seed=W1_SEED,
    )
    w1_elapsed = time.perf_counter() - t_w1
    total_elapsed = time.perf_counter() - t0

    print(f"  W1 gate ({w1_elapsed:.1f}s): {_verdict_str(w1_result)}")
    print(f"  Total elapsed: {total_elapsed:.1f}s")
    print(
        f"  smoke-B: {'PASS' if w1_result.verdict in ('PASS', 'SKIP') else 'CHECK (flip?)'}"
    )
    return {
        "status": "OK",
        "model": model,
        "recipe": recipe_path.stem,
        "w1_verdict": w1_result.verdict,
        "max_w1_sigma": float(w1_result.max_w1_sigma),
        "floor_of_max": float(w1_result.floor_of_max),
        "n_dims": w1_result.n_dims,
        "resample_elapsed_s": round(resample_elapsed, 1),
        "w1_elapsed_s": round(w1_elapsed, 1),
        "elapsed_s": round(total_elapsed, 1),
    }


def smoke_4() -> dict:
    """Step 4: irt_2pl×chees at tiny N — just checks E2E pipeline doesn't crash."""
    print(f"\n{PRINT_SEP}")
    print("smoke-4: Step 4 irt_2pl×chees at tiny N (nc=4, n_warmup=50, n_samples=30)")
    print(PRINT_SEP)

    t0 = time.perf_counter()
    try:
        draws, gate_stats = run_irt2pl_chees(
            num_chains=SMOKE_CHEES_N_CHAINS,
            n_warmup=SMOKE_CHEES_N_WARMUP,
            n_samples=SMOKE_CHEES_N_SAMPLES,
        )
    except Exception as exc:
        import traceback as tb

        print(f"  ERROR in run_irt2pl_chees: {exc}")
        tb.print_exc()
        return {"status": "ERROR", "error": str(exc)}

    print(f"  Draws: {{{', '.join(f'{k}: {v.shape}' for k, v in draws.items())}}}")
    print(f"  gate_stats: {gate_stats}")

    gt_data = load_gt_data("irt_2pl")
    if gt_data is None:
        print("  SKIP: no GT data")
        return {"status": "SKIP", "reason": "no GT"}
    gt_draws_per_site, gt_summary_per_site = gt_data

    try:
        w1_result = apply_w1_gate(
            draws,
            gt_summary_per_site,
            gt_draws_per_site,
            B=SMOKE_W1_B,
            alpha=W1_ALPHA,
            seed=W1_SEED,
        )
    except Exception as exc:
        import traceback as tb

        print(f"  ERROR in apply_w1_gate: {exc}")
        tb.print_exc()
        return {"status": "ERROR", "error": str(exc)}

    total_elapsed = time.perf_counter() - t0
    print(f"  W1 gate: {_verdict_str(w1_result)}")
    print(f"  Total elapsed: {total_elapsed:.1f}s")
    print("  smoke-4: pipeline OK (verdict not checked at tiny N)")
    return {
        "status": "OK",
        "w1_verdict": w1_result.verdict,
        "max_w1_sigma": float(w1_result.max_w1_sigma),
        "floor_of_max": float(w1_result.floor_of_max),
        "n_dims": w1_result.n_dims,
        "n_heavy_tail_dims": w1_result.n_heavy_tail_dims,
        "elapsed_s": round(total_elapsed, 1),
    }


def main() -> None:
    print("W1 full-catalog re-validation — SMOKE TEST")
    print(f"JAX_PLATFORM_NAME={os.environ.get('JAX_PLATFORM_NAME', 'not set')}")
    print(f"SMOKE_W1_B={SMOKE_W1_B}  SMOKE_RESAMPLE_N={SMOKE_RESAMPLE_N}")
    print(
        f"SMOKE_CHEES: nc={SMOKE_CHEES_N_CHAINS}, n_warmup={SMOKE_CHEES_N_WARMUP}, "
        f"n_samples={SMOKE_CHEES_N_SAMPLES}"
    )

    results = {}
    t_total = time.perf_counter()

    results["smoke_a"] = smoke_a()
    results["smoke_b"] = smoke_b()
    results["smoke_4"] = smoke_4()

    total_elapsed = time.perf_counter() - t_total

    print(f"\n{PRINT_SEP}")
    print("SMOKE SUMMARY")
    print(PRINT_SEP)
    errors = 0
    for name, r in results.items():
        status = r.get("status", "?")
        elapsed = r.get("elapsed_s", 0)
        if status == "ERROR":
            errors += 1
            print(f"  {name}: ERROR — {r.get('error', '')[:80]}")
        elif status == "SKIP":
            print(f"  {name}: SKIP ({r.get('reason', '')})")
        else:
            w1v = r.get("w1_verdict", "?")
            print(f"  {name}: OK  W1={w1v}  wall={elapsed:.1f}s")

    # Per-cell time estimates for full sweep projection
    print(f"\nTotal smoke wall: {total_elapsed:.1f}s")

    # Estimate from smoke-A (W1-only) and smoke-B (resample + W1)
    a_res = results.get("smoke_a", {})
    b_res = results.get("smoke_b", {})
    if a_res.get("w1_elapsed_s") and b_res.get("elapsed_s"):
        t_a = a_res["w1_elapsed_s"]  # cached: W1 only
        t_b_w1 = b_res.get("w1_elapsed_s", 0)
        t_b_resample = b_res.get("resample_elapsed_s", 0)
        t_b = t_b_resample + t_b_w1
        # Extrapolate: 20 A cells, 102 B cells, 9 C cells (estimate C ≈ 3× B)
        est_min = (20 * t_a + 102 * t_b + 9 * 3 * t_b) / 60
        print(
            f"\nFull sweep estimate: ~{est_min:.0f} min (smoke-A={t_a:.1f}s, smoke-B={t_b:.1f}s)"
        )
        if est_min > 30:
            print(
                "WARNING: projected >30 min — must report wall estimate to @tl before full run."
            )
        else:
            print("Projected ≤30 min — OK to proceed directly.")
    else:
        print("Could not project full sweep time (smoke-A or smoke-B skipped/errored).")

    if errors > 0:
        print(f"\n{errors} smoke test(s) FAILED — fix before running full sweep.")
        sys.exit(1)
    else:
        print("\nAll smoke tests passed. Safe to run full sweep.")
        sys.exit(0)


if __name__ == "__main__":
    main()
