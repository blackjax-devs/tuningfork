"""Targeted re-emission for the 9 FLAG_FAIL recipes from the GT migration coherence screen.

TL directive 2026-07-13: re-run each cell under new GT (summary_v2.json path now wired
in the production path), record before/after verdict + z.

Recipes that flip on re-run: reported, NOT deleted or re-tuned in this branch.
Verdict-evidence updates ride along in the PR.

Order: cheap CPU models first (banana, eight_schools×2, german_credit×2, ill_cond_50,
mvn_10×2), radon GPU LAST (colossus, num_chains=128).

Usage:
    JAX_PLATFORM_NAME=cpu uv run python experiments/gt_migration_reemit.py
    # radon cell only (GPU colossus):
    JAX_PLATFORM_NAME=cuda uv run python experiments/gt_migration_reemit.py --radon-only
"""

import argparse
import json
import os
import pathlib
import sys
import time

# Ensure repo root is importable
_REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

CATALOG = _REPO / "tuningfork" / "catalog"
RESULTS_FILE = pathlib.Path(__file__).parent / "gt_migration_reemit_results.json"

# ---------------------------------------------------------------------------
# Recipe configs: (model, stem, n_warmup_override, n_samples, num_chains, extra_kwargs)
# "extra_kwargs" feeds into emit_low_recipe_for_cell as keyword args.
# ---------------------------------------------------------------------------
CPU_CONFIGS = [
    # banana: statistician-certified VMAP run; re-emit with same n_warmup/n_samples
    # and ta=0.9 (from committed warmup params). Effort=MEDIUM to honour calibration.
    dict(
        model="banana",
        stem="medium__adjusted_mclmc_dynamic__adjusted_mclmc_tuning",
        n_warmup=5000,
        n_samples=5000,
        num_chains=4,
        target_acceptance=0.9,
        effort="medium",
    ),
    # eight_schools_ncp low HMC with LR-IMM; num_integration_steps=64 is key
    dict(
        model="eight_schools_ncp",
        stem="low__hmc__window_adaptation_low_rank_imm",
        n_warmup=1000,
        n_samples=1000,
        num_chains=4,
        sampler_kwargs_override={"num_integration_steps": 64},
        effort="low",
    ),
    # eight_schools_ncp low dmhmc with dense IMM
    dict(
        model="eight_schools_ncp",
        stem="low__dmhmc__window_adaptation_dense_imm",
        n_warmup=1000,
        n_samples=1000,
        num_chains=4,
        effort="low",
    ),
    # german_credit medium HMC with diag IMM; nis=5
    dict(
        model="german_credit",
        stem="medium__hmc__window_adaptation_diag_imm",
        n_warmup=1000,
        n_samples=1000,
        num_chains=4,
        sampler_kwargs_override={"num_integration_steps": 5},
        effort="medium",
    ),
    # german_credit low dynamic_hmc with CHEES warmup; ta=0.651
    dict(
        model="german_credit",
        stem="low__dynamic_hmc__chees",
        n_warmup=1000,
        n_samples=1000,
        num_chains=4,
        target_acceptance=0.651,
        effort="low",
    ),
    # ill_cond_50 medium dmhmc with step_policy (empirical oracle)
    dict(
        model="ill_cond_50",
        stem="medium__dmhmc__window_adaptation_diag_imm__policy_v7-empirical-oracle",
        n_warmup=1000,
        n_samples=1000,
        num_chains=4,
        step_policy={
            "kind": "empirical",
            "values": [15, 31, 47, 63, 95, 127, 159, 191],
            "weights": [
                0.00025,
                0.02525,
                0.00025,
                0.61525,
                0.00975,
                0.34825,
                0.00025,
                0.00075,
            ],
        },
        policy_tag="v7-empirical-oracle",
        effort="medium",
    ),
    # mvn_10 medium HMC dense IMM; nis=5
    dict(
        model="mvn_10",
        stem="medium__hmc__window_adaptation_dense_imm",
        n_warmup=1000,
        n_samples=1000,
        num_chains=4,
        sampler_kwargs_override={"num_integration_steps": 5},
        effort="medium",
    ),
    # mvn_10 medium HMC diag IMM; nis=5
    dict(
        model="mvn_10",
        stem="medium__hmc__window_adaptation_diag_imm",
        n_warmup=1000,
        n_samples=1000,
        num_chains=4,
        sampler_kwargs_override={"num_integration_steps": 5},
        effort="medium",
    ),
]

GPU_CONFIGS = [
    # radon medium dynamic_hmc with CHEES; num_chains=128 — GPU only (colossus)
    dict(
        model="radon",
        stem="medium__dynamic_hmc__chees",
        n_warmup=2000,
        n_samples=1000,
        num_chains=128,
        effort="medium",
    ),
]


def parse_stem(stem: str):
    """Parse recipe stem → (effort, sampler, warmup, inner, policy_tag)."""
    parts = stem.split("__")
    effort = parts[0]
    sampler = parts[1]
    warmup_parts = parts[2:]
    # Strip inner_kernel and policy tags
    inner = None
    policy_tag = None
    warmup_name_parts = []
    for p in warmup_parts:
        if p.startswith("inner_"):
            inner = p[len("inner_") :]
        elif p.startswith("policy_"):
            policy_tag = p[len("policy_") :]
        else:
            warmup_name_parts.append(p)
    warmup = "__".join(warmup_name_parts)
    return effort, sampler, warmup, inner, policy_tag


def read_committed_verdict(model: str, stem: str) -> dict:
    """Read the committed gate_evidence.auto from the stored recipe."""
    p = CATALOG / model / "recipes" / f"{stem}.json"
    if not p.exists():
        return {}
    r = json.loads(p.read_text())
    return r.get("gate_evidence", {}).get("auto", {})


def reemit_cell(cfg: dict) -> dict:
    """Run emit_low_recipe_for_cell for one cell and return result dict."""
    from tuningfork.recipes._recipe_runner import Effort, emit_low_recipe_for_cell

    model = cfg["model"]
    stem = cfg["stem"]
    n_warmup = cfg["n_warmup"]
    n_samples = cfg["n_samples"]
    num_chains = cfg.get("num_chains", 4)
    effort_str = cfg.get("effort", "low")
    effort = {"low": Effort.LOW, "medium": Effort.MEDIUM, "high": Effort.HIGH}.get(
        effort_str.lower(), Effort.LOW
    )

    # Parse the stem to get sampler + warmup names
    _effort, sampler, warmup, inner, _policy_tag_from_stem = parse_stem(stem)
    policy_tag = cfg.get("policy_tag", _policy_tag_from_stem)

    committed = read_committed_verdict(model, stem)

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"Re-emitting: {model}/{stem}")
    print(f"  sampler={sampler}, warmup={warmup}, inner={inner}")
    print(
        f"  effort={effort_str}, n_warmup={n_warmup}, n_samples={n_samples}, num_chains={num_chains}"
    )
    print(f"  policy_tag={policy_tag}")
    print(
        f"  committed: z={committed.get('max_abs_mean_z'):.4f}, verdict={committed.get('verdict')}"
    )
    print(sep)

    t0 = time.perf_counter()
    result = emit_low_recipe_for_cell(
        model_name=model,
        warmup_name=warmup,
        sampler_name=sampler,
        n_warmup=n_warmup,
        n_samples=n_samples,
        num_chains=num_chains,
        effort=effort,
        target_acceptance=cfg.get("target_acceptance"),
        sampler_kwargs_override=cfg.get("sampler_kwargs_override"),
        warmup_kwargs_override=cfg.get("warmup_kwargs_override"),
        step_policy=cfg.get("step_policy"),
        policy_tag=policy_tag,
        warmup_inner_kernel=inner,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    # Read new gate_evidence from updated recipe (if PASS) or from result
    new_verdict = result.verdict
    recipe_path = CATALOG / model / "recipes" / f"{stem}.json"
    if recipe_path.exists():
        updated = json.loads(recipe_path.read_text())
        ae = updated.get("gate_evidence", {}).get("auto", {})
        new_z = ae.get("max_abs_mean_z")
    else:
        ae = {}
        new_z = None

    # Determine the actual new z from result's CellResult
    if new_z is None and hasattr(result, "max_abs_mean_z"):
        new_z = getattr(result, "max_abs_mean_z", None)

    old_verdict = committed.get("verdict", "?")
    old_z = committed.get("max_abs_mean_z")
    flip = f"{old_verdict}→{new_verdict}" if old_verdict != new_verdict else "unchanged"

    print(
        f"  [RESULT] {old_verdict}(z={old_z:.4f}) → {new_verdict}(z={new_z}) in {elapsed:.1f}s"
    )
    if old_verdict != new_verdict:
        print(f"  *** FLIP: {flip} ***")

    return {
        "model": model,
        "stem": stem,
        "old_verdict": old_verdict,
        "old_z": old_z,
        "new_verdict": new_verdict,
        "new_z": new_z,
        "flip": flip,
        "elapsed_s": round(elapsed, 1),
        "rhat_max": ae.get("rhat_max"),
        "min_bulk_ess": ae.get("min_bulk_ess"),
        "n_divergences": ae.get("n_divergences"),
        "gt_cert_coverage": ae.get("gt_cert_coverage"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--radon-only", action="store_true", help="Only run the radon GPU cell"
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Only run CPU cells (skip radon GPU)"
    )
    args = parser.parse_args()

    os.environ.setdefault("JAX_ENABLE_X64", "1")

    if args.radon_only:
        configs = GPU_CONFIGS
    elif args.cpu_only:
        configs = CPU_CONFIGS
    else:
        configs = CPU_CONFIGS + GPU_CONFIGS

    existing_results = {}
    if RESULTS_FILE.exists():
        existing_results = json.loads(RESULTS_FILE.read_text())

    results = dict(existing_results)

    for cfg in configs:
        key = f"{cfg['model']}/{cfg['stem']}"
        try:
            r = reemit_cell(cfg)
            results[key] = r
        except Exception as exc:
            import traceback

            print(f"  [ERROR] {key}: {exc}")
            traceback.print_exc()
            results[key] = {
                "model": cfg["model"],
                "stem": cfg["stem"],
                "error": str(exc),
                "new_verdict": "ERROR",
            }
        # Save incrementally so a crash doesn't lose earlier results
        RESULTS_FILE.write_text(json.dumps(results, indent=2))

    # Summary
    print("\n" + "=" * 65)
    print("RE-EMISSION SUMMARY")
    print("=" * 65)
    flips = [
        (k, v) for k, v in results.items() if v.get("flip", "unchanged") != "unchanged"
    ]
    unchanged = [
        (k, v) for k, v in results.items() if v.get("flip", "unchanged") == "unchanged"
    ]
    errors = [(k, v) for k, v in results.items() if v.get("new_verdict") == "ERROR"]

    print(
        f"Total: {len(results)} | UNCHANGED: {len(unchanged)} | FLIP: {len(flips)} | ERROR: {len(errors)}"
    )
    if flips:
        print("\nFLIPS:")
        for k, v in sorted(flips, key=lambda x: x[1].get("flip", "")):
            print(
                f"  {k}: {v['flip']} (old_z={v.get('old_z'):.4f}, new_z={v.get('new_z')})"
            )
    if unchanged:
        print("\nUNCHANGED:")
        for k, v in unchanged:
            print(
                f"  {k}: {v.get('new_verdict')} (old_z={v.get('old_z'):.4f}, new_z={v.get('new_z')})"
            )
    if errors:
        print("\nERRORS:")
        for k, v in errors:
            print(f"  {k}: {v.get('error')}")

    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
