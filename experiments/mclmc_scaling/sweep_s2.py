"""S2 driver: optimal step at fixed GT-dense IMM (per-variant criterion).

Unadjusted (bias-bounded #1): fix IMM=GT-dense (kappa_eff=1) and L=S1-tuned,
sweep step_size; measure 2nd-moment bias vs GT + ess/grad + EEVPD. The
bias-bounded optimum = largest step with max_bias <= tau. Also locate where the
EEVPD=5e-4 tuned step lands on the bias curve (the "is 5e-4 principled" audit).

Adjusted (efficiency-optimal #2): adjusted_mclmc tuned at GT-dense IMM,
target acc 0.9; report (step, L, N, acc, ess/grad). Asymptotically unbiased.

Large NS keeps the no-burn-in transient negligible (<0.5%).

  uv run python sweep_s2.py --smoke
  uv run python sweep_s2.py
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from gt_imm import gt_cov, gt_lrd_imm
from run_fixed_imm import run_mclmc_fixed_imm

SMOKE = "--smoke" in sys.argv
NS = 200 if SMOKE else 5000
NS_ADJ = 200 if SMOKE else 3000
NC = 2 if SMOKE else 4
NW_ADJ = 100 if SMOKE else 2000
SEED = 20260616

cfg = {
    "ill_cond_50": {
        "k": 50,
        "L_fixed": 5.82,
        "steps": (
            [2, 8, 40] if SMOKE else [1, 2, 4, 6, 8, 8.54, 10, 12, 15, 20, 30, 40, 50]
        ),
    },
    "mvn_10": {
        "k": 10,
        "L_fixed": 2.63,
        "steps": [2, 5] if SMOKE else [1, 2, 2.63, 3.66, 5, 7, 10, 15],
    },
}

results = {"unadjusted_step_sweep": [], "adjusted_tuned": []}

print(f"S2 | smoke={SMOKE} | NS={NS} NS_adj={NS_ADJ} NC={NC} seed={SEED}")
print("\n=== UNADJUSTED step sweep (fixed GT-dense IMM, fixed L; bias-bounded) ===")
print(
    f"{'model':12s} {'step':>6s} {'L':>5s} {'maxbias':>8s} {'meanbias':>8s} "
    f"{'minESS':>8s} {'ess/grad':>9s} {'eevpd':>9s} {'div':>5s}"
)
for model, c in cfg.items():
    Sigma, _ = gt_cov(model)
    imm = gt_lrd_imm(Sigma, c["k"])
    for s in c["steps"]:
        r = run_mclmc_fixed_imm(
            model,
            imm,
            n_warmup=0,
            n_samples=NS,
            num_chains=NC,
            seed=SEED,
            fixed_step_size=float(s),
            fixed_L=float(c["L_fixed"]),
        )
        r.update(
            {"model": model, "fixed_step": float(s), "fixed_L": float(c["L_fixed"])}
        )
        results["unadjusted_step_sweep"].append(r)
        print(
            f"{model:12s} {s:>6.2f} {c['L_fixed']:>5.2f} {r['max_bias']:>8.4f} "
            f"{r['mean_bias']:>8.4f} {r['min_bulk_ess']:>8.1f} {r['ess_per_grad']:>9.5f} "
            f"{r['eevpd']:>9.2e} {r['div_rate']:>5.2f}"
        )
        sys.stdout.flush()

print("\n=== ADJUSTED tuned (fixed GT-dense IMM, target acc 0.9; efficiency) ===")
print(
    f"{'model':12s} {'step':>6s} {'L':>6s} {'N':>4s} {'acc':>6s} {'maxbias':>8s} "
    f"{'minESS':>8s} {'ess/grad':>9s}"
)
for model, c in cfg.items():
    Sigma, _ = gt_cov(model)
    imm = gt_lrd_imm(Sigma, c["k"])
    r = run_mclmc_fixed_imm(
        model,
        imm,
        n_warmup=NW_ADJ,
        n_samples=NS_ADJ,
        num_chains=NC,
        seed=SEED,
        adjusted=True,
        floor_factor=1.5,
        adj_target=0.9,
    )
    r.update({"model": model})
    results["adjusted_tuned"].append(r)
    print(
        f"{model:12s} {r['step_size']:>6.2f} {r['L']:>6.2f} {r['n_steps_median']:>4.0f} "
        f"{r['acceptance_rate']:>6.3f} {r['max_bias']:>8.4f} {r['min_bulk_ess']:>8.1f} "
        f"{r['ess_per_grad']:>9.5f}"
    )
    sys.stdout.flush()

out = os.path.join(_HERE, "s2_results_smoke.json" if SMOKE else "s2_results.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWROTE {out}")
