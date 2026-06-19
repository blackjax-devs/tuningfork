"""S1 sweep driver (TL/statistician 'run' layer; consumes SWE's harness).

Sweeps the GT-LRD IMM rank k for ill_cond_50 + mvn_10, running fixed-IMM
unadjusted MCLMC at each k, recording (kappa_eff, step, L, bias, ESS,
ESS/grad, EEVPD, div). The empirical k* = smallest k where sampling quality
(ESS/grad up, bias down) plateaus.

  uv run python sweep_s1.py --smoke   # tiny N, fast sanity of THIS driver
  uv run python sweep_s1.py           # full N
"""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from gt_imm import gt_cov, gt_lrd_imm, kappa_eff
from run_fixed_imm import run_mclmc_fixed_imm

SMOKE = "--smoke" in sys.argv
if SMOKE:
    NW, NS, NC = 20, 50, 2
    plan = {"ill_cond_50": [0, 5, 50], "mvn_10": [0, 10]}
else:
    NW, NS, NC = 2000, 2000, 4
    plan = {
        "ill_cond_50": [0, 1, 3, 5, 9, 16, 28, 40, 49, 50],
        "mvn_10": [0, 5, 10],
    }
SEED = 20260616

print(
    f"S1 sweep | smoke={SMOKE} | n_warmup={NW} n_samples={NS} num_chains={NC} seed={SEED}"
)
print(
    f"{'model':12s} {'k':>3s} {'kappa_eff':>9s} {'step':>7s} {'L':>7s} "
    f"{'maxbias':>8s} {'meanbias':>8s} {'minESS':>8s} {'ess/grad':>9s} {'eevpd':>9s} {'div':>5s} {'wall':>6s}"
)
print("-" * 104)

results = []
for model, ks in plan.items():
    Sigma, _ = gt_cov(model)
    for k in ks:
        t0 = time.time()
        imm = gt_lrd_imm(Sigma, k)
        keff = float(kappa_eff(Sigma, imm))
        r = run_mclmc_fixed_imm(
            model, imm, n_warmup=NW, n_samples=NS, num_chains=NC, seed=SEED
        )
        wall = round(time.time() - t0, 1)
        r.update({"model": model, "k": int(k), "kappa_eff": keff, "wall_s": wall})
        results.append(r)
        print(
            f"{model:12s} {k:>3d} {keff:>9.2f} {r['step_size']:>7.3f} {r['L']:>7.3f} "
            f"{r['max_bias']:>8.4f} {r['mean_bias']:>8.4f} {r['min_bulk_ess']:>8.1f} "
            f"{r['ess_per_grad']:>9.5f} {r['eevpd']:>9.2e} {r['div_rate']:>5.2f} {wall:>6.1f}"
        )
        sys.stdout.flush()

out = os.path.join(_HERE, "s1_results_smoke.json" if SMOKE else "s1_results.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWROTE {out}  ({len(results)} rows)")
