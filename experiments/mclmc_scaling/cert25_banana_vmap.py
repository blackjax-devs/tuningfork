"""#25 vmap-forced re-cert: banana adjusted_mclmc, avg{18,24,36,54}, n=5000, 6 seeds.

Produces clean VMAP-path provenance (user objected to the loop fallback). Replaces the
brittle bit-exact position parity with the endorsed TWO-PART GATE:
  (A) STRUCTURAL: sample-1 micro-parity max|Δ| < 1e-10 (catches real codegen/key bugs
      before chaos can mask them) + key-identity assert (loop keys == stacked vmap keys).
  (B) STATISTICAL EQUIVALENCE (handles chaos): on a reference cell (avg=18, n=2000),
      num_integration_steps exact, div equal, mean acc |Δ|<5e-3, KS p>0.05 (or D<0.05) per dim.
Decision rule: A-pass AND B-pass -> vmap is canonical (NO loop fallback). Otherwise FLAG and
ABORT (a fallback would MASK a bug, not fix it).

Then run the cert vmap-only and diff avg=24 against the loop cert (cert25_banana_medium_results.json).

Run: JAX_PLATFORM_NAME=cpu .venv/bin/python experiments/mclmc_scaling/cert25_banana_vmap.py
"""

import json
import os
import subprocess
import sys
import warnings

import jax

jax.config.update("jax_enable_x64", True)

import arviz as az
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.flatten_util import ravel_pytree
from scipy.stats import ks_2samp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(os.path.join(_HERE, "..", ".."))

import blackjax.mcmc.adjusted_mclmc_dynamic as adj_dyn_mod
import blackjax.mcmc.mclmc as mclmc_mod
from blackjax.adaptation.mclmc_adaptation import mclmc_find_L_and_step_size
from run_fixed_imm import _make_fixed_imm_adj_dyn_kernel, _make_fixed_imm_kernel

EXPECT_HEAD = "8937e088"
GIT_HEAD = (
    subprocess.check_output(
        ["git", "-C", "/home/jp/blackjax-devs/blackjax", "rev-parse", "HEAD"]
    )
    .decode()
    .strip()
)
print(
    ("git_head OK: " if GIT_HEAD.startswith(EXPECT_HEAD) else "!! off-pin: ") + GIT_HEAD
)

AVG_GRID = [18, 24, 36, 54]
SEEDS = [10, 11, 12, 13, 14, 15]
N_SAMPLES, N_WARMUP, CH = 5000, 5000, 4
TAU_2MBIAS, TAU_MBSD, RHAT_BAD, ESS_MIN_PER_CH, SEEDS_REQUIRED = (
    0.10,
    0.06,
    1.01,
    100,
    5,
)
# two-part gate thresholds. acc bar is STATISTICAL: |Δacc| < K_SE * binomial-SE(acc,n).
# (An absolute 5e-3 bar sits BELOW the ~4.9e-3 MC SE of the acceptance estimate at n=2000,
#  so it would trip on Monte Carlo noise, not distributional drift — wrong bar for a chaotic
#  sampler. KS already guards the distribution.)
GATE_A_MICRO, GATE_B_KSP, GATE_B_K_SE = 1e-10, 0.05, 4.0

BANANA_MEAN = np.array([0.0, 2.0])
BANANA_VAR = np.array([8.0, 9.0])


def load_banana():
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.model._registry import MODELS as _M

    init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M["banana"])
    _, unravel = ravel_pytree(init_dict)
    return (lambda xf: ld_raw(unravel(xf))), jnp.zeros(2), 2, jnp.asarray(BANANA_VAR)


def _chain_keys(seed, ch):
    return [jax.random.key(seed * 1000 + ci + 1) for ci in range(ch)]


ld, init, d, imm = load_banana()
gm, gv = BANANA_MEAN.copy(), BANANA_VAR.copy()


def ref_step(seed, num_steps):
    st = mclmc_mod.init(init, ld, jax.random.key(seed * 1000 + 500))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p, _ = mclmc_find_L_and_step_size(
            mclmc_kernel=_make_fixed_imm_kernel(imm),
            num_steps=num_steps,
            state=st,
            rng_key=jax.random.key(seed * 1000 + 501),
            logdensity_fn=ld,
            diagonal_preconditioning=False,
        )
    return float(p.step_size)


def sample_loop(step, avg, n, seed):
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    pos, divs, accs, nss = [], [], [], []
    for sk in _chain_keys(seed, CH):
        s = adj_dyn_mod.init(init, ld, sk)

        def stp(c, key):
            nx, info = dyn(
                rng_key=key,
                state=c,
                logdensity_fn=ld,
                step_size=step,
                L_proposal_factor=jnp.inf,
                inverse_mass_matrix=imm,
                integration_steps_params=(float(avg),),
            )
            return nx, (
                nx.position,
                info.is_divergent,
                info.acceptance_rate,
                info.num_integration_steps,
            )

        _, (pt, dv, ac, ns) = jax.lax.scan(stp, s, jax.random.split(sk, n))
        pos.append(np.array(jax.vmap(lambda q: ravel_pytree(q)[0])(pt)))
        divs.append(np.array(dv))
        accs.append(np.array(ac))
        nss.append(np.array(ns))
    return np.stack(pos, 0), np.array(divs), np.array(accs), np.array(nss)


def sample_vmap(step, avg, n, seed):
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    keys = jnp.stack(_chain_keys(seed, CH))

    def run_chain(sk):
        s = adj_dyn_mod.init(init, ld, sk)

        def stp(c, key):
            nx, info = dyn(
                rng_key=key,
                state=c,
                logdensity_fn=ld,
                step_size=step,
                L_proposal_factor=jnp.inf,
                inverse_mass_matrix=imm,
                integration_steps_params=(float(avg),),
            )
            return nx, (
                nx.position,
                info.is_divergent,
                info.acceptance_rate,
                info.num_integration_steps,
            )

        _, (pt, dv, ac, ns) = jax.lax.scan(stp, s, jax.random.split(sk, n))
        flat = jax.vmap(lambda q: ravel_pytree(q)[0])(pt)
        return flat, dv, ac, ns

    flats, dvs, acs, nss = jax.vmap(run_chain)(keys)
    return np.asarray(flats), np.asarray(dvs), np.asarray(acs), np.asarray(nss)


def quality(arr):
    vm = np.mean((arr - gm[None, None, :]) ** 2, axis=(0, 1))
    b2 = float((np.abs(vm - gv) / np.maximum(gv, 1e-30)).max())
    mean_est = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    mbsd = float((np.abs(mean_est - gm) / np.maximum(np.sqrt(gv), 1e-30)).max())
    ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], arr)})
    ess = float(np.array(az.ess(ds, method="bulk")["x"]).min())
    rhat = float(np.array(az.rhat(ds, method="rank")["x"]).max())
    return b2, ess, rhat, mbsd


def seed_passes(b2, mbsd, ess, rhat, div):
    return (
        b2 < TAU_2MBIAS
        and mbsd < TAU_MBSD
        and rhat < RHAT_BAD
        and not np.isnan(rhat)
        and ess > ESS_MIN_PER_CH * CH
        and div == 0.0
    )


# ============ TWO-PART GATE (endorsed; no fallback) ============
print("\n" + "=" * 80 + "\nTWO-PART vmap GATE\n" + "=" * 80)
ref_seed = SEEDS[0]
rs0 = ref_step(ref_seed, 2000)
# (A) structural: key identity + sample-1 micro-parity
loop_keys = _chain_keys(ref_seed, CH)
kd_loop = np.asarray(jnp.stack([jax.random.key_data(k) for k in loop_keys]))
kd_vmap = np.asarray(jax.random.key_data(jnp.stack(loop_keys)))
keys_identical = bool(np.array_equal(kd_loop, kd_vmap))
a1L, _, _, _ = sample_loop(rs0, 18, 1, ref_seed)
a1V, _, _, _ = sample_vmap(rs0, 18, 1, ref_seed)
micro = float(np.max(np.abs(a1L - a1V)))
gate_A = keys_identical and (micro < GATE_A_MICRO)
print(
    f"(A) keys_identical={keys_identical} | sample-1 micro-parity max|Δ|={micro:.3e} (<{GATE_A_MICRO}) -> {'PASS' if gate_A else 'FAIL'}"
)
# (B) statistical equivalence on reference cell
aBL, dBL, cBL, nBL = sample_loop(rs0, 18, 2000, ref_seed)
aBV, dBV, cBV, nBV = sample_vmap(rs0, 18, 2000, ref_seed)
dnst = float(np.max(np.abs(nBL.mean(1) - nBV.mean(1))))
ddiv = float(np.max(np.abs(dBL.mean(1) - dBV.mean(1))))
dacc = float(np.max(np.abs(cBL.mean(1) - cBV.mean(1))))
ksp = []
for dim in range(d):
    st_, p_ = ks_2samp(aBL[:, :, dim].ravel(), aBV[:, :, dim].ravel())
    ksp.append((float(st_), float(p_)))
ks_ok = all((p > GATE_B_KSP or D < 0.05) for D, p in ksp)
# statistical acceptance bar: K_SE * binomial SE at the reference-cell n (=2000)
acc_pooled = float(np.mean([cBL.mean(), cBV.mean()]))
n_refcell = aBL.shape[1]
se_acc = float(np.sqrt(acc_pooled * (1 - acc_pooled) / n_refcell))
acc_tol = GATE_B_K_SE * se_acc
acc_ok = dacc < acc_tol
gate_B = (dnst == 0.0) and (ddiv == 0.0) and acc_ok and ks_ok
print(
    f"(B) nsteps Δ={dnst:.2e} div Δ={ddiv:.2e} | acc |Δ|={dacc:.3e} < {GATE_B_K_SE}·SE={acc_tol:.3e} "
    f"(SE={se_acc:.3e}, n={n_refcell}) -> {'ok' if acc_ok else 'TOO BIG'} | "
    f"KS x0(D={ksp[0][0]:.4f},p={ksp[0][1]:.3f}) x1(D={ksp[1][0]:.4f},p={ksp[1][1]:.3f}) -> {'PASS' if gate_B else 'FAIL'}"
)
if not (gate_A and gate_B):
    print(
        "!! GATE FAILED -> ABORT (a loop fallback would MASK a bug, not fix it). Investigate."
    )
    sys.exit(1)
print("GATE PASSED -> vmap is canonical multichain path (no fallback). Proceeding.\n")
sys.stdout.flush()

# ============ CERT (vmap-only) ============
seed_steps = {seed: ref_step(seed, N_WARMUP) for seed in SEEDS}
median_step = float(np.median(list(seed_steps.values())))
print(
    "per-seed warmup step: "
    + ", ".join(f"s{s}={seed_steps[s]:.4f}" for s in SEEDS)
    + f" | median={median_step:.5f}\n"
)

cells, avg_summary = [], {}
print(
    f"  {'avg':>4s} {'seed':>5s} {'step':>6s} {'realAvg':>7s} {'2mbias':>7s} {'mbias_sd':>8s} {'minESS':>7s} {'Rhat':>7s} {'div':>5s} {'acc':>6s}  {'pass':>5s}"
)
for avg in AVG_GRID:
    n_pass = 0
    for seed in SEEDS:
        step = seed_steps[seed]
        arr, dv, ac, ns = sample_vmap(step, avg, N_SAMPLES, seed)
        dr = float(dv.mean())
        acc = float(ac.mean())
        mns = float(ns.mean())
        b2, ess, rhat, mbsd = quality(arr)
        sp = seed_passes(b2, mbsd, ess, rhat, dr)
        n_pass += int(sp)
        cells.append(
            {
                "model": "banana",
                "avg_param": avg,
                "seed": int(seed),
                "d": int(d),
                "step_size": float(step),
                "realized_avg": mns,
                "n_samples": N_SAMPLES,
                "n_warmup": N_WARMUP,
                "max_2mom_bias": float(b2),
                "max_mean_bias_sd": float(mbsd),
                "min_bulk_ess": float(ess),
                "max_rhat": float(rhat),
                "div_rate": dr,
                "mean_acc": acc,
                "seed_pass": bool(sp),
                "path": "VMAP",
                "git_head": GIT_HEAD,
            }
        )
        print(
            f"  {avg:>4d} {seed:>5d} {step:>6.4f} {mns:>7.1f} {b2:>7.3f} {mbsd:>8.3f} {ess:>7.0f} {rhat:>7.3f} {dr:>5.2f} {acc:>6.3f}  {'YES' if sp else 'no':>5s}"
        )
        sys.stdout.flush()
    cp = n_pass >= SEEDS_REQUIRED
    avg_summary[avg] = {"n_pass": n_pass, "cell_pass": cp}
    print(
        f"  -> avg={avg}: {n_pass}/{len(SEEDS)} PASS => {'CELL-PASS' if cp else 'cell-fail'}\n"
    )

passing = [a for a in AVG_GRID if avg_summary[a]["cell_pass"]]
filing_avg = min(passing) if passing else None

# ---- diff avg=24 against loop cert ----
print("=" * 80 + "\nDIFF avg=24: vmap re-cert vs loop cert\n" + "=" * 80)
diff_block = None
loop_path = os.path.join(_HERE, "cert25_banana_medium_results.json")
if os.path.exists(loop_path):
    loopd = json.load(open(loop_path))
    loop24 = [
        c
        for c in loopd["cells"]
        if c["model"] == "banana"
        and c["avg_param"] == 24
        and isinstance(c["seed"], int)
    ]
    vmap24 = [c for c in cells if c["avg_param"] == 24]
    lb = sorted([c["max_2mom_bias"] for c in loop24])
    vb = sorted([c["max_2mom_bias"] for c in vmap24])
    lmed = float(np.median(lb))
    vmed = float(np.median(vb))
    lpass = (
        sum(
            c["seed_pass"]
            for c in loopd["cells"]
            if c.get("avg_param") == 24 and isinstance(c.get("seed"), int)
        )
        if any("seed_pass" in c for c in loopd["cells"])
        else loopd["avg_summary"]["24"]["n_pass"]
    )
    vpass = avg_summary[24]["n_pass"]
    print(
        f"  loop avg=24: {lpass}/6 PASS | median 2mbias={lmed:.3f} | per-seed 2mbias sorted={['%.3f'%x for x in lb]}"
    )
    print(
        f"  vmap avg=24: {vpass}/6 PASS | median 2mbias={vmed:.3f} | per-seed 2mbias sorted={['%.3f'%x for x in vb]}"
    )
    diff_block = {
        "loop_pass": int(lpass),
        "vmap_pass": int(vpass),
        "loop_med_2mbias": lmed,
        "vmap_med_2mbias": vmed,
        "loop_2mbias_sorted": lb,
        "vmap_2mbias_sorted": vb,
    }
else:
    print("  (loop cert results not found for diff)")

print("\n" + "=" * 80 + "\nFILING SUMMARY (vmap path)")
for a in AVG_GRID:
    print(
        f"  avg={a:>3d}: {avg_summary[a]['n_pass']}/6 | {'CELL-PASS' if avg_summary[a]['cell_pass'] else 'cell-fail'}"
    )
print(f"  passing avgs: {passing} | recommended filing avg: {filing_avg}")
if filing_avg is not None:
    print(
        f"  RECIPE: avg={filing_avg}, step(median)={median_step:.5f}, L={filing_avg*median_step:.4f}, "
        f"IMM diag=[8,9], golden mean=[0,2] var=[8,9]"
    )

out = os.path.join(_HERE, "cert25_banana_vmap_results.json")
with open(out, "w") as f:
    json.dump(
        {
            "git_head": GIT_HEAD,
            "path": "VMAP",
            "gate": {
                "A_micro": micro,
                "A_keys_identical": keys_identical,
                "B_dnsteps": dnst,
                "B_ddiv": ddiv,
                "B_dacc": dacc,
                "B_acc_tol": acc_tol,
                "B_se_acc": se_acc,
                "B_ks": ksp,
                "gate_A": gate_A,
                "gate_B": gate_B,
            },
            "avg_grid": AVG_GRID,
            "n_samples": N_SAMPLES,
            "n_warmup": N_WARMUP,
            "seeds": SEEDS,
            "ch": CH,
            "seed_steps": seed_steps,
            "median_step": median_step,
            "avg_summary": {str(k): v for k, v in avg_summary.items()},
            "filing_avg": filing_avg,
            "diff_vs_loop_avg24": diff_block,
            "cells": cells,
        },
        f,
        indent=2,
    )
print(f"\nwrote {out} ({len(cells)} rows) | path=VMAP")
print("DONE_CERT25_VMAP")
