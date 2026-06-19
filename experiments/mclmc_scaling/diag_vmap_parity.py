"""#25-followup: vmap-vs-loop multichain DIAGNOSIS on banana (adjusted_mclmc_dynamic).

Question (from user via tl): is vmap-over-chains statistically sound as the standing
multichain path, or must we keep the sequential-loop fallback? The parity gate flagged
max|loop-vmap| ~ 14.8 on banana avg=18. Hypothesis: positive-Lyapunov fp-ordering
amplification (XLA SIMD width-4 vs scalar) on a chaotic curved target -- NOT a code bug.

This script (parts 1-3 of the diagnosis):
  PART 1 fp-ordering curve  : per-step max|loop-vmap| growth from machine-eps, in float64.
                              avg=1 (per-integration-step granularity) + avg=18 (cert config).
                              Fit log|Delta| ~ lambda*step on the growth region.
  PART 2 statistical equiv  : per-chain 2mbias/ESS/rhat/acc/nsteps loop-vs-vmap across seeds,
                              + KS test on pooled marginals (x1,x2) loop-ensemble vs vmap-ensemble.
  PART 3 key identity       : confirm vmap & loop consume identical RNG keys (rule out a
                              key-broadcast difference as a non-fp cause).

Run: JAX_PLATFORM_NAME=cpu .venv/bin/python experiments/mclmc_scaling/diag_vmap_parity.py
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

GIT_HEAD = (
    subprocess.check_output(
        ["git", "-C", "/home/jp/blackjax-devs/blackjax", "rev-parse", "HEAD"]
    )
    .decode()
    .strip()
)
print("git_head:", GIT_HEAD)
print("jax x64 enabled:", jax.config.read("jax_enable_x64"))

BANANA_MEAN = np.array([0.0, 2.0])
BANANA_VAR = np.array([8.0, 9.0])
CH = 4


def load_banana():
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.model._registry import MODELS as _M

    init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M["banana"])
    _, unravel = ravel_pytree(init_dict)
    imm = jnp.asarray(BANANA_VAR)
    return (lambda xf: ld_raw(unravel(xf))), jnp.zeros(2), imm


def ref_step(ld, init, imm, seed=101):
    st = mclmc_mod.init(init, ld, jax.random.key(seed * 7 + 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p, _ = mclmc_find_L_and_step_size(
            mclmc_kernel=_make_fixed_imm_kernel(imm),
            num_steps=5000,
            state=st,
            rng_key=jax.random.key(seed * 7 + 2),
            logdensity_fn=ld,
            diagonal_preconditioning=False,
        )
    return float(p.step_size)


def _chain_keys(seed, ch):
    return [jax.random.key(seed * 1000 + ci + 1) for ci in range(ch)]


def _run_chain_scan(ld, init, imm, dyn, step, avg, sk, n):
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


def sample_loop(ld, init, imm, step, avg, n, ch, seed):
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    pos, divs, accs, nss = [], [], [], []
    for sk in _chain_keys(seed, ch):
        flat, dv, ac, ns = _run_chain_scan(ld, init, imm, dyn, step, avg, sk, n)
        pos.append(np.array(flat))
        divs.append(np.array(dv))
        accs.append(np.array(ac))
        nss.append(np.array(ns))
    return np.stack(pos, 0), np.stack(divs, 0), np.stack(accs, 0), np.stack(nss, 0)


def sample_vmap(ld, init, imm, step, avg, n, ch, seed):
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    keys = jnp.stack(_chain_keys(seed, ch))

    def run_chain(sk):
        return _run_chain_scan(ld, init, imm, dyn, step, avg, sk, n)

    flats, dvs, acs, nss = jax.vmap(run_chain)(keys)
    return np.asarray(flats), np.asarray(dvs), np.asarray(acs), np.asarray(nss)


def per_chain_quality(arr_1chain):
    """arr_1chain: (n, d). Single-chain 2mbias, mean-bias-sd, bulk-ESS, acc not here."""
    a = arr_1chain[None, :, :]  # (1, n, d)
    vm = np.mean((a - BANANA_MEAN[None, None, :]) ** 2, axis=(0, 1))
    b2 = float((np.abs(vm - BANANA_VAR) / BANANA_VAR).max())
    me = a.reshape(-1, a.shape[-1]).mean(0)
    mbsd = float((np.abs(me - BANANA_MEAN) / np.sqrt(BANANA_VAR)).max())
    ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], a)})
    ess = float(np.array(az.ess(ds, method="bulk")["x"]).min())
    return b2, mbsd, ess


ld, init, imm = load_banana()
STEP = ref_step(ld, init, imm, seed=101)
print(f"banana ref_step(seed101) = {STEP:.5f}\n")
out = {"git_head": GIT_HEAD, "x64": True, "step": STEP}

# ===================== PART 3 (first, cheap): KEY IDENTITY =====================
print("=" * 90)
print("PART 3 -- RNG KEY IDENTITY (loop vs vmap)")
ck_list = _chain_keys(101, CH)  # loop: python list of per-chain keys
ck_stack = jnp.stack(ck_list)  # vmap: stacked, vmapped over axis 0
key_match = []
for ci in range(CH):
    loop_split = jax.random.split(ck_list[ci], 50)
    vmap_split = jax.random.split(ck_stack[ci], 50)
    same = bool(
        np.array_equal(
            np.asarray(jax.random.key_data(loop_split)),
            np.asarray(jax.random.key_data(vmap_split)),
        )
    )
    key_match.append(same)
all_keys_identical = all(key_match)
print(
    f"  per-chain key_data identical (loop vs stacked, incl. 50-way split): {key_match}"
)
print(
    f"  ==> ALL KEYS IDENTICAL: {all_keys_identical}  (rules out key-broadcast as a cause)"
)
out["part3_keys_identical"] = all_keys_identical
sys.stdout.flush()

# ===================== PART 1: fp-ORDERING DIVERGENCE CURVE =====================
print("\n" + "=" * 90)
print("PART 1 -- fp-ORDERING DIVERGENCE CURVE (float64), chain 0, seed 101")
out["part1"] = {}
for avg, n in [(1, 400), (18, 400)]:
    aL, _, acL, nsL = sample_loop(ld, init, imm, STEP, avg, n, CH, 101)
    aV, _, acV, nsV = sample_vmap(ld, init, imm, STEP, avg, n, CH, 101)
    d0L, d0V = aL[0], aV[0]  # chain-0 position traj, (n, d)
    dvec = np.max(np.abs(d0L - d0V), axis=1)  # per-sample max-abs position diff
    dvec = np.maximum(dvec, 1e-300)
    # exp fit on the growth region: from first sample to where it first exceeds 1e-2
    sat_idx = int(np.argmax(dvec > 1e-2)) if (dvec > 1e-2).any() else n
    fit_hi = max(5, min(sat_idx, n))
    xs = np.arange(fit_hi)
    lam = (
        float(np.polyfit(xs, np.log(dvec[:fit_hi]), 1)[0])
        if fit_hi >= 3
        else float("nan")
    )
    # acc/nsteps aggregate match (chain 0)
    acc_match = float(np.max(np.abs(acL[0] - acV[0])))
    ns_match = float(np.max(np.abs(nsL[0] - nsV[0])))
    print(f"\n  avg={avg}, n={n}:")
    print(f"    |Delta| at sample 1   = {dvec[0]:.3e}   (machine-eps scale expected)")
    print(f"    |Delta| at sample 5   = {dvec[min(5,n-1)]:.3e}")
    print(f"    |Delta| at sample 20  = {dvec[min(20,n-1)]:.3e}")
    print(f"    |Delta| at sample 100 = {dvec[min(100,n-1)]:.3e}")
    print(f"    |Delta| at sample {n-1} = {dvec[n-1]:.3e}")
    print(f"    saturation (>1e-2) first at sample = {sat_idx}")
    print(
        f"    exp-growth fit lambda (per sample, over growth region [0:{fit_hi}]) = {lam:.4f}"
    )
    print(
        f"    chain-0 per-step acc max|loop-vmap| = {acc_match:.3e}; nsteps max|d| = {ns_match:.3e}"
    )
    # subsampled curve
    idxs = [i for i in range(0, n, max(1, n // 20))]
    print(f"    curve[sample:|Delta|]: " + " ".join(f"{i}:{dvec[i]:.2e}" for i in idxs))
    out["part1"][f"avg{avg}"] = {
        "delta_sample1": float(dvec[0]),
        "delta_last": float(dvec[n - 1]),
        "saturation_sample": sat_idx,
        "lambda_per_sample": lam,
        "acc_maxabs": acc_match,
        "nsteps_maxabs": ns_match,
        "curve": {int(i): float(dvec[i]) for i in idxs},
    }
    sys.stdout.flush()

# ===================== PART 2: STATISTICAL EQUIVALENCE =====================
print("\n" + "=" * 90)
print("PART 2 -- STATISTICAL EQUIVALENCE (loop vs vmap), avg=18, n=2000")
SEEDS = [101, 102, 103]
NS = 2000
AVG = 18
out["part2"] = {"seeds": SEEDS, "n": NS, "avg": AVG, "per_seed": []}
print(
    f"  {'seed':>5} {'path':>5} {'2mbias_pc':>20} {'minESS_pc':>22} {'acc':>7} {'nsteps':>7}"
)
ks_all = {"x1": [], "x2": []}
for seed in SEEDS:
    aL, dL, acL, nsL = sample_loop(ld, init, imm, STEP, AVG, NS, CH, seed)
    aV, dV, acV, nsV = sample_vmap(ld, init, imm, STEP, AVG, NS, CH, seed)
    # per-chain quality both paths
    qL = [per_chain_quality(aL[c]) for c in range(CH)]
    qV = [per_chain_quality(aV[c]) for c in range(CH)]
    b2L = [q[0] for q in qL]
    b2V = [q[0] for q in qV]
    essL = [q[2] for q in qL]
    essV = [q[2] for q in qV]
    accL, accV = float(acL.mean()), float(acV.mean())
    nsL_m, nsV_m = float(nsL.mean()), float(nsV.mean())
    # pooled-ensemble marginal KS (the real "same distribution" test)
    poolL = aL.reshape(-1, 2)
    poolV = aV.reshape(-1, 2)
    ks_x1 = ks_2samp(poolL[:, 0], poolV[:, 0])
    ks_x2 = ks_2samp(poolL[:, 1], poolV[:, 1])
    ks_all["x1"].append((float(ks_x1.statistic), float(ks_x1.pvalue)))
    ks_all["x2"].append((float(ks_x2.statistic), float(ks_x2.pvalue)))
    print(
        f"  {seed:>5} {'loop':>5} {str([round(b,3) for b in b2L]):>20} {str([int(e) for e in essL]):>22} {accL:>7.4f} {nsL_m:>7.3f}"
    )
    print(
        f"  {seed:>5} {'vmap':>5} {str([round(b,3) for b in b2V]):>20} {str([int(e) for e in essV]):>22} {accV:>7.4f} {nsV_m:>7.3f}"
    )
    print(
        f"        KS x1: D={ks_x1.statistic:.4f} p={ks_x1.pvalue:.3f} | KS x2: D={ks_x2.statistic:.4f} p={ks_x2.pvalue:.3f}"
        f" | acc|d|={abs(accL-accV):.2e} nsteps|d|={abs(nsL_m-nsV_m):.2e} div: {float(dL.mean()):.3f}/{float(dV.mean()):.3f}"
    )
    out["part2"]["per_seed"].append(
        {
            "seed": seed,
            "b2_loop": b2L,
            "b2_vmap": b2V,
            "ess_loop": essL,
            "ess_vmap": essV,
            "acc_loop": accL,
            "acc_vmap": accV,
            "nsteps_loop": nsL_m,
            "nsteps_vmap": nsV_m,
            "ks_x1": [float(ks_x1.statistic), float(ks_x1.pvalue)],
            "ks_x2": [float(ks_x2.statistic), float(ks_x2.pvalue)],
            "div_loop": float(dL.mean()),
            "div_vmap": float(dV.mean()),
        }
    )
    sys.stdout.flush()

ks_min_p = min([p for _, p in ks_all["x1"]] + [p for _, p in ks_all["x2"]])
ks_max_D = max([D for D, _ in ks_all["x1"]] + [D for D, _ in ks_all["x2"]])
print(
    f"\n  KS across all seeds/marginals: max D = {ks_max_D:.4f}, min p = {ks_min_p:.3f}"
)
print(f"  (large p / small D => loop & vmap ensembles are the SAME distribution)")
out["part2"]["ks_max_D"] = ks_max_D
out["part2"]["ks_min_p"] = ks_min_p

outp = os.path.join(_HERE, "diag_vmap_parity_results.json")
with open(outp, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote {outp}")
print("DONE_DIAG_VMAP")
