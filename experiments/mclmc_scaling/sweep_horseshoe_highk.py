"""High-k trajectory probe for horseshoe (d=204) — settles constant-vs-adaptive L.

The old pre-#941 recipe ran horseshoe at L=14.28 / step=0.230 => realized avg ~62
(only REVIEW-tier). The main sweep (sweep_stiff_k.py) maxes at k=8 and shows no
recovery. This probe pushes a LARGE CONSTANT k up to 64 (≈ the old operating point)
to answer: does long constant L finally recover mixing (rhat->1), or does it just
trade the rhat blowup for a divergence storm / climbing 2nd-moment bias (=> adaptive
/ position-dependent L is the only real fix)?

Design difference vs the main sweep: FIXED n_samples per chain (not fixed grad budget),
so rhat/ESS/bias are equally well-powered at every k (the fixed-budget design starves
high k to ~50 draws, which would make rhat meaningless). Cost (total grads) is recorded.

Run:  JAX_PLATFORM_NAME=cpu .venv/bin/python experiments/mclmc_scaling/sweep_horseshoe_highk.py
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(os.path.join(_HERE, "..", ".."))

import blackjax.mcmc.adjusted_mclmc_dynamic as adj_dyn_mod
import blackjax.mcmc.mclmc as mclmc_mod
from blackjax.adaptation.mclmc_adaptation import mclmc_find_L_and_step_size
from gt_imm import gt_from_draws
from run_fixed_imm import _make_fixed_imm_adj_dyn_kernel, _make_fixed_imm_kernel

EXPECT_HEAD = "8937e088"
try:
    GIT_HEAD = (
        subprocess.check_output(
            ["git", "-C", "/home/jp/blackjax-devs/blackjax", "rev-parse", "HEAD"]
        )
        .decode()
        .strip()
    )
except Exception as e:  # pragma: no cover
    GIT_HEAD = f"UNKNOWN({e})"
print(
    ("git_head OK: " if GIT_HEAD.startswith(EXPECT_HEAD) else f"!! WARNING off-pin: ")
    + GIT_HEAD
)

K_VALUES = [8, 16, 32, 64]  # 8 = continuity bridge to the main sweep; 62 = old op-point
SEEDS = [0, 1]
N_SAMPLES = 500  # fixed per chain (well-powered rhat at every k)
CH = 4


def ref_step(ld, init, imm):
    st = mclmc_mod.init(init, ld, jax.random.key(11))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p, _ = mclmc_find_L_and_step_size(
            mclmc_kernel=_make_fixed_imm_kernel(imm),
            num_steps=2000,
            state=st,
            rng_key=jax.random.key(12),
            logdensity_fn=ld,
            diagonal_preconditioning=False,
        )
    return float(p.step_size)


def sample(ld, init, imm, step, avg, n, ch, seed):
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    pos, divs, accs, nss = [], [], [], []
    for ci in range(ch):
        sk = jax.random.key(seed * 1000 + ci + 1)
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
    return (
        np.stack(pos, 0),
        float(np.mean(np.concatenate(divs))),
        float(np.mean(np.concatenate(accs))),
        float(np.mean(np.concatenate(nss))),
    )


def quality(arr, gm, gv):
    vm = np.mean((arr - gm[None, None, :]) ** 2, axis=(0, 1))
    max_2mom_bias = float((np.abs(vm - gv) / np.maximum(gv, 1e-30)).max())
    mean_est = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    max_mean_bias_sd = float(
        (np.abs(mean_est - gm) / np.maximum(np.sqrt(gv), 1e-30)).max()
    )
    ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], arr)})
    min_ess = float(np.array(az.ess(ds, method="bulk")["x"]).min())
    max_rhat = float(np.array(az.rhat(ds, method="rank")["x"]).max())
    return max_2mom_bias, min_ess, max_rhat, max_mean_bias_sd


imm, gt_var, gt_mean, d = gt_from_draws("horseshoe")
gm, gv = np.asarray(gt_mean), np.asarray(gt_var)
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model._registry import MODELS as _M

init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M["horseshoe"])
_, unravel = ravel_pytree(init_dict)
ld = lambda xf: ld_raw(unravel(xf))
init = jnp.asarray(gt_mean, dtype=jnp.float64)
rs = ref_step(ld, init, imm)

print(
    f"horseshoe high-k probe | d={d} | ref_step={rs:.3f} | fixed n={N_SAMPLES}/chain x {CH} chains x seeds {SEEDS}"
)
print(
    f"  {'k':>4s} {'realAvg':>7s} {'2mbias':>7s} {'mbias_sd':>8s} {'minESS':>7s} {'Rhat':>7s} {'div':>6s} {'acc':>6s} {'grads/ch':>9s}"
)

cells = []
for k in K_VALUES:
    biases, esss, rhats, divs, accs, avgs, mbsds = [], [], [], [], [], [], []
    for seed in SEEDS:
        arr, dr, ac, mns = sample(ld, init, imm, rs, k, N_SAMPLES, CH, seed)
        mb, me, mr, mbsd = quality(arr, gm, gv)
        biases.append(mb)
        esss.append(me)
        rhats.append(mr)
        divs.append(dr)
        accs.append(ac)
        avgs.append(mns)
        mbsds.append(mbsd)
        cells.append(
            {
                "model": "horseshoe",
                "k": k,
                "seed": int(seed),
                "d": int(d),
                "realized_avg": float(mns),
                "n_samples": N_SAMPLES,
                "max_2mom_bias": float(mb),
                "max_mean_bias_sd": float(mbsd),
                "min_bulk_ess": float(me),
                "max_rhat": float(mr),
                "div_rate": float(dr),
                "mean_acc": float(ac),
                "git_head": GIT_HEAD,
            }
        )
    bias, ess, div, acc = np.mean(biases), np.mean(esss), np.mean(divs), np.mean(accs)
    rhat = np.nanmax(rhats)
    avg_u = np.mean(avgs)
    mbsd_m = float(np.mean(mbsds))
    grads = int(2 * avg_u * N_SAMPLES)
    cells.append(
        {
            "model": "horseshoe",
            "k": k,
            "seed": "AGG",
            "d": int(d),
            "realized_avg": float(avg_u),
            "n_samples": N_SAMPLES,
            "max_2mom_bias": float(bias),
            "max_mean_bias_sd": mbsd_m,
            "min_bulk_ess": float(ess),
            "max_rhat": float(rhat),
            "div_rate": float(div),
            "mean_acc": float(acc),
            "grads_per_chain": grads,
            "n_seeds": len(SEEDS),
            "git_head": GIT_HEAD,
        }
    )
    print(
        f"  {k:>4d} {avg_u:>7.1f} {bias:>7.3f} {mbsd_m:>8.3f} {ess:>7.0f} {rhat:>7.3f} {div:>6.3f} {acc:>6.3f} {grads:>9d}"
    )
    sys.stdout.flush()

out = os.path.join(_HERE, "sweep_horseshoe_highk_results.json")
with open(out, "w") as f:
    json.dump(
        {
            "git_head": GIT_HEAD,
            "k_values": K_VALUES,
            "n_samples": N_SAMPLES,
            "cells": cells,
        },
        f,
        indent=2,
    )
print(f"wrote {out} ({len(cells)} rows)")
print("DONE_HIGHK_PROBE")
