"""Focused avg=k trajectory sweep for the two stiff/curved models (banana, horseshoe).

Regenerates the per-k table tl/user asked for after the original sweep_robust_traj.py
stdout was lost (that script only printed). Differences from sweep_robust_traj.py:
  - MODELS = [banana, horseshoe] only.
  - banana gets a NEW analytic-GT branch (it has no catalog draws): x1~N(0,8),
    x2|x1~N(x1^2/4,1) => mean=[0,2], var=[8,9], Cov(x1,x2)=0; IMM = marginal cov diag([8,9]).
  - stdout is redirected (by the launcher) to a committed log, AND every per-cell row
    (aggregated + per-seed) is dumped to sweep_stiff_k_results.json with the blackjax git_head.

Strategies (unchanged): S0_mala(avg=1) / S_short(avg=2) / S_long(avg=8) /
S_search(grid{1,2,4,8}, ess/grad) / S_smallstep(avg=8, step*0.5).
Fixed sampling-grad budget; horseshoe d=204 -> budget=3000, seeds[0,1]; banana d=2 -> budget=20000, seeds[0,1,2].

Run:  JAX_PLATFORM_NAME=cpu .venv/bin/python experiments/mclmc_scaling/sweep_stiff_k.py
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
from gt_imm import gt_cov, gt_from_draws, gt_lrd_imm
from run_fixed_imm import _make_fixed_imm_adj_dyn_kernel, _make_fixed_imm_kernel

# --- git_head guard (must be the avg=2 fix pin) ------------------------------
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
if not GIT_HEAD.startswith(EXPECT_HEAD):
    print(
        f"!! WARNING: blackjax HEAD={GIT_HEAD} does NOT start with {EXPECT_HEAD} — results NOT on the cert pin"
    )
else:
    print(f"git_head OK: {GIT_HEAD}")

DIAG = "--diag" in sys.argv
MODELS = ["banana", "horseshoe"]
SEEDS = [0, 1, 2]
BUDGET = 20000
CH = 4

# (avg, step_scale); avg="search" => ess/grad grid {1,2,4,8}
STRATEGIES = {
    "S0_mala": (1, 1.0),
    "S_short": (2, 1.0),
    "S_long": (8, 1.0),
    "S_search": ("search", 1.0),
    "S_smallstep": (8, 0.5),
}
SEARCH_GRID = [1, 2, 4, 8]
TAU_BIAS, RHAT_BAD, ESS_MIN = 0.10, 1.01, 100  # ESS_MIN per chain

# banana analytic GT (no catalog draws): x1~N(0,8); x2|x1~N(x1^2/4,1)
#   E[x2]=Var(x1)/4=2 ; Var(x2)=Var(x1^2)/16+1 = 2*8^2/16+1 = 9 ; Cov(x1,x2)=0
BANANA_MEAN = np.array([0.0, 2.0])
BANANA_VAR = np.array([8.0, 9.0])


def load(model):
    if model in ("mvn_10", "ill_cond_50"):
        Sigma, _ = gt_cov(model)
        d = Sigma.shape[0]
        Sinv = jnp.asarray(np.linalg.inv(Sigma))
        imm = jnp.asarray(np.diag(Sigma)) if DIAG else gt_lrd_imm(Sigma, d)
        return (
            (lambda x: -0.5 * jnp.dot(x, Sinv @ x)),
            jnp.zeros(d),
            d,
            imm,
            np.zeros(d),
            np.diag(Sigma),
        )
    if model == "banana":
        # Real curved banana logdensity; IMM = marginal-cov diagonal (the "GT dense" analog
        # for an uncorrelated marginal). Curvature is NOT captured by a constant metric —
        # that is exactly what the trajectory-length strategies are being tested against.
        from tuningfork.model._numpyro import build_logdensity_fn
        from tuningfork.model._registry import MODELS as _M

        init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M["banana"])
        _, unravel = ravel_pytree(init_dict)
        d = 2
        # Marginal cov is diagonal (Cov(x1,x2)=0), so the exact GT metric is the 1-D
        # diagonal IMM = marginal variances [8,9]. (gt_lrd_imm is for dense covariances.)
        imm = jnp.asarray(BANANA_VAR)
        return (
            (lambda xf: ld_raw(unravel(xf))),
            jnp.zeros(d),
            d,
            imm,
            BANANA_MEAN.copy(),
            BANANA_VAR.copy(),
        )
    imm_dense, gt_var, gt_mean, d = gt_from_draws(model)
    imm = jnp.asarray(gt_var) if DIAG else imm_dense
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.model._registry import MODELS as _M

    init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M[model])
    _, unravel = ravel_pytree(init_dict)
    return (
        (lambda xf: ld_raw(unravel(xf))),
        jnp.asarray(gt_mean, dtype=jnp.float64),
        d,
        imm,
        np.asarray(gt_mean),
        np.asarray(gt_var),
    )


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
    """Return positions (ch,n,d), div_rate, mean_acc, mean_nsteps."""
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
    """max 2nd-moment (variance) bias, min bulk-ESS, max split-Rhat, + max mean-bias/sd."""
    vm = np.mean((arr - gm[None, None, :]) ** 2, axis=(0, 1))
    max_bias = float((np.abs(vm - gv) / np.maximum(gv, 1e-30)).max())
    # ESS-independent mean-bias in sd units (the cert metric), for cross-reference
    mean_est = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    max_mean_bias_sd = float(
        (np.abs(mean_est - gm) / np.maximum(np.sqrt(gv), 1e-30)).max()
    )
    ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], arr)})
    min_ess = float(np.array(az.ess(ds, method="bulk")["x"]).min())
    max_rhat = float(np.array(az.rhat(ds, method="rank")["x"]).max())
    return max_bias, min_ess, max_rhat, max_mean_bias_sd


def search_avg(ld, init, imm, step, ch, seed):
    """B1-style: pick avg maximizing ess/grad on a short pilot."""
    best, best_eg = 1, -1.0
    for a in SEARCH_GRID:
        arr, _, _, mns = sample(ld, init, imm, step, a, 300, ch, seed + 99)
        ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], arr)})
        me = float(np.array(az.ess(ds, method="bulk")["x"]).min())
        eg = me / max(2 * mns * 500 * ch, 1)
        if eg > best_eg:
            best_eg, best = eg, a
    return best


print(
    f"sweep_stiff_k | IMM={'DIAGONAL' if DIAG else 'GT-dense'} | budget={BUDGET} grads/chain | {CH} chains"
)
print(
    f"classify: pass (bias<{TAU_BIAS}, rhat<{RHAT_BAD}, minESS>{ESS_MIN}/chain) | "
    f"loud-fail (bias high AND rhat>{RHAT_BAD} or div>0.01) | silent-fail (bias high, no flag)\n"
)

scoreboard = {}
cells = []  # dumped to json
for model in MODELS:
    ld, init, d, imm, gm, gv = load(model)
    rs = ref_step(ld, init, imm)
    budget = 3000 if d > 200 else (6000 if d > 50 else BUDGET)
    seeds = [0, 1] if d > 50 else SEEDS
    print(
        f"{'='*98}\n{model} (d={d}) | ref_step={rs:.3f} | budget={budget} seeds={seeds}"
    )
    print(
        f"  {'strategy':12s} {'avg':>4s} {'N':>6s} {'2mbias':>7s} {'mbias_sd':>8s} {'minESS':>7s} {'Rhat':>6s} "
        f"{'div':>6s} {'acc':>6s}  {'verdict':>11s}"
    )
    for sname, (aspec, ss) in STRATEGIES.items():
        biases, esss, rhats, divs, accs, avgs_used, mbsds = [], [], [], [], [], [], []
        for seed in seeds:
            step = ss * rs
            avg = (
                search_avg(ld, init, imm, step, CH, seed)
                if aspec == "search"
                else aspec
            )
            n = max(int(budget / (2 * avg)), 50)
            arr, dr, ac, mns = sample(ld, init, imm, step, avg, n, CH, seed)
            mb, me, mr, mbsd = quality(arr, gm, gv)
            biases.append(mb)
            esss.append(me)
            rhats.append(mr)
            divs.append(dr)
            accs.append(ac)
            avgs_used.append(mns)
            mbsds.append(mbsd)
            cells.append(
                {
                    "model": model,
                    "strategy": sname,
                    "seed": int(seed),
                    "d": int(d),
                    "realized_avg": float(mns),
                    "n_steps": int(n),
                    "min_bulk_ess": float(me),
                    "max_rhat": float(mr),
                    "max_2mom_bias": float(mb),
                    "max_mean_bias_sd": float(mbsd),
                    "div_rate": float(dr),
                    "mean_acc": float(ac),
                    "git_head": GIT_HEAD,
                }
            )
        bias, ess, div, acc = (
            np.mean(biases),
            np.mean(esss),
            np.mean(divs),
            np.mean(accs),
        )
        mbsd_mean = float(np.mean(mbsds))
        rhat = np.nanmax(rhats) if not np.all(np.isnan(rhats)) else np.nan
        avg_u = np.mean(avgs_used)
        n_u = max(int(budget / (2 * max(avg_u, 1e-9))), 50)
        bias_ok = bias < TAU_BIAS
        rhat_ok = (rhat < RHAT_BAD) and not np.isnan(rhat)
        ess_ok = ess > ESS_MIN * CH
        loud_signal = (rhat > RHAT_BAD) or np.isnan(rhat) or (div > 0.01) or (acc < 0.1)
        if bias_ok and rhat_ok and ess_ok:
            verdict = "PASS"
        elif (not bias_ok) and loud_signal:
            verdict = "loud-fail"
        elif not bias_ok:
            verdict = "SILENT-FAIL"
        else:
            verdict = "marginal"
        scoreboard[(model, sname)] = verdict
        cells.append(
            {
                "model": model,
                "strategy": sname,
                "seed": "AGG",
                "d": int(d),
                "realized_avg": float(avg_u),
                "n_steps": int(n_u),
                "min_bulk_ess": float(ess),
                "max_rhat": float(rhat) if not np.isnan(rhat) else None,
                "max_2mom_bias": float(bias),
                "max_mean_bias_sd": mbsd_mean,
                "div_rate": float(div),
                "mean_acc": float(acc),
                "verdict": verdict,
                "n_seeds": len(seeds),
                "git_head": GIT_HEAD,
            }
        )
        print(
            f"  {sname:12s} {avg_u:>4.1f} {n_u:>6d} {bias:>7.3f} {mbsd_mean:>8.3f} {ess:>7.0f} {rhat:>6.3f} "
            f"{div:>6.3f} {acc:>6.3f}  {verdict:>11s}"
        )
        sys.stdout.flush()
    print()

out_path = os.path.join(_HERE, "sweep_stiff_k_results.json")
with open(out_path, "w") as f:
    json.dump(
        {"git_head": GIT_HEAD, "strategies": STRATEGIES, "cells": cells}, f, indent=2
    )
print(f"wrote {out_path}  ({len(cells)} rows)")

print(f"{'='*98}\nSCOREBOARD (verdict per strategy)")
for sname in STRATEGIES:
    print(
        f"  {sname:12s} " + "  ".join(f"{m}={scoreboard[(m, sname)]}" for m in MODELS)
    )
print("DONE_SWEEP_STIFF_K")
