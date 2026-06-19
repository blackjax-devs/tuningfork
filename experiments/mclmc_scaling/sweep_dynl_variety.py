"""Dynamic-L variety sweep: adjusted_mclmc_dynamic across avg ladder (2,6,18,54,108).

User-approved config (via tl): 4 models [banana, horseshoe, mvn_10, ill_cond_50],
4 chains VMAPPED x 3 seeds, fixed n=500/chain, GT-dense IMM. Gate on 2nd-moment bias
vs the independent reference (NOT ess/grad — per the #22 design constraint). Per cell:
realized_avg, minESS, rhat, 2mbias, mbias_sd, div, acc.

vmap is a new code path => PARITY GATE at startup: vmapped 4-chain result must match the
sequential-loop result to fp tolerance (same per-chain RNG keys). If parity fails, fall
back to the sequential loop rather than ship a vmap bug.

Run: JAX_PLATFORM_NAME=cpu .venv/bin/python experiments/mclmc_scaling/sweep_dynl_variety.py
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
    ("git_head OK: " if GIT_HEAD.startswith(EXPECT_HEAD) else "!! WARNING off-pin: ")
    + GIT_HEAD
)

MODELS = ["banana", "mvn_10", "ill_cond_50", "horseshoe"]
AVG_LADDER = [2, 6, 18, 54, 108]
SEEDS = [0, 1, 2]
N_SAMPLES = 500
CH = 4
TAU_BIAS, RHAT_BAD, ESS_MIN = 0.10, 1.01, 100

BANANA_MEAN = np.array([0.0, 2.0])
BANANA_VAR = np.array([8.0, 9.0])


def load(model):
    if model in ("mvn_10", "ill_cond_50"):
        Sigma, _ = gt_cov(model)
        d = Sigma.shape[0]
        Sinv = jnp.asarray(np.linalg.inv(Sigma))
        imm = gt_lrd_imm(Sigma, d)
        return (
            (lambda x: -0.5 * jnp.dot(x, Sinv @ x)),
            jnp.zeros(d),
            d,
            imm,
            np.zeros(d),
            np.diag(Sigma),
        )
    if model == "banana":
        from tuningfork.model._numpyro import build_logdensity_fn
        from tuningfork.model._registry import MODELS as _M

        init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M["banana"])
        _, unravel = ravel_pytree(init_dict)
        d = 2
        imm = jnp.asarray(BANANA_VAR)  # exact diagonal marginal cov (Cov(x1,x2)=0)
        return (
            (lambda xf: ld_raw(unravel(xf))),
            jnp.zeros(d),
            d,
            imm,
            BANANA_MEAN.copy(),
            BANANA_VAR.copy(),
        )
    imm_dense, gt_var, gt_mean, d = gt_from_draws(model)
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.model._registry import MODELS as _M

    init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M[model])
    _, unravel = ravel_pytree(init_dict)
    return (
        (lambda xf: ld_raw(unravel(xf))),
        jnp.asarray(gt_mean, dtype=jnp.float64),
        d,
        imm_dense,
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


def _chain_keys(seed, ch):
    return [jax.random.key(seed * 1000 + ci + 1) for ci in range(ch)]


def sample_loop(ld, init, imm, step, avg, n, ch, seed):
    """Original sequential per-chain loop (parity reference)."""
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    pos, divs, accs, nss = [], [], [], []
    for sk in _chain_keys(seed, ch):
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


def sample_vmap(ld, init, imm, step, avg, n, ch, seed):
    """Vmapped 4-chain version. Same per-chain keys as sample_loop => parity expected."""
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    keys = jnp.stack(_chain_keys(seed, ch))

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
        flat = jax.vmap(lambda q: ravel_pytree(q)[0])(pt)  # (n, d)
        return flat, dv, ac, ns

    flats, dvs, acs, nss = jax.vmap(run_chain)(keys)  # leading dim = ch
    arr = np.asarray(flats)  # (ch, n, d)
    return (
        arr,
        float(np.mean(np.asarray(dvs))),
        float(np.mean(np.asarray(acs))),
        float(np.mean(np.asarray(nss))),
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


# ---- PARITY GATE: vmap must match loop on one cell (mvn_10, avg=2, n=200) ----
print("PARITY CHECK: vmap vs sequential loop (mvn_10, avg=2, n=200, 4 chains)...")
_ld, _init, _d, _imm, _gm, _gv = load("mvn_10")
_rs = ref_step(_ld, _init, _imm)
_a_loop, _dl, _al, _nl = sample_loop(_ld, _init, _imm, _rs, 2, 200, CH, 0)
_a_vmap, _dv, _av, _nv = sample_vmap(_ld, _init, _imm, _rs, 2, 200, CH, 0)
_max_abs = float(np.max(np.abs(_a_loop - _a_vmap)))
_stat_close = abs(_dl - _dv) < 1e-9 and abs(_al - _av) < 1e-9 and abs(_nl - _nv) < 1e-9
print(
    f"  positions max|loop-vmap| = {_max_abs:.3e} | div {_dl:.6f}/{_dv:.6f} acc {_al:.6f}/{_av:.6f} nsteps {_nl:.4f}/{_nv:.4f}"
)
USE_VMAP = (_max_abs < 1e-8) and _stat_close
if USE_VMAP:
    print("  PARITY OK -> using vmap (fast path)")
    sample = sample_vmap
else:
    print(
        f"  !! PARITY FAILED (max|d|={_max_abs:.3e}) -> FALLING BACK to sequential loop"
    )
    sample = sample_loop
sys.stdout.flush()

print(
    f"\nsweep_dynl_variety | path={'VMAP' if USE_VMAP else 'LOOP'} | n={N_SAMPLES}/chain x {CH}ch x seeds {SEEDS} | ladder={AVG_LADDER}"
)
print(
    f"gate: 2nd-moment bias vs independent reference (TAU_BIAS={TAU_BIAS}); NOT ess/grad\n"
)

scoreboard = {}
cells = []
for model in MODELS:
    ld, init, d, imm, gm, gv = load(model)
    rs = ref_step(ld, init, imm)
    print(f"{'='*100}\n{model} (d={d}) | ref_step={rs:.3f}")
    print(
        f"  {'avg':>4s} {'realAvg':>7s} {'2mbias':>7s} {'mbias_sd':>8s} {'minESS':>7s} {'Rhat':>7s} {'div':>6s} {'acc':>6s}  {'verdict':>11s}"
    )
    for avg in AVG_LADDER:
        biases, esss, rhats, divs, accs, ravg, mbsds = [], [], [], [], [], [], []
        for seed in SEEDS:
            arr, dr, ac, mns = sample(ld, init, imm, rs, avg, N_SAMPLES, CH, seed)
            mb, me, mr, mbsd = quality(arr, gm, gv)
            biases.append(mb)
            esss.append(me)
            rhats.append(mr)
            divs.append(dr)
            accs.append(ac)
            ravg.append(mns)
            mbsds.append(mbsd)
            cells.append(
                {
                    "model": model,
                    "avg_param": avg,
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
        bias, ess, div, acc = (
            np.mean(biases),
            np.mean(esss),
            np.mean(divs),
            np.mean(accs),
        )
        rhat = np.nanmax(rhats)
        avg_u = np.mean(ravg)
        mbsd_m = float(np.mean(mbsds))
        bias_ok = bias < TAU_BIAS
        rhat_ok = (rhat < RHAT_BAD) and not np.isnan(rhat)
        ess_ok = ess > ESS_MIN * CH
        loud = (rhat > RHAT_BAD) or np.isnan(rhat) or (div > 0.01) or (acc < 0.1)
        verdict = (
            "PASS"
            if (bias_ok and rhat_ok and ess_ok)
            else (
                "loud-fail"
                if (not bias_ok and loud)
                else "SILENT-FAIL" if not bias_ok else "marginal"
            )
        )
        scoreboard[(model, avg)] = verdict
        cells.append(
            {
                "model": model,
                "avg_param": avg,
                "seed": "AGG",
                "d": int(d),
                "realized_avg": float(avg_u),
                "n_samples": N_SAMPLES,
                "max_2mom_bias": float(bias),
                "max_mean_bias_sd": mbsd_m,
                "min_bulk_ess": float(ess),
                "max_rhat": float(rhat) if not np.isnan(rhat) else None,
                "div_rate": float(div),
                "mean_acc": float(acc),
                "verdict": verdict,
                "n_seeds": len(SEEDS),
                "git_head": GIT_HEAD,
            }
        )
        print(
            f"  {avg:>4d} {avg_u:>7.1f} {bias:>7.3f} {mbsd_m:>8.3f} {ess:>7.0f} {rhat:>7.3f} {div:>6.3f} {acc:>6.3f}  {verdict:>11s}"
        )
        sys.stdout.flush()
    print()

out = os.path.join(_HERE, "sweep_dynl_variety_results.json")
with open(out, "w") as f:
    json.dump(
        {
            "git_head": GIT_HEAD,
            "path": "VMAP" if USE_VMAP else "LOOP",
            "avg_ladder": AVG_LADDER,
            "n_samples": N_SAMPLES,
            "seeds": SEEDS,
            "parity_max_abs": _max_abs,
            "cells": cells,
        },
        f,
        indent=2,
    )
print(
    f"wrote {out} ({len(cells)} rows) | path={'VMAP' if USE_VMAP else 'LOOP'} | parity_max_abs={_max_abs:.3e}"
)
print("DONE_DYNL_VARIETY")
