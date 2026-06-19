"""#25 cert: banana adjusted_mclmc at the grid-search PASS window (avg in {18,24,36,54}).

User-approved config (via tl): banana ONLY, adjusted_mclmc_dynamic with avg PINNED constant,
sweep avg in {18,24,36,54}, n=5000/chain, 4 chains VMAPPED, n_warmup=5000, 6 FRESH seeds.
IMM = banana analytic-GT branch (diag marginal var [8,9]; golden mean=[0,2]). step from
per-seed EEVPD warmup; file the MEDIAN step as the golden.

GATE (PASS@medium__), per seed: 2mbias<0.10 AND mbias_sd<0.06 AND rhat<1.01 AND
min_bulk_ess>100/ch (=>400 total over 4 ch) AND div==0. Filing requires the gate to hold on
>=5/6 seeds. Filing point = the LOWEST avg that clears >=5/6 (shortest robust L: cheapest +
furthest from the overshoot edge). If a higher avg is materially more seed-stable, it is
flagged with numbers.

vmap is the validated fast path (parity 1.15e-14 prior); re-run the 1-cell PARITY GATE at
startup (banana, avg=18, n=200) and fall back to the sequential loop if it does not hold.

Run: JAX_PLATFORM_NAME=cpu .venv/bin/python experiments/mclmc_scaling/cert25_banana_medium.py
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

AVG_GRID = [18, 24, 36, 54]
SEEDS = [10, 11, 12, 13, 14, 15]  # FRESH seeds (sweep used 0,1,2)
N_SAMPLES = 5000
N_WARMUP = 5000
CH = 4
# Gate thresholds (PASS@medium__)
TAU_2MBIAS, TAU_MBSD, RHAT_BAD, ESS_MIN_PER_CH = 0.10, 0.06, 1.01, 100
SEEDS_REQUIRED = 5  # of 6

BANANA_MEAN = np.array([0.0, 2.0])
BANANA_VAR = np.array([8.0, 9.0])


def load_banana():
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.model._registry import MODELS as _M

    init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M["banana"])
    _, unravel = ravel_pytree(init_dict)
    d = 2
    imm = jnp.asarray(BANANA_VAR)  # exact diagonal marginal cov (Cov(x1,x2)=0)
    return (lambda xf: ld_raw(unravel(xf))), jnp.zeros(d), d, imm


def ref_step(ld, init, imm, seed, num_steps):
    """Per-seed EEVPD warmup -> step_size. Warmup keys disjoint from chain keys."""
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


def _chain_keys(seed, ch):
    return [jax.random.key(seed * 1000 + ci + 1) for ci in range(ch)]


def sample_loop(ld, init, imm, step, avg, n, ch, seed):
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
        flat = jax.vmap(lambda q: ravel_pytree(q)[0])(pt)
        return flat, dv, ac, ns

    flats, dvs, acs, nss = jax.vmap(run_chain)(keys)
    arr = np.asarray(flats)
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


def seed_passes(b2, mbsd, ess, rhat, div):
    return (
        b2 < TAU_2MBIAS
        and mbsd < TAU_MBSD
        and rhat < RHAT_BAD
        and not np.isnan(rhat)
        and ess > ESS_MIN_PER_CH * CH
        and div == 0.0
    )


# ---- PARITY GATE (banana, avg=18, n=200) ----
print("PARITY CHECK: vmap vs sequential loop (banana, avg=18, n=200, 4 chains)...")
_ld, _init, _d, _imm = load_banana()
_rs = ref_step(_ld, _init, _imm, SEEDS[0], 2000)
_a_loop, _dl, _al, _nl = sample_loop(_ld, _init, _imm, _rs, 18, 200, CH, SEEDS[0])
_a_vmap, _dv, _av, _nv = sample_vmap(_ld, _init, _imm, _rs, 18, 200, CH, SEEDS[0])
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
    f"\ncert25_banana_medium | path={'VMAP' if USE_VMAP else 'LOOP'} | n={N_SAMPLES}/ch x {CH}ch x seeds {SEEDS} | nw={N_WARMUP} | avg grid={AVG_GRID}"
)
print(
    f"gate PASS@medium__: 2mbias<{TAU_2MBIAS} AND mbias_sd<{TAU_MBSD} AND rhat<{RHAT_BAD} "
    f"AND minESS>{ESS_MIN_PER_CH}/ch AND div==0, holding >={SEEDS_REQUIRED}/{len(SEEDS)} seeds\n"
)

ld, init, d, imm = load_banana()
gm, gv = BANANA_MEAN.copy(), BANANA_VAR.copy()

# per-seed warmup step
seed_steps = {seed: ref_step(ld, init, imm, seed, N_WARMUP) for seed in SEEDS}
median_step = float(np.median(list(seed_steps.values())))
print(
    "per-seed warmup step_size: "
    + ", ".join(f"s{seed}={seed_steps[seed]:.4f}" for seed in SEEDS)
)
print(f"  median step = {median_step:.4f}\n")

cells = []
avg_summary = {}
print(
    f"  {'avg':>4s} {'seed':>5s} {'step':>6s} {'realAvg':>7s} {'2mbias':>7s} {'mbias_sd':>8s} {'minESS':>7s} {'Rhat':>7s} {'div':>5s} {'acc':>6s}  {'pass':>5s}"
)
for avg in AVG_GRID:
    n_pass = 0
    for seed in SEEDS:
        step = seed_steps[seed]
        arr, dr, ac, mns = sample(ld, init, imm, step, avg, N_SAMPLES, CH, seed)
        b2, ess, rhat, mbsd = quality(arr, gm, gv)
        sp = seed_passes(b2, mbsd, ess, rhat, dr)
        n_pass += int(sp)
        cells.append(
            {
                "model": "banana",
                "avg_param": avg,
                "seed": int(seed),
                "d": int(d),
                "step_size": float(step),
                "realized_avg": float(mns),
                "n_samples": N_SAMPLES,
                "n_warmup": N_WARMUP,
                "max_2mom_bias": float(b2),
                "max_mean_bias_sd": float(mbsd),
                "min_bulk_ess": float(ess),
                "max_rhat": float(rhat),
                "div_rate": float(dr),
                "mean_acc": float(ac),
                "seed_pass": bool(sp),
                "git_head": GIT_HEAD,
            }
        )
        print(
            f"  {avg:>4d} {seed:>5d} {step:>6.4f} {mns:>7.1f} {b2:>7.3f} {mbsd:>8.3f} {ess:>7.0f} {rhat:>7.3f} {dr:>5.2f} {ac:>6.3f}  {'YES' if sp else 'no':>5s}"
        )
        sys.stdout.flush()
    cell_pass = n_pass >= SEEDS_REQUIRED
    avg_summary[avg] = {"n_pass": n_pass, "cell_pass": cell_pass}
    print(
        f"  -> avg={avg}: {n_pass}/{len(SEEDS)} seeds PASS => {'CELL-PASS' if cell_pass else 'cell-fail'}\n"
    )

# Filing recommendation: lowest avg with >=SEEDS_REQUIRED passing seeds
passing_avgs = [a for a in AVG_GRID if avg_summary[a]["cell_pass"]]
filing_avg = min(passing_avgs) if passing_avgs else None
filing = None
if filing_avg is not None:
    filing = {
        "kernel": "adjusted_mclmc_dynamic",
        "avg_integration_steps": filing_avg,
        "step_size_golden_median": median_step,
        "L_trajectory_golden": filing_avg * median_step,
        "inverse_mass_matrix_diag": [8.0, 9.0],
        "golden_mean": [0.0, 2.0],
        "golden_var": [8.0, 9.0],
        "n_seeds_pass": avg_summary[filing_avg]["n_pass"],
        "tier": "medium__",
    }

print("=" * 90)
print("FILING SUMMARY")
for a in AVG_GRID:
    print(
        f"  avg={a:>3d}: {avg_summary[a]['n_pass']}/{len(SEEDS)} pass | {'CELL-PASS' if avg_summary[a]['cell_pass'] else 'cell-fail'}"
    )
print(f"  passing avgs: {passing_avgs}")
print(f"  recommended filing avg: {filing_avg}")
if filing:
    print(
        f"  RECIPE: avg={filing_avg}, step(median)={median_step:.4f}, L={filing['L_trajectory_golden']:.4f}, "
        f"IMM diag=[8,9], golden mean=[0,2] var=[8,9]"
    )
else:
    print(
        "  NO avg cleared >=5/6 seeds -> KEEP failed__ (n=500 window did NOT hold at production budget)"
    )

out = os.path.join(_HERE, "cert25_banana_medium_results.json")
with open(out, "w") as f:
    json.dump(
        {
            "git_head": GIT_HEAD,
            "path": "VMAP" if USE_VMAP else "LOOP",
            "avg_grid": AVG_GRID,
            "n_samples": N_SAMPLES,
            "n_warmup": N_WARMUP,
            "seeds": SEEDS,
            "ch": CH,
            "parity_max_abs": _max_abs,
            "seed_steps": seed_steps,
            "median_step": median_step,
            "gate": {
                "tau_2mbias": TAU_2MBIAS,
                "tau_mbsd": TAU_MBSD,
                "rhat_bad": RHAT_BAD,
                "ess_min_per_ch": ESS_MIN_PER_CH,
                "seeds_required": SEEDS_REQUIRED,
            },
            "avg_summary": {str(k): v for k, v in avg_summary.items()},
            "filing_avg": filing_avg,
            "filing_recipe": filing,
            "cells": cells,
        },
        f,
        indent=2,
    )
print(
    f"\nwrote {out} ({len(cells)} rows) | path={'VMAP' if USE_VMAP else 'LOOP'} | parity_max_abs={_max_abs:.3e}"
)
print("DONE_CERT25_BANANA")
