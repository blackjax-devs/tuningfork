"""Robustness-first trajectory tuning for upstream adjusted_mclmc_dynamic (first cut).

Priorities (user 2026-06-17, upstream framing): (1) works for more models,
(2) less bias at SIMILAR COMPUTE, (3) quality (ESS+Rhat) with minimal tuning +
FAIL LOUD. NOT ess/grad.

Compares 5 trajectory-length strategies at a FIXED sampling-grad budget (so longer
trajectories get fewer samples), 4 chains x 3 seeds, GT dense IMM (isolate trajectory
length). Metrics: 2nd-moment bias vs GT, min bulk-ESS, max split-Rhat, divergence rate,
acceptance. Classify each (model,strategy): pass / loud-fail / silent-fail.

Smoke:  JAX_PLATFORM_NAME=cpu uv run python sweep_robust_traj.py --smoke
Full:   JAX_PLATFORM_NAME=cpu uv run python sweep_robust_traj.py
"""

import os
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

SMOKE = "--smoke" in sys.argv
DIAG = (
    "--diag" in sys.argv
)  # use GT DIAGONAL IMM (M^-1=diag(Sigma)) instead of GT dense
HIGHD = "--highd" in sys.argv  # high-d confirmation: irt_1pl (d=500), reduced budget
if SMOKE:
    # diag-regime smoke: small models (fast) — verifies the 1-D IMM path + the key
    # ill_cond_50 case (diagonal can't whiten rotated kappa=1000).
    MODELS = ["mvn_10", "ill_cond_50"]
    SEEDS = [0]
    BUDGET = 800  # sampling grads per chain (fixed)
    CH = 2
else:
    # broad panel across geometry classes (priority 1: works for more models)
    MODELS = [
        "mvn_10",  # smooth iso
        "german_credit",  # correlated (diag-OK)
        "ill_cond_50",  # rotated κ=1000 (whitened by GT dense IMM)
        "eight_schools_ncp",  # NCP funnel
        "neals_funnel",  # centered funnel (fail-loud probe)
        "horseshoe",  # heavy-tail (Cat C)
        "irt_2pl",  # hierarchical funnel
    ]
    SEEDS = [0, 1, 2]
    BUDGET = 20000
    CH = 4

if HIGHD:  # high-d confirmation overrides the panel to irt_1pl (d=500)
    MODELS = ["irt_1pl"]
    if SMOKE:
        BUDGET = 400

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


def load(model):
    if model in ("mvn_10", "ill_cond_50"):
        Sigma, _ = gt_cov(model)
        d = Sigma.shape[0]
        Sinv = jnp.asarray(np.linalg.inv(Sigma))
        # DIAG: 1-D inverse_mass_matrix M^-1 = diag(Sigma) (the variances). Else GT dense LRD.
        imm = jnp.asarray(np.diag(Sigma)) if DIAG else gt_lrd_imm(Sigma, d)
        return (
            (lambda x: -0.5 * jnp.dot(x, Sinv @ x)),
            jnp.zeros(d),
            d,
            imm,
            np.zeros(d),
            np.diag(Sigma),
        )
    imm_dense, gt_var, gt_mean, d = gt_from_draws(model)
    imm = jnp.asarray(gt_var) if DIAG else imm_dense  # 1-D diagonal M^-1 = GT variances
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
    """max 2nd-moment bias, min bulk-ESS, max split-Rhat."""
    vm = np.mean((arr - gm[None, None, :]) ** 2, axis=(0, 1))
    max_bias = float((np.abs(vm - gv) / np.maximum(gv, 1e-30)).max())
    ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], arr)})
    min_ess = float(np.array(az.ess(ds, method="bulk")["x"]).min())
    max_rhat = float(np.array(az.rhat(ds, method="rank")["x"]).max())
    return max_bias, min_ess, max_rhat


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
    f"robust_traj | smoke={SMOKE} | IMM={'DIAGONAL' if DIAG else 'GT-dense'} | budget={BUDGET} grads/chain | {CH} chains x {len(SEEDS)} seeds"
)
print(
    f"classify: pass (bias<{TAU_BIAS}, rhat<{RHAT_BAD}, minESS>{ESS_MIN}/chain) | "
    f"loud-fail (bias high AND rhat>{RHAT_BAD} or div>0.01) | silent-fail (bias high, no flag)\n"
)

scoreboard = {}
for model in MODELS:
    ld, init, d, imm, gm, gv = load(model)
    rs = ref_step(ld, init, imm)
    # big models (d>50) are expensive + loud-fail anyway: smaller budget / fewer seeds
    if SMOKE:
        budget, seeds = BUDGET, SEEDS
    else:
        budget = 3000 if d > 200 else (6000 if d > 50 else BUDGET)
        seeds = [0, 1] if d > 50 else SEEDS
    print(
        f"{'='*92}\n{model} (d={d}) | ref_step={rs:.3f} | budget={budget} seeds={seeds}"
    )
    print(
        f"  {'strategy':12s} {'avg':>4s} {'N':>6s} {'bias':>7s} {'minESS':>7s} {'Rhat':>6s} "
        f"{'div':>6s} {'acc':>6s}  {'verdict':>11s}"
    )
    for sname, (aspec, ss) in STRATEGIES.items():
        biases, esss, rhats, divs, accs, avgs_used = [], [], [], [], [], []
        for seed in seeds:
            step = ss * rs
            avg = (
                search_avg(ld, init, imm, step, CH, seed)
                if aspec == "search"
                else aspec
            )
            n = max(int(budget / (2 * avg)), 50)
            arr, dr, ac, mns = sample(ld, init, imm, step, avg, n, CH, seed)
            mb, me, mr = quality(arr, gm, gv)
            biases.append(mb)
            esss.append(me)
            rhats.append(mr)
            divs.append(dr)
            accs.append(ac)
            avgs_used.append(avg)
        bias, ess, div, acc = (
            np.mean(biases),
            np.mean(esss),
            np.mean(divs),
            np.mean(accs),
        )
        rhat = np.nanmax(rhats) if not np.all(np.isnan(rhats)) else np.nan
        avg_u = np.mean(avgs_used)
        n_u = max(int(budget / (2 * avg_u)), 50)
        bias_ok = bias < TAU_BIAS
        rhat_ok = (rhat < RHAT_BAD) and not np.isnan(rhat)
        ess_ok = ess > ESS_MIN * CH
        # FAIL-LOUD signals: high Rhat, nan-Rhat (frozen chains), divergences, or acc collapse.
        loud_signal = (rhat > RHAT_BAD) or np.isnan(rhat) or (div > 0.01) or (acc < 0.1)
        if bias_ok and rhat_ok and ess_ok:
            verdict = "PASS"
        elif (not bias_ok) and loud_signal:
            verdict = "loud-fail"
        elif not bias_ok:
            verdict = "SILENT-FAIL"
        else:
            verdict = "marginal"  # bias ok but ess/rhat short
        scoreboard[(model, sname)] = verdict
        print(
            f"  {sname:12s} {avg_u:>4.1f} {n_u:>6d} {bias:>7.3f} {ess:>7.0f} {rhat:>6.3f} "
            f"{div:>6.3f} {acc:>6.3f}  {verdict:>11s}"
        )
        sys.stdout.flush()
    print()

print(f"{'='*92}\nSCOREBOARD (verdict counts per strategy across {len(MODELS)} models)")
for sname in STRATEGIES:
    verds = [scoreboard[(m, sname)] for m in MODELS]
    counts = {v: verds.count(v) for v in set(verds)}
    print(f"  {sname:12s} {counts}  silent-fails={verds.count('SILENT-FAIL')}")
print(
    "\nKey upstream questions: (1) which strategy PASSes the most models? "
    "(2) does any have SILENT-FAILs (bias w/o flag = disqualifying)? "
    "(3) does a FIXED default (S_short/S_long) match S_search (minimal tuning)?"
)
print("DONE")
