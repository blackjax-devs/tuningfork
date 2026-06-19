"""B1.5: joint (step_scale × avg) tradeoff for adjusted_mclmc_dynamic.

Question (user 2026-06-17): on heterogeneous geometry, does a SMALLER step + LONGER
trajectory beat a BIGGER step + SHORTER one? B1 tunes avg at a FIXED step, so it
under-explores the small-step/long-trajectory regime. Here we map the 2-D grid.

For each (step_scale, avg): run an adjusted_mclmc_dynamic pilot at step = step_scale ×
ref_step (ref = unadjusted-tuned step), GT dense IMM, integration_steps_params=(avg,).
Report ess/grad, acceptance, max 2nd-moment bias.

Predictions:
  smooth (mvn_10)            -> optimum at high step_scale / low avg
  funnel (eight_schools/neals)-> optimum at low step_scale / high avg (the user's regime)

Smoke:  JAX_PLATFORM_NAME=cpu uv run python sweep_step_x_traj.py --smoke
Full:   JAX_PLATFORM_NAME=cpu uv run python sweep_step_x_traj.py
"""

import os
import sys

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
if SMOKE:
    STEP_SCALES, AVGS, N_PILOT, CH = [0.5, 1.0], [1, 4], 80, 2
    MODELS = ["mvn_10"]
else:
    STEP_SCALES, AVGS, N_PILOT, CH = [0.25, 0.5, 1.0, 2.0], [1, 2, 4, 8, 16], 1500, 4
    MODELS = ["mvn_10", "eight_schools_ncp", "neals_funnel"]
SEED = 20260617


def load(model):
    if model in ("mvn_10", "ill_cond_50"):
        Sigma, _ = gt_cov(model)
        d = Sigma.shape[0]
        Sinv = jnp.asarray(np.linalg.inv(Sigma))
        return (
            (lambda x: -0.5 * jnp.dot(x, Sinv @ x)),
            jnp.zeros(d),
            d,
            gt_lrd_imm(Sigma, d),
            np.zeros(d),
            np.diag(Sigma),
        )
    imm, gt_var, gt_mean, d = gt_from_draws(model)
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.model._registry import MODELS as _M

    init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(SEED), _M[model])
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
    import warnings

    st = mclmc_mod.init(init, ld, jax.random.key(SEED))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p, _ = mclmc_find_L_and_step_size(
            mclmc_kernel=_make_fixed_imm_kernel(imm),
            num_steps=2000,
            state=st,
            rng_key=jax.random.key(SEED + 1),
            logdensity_fn=ld,
            diagonal_preconditioning=False,
        )
    return float(p.step_size)


def run_cell(ld, init, imm, step, avg, gm, gv, d):
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    pos, ns, ac = [], [], []
    for ci in range(CH):
        sk = jax.random.key(SEED + 300 + ci)
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
            return nx, (nx.position, info.num_integration_steps, info.acceptance_rate)

        _, (pt, n, a) = jax.lax.scan(stp, s, jax.random.split(sk, N_PILOT))
        pos.append(np.array(jax.vmap(lambda q: ravel_pytree(q)[0])(pt)))
        ns.append(np.array(n))
        ac.append(np.array(a))
    arr = np.stack(pos, 0)  # (CH, N_PILOT, d)
    finite = np.isfinite(arr).all(axis=(0, 2)).all() if arr.size else False
    vm = np.mean((arr - gm[None, None, :]) ** 2, axis=(0, 1))
    max_bias = float((np.abs(vm - gv) / np.maximum(gv, 1e-30)).max())
    ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], arr)})
    min_ess = float(np.array(az.ess(ds, method="bulk")["x"]).min())
    mean_ns = float(np.mean(np.concatenate(ns)))
    eg = min_ess / max(2 * mean_ns * N_PILOT * CH, 1)
    return eg, float(np.mean(np.concatenate(ac))), max_bias, mean_ns


for model in MODELS:
    ld, init, d, imm, gm, gv = load(model)
    rs = ref_step(ld, init, imm)
    print(
        f"\n{'='*78}\n{model} (d={d}) | ref_step(unadj)={rs:.3f} | grid step_scale × avg, GT dense IMM"
    )
    print(f"{'='*78}")
    results = {}
    for ss in STEP_SCALES:
        for avg in AVGS:
            eg, acc, mb, mns = run_cell(ld, init, imm, ss * rs, avg, gm, gv, d)
            results[(ss, avg)] = (eg, acc, mb, mns)
            sys.stdout.flush()
    # ess/grad table
    print(f"\ness/grad   (rows=step_scale, cols=avg)   [step = scale × {rs:.3f}]")
    print("  scale\\avg " + "".join(f"{a:>10d}" for a in AVGS))
    for ss in STEP_SCALES:
        print(f"  {ss:>7.2f}  " + "".join(f"{results[(ss,a)][0]:>10.5f}" for a in AVGS))
    print("acceptance (rows=step_scale, cols=avg)")
    for ss in STEP_SCALES:
        print(f"  {ss:>7.2f}  " + "".join(f"{results[(ss,a)][1]:>10.3f}" for a in AVGS))
    print("max_bias   (rows=step_scale, cols=avg)")
    for ss in STEP_SCALES:
        print(f"  {ss:>7.2f}  " + "".join(f"{results[(ss,a)][2]:>10.3f}" for a in AVGS))
    best = max(results, key=lambda k: results[k][0])
    eg, acc, mb, mns = results[best]
    print(
        f"\n  OPTIMUM: step_scale={best[0]}, avg={best[1]} -> ess/grad={eg:.5f}, acc={acc:.3f}, max_bias={mb:.3f}, mean_nsteps={mns:.1f}"
    )
    sys.stdout.flush()
print("\nDONE")
