"""SCOPE B: does enabling the canonical L-tuners escape the MALA collapse?

The harness sets frac_tune2=0, frac_tune3=0 -> L stays at init -> avg=L/step~1 -> MALA.
Canonical blackjax has two L-tuners:
  frac_tune2: variance-based, L = sqrt(sum var)*tuning_factor (typical-set radius)
  frac_tune3: autocorrelation-based (adjusted_mclmc_make_adaptation_L, ~10 eff samples)
Test 3 settings; for each report tuned (L, step), avg=L/step, then SAMPLE the dynamic
kernel at that avg and report acc / mean_nsteps / ess / ess_per_sampgrad / bias.

If a setting gives avg>1 with ess/grad >= unadjusted -> wire it in (B = enable+validate).
If all stay ~MALA or unstable -> B needs an explicit ess/grad-targeting trajectory tuner.
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

sys.path.insert(0, "/home/jp/blackjax-devs/tuningfork/experiments/mclmc_scaling")
os.chdir("/home/jp/blackjax-devs/tuningfork")

import blackjax.mcmc.adjusted_mclmc as adj_mclmc_mod
import blackjax.mcmc.adjusted_mclmc_dynamic as adj_dyn_mod
import blackjax.mcmc.mclmc as mclmc_mod
from blackjax.adaptation.adjusted_mclmc_adaptation import (
    adjusted_mclmc_find_L_and_step_size,
)
from blackjax.adaptation.mclmc_adaptation import (
    MCLMCAdaptationState,
    mclmc_find_L_and_step_size,
)
from gt_imm import gt_cov, gt_from_draws, gt_lrd_imm
from run_fixed_imm import (
    _make_fixed_imm_adj_dyn_kernel,
    _make_fixed_imm_adj_kernel_for_tuning,
    _make_fixed_imm_kernel,
)

NW, NS, CH, SEED = 2000, 2000, 4, 20260617


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
    from tuningfork.model._registry import MODELS

    init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(SEED), MODELS[model])
    _, unravel = ravel_pytree(init_dict)
    return (
        (lambda xf: ld_raw(unravel(xf))),
        jnp.asarray(gt_mean, dtype=jnp.float64),
        d,
        imm,
        np.asarray(gt_mean),
        np.asarray(gt_var),
    )


def essbias(arr, gm, gv):
    vm = np.mean((arr - gm[None, None, :]) ** 2, axis=(0, 1))
    bias = float((np.abs(vm - gv) / np.maximum(gv, 1e-30)).max())
    ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], arr)})
    return float(np.array(az.ess(ds, method="bulk")["x"]).min()), bias


def unadj_baseline(ld, init, imm, gm, gv):
    k = _make_fixed_imm_kernel(imm)
    st = mclmc_mod.init(init, ld, jax.random.key(SEED))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p, _ = mclmc_find_L_and_step_size(
            mclmc_kernel=k,
            num_steps=NW,
            state=st,
            rng_key=jax.random.key(SEED + 1),
            logdensity_fn=ld,
            diagonal_preconditioning=False,
        )
    L, step = float(p.L), float(p.step_size)
    kern = mclmc_mod.build_kernel()
    pos = []
    for ci in range(CH):
        s = mclmc_mod.init(init, ld, jax.random.key(SEED + 10 + ci))

        def stp(c, key):
            nx = kern(
                rng_key=key,
                state=c,
                logdensity_fn=ld,
                L=L,
                step_size=step,
                inverse_mass_matrix=imm,
            )[0]
            return nx, nx.position

        _, pt = jax.lax.scan(
            stp, s, jax.random.split(jax.random.key(SEED + 100 + ci), NS)
        )
        pos.append(np.array(jax.vmap(lambda q: ravel_pytree(q)[0])(pt)))
    ess, bias = essbias(np.stack(pos, 0), gm, gv)
    return ess, bias, ess / (2 * NS * CH), L, step


def adj_tune_sample(ld, init, imm, ft1, ft2, ft3, gm, gv):
    tune_kernel = _make_fixed_imm_adj_kernel_for_tuning(imm)
    init_state = adj_mclmc_mod.init(
        init, ld
    )  # STATIC adjusted_mclmc state (3-field) for tuning
    params0 = MCLMCAdaptationState(
        L=jnp.array(jnp.sqrt(float(init.shape[0]))),
        step_size=jnp.array(0.5),
        inverse_mass_matrix=imm.sigma,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p, _ = adjusted_mclmc_find_L_and_step_size(
            mclmc_kernel=tune_kernel,
            logdensity_fn=ld,
            num_steps=NW,
            state=init_state,
            rng_key=jax.random.key(SEED + 6),
            target=0.9,
            frac_tune1=ft1,
            frac_tune2=ft2,
            frac_tune3=ft3,
            diagonal_preconditioning=False,
            params=params0,
        )
    L, step = float(p.L), float(p.step_size)
    avg = max(1.0, L / step)
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    pos, ns, ac = [], [], []
    for ci in range(CH):
        sk = jax.random.key(SEED + 200 + ci)
        s = adj_dyn_mod.init(init, ld, sk)

        def stp(c, key):
            nx, info = dyn(
                rng_key=key,
                state=c,
                logdensity_fn=ld,
                step_size=step,
                L_proposal_factor=jnp.inf,
                inverse_mass_matrix=imm,
                integration_steps_params=(avg,),
            )
            return nx, (nx.position, info.num_integration_steps, info.acceptance_rate)

        _, (pt, n, a) = jax.lax.scan(stp, s, jax.random.split(sk, NS))
        pos.append(np.array(jax.vmap(lambda q: ravel_pytree(q)[0])(pt)))
        ns.append(np.array(n))
        ac.append(np.array(a))
    ess, bias = essbias(np.stack(pos, 0), gm, gv)
    mns = float(np.mean(np.concatenate(ns)))
    return (
        L,
        step,
        avg,
        mns,
        float(np.mean(np.concatenate(ac))),
        ess,
        bias,
        ess / max(2 * mns * NS * CH, 1),
    )


for model in ["mvn_10", "ill_cond_50"]:
    ld, init, d, imm, gm, gv = load(model)
    ue, ub, ueg, uL, ustep = unadj_baseline(ld, init, imm, gm, gv)
    print(
        f"\n=== {model} (d={d}) | UNADJ ess/sampgrad={ueg:.5f} bias={ub:.4f} (L={uL:.2f} step={ustep:.2f}) ==="
    )
    print(
        f"  {'fracs(1,2,3)':>14s} {'L':>7s} {'step':>6s} {'avg':>5s} {'mean_nstep':>10s} {'acc':>6s} {'ess':>8s} {'bias':>7s} {'ess/sampgrad':>12s} {'vs_unadj':>8s}"
    )
    for ft1, ft2, ft3 in [(0.5, 0.0, 0.0), (0.4, 0.3, 0.0), (0.2, 0.2, 0.3)]:
        try:
            L, step, avg, mns, acc, ess, bias, eg = adj_tune_sample(
                ld, init, imm, ft1, ft2, ft3, gm, gv
            )
            print(
                f"  {f'({ft1},{ft2},{ft3})':>14s} {L:>7.2f} {step:>6.2f} {avg:>5.2f} {mns:>10.1f} {acc:>6.3f} {ess:>8.1f} {bias:>7.4f} {eg:>12.6f} {eg/max(ueg,1e-30):>7.2f}x"
            )
        except Exception as exc:
            print(f"  ({ft1},{ft2},{ft3}) ERROR: {exc}")
    sys.stdout.flush()
print("\nDONE")
