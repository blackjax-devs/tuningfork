"""Does adjusted_mclmc_dynamic's efficiency recover with PROPER trajectory length?

The harness pins avg_steps = L/step. The MCLMC sqrt(d) law gives L=0.85sqrt(d) <
step=1.22sqrt(d), so L/step~0.7 -> 1-step trajectories = MALA (the "adjusted worse"
artifact). Here we DECOUPLE trajectory length from the unadjusted L: sweep
integration_steps_params=(avg_steps,) over {1,2,4,8,16,32} at a fixed step, and
measure steady-state ess per SAMPLING grad (warmup amortizes away on real runs).

If ess/grad peaks well above the avg_steps=1 (MALA) value and approaches/exceeds
unadjusted, then adjusted_dynamic CAN be competitive on smooth targets with proper
trajectory tuning -> "always adjusted_dynamic" is viable. If not, unadjusted wins.
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

import blackjax
import blackjax.mcmc.adjusted_mclmc_dynamic as adj_dyn_mod
from blackjax.adaptation.mclmc_adaptation import mclmc_find_L_and_step_size
from gt_imm import gt_cov, gt_from_draws, gt_lrd_imm
from run_fixed_imm import _make_fixed_imm_adj_dyn_kernel, _make_fixed_imm_kernel

N_WARMUP, N_SAMPLES, NUM_CHAINS, SEED = 1500, 2000, 4, 20260617
AVG_STEPS_GRID = [1, 2, 4, 8, 16, 32]


def load(model):
    if model in ("mvn_10", "ill_cond_50"):
        Sigma, _ = gt_cov(model)
        d = Sigma.shape[0]
        Sinv = jnp.asarray(np.linalg.inv(Sigma))
        ld = lambda x: -0.5 * jnp.dot(x, Sinv @ x)  # noqa: E731
        imm = gt_lrd_imm(Sigma, k=d)
        return ld, jnp.zeros(d), d, imm, np.zeros(d), np.diag(Sigma)
    imm, gt_var, gt_mean, d = gt_from_draws(model)
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.model._registry import MODELS

    entry = MODELS[model]
    init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(SEED), entry)
    _, unravel = ravel_pytree(init_dict)
    ld = lambda xf: ld_raw(unravel(xf))  # noqa: E731
    return (
        ld,
        jnp.asarray(gt_mean, dtype=jnp.float64),
        d,
        imm,
        np.asarray(gt_mean),
        np.asarray(gt_var),
    )


def ess_bias(positions_arr, gt_mean, gt_var):
    var_mcmc = np.mean((positions_arr - gt_mean[None, None, :]) ** 2, axis=(0, 1))
    bias = float((np.abs(var_mcmc - gt_var) / np.maximum(gt_var, 1e-30)).max())
    ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], positions_arr)})
    return float(np.array(az.ess(ds, method="bulk")["x"]).min()), bias


def tune_unadj(ld, init, imm, d):
    fixed_kernel = _make_fixed_imm_kernel(imm)
    st = blackjax.mcmc.mclmc.init(init, ld, jax.random.key(SEED))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, params, _ = mclmc_find_L_and_step_size(
            mclmc_kernel=fixed_kernel,
            num_steps=N_WARMUP,
            state=st,
            rng_key=jax.random.key(SEED + 1),
            logdensity_fn=ld,
            diagonal_preconditioning=False,
        )
    return float(params.L), float(params.step_size)


def run_unadj(ld, init, imm, L, step, gt_mean, gt_var):
    kernel = blackjax.mcmc.mclmc.build_kernel()
    pos = []
    for ci in range(NUM_CHAINS):
        st = blackjax.mcmc.mclmc.init(init, ld, jax.random.key(SEED + 10 + ci))
        keys = jax.random.split(jax.random.key(SEED + 100 + ci), N_SAMPLES)

        def stp(s, k):
            ns, _ = kernel(
                rng_key=k,
                state=s,
                logdensity_fn=ld,
                L=L,
                step_size=step,
                inverse_mass_matrix=imm,
            )
            return ns, ns.position

        _, pt = jax.lax.scan(stp, st, keys)
        pos.append(np.array(jax.vmap(lambda p: ravel_pytree(p)[0])(pt)))
    arr = np.stack(pos, 0)
    ess, bias = ess_bias(arr, gt_mean, gt_var)
    grads = 2 * N_SAMPLES * NUM_CHAINS
    return ess, bias, ess / grads


def run_adj(ld, init, imm, step, avg_steps, gt_mean, gt_var):
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    pos, nsteps, accs = [], [], []
    for ci in range(NUM_CHAINS):
        sk = jax.random.key(SEED + 200 + ci)
        st = adj_dyn_mod.init(init, ld, sk)
        keys = jax.random.split(sk, N_SAMPLES)

        def stp(s, k):
            ns, info = dyn(
                rng_key=k,
                state=s,
                logdensity_fn=ld,
                step_size=step,
                L_proposal_factor=jnp.inf,
                inverse_mass_matrix=imm,
                integration_steps_params=(float(avg_steps),),
            )
            return ns, (ns.position, info.num_integration_steps, info.acceptance_rate)

        _, (pt, nst, ac) = jax.lax.scan(stp, st, keys)
        pos.append(np.array(jax.vmap(lambda p: ravel_pytree(p)[0])(pt)))
        nsteps.append(np.array(nst))
        accs.append(np.array(ac))
    arr = np.stack(pos, 0)
    ess, bias = ess_bias(arr, gt_mean, gt_var)
    mean_steps = float(np.mean(np.concatenate(nsteps)))
    grads = 2 * mean_steps * N_SAMPLES * NUM_CHAINS
    return (
        ess,
        bias,
        ess / max(grads, 1),
        mean_steps,
        float(np.mean(np.concatenate(accs))),
    )


for model in ["mvn_10", "german_credit", "ill_cond_50"]:
    ld, init, d, imm, gt_mean, gt_var = load(model)
    L_u, step_u = tune_unadj(ld, init, imm, d)
    ue, ub, ueg = run_unadj(ld, init, imm, L_u, step_u, gt_mean, gt_var)
    print(f"\n=== {model} (d={d}) | unadj step={step_u:.2f} L={L_u:.2f} ===")
    print(f"  UNADJ      ess={ue:7.1f} bias={ub:.4f} ess/sampgrad={ueg:.6f}")
    print(
        f"  {'avg_steps':>9s} {'real_nstep':>10s} {'acc':>6s} {'ess':>8s} {'bias':>7s} {'ess/sampgrad':>13s} {'vs_unadj':>9s}"
    )
    for a in AVG_STEPS_GRID:
        ae, ab, aeg, ms, acc = run_adj(ld, init, imm, step_u, a, gt_mean, gt_var)
        print(
            f"  {a:>9d} {ms:>10.1f} {acc:>6.3f} {ae:>8.1f} {ab:>7.4f} {aeg:>13.6f} {aeg/max(ueg,1e-30):>8.2f}x"
        )
    sys.stdout.flush()
print("\nDONE")
