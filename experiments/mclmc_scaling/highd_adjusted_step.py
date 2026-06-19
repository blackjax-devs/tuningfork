"""High-d confirmation done RIGHT: avg=1 (MALA) vs avg=2 at the ADJUSTED tuner's step.

The robustness sweep used the unadjusted MCLMC step as reference; at d=500 that step
(~24) is far too big for the adjusted MH kernel -> acc=0, frozen. The UPSTREAM fix
uses the adjusted tuner's step (tuned to acc=0.9), then sets avg=2 (L=2*step). This
replicates that real config on irt_1pl (d=500) and checks avg=2 vs avg=1 (MALA).
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

import blackjax.mcmc.adjusted_mclmc as adj_mclmc_mod
import blackjax.mcmc.adjusted_mclmc_dynamic as adj_dyn_mod
from blackjax.adaptation.adjusted_mclmc_adaptation import (
    adjusted_mclmc_find_L_and_step_size,
)
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from gt_imm import gt_from_draws
from run_fixed_imm import (
    _make_fixed_imm_adj_dyn_kernel,
    _make_fixed_imm_adj_kernel_for_tuning,
)

model = "irt_1pl"
imm, gt_var, gt_mean, d = gt_from_draws(model)
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model._registry import MODELS as _M

init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M[model])
_, unravel = ravel_pytree(init_dict)
ld = lambda xf: ld_raw(unravel(xf))  # noqa: E731
init = jnp.asarray(gt_mean, dtype=jnp.float64)
gm, gv = np.asarray(gt_mean), np.asarray(gt_var)
CH, BUDGET = 4, 6000  # grads/chain (fixed); N = BUDGET/(2*avg)

# ADJUSTED tuner -> step tuned to acc=0.9 (the real upstream basis)
tune_kernel = _make_fixed_imm_adj_kernel_for_tuning(imm)
init_state = adj_mclmc_mod.init(init, ld)
params0 = MCLMCAdaptationState(
    L=jnp.array(jnp.sqrt(float(d))),
    step_size=jnp.array(0.5),
    inverse_mass_matrix=imm.sigma,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _, p, _ = adjusted_mclmc_find_L_and_step_size(
        tune_kernel,
        logdensity_fn=ld,
        num_steps=3000,
        state=init_state,
        rng_key=jax.random.key(3),
        target=0.9,
        frac_tune1=0.5,
        frac_tune2=0.0,
        frac_tune3=0.0,
        diagonal_preconditioning=False,
        params=params0,
    )
adj_step = float(p.step_size)
print(
    f"irt_1pl (d={d}) | ADJUSTED tuner step={adj_step:.4f} (vs unadjusted ~24 which froze)\n"
)
print(
    f"{'avg':>4s} {'N':>5s} {'acc':>6s} {'bias':>7s} {'minESS':>8s} {'Rhat':>6s} {'med_ns':>6s}  verdict"
)


def run(avg, seed):
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    n = max(int(BUDGET / (2 * avg)), 50)
    pos, accs, nss = [], [], []
    for ci in range(CH):
        sk = jax.random.key(seed * 100 + ci + 1)
        s = adj_dyn_mod.init(init, ld, sk)

        def stp(c, k):
            nx, info = dyn(
                rng_key=k,
                state=c,
                logdensity_fn=ld,
                step_size=adj_step,
                L_proposal_factor=jnp.inf,
                inverse_mass_matrix=imm,
                integration_steps_params=(float(avg),),
            )
            return nx, (nx.position, info.acceptance_rate, info.num_integration_steps)

        _, (pt, ac, ns) = jax.lax.scan(stp, s, jax.random.split(sk, n))
        pos.append(np.array(jax.vmap(lambda q: ravel_pytree(q)[0])(pt)))
        accs.append(np.array(ac))
        nss.append(np.array(ns))
    arr = np.stack(pos, 0)
    vm = np.mean((arr - gm[None, None, :]) ** 2, axis=(0, 1))
    bias = float((np.abs(vm - gv) / np.maximum(gv, 1e-30)).max())
    ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], arr)})
    ess = float(np.array(az.ess(ds, method="bulk")["x"]).min())
    rhat = float(np.array(az.rhat(ds, method="rank")["x"]).max())
    return (
        n,
        float(np.mean(np.concatenate(accs))),
        bias,
        ess,
        rhat,
        float(np.median(np.concatenate(nss))),
    )


for avg in [1, 2]:
    rs = [run(avg, s) for s in [0, 1]]
    n = rs[0][0]
    acc = np.mean([r[1] for r in rs])
    bias = np.mean([r[2] for r in rs])
    ess = np.mean([r[3] for r in rs])
    rhat = np.nanmax([r[4] for r in rs])
    mns = np.mean([r[5] for r in rs])
    ok = bias < 0.1 and rhat < 1.01 and ess > 100 * CH
    loud = (not ok) and (rhat > 1.01 or np.isnan(rhat) or acc < 0.1)
    verdict = "PASS" if ok else ("loud-fail" if loud else "SILENT-FAIL")
    tag = " (MALA)" if avg == 1 else ""
    print(
        f"{avg:>4d} {n:>5d} {acc:>6.3f} {bias:>7.3f} {ess:>8.0f} {rhat:>6.3f} {mns:>6.1f}  {verdict}{tag}"
    )
print("\nDONE")
