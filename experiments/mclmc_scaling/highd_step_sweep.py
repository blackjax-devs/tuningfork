"""Isolate: at d=500, does avg=2 work with a PROPERLY SMALL step?

The adjusted tuner gave step=19.24 (acc 0.06) — a tuner-convergence failure, not an
avg=2 failure. Manually sweep small steps × {avg=1, avg=2} on irt_1pl to find the
acc~0.9 step and confirm avg=2 is fine (and beats MALA) there. If yes -> the d=500
issue is the STEP tuner, orthogonal to the avg=2 trajectory fix.
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
from gt_imm import gt_from_draws
from run_fixed_imm import _make_fixed_imm_adj_dyn_kernel

imm, gt_var, gt_mean, d = gt_from_draws("irt_1pl")
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model._registry import MODELS as _M

init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M["irt_1pl"])
_, unravel = ravel_pytree(init_dict)
ld = lambda xf: ld_raw(unravel(xf))  # noqa: E731
init = jnp.asarray(gt_mean, dtype=jnp.float64)
gm, gv = np.asarray(gt_mean), np.asarray(gt_var)
CH, BUDGET = 4, 8000
dyn = _make_fixed_imm_adj_dyn_kernel(imm)


def run(step, avg):
    n = max(int(BUDGET / (2 * avg)), 50)
    pos, accs = [], []
    for ci in range(CH):
        sk = jax.random.key(ci + 1)
        s = adj_dyn_mod.init(init, ld, sk)

        def stp(c, k):
            nx, info = dyn(
                rng_key=k,
                state=c,
                logdensity_fn=ld,
                step_size=step,
                L_proposal_factor=jnp.inf,
                inverse_mass_matrix=imm,
                integration_steps_params=(float(avg),),
            )
            return nx, (nx.position, info.acceptance_rate)

        _, (pt, ac) = jax.lax.scan(stp, s, jax.random.split(sk, n))
        pos.append(np.array(jax.vmap(lambda q: ravel_pytree(q)[0])(pt)))
        accs.append(np.array(ac))
    arr = np.stack(pos, 0)
    vm = np.mean((arr - gm[None, None, :]) ** 2, axis=(0, 1))
    bias = float((np.abs(vm - gv) / np.maximum(gv, 1e-30)).max())
    ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], arr)})
    ess = float(np.array(az.ess(ds, method="bulk")["x"]).min())
    rhat = float(np.array(az.rhat(ds, method="rank")["x"]).max())
    return n, float(np.mean(np.concatenate(accs))), bias, ess, rhat


print(f"irt_1pl (d={d}) | manual step sweep, GT dense IMM, budget={BUDGET}/chain\n")
print(
    f"{'step':>5s} {'avg':>4s} {'N':>5s} {'acc':>6s} {'bias':>7s} {'minESS':>8s} {'Rhat':>6s}  verdict"
)
for step in [1.0, 2.0, 4.0, 8.0]:
    for avg in [1, 2]:
        n, acc, bias, ess, rhat = run(step, avg)
        ok = bias < 0.1 and rhat < 1.01 and ess > 100 * CH
        v = (
            "PASS"
            if ok
            else ("loud-fail" if (rhat > 1.01 or acc < 0.1) else "SILENT-FAIL")
        )
        print(
            f"{step:>5.1f} {avg:>4d} {n:>5d} {acc:>6.3f} {bias:>7.3f} {ess:>8.0f} {rhat:>6.3f}  {v}"
        )
    sys.stdout.flush()
print("\nDONE")
