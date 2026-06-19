"""#29 backfill: max_abs_mean_z for banana medium__adjusted_mclmc_dynamic.

Computes per-dim mean bias in MCSE units:
    z_d = |mean_est_d - gm_d| / (sample_sd_d / sqrt(ess_d))
and files the worst-over-seeds, max-over-dims value (matching the recipe's
worst-case convention for min_bulk_ess / max_2mom_bias / max_mean_bias_sd).

PROTOCOL: each seed runs at ITS OWN EEVPD warmup step (the seed_steps below,
from cert25_banana_vmap_results.json) — the SAME protocol that produced the
recipe's filed min_ess 6191 / max_2mbias 0.1126 / max_mbias_sd 0.0382.  This is
deliberately NOT the single filed median step (0.21126): max_abs_mean_z must
sit on the same protocol as its sibling worst-case fields.  A pinned-median-step
variant is reported for the record but NOT filed.

Bit-faithful to cert25_banana_vmap: reproduces seed15 ESS=6191.263631501678 and
seed13 mbias_sd=0.038174... exactly.

Run: JAX_PLATFORM_NAME=cpu .venv/bin/python experiments/mclmc_scaling/meanz_probe.py
Filed value: max_abs_mean_z = 3.736365805232391 (seed13, dim1, own-step).
"""
import json, os, sys
import jax
jax.config.update("jax_enable_x64", True)
import arviz as az, jax.numpy as jnp, numpy as np, xarray as xr
from jax.flatten_util import ravel_pytree

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

import blackjax.mcmc.adjusted_mclmc_dynamic as adj_dyn_mod
from blackjax.mcmc.adjusted_mclmc_dynamic import make_random_trajectory_length_fn

AVG, N, CH = 24, 5000, 4
MEDIAN_STEP = 0.21125994255157865
SEED_STEPS = {10: 0.21315964960919145, 11: 0.21816698957090083,
              12: 0.19692653512480268, 13: 0.2220082827266748,
              14: 0.20936023549396587, 15: 0.11712712533301083}
SEEDS = [10, 11, 12, 13, 14, 15]
gm, gv = np.array([0.0, 2.0]), np.array([8.0, 9.0])

from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model._registry import MODELS as _M
init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M["banana"])
_, unravel = ravel_pytree(init_dict)
ld = lambda xf: ld_raw(unravel(xf))
init, imm = jnp.zeros(2), jnp.asarray(gv)
steps_fn = make_random_trajectory_length_fn(True)
base = adj_dyn_mod.build_kernel(integration_steps_fn=steps_fn)

def sample(step, seed):
    keys = jnp.stack([jax.random.key(seed * 1000 + ci + 1) for ci in range(CH)])
    def run_chain(sk):
        s = adj_dyn_mod.init(init, ld, sk)
        def stp(c, key):
            nx, info = base(rng_key=key, state=c, logdensity_fn=ld, step_size=step,
                            L_proposal_factor=jnp.inf, inverse_mass_matrix=imm,
                            integration_steps_params=(float(AVG),))
            return nx, (nx.position, info.is_divergent, info.acceptance_rate,
                        info.num_integration_steps)
        _, (pt, dv, ac, ns) = jax.lax.scan(stp, s, jax.random.split(sk, N))
        return jax.vmap(lambda q: ravel_pytree(q)[0])(pt), dv, ac, ns
    flats, dvs, acs, nss = jax.vmap(run_chain)(keys)
    arr = np.asarray(flats); flat = arr.reshape(-1, 2)
    mean_est, samp_sd = flat.mean(0), flat.std(0, ddof=1)
    ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], arr)})
    ess = np.array(az.ess(ds, method="bulk")["x"])
    z = np.abs(mean_est - gm) / (samp_sd / np.sqrt(ess))
    mbsd = np.abs(mean_est - gm) / np.sqrt(gv)
    return dict(seed=seed, step=step, realized_avg=float(np.mean(np.asarray(nss))),
                div_rate=float(np.mean(np.asarray(dvs))), mean_acc=float(np.mean(np.asarray(acs))),
                z_per_dim=z.tolist(), max_abs_mean_z=float(z.max()), argmax_dim=int(np.argmax(z)),
                min_bulk_ess=float(ess.min()), max_mbias_sd=float(mbsd.max()),
                ess_per_dim=ess.tolist(), mbias_sd_per_dim=mbsd.tolist())

def run(mode):
    cells = [sample(SEED_STEPS[s] if mode == "own" else MEDIAN_STEP, s) for s in SEEDS]
    zmax = max(c["max_abs_mean_z"] for c in cells)
    return dict(cells=cells, worst_over_seeds=zmax,
                worst_seed=max(cells, key=lambda c: c["max_abs_mean_z"])["seed"],
                median_over_seeds=float(np.median([c["max_abs_mean_z"] for c in cells])))

own, med = run("own"), run("median")
print(f"OWN-STEP (FILED):  worst={own['worst_over_seeds']!r} (seed{own['worst_seed']}) median={own['median_over_seeds']:.4f}")
print(f"MEDIAN-STEP (ref): worst={med['worst_over_seeds']:.4f} (seed{med['worst_seed']}) median={med['median_over_seeds']:.4f}")
out = os.path.join(_HERE, "mean_z_results.json")
json.dump({"metric": "max_abs_mean_z = max_d |mean_est_d - gm_d|/(sample_sd_d/sqrt(ess_d))",
           "config": {"L": 5.070238621237888, "imm": [8, 9], "avg": AVG, "seeds": SEEDS,
                      "n_samples": N, "ch": CH, "median_step": MEDIAN_STEP, "seed_steps": SEED_STEPS},
           "FILED_max_abs_mean_z": own["worst_over_seeds"], "filed_basis": "own_step worst-over-seeds (seed13,dim1)",
           "own_step": own, "median_step_reference": med}, open(out, "w"), indent=2)
print(f"wrote {out}")
print("DONE_MEANZ_PROBE")
