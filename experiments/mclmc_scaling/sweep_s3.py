"""S3 (isotropic-Gaussian dimension sweep): the a/b scaling law vs d.

At the matched DENSE GT IMM (k=d), whitening maps any Gaussian to N(0,I)_d, so
the matched-geometry a,b for ill_cond_50 == the isotropic problem at d=50. This
driver runs MCLMC (EEVPD-tuned) + NUTS (window_adaptation) on N(0,I)_d across d
and reports:
    a = step_mclmc / eps_nuts                 (step-size ratio)
    b = L_mclmc / (L_nuts * eps_nuts) = L_mclmc / T_nuts   (length ratio)

Cross-check: the d=50 row should reproduce the ill_cond_50 anchor (a≈17.9,
b≈1.74), confirming matched-dense ill_cond_50 ≡ isotropic d=50.

  uv run python sweep_s3.py --smoke
  uv run python sweep_s3.py
"""

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import blackjax
import blackjax.mcmc.mclmc as mclmc_mod
from blackjax.adaptation.mclmc_adaptation import mclmc_find_L_and_step_size

SMOKE = "--smoke" in sys.argv
DS = [10, 50] if SMOKE else [10, 30, 50, 100, 200, 500]
NW = 200 if SMOKE else 2000
NS = 500 if SMOKE else 3000
SEED = 20260616


def logdensity(x):  # N(0, I_d)
    return -0.5 * jnp.dot(x, x)


def run_nuts(d, key):
    init = jnp.zeros(d, dtype=jnp.float64)
    k1, k2 = jax.random.split(key)
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, params), _ = warmup.run(k1, init, num_steps=NW)
    nuts = blackjax.nuts(logdensity, **params)
    keys = jax.random.split(k2, NS)

    def step(s, kk):
        ns, info = nuts.step(kk, s)
        return ns, info.num_integration_steps

    _, nsteps = jax.lax.scan(step, state, keys)
    eps = float(params["step_size"])
    L_nuts = float(np.median(np.asarray(nsteps)))
    return eps, L_nuts


def run_mclmc(d, key):
    init = jnp.zeros(d, dtype=jnp.float64)
    k1, k2 = jax.random.split(key)
    state = mclmc_mod.init(init, logdensity, k1)
    kernel = mclmc_mod.build_kernel()
    _, params, _ = mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=NW,
        state=state,
        rng_key=k2,
        logdensity_fn=logdensity,
        diagonal_preconditioning=True,
    )
    return float(params.step_size), float(params.L)


rows = []
print(f"S3 isotropic | smoke={SMOKE} | NW={NW} NS={NS}")
print(
    f"{'d':>5s} {'mclmc_step':>11s} {'mclmc_L':>8s} {'nuts_eps':>9s} {'nuts_L':>7s} "
    f"{'T_nuts':>7s} {'a=step/eps':>11s} {'b=L/T':>7s}"
)
for d in DS:
    k_m, k_n = jax.random.split(jax.random.key(SEED + d))
    step_m, L_m = run_mclmc(d, k_m)
    eps_n, L_n = run_nuts(d, k_n)
    T_n = eps_n * L_n
    a = step_m / eps_n
    b = L_m / T_n
    rows.append(
        {
            "d": d,
            "mclmc_step": step_m,
            "mclmc_L": L_m,
            "nuts_eps": eps_n,
            "nuts_L": L_n,
            "T_nuts": T_n,
            "a": a,
            "b": b,
        }
    )
    print(
        f"{d:>5d} {step_m:>11.3f} {L_m:>8.3f} {eps_n:>9.4f} {L_n:>7.1f} "
        f"{T_n:>7.3f} {a:>11.2f} {b:>7.3f}"
    )
    sys.stdout.flush()

out = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "s3_iso_smoke.json" if SMOKE else "s3_iso.json",
)
with open(out, "w") as f:
    json.dump(rows, f, indent=2)
print(f"WROTE {out}")
