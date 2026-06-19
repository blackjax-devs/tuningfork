"""#25 vmap-vs-loop diagnosis, PHASE 1b — CORRECTED mechanism demo (banana).

Fixes two flaws in p1:
 (A1b) fp SEED: batched-grad over 4 DIFFERENT positions (vmap) vs scalar-grad per
       position. With identical broadcast inputs p1 trivially got 0; the real fp
       rounding only appears with distinct lanes. Report max|Δ| per lane.
 (A2)  REAL divergence curve from the ACTUAL sampler (bounded, no NaN): run sample_loop
       and sample_vmap, extract chain 0, per-sample max|Δ|_n. Do avg=1 (MALA-like, 1
       integration step/sample) vs avg=18 to show amplification + trajectory-length
       dependence. Fit log-slope on the pre-saturation region only.

Run: JAX_PLATFORM_NAME=cpu .venv/bin/python experiments/mclmc_scaling/diag_vmap_banana_p1b.py
"""

import json
import os
import subprocess
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(os.path.join(_HERE, "..", ".."))

import blackjax.mcmc.adjusted_mclmc_dynamic as adj_dyn_mod
from run_fixed_imm import _make_fixed_imm_adj_dyn_kernel

GIT_HEAD = (
    subprocess.check_output(
        ["git", "-C", "/home/jp/blackjax-devs/blackjax", "rev-parse", "HEAD"]
    )
    .decode()
    .strip()
)
print("git_head:", GIT_HEAD, "| x64:", jax.config.jax_enable_x64)

BANANA_VAR = np.array([8.0, 9.0])
CH = 4
STEP = 0.21126  # cert median


def load_banana():
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.model._registry import MODELS as _M

    init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M["banana"])
    _, unravel = ravel_pytree(init_dict)
    return (lambda xf: ld_raw(unravel(xf))), jnp.zeros(2), 2, jnp.asarray(BANANA_VAR)


ld, init, d, imm = load_banana()
gradfn = jax.grad(ld)


def _chain_keys(seed, ch):
    return [jax.random.key(seed * 1000 + ci + 1) for ci in range(ch)]


# ---------- A1b: batched-grad over DIFFERENT positions vs scalar ----------
print(
    "\n"
    + "=" * 78
    + "\nA1b: fp SEED — batched (vmap) grad over 4 DIFFERENT positions vs scalar-per-position\n"
    + "=" * 78
)
np.random.seed(1)
X = jnp.asarray(np.random.randn(CH, d) * np.sqrt(BANANA_VAR))  # 4 distinct positions
g_batched = jax.vmap(gradfn)(X)  # vectorized
g_scalar = jnp.stack([gradfn(X[i]) for i in range(CH)])  # scalar per lane
ld_batched = jax.vmap(ld)(X)
ld_scalar = jnp.stack([ld(X[i]) for i in range(CH)])
dg_lane = np.max(np.abs(np.asarray(g_batched - g_scalar)), axis=1)
dld_lane = np.abs(np.asarray(ld_batched - ld_scalar))
for i in range(CH):
    print(f"  lane {i}: |Δld| = {dld_lane[i]:.3e}   max|Δgrad| = {dg_lane[i]:.3e}")
fp_seed = float(max(dg_lane.max(), dld_lane.max()))
print(
    f"  => max fp discrepancy (batched vs scalar, distinct inputs) = {fp_seed:.3e}  (fp64 eps ~2.2e-16)"
)


# ---------- A2: REAL sampler chain-0 per-sample divergence curve ----------
def sample_full(step, avg, n, seed, vmap):
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    keys = _chain_keys(seed, CH)

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
            return nx, ravel_pytree(nx.position)[0]

        _, pt = jax.lax.scan(stp, s, jax.random.split(sk, n))
        return pt  # (n, d)

    if vmap:
        flats = jax.vmap(run_chain)(jnp.stack(keys))  # (ch, n, d)
        return np.asarray(flats)
    else:
        return np.stack([np.asarray(run_chain(k)) for k in keys], 0)


def curve(avg, n, seed):
    aL = sample_full(STEP, avg, n, seed, vmap=False)
    aV = sample_full(STEP, avg, n, seed, vmap=True)
    c0L, c0V = aL[0], aV[0]  # (n, d) chain 0
    delta = np.max(np.abs(c0L - c0V), axis=1)  # (n,)
    return delta


sat = float(np.sqrt(BANANA_VAR).max())
results = {}
for avg in [1, 18]:
    print(
        "\n"
        + "=" * 78
        + f"\nA2: REAL sampler chain-0 divergence curve | avg={avg}, seed=10, step={STEP}\n"
        + "=" * 78
    )
    delta = curve(avg, 300, 10)
    pts = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 299]
    for nn in pts:
        print(f"    sample {nn:>4d}: max|Δ| = {delta[nn]:.3e}")
    # log-slope on the growth region (delta in (0, 0.1*sat), finite)
    mask = np.isfinite(delta) & (delta > 0) & (delta < 0.1 * sat)
    ns = np.where(mask)[0]
    lam = None
    if len(ns) >= 5:
        A = np.vstack([ns, np.ones_like(ns)]).T
        lam, b = np.linalg.lstsq(A, np.log(delta[ns]), rcond=None)[0]
        sat_idx = (
            int(np.argmax(delta >= 0.5 * sat)) if np.any(delta >= 0.5 * sat) else None
        )
        print(
            f"  growth-region log-slope λ = {lam:.4f} /sample; "
            f"reaches 0.5·sat({0.5*sat:.2f}) at sample {sat_idx}; saturates ~{sat:.2f}"
        )
    else:
        print(
            f"  no sustained growth region (n_growth={len(ns)}) — delta stays ~{np.nanmax(delta):.2e} "
            f"(avg=1 expected to grow far slower / barely)"
        )
    results[str(avg)] = {
        "delta_at": {str(p): float(delta[p]) for p in pts},
        "lambda_per_sample": (float(lam) if lam is not None else None),
        "max_delta": float(np.nanmax(delta)),
    }

out = os.path.join(_HERE, "diag_vmap_banana_p1b_results.json")
with open(out, "w") as f:
    json.dump(
        {
            "git_head": GIT_HEAD,
            "fp_seed_batched_vs_scalar": fp_seed,
            "a1b_dgrad_lane": dg_lane.tolist(),
            "a1b_dld_lane": dld_lane.tolist(),
            "saturation": sat,
            "curves": results,
        },
        f,
        indent=2,
    )
print(f"\nwrote {out}")
print("DONE_DIAG_P1B")
